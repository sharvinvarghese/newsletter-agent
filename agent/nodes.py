"""Node implementations of the newsletter workflow.

Each method corresponds to exactly one LangGraph node and returns a plain dict
of state updates (see :func:`agent.state.node_update`). Nodes only ever touch
validated Pydantic objects; they never pass raw dictionaries between each other.

Prompting philosophy
--------------------
* The model gets *facts* (articles, summaries, deterministic checks) and is asked
  for a specific Pydantic schema - never for free-form JSON.
* Facts that can be verified in Python (URLs, attribution, recency, section
  counts) are computed here and in :mod:`agent.validation`, not by the model.
* When an LLM call fails, the node records the problem and uses a *logged*
  deterministic fallback instead of silently degrading.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from config.settings import Settings
from langgraph.types import interrupt
from llm.client import LLMUnavailableError, MissingAPIKeyError, redact_secrets
from llm.structured import StructuredOutputError
from renderers import (
    _write_json,
    build_context,
    dump_articles_excel,
    save_newsletter,
    write_run_artifact,
)
from schemas import (
    AgentMode,
    AgentPlan,
    AgentResult,
    ArticleEvaluation,
    ArticleEvaluationBatch,
    ArticleSummary,
    ArticleSummaryBatch,
    CritiqueResult,
    EvaluatedArticle,
    GoalSpec,
    HumanApproval,
    NewsArticle,
    Newsletter,
    NewsletterSection,
    ResearchStats,
)
from tools.deduplication import canonicalize_url
from tools.deduplication import deduplicate_articles as dedupe_articles
from tools.news_research import SourceConfigError, extract_terms, load_sources

from agent.deps import AgentDeps
from agent.routing import MAX_REVISIONS
from agent.state import AgentState, node_update
from agent.validation import (
    audit_newsletter,
    clamp,
    deterministic_evaluation,
    fallback_goal_spec,
    fallback_plan,
    sanitize_newsletter,
    select_articles,
)

logger = logging.getLogger("newsletter_agent.nodes")

#: Text budget per article while scoring candidates (evaluation is cheap and wide).
EVALUATION_TEXT_CHARS = 700

#: Errors that mean "the LLM path is unavailable" - always trigger a fallback.
_LLM_PROBLEMS = (StructuredOutputError, LLMUnavailableError, MissingAPIKeyError)


# ---------------------------------------------------------------------------
# Shared helpers (private; the nodes above this point are the public surface)
# ---------------------------------------------------------------------------

#: Editorial limits used in the prompts (mirrors the intent of tuning.yaml).
GEN_MIN_SECTIONS, GEN_MAX_SECTIONS = 3, 7
GEN_MIN_SENTENCES, GEN_MAX_SENTENCES = 2, 6
GEN_MIN_SECTION_SENTENCES, GEN_MAX_SECTION_SENTENCES = 4, 8
SUM_MIN_SENTENCES, SUM_MAX_SENTENCES = 3, 6
SUM_MIN_POINTS, SUM_MAX_POINTS = 2, 5
MIN_RELEVANCE = 0.35


def _as_json(value: Any) -> str:
    """Readable JSON for prompt payloads (Pydantic models, lists, plain data)."""
    from pydantic import BaseModel

    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    if isinstance(value, (list, tuple)):
        items = [item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in value]
        return json.dumps(items, indent=2, default=str)
    if isinstance(value, dict):
        return json.dumps(value, indent=2, default=str)
    return str(value)


def _digest(articles: Sequence[NewsArticle], *, excerpt_chars: int = 500) -> list[dict[str, str]]:
    """Small per-article fact sheet for the evaluation prompt."""
    return [
        {
            "title": article.title,
            "url": article.url,
            "source": article.source,
            "published_at": (article.published_at or "")[:10],
            "excerpt": article.content_for_llm(excerpt_chars),
        }
        for article in articles
    ]


def _fallback_summary(article: NewsArticle) -> ArticleSummary:
    """Deterministic summary used when the summarization call fails."""
    text = (article.content_for_llm(700) or article.description or "").strip()
    if not text:
        text = f"{article.title} - the source feed did not include the full text."
    sentence = text[:240]
    cut = sentence.rfind(". ")
    if cut > 40:
        sentence = sentence[: cut + 1]
    summary = f"{article.title}. {sentence}".strip()
    if len(summary) < 60:
        summary = f"{summary} Open the article for the full story."
    point = (sentence or article.title)[:140].strip() or article.title
    return ArticleSummary(
        title=article.title,
        source=article.source,
        url=article.url,
        summary=summary,
        key_points=[point],
        why_it_matters=(
            f"Reported by {article.source or 'the source'}; open the article to judge "
            "the impact for your readers."
        ),
    )


def _invoke_prompt(
    deps: AgentDeps,
    schema: type[Any],
    prompt_name: str,
    values: dict[str, Any],
) -> tuple[Any | None, str]:
    """Render a named prompt, call the LLM; returns ``(model, error)``."""
    from llm.prompts import PromptError, get_prompts

    try:
        messages = get_prompts().render(prompt_name, **values).as_messages()
    except PromptError as exc:
        return None, f"prompt '{prompt_name}': {exc}"
    try:
        return deps.llm.invoke(schema, messages), ""
    except _LLM_PROBLEMS as exc:
        return None, redact_secrets(exc)
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"{type(exc).__name__}: {redact_secrets(exc)}"


def _requested_articles(state: AgentState, deps: AgentDeps) -> int:
    """Target article count from the goal spec, clamped to a sane range."""
    spec = state.get("goal_spec")
    raw = getattr(spec, "requested_article_count", None) or deps.settings.target_articles
    return clamp(raw, 1, 12)


def parse_goal(state: AgentState, *, deps: AgentDeps) -> dict[str, Any]:
    """Interpret the plain-English goal (LLM first, rule-based parser as fallback)."""
    goal = (state.get("goal") or "").strip()
    resolved_mode = state.get("mode") or AgentMode.FULLY_AUTONOMOUS
    settings = deps.settings
    values = {
        "goal": goal,
        "articles": settings.target_articles,
        "min_articles": GEN_MIN_SECTIONS,
        "max_articles": GEN_MAX_SECTIONS,
        "hard_max_articles": 12,
        "default_frequency": "weekly",
        "frequencies": "daily, weekly, biweekly, monthly",
        "default_format": "html+markdown",
        "output_formats": "html+markdown, html, markdown",
        "mode": getattr(resolved_mode, "value", str(resolved_mode)),
    }
    spec, error = _invoke_prompt(deps, GoalSpec, "goal_parsing", values)
    usage = getattr(deps.llm, "last_token_usage", {}) or {}
    input_tokens = int(usage.get("input", 0))
    output_tokens = int(usage.get("output", 0))
    tokens = [f"input_tokens={input_tokens}", f"output_tokens={output_tokens}"]
    if spec is None:
        spec = fallback_goal_spec(
            goal,
            requested_articles=state.get("target_articles") or settings.target_articles,
            mode=resolved_mode,
        )
        log = [f"Goal parsing used the deterministic fallback ({error or 'LLM unavailable'})."]
        errors = [f"goal_parsing: {error}"] if error else []
    else:
        log = [f"Goal parsed: topic='{spec.topic}', {spec.requested_article_count} article(s), {spec.frequency}."]
        errors = []
    if state.get("target_articles"):  # UI override wins over both LLM and heuristics
        spec = spec.model_copy(update={"requested_article_count": clamp(state["target_articles"], 1, 12)})
    return node_update(state, goal_spec=spec, log=log, errors=errors, _progress_tokens=tokens)


def plan_research(state: AgentState, *, deps: AgentDeps) -> dict[str, Any]:
    """Turn the goal spec into queries and source hints (LLM first, rules second)."""
    spec = state.get("goal_spec")
    if spec is None:  # parse_goal always sets this; stay defensive
        spec = fallback_goal_spec(state.get("goal", ""), mode=state.get("mode") or AgentMode.FULLY_AUTONOMOUS)
    settings = deps.settings
    try:
        enabled_ids = [source.id for source in load_sources().sources if getattr(source, "enabled", True)]
    except SourceConfigError as exc:
        enabled_ids = []
        logger.error("Source configuration unavailable: %s", exc)
    values = {
        "goal_spec": _as_json(spec),
        "sources": _as_json(enabled_ids),
        "collect_ceiling": 100,
        "min_queries": 2,
        "max_queries": 6,
        "min_collect": 10,
        "max_collect": 48,
        "min_articles": GEN_MIN_SECTIONS,
        "max_articles": GEN_MAX_SECTIONS,
        "min_requirements": 2,
        "max_requirements": 6,
    }
    plan, error = _invoke_prompt(deps, AgentPlan, "planning", values)
    usage = getattr(deps.llm, "last_token_usage", {}) or {}
    tokens = [f"input_tokens={int(usage.get('input', 0))}", f"output_tokens={int(usage.get('output', 0))}"]
    if plan is None:
        plan = fallback_plan(
            spec,
            available_sources=enabled_ids,
            max_articles_to_collect=settings.max_articles_to_collect,
        )
        log = [f"Planning used the deterministic fallback ({error or 'LLM unavailable'})."]
        errors = [f"planning: {error}"] if error else []
    else:
        log = [f"Plan ready: {len(plan.research_queries)} quer(y/ies), {len(plan.sources)} source hint(s)."]
        errors = []
    if state.get("target_articles"):
        plan = plan.model_copy(update={"target_articles": clamp(state["target_articles"], 1, 12)})
    if state.get("max_articles_to_collect"):
        plan = plan.model_copy(update={"max_articles_to_collect": clamp(state["max_articles_to_collect"], 5, 100)})
    return node_update(state, plan=plan, search_queries=list(plan.research_queries), log=log, errors=errors, _progress_tokens=tokens)

def research(state: AgentState, *, deps: AgentDeps) -> dict[str, Any]:
    """Collect candidates from the planned sources (RSS, sitemaps, Hacker News)."""
    plan = state.get("plan")
    queries = list(state.get("search_queries") or (plan.research_queries if plan else []))
    window = state.get("recency_window_days") or deps.settings.recency_window_days
    try:
        outcome = deps.research.research(
            queries=queries,
            sources=((plan.sources if plan else None) or None),
            max_articles=(plan.max_articles_to_collect if plan else None),
            recency_window_days=int(window) if window else None,
            enrich=None,
        )
    except Exception as exc:  # one broken tool call must never kill the run
        message = f"{type(exc).__name__}: {redact_secrets(exc)}"
        return node_update(
            state,
            raw_articles=[],
            candidate_articles=[],
            research_stats=ResearchStats(notes=[f"research failed: {message}"]),
            log=["Research failed; continuing without candidates."],
            errors=[f"research: {message}"],
        )
    deduped = dedupe_articles(outcome.articles)
    log = [
        (f"Research: {len(outcome.articles)} raw article(s) from "
        f"{outcome.stats.sources_succeeded}/{outcome.stats.sources_attempted} source(s)."),
        f"After de-duplication: {len(deduped.kept_articles)} candidate(s).",
    ]
    if outcome.stats.failed_sources:
        log.append("Failed source(s): " + ", ".join(outcome.stats.failed_sources[:6]))
    return node_update(
        state,
        raw_articles=outcome.articles,
        candidate_articles=deduped.kept_articles,
        research_stats=outcome.stats,
        log=log,
        _progress_tokens=[
            f"raw_articles={len(outcome.articles)}",
            f"articles={len(deduped.kept_articles)}",
        ],
    )


def evaluate(state: AgentState, *, deps: AgentDeps) -> dict[str, Any]:
    """Score every candidate in small LLM batches; failures fall back per article."""
    candidates = list(state.get("candidate_articles") or [])
    if not candidates:
        return node_update(state, evaluated_articles=[], log=["No candidates to evaluate."])
    spec = state.get("goal_spec") or fallback_goal_spec(state.get("goal", ""))
    plan = state.get("plan") or fallback_plan(spec)
    settings = deps.settings
    window = int(state.get("recency_window_days") or settings.recency_window_days)
    target = _requested_articles(state, deps)
    terms = extract_terms(list(state.get("search_queries") or plan.research_queries))
    batch_size = max(1, int(getattr(settings, "evaluation_batch_size", 6)))

    evaluated: list[EvaluatedArticle] = []
    errors: list[str] = []
    fallback_count = 0
    input_tokens = 0
    output_tokens = 0
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        values = {
            "goal": state.get("goal", ""),
            "audience": getattr(spec, "audience", None) or "a technical audience",
            "frequency": getattr(spec, "frequency", "weekly"),
            "count": len(batch),
            "candidates": _as_json(_digest(batch, excerpt_chars=EVALUATION_TEXT_CHARS)),
            "min_score": MIN_RELEVANCE,
            "max_score": 1.0,
            "min_relevance": MIN_RELEVANCE,
            "max_selected": target,
            "requirements": _as_json(plan.newsletter_requirements),
        }
        parsed, error = _invoke_prompt(deps, ArticleEvaluationBatch, "evaluation", values)
        usage = getattr(deps.llm, "last_token_usage", {}) or {}
        input_tokens += int(usage.get("input", 0))
        output_tokens += int(usage.get("output", 0))
        if parsed is None:
            errors.append(f"evaluation batch {start // batch_size + 1}: {error}")
            fallback_count += len(batch)
            evaluated.extend(
                deterministic_evaluation(article, terms=terms, window_days=window) for article in batch
            )
            continue
        by_url = {canonicalize_url(article.url): article for article in batch}
        seen: set[str] = set()
        for item in parsed.evaluations:
            key = canonicalize_url(item.url)
            if key not in by_url or key in seen:
                continue
            seen.add(key)
            evaluation = ArticleEvaluation.model_validate(item.model_dump(exclude={"url"}))
            evaluated.append(EvaluatedArticle(article=by_url[key], evaluation=evaluation))
        missing = [article for article in batch if canonicalize_url(article.url) not in seen]
        if missing:
            fallback_count += len(missing)
            evaluated.extend(
                deterministic_evaluation(article, terms=terms, window_days=window) for article in missing
            )

    log = [
        f"Evaluated {len(evaluated)} candidate(s); {fallback_count} used the deterministic fallback.",
    ]
    return node_update(
        state,
        evaluated_articles=evaluated,
        log=log,
        errors=errors,
        _progress_tokens=[f"input_tokens={input_tokens}", f"output_tokens={output_tokens}"],
    )


def select(state: AgentState, *, deps: AgentDeps) -> dict[str, Any]:
    """Pick the strongest non-duplicate articles up to the requested count."""
    evaluated = list(state.get("evaluated_articles") or [])
    target = _requested_articles(state, deps)
    chosen = select_articles(evaluated, target=target, min_relevance=MIN_RELEVANCE)
    selected = [item.article for item in chosen]
    return node_update(
        state,
        selected_articles=selected,
        log=[f"Selected {len(selected)} of {len(evaluated)} candidate(s) (target was {target})."],
    )

def _fallback_newsletter(summaries: Sequence[ArticleSummary], spec: GoalSpec | None) -> Newsletter:
    """Deterministic newsletter used when the generation call failed."""
    topic = getattr(spec, "topic", "technology") or "technology"
    frequency = getattr(spec, "frequency", "weekly") or "weekly"
    sections = [
        NewsletterSection(
            headline=summary.title,
            article_url=summary.url,
            source=summary.source,
            summary=summary.summary,
            why_it_matters=summary.why_it_matters,
        )
        for summary in summaries
    ]
    return Newsletter(
        subject=f"{topic}: this {frequency}'s essentials",
        preheader=f"{len(sections)} story(ies) on {topic}, summarised from this {frequency}'s research.",
        introduction=(
            f"This {frequency} edition covers {len(sections)} development(s) around {topic}. "
            "Each section summarises a single source and explains why it matters."
        ),
        sections=sections,
        conclusion="Open the linked sources for the full stories and judge the impact for your own work.",
    )


def summarize(state: AgentState, *, deps: AgentDeps) -> dict[str, Any]:
    """Produce one validated summary per selected article.

    All articles are sent to the LLM in a single batched call to conserve API
    budget. If the batch call fails, each article is retried individually so a
    single stubborn article never blocks the whole newsletter.
    """
    selected = list(state.get("selected_articles") or [])
    if not selected:
        return node_update(state, summaries=[], log=["No articles were selected; skipping summarization."])
    spec = state.get("goal_spec")
    settings = deps.settings
    audience = getattr(spec, "audience", None) or "a technical audience"
    goal = state.get("goal", "")

    summaries, input_tokens, output_tokens, fallback_count = _batch_summarize(
        selected, deps, goal=goal, audience=audience, settings=settings
    )
    log = [
        (f"Summarized {len(summaries)} article(s) in {len(selected) // max(1, getattr(settings, 'summarization_batch_size', 8)) + 1} batched call(s); "
        f"{fallback_count} used the deterministic fallback. "
        f"~{input_tokens + output_tokens} tokens used ({input_tokens} in, {output_tokens} out)."),
    ]
    return node_update(
        state,
        summaries=summaries,
        log=log,
        errors=[],
        _progress_tokens=[f"input_tokens={input_tokens}", f"output_tokens={output_tokens}"],
    )


def _batch_summarize(
    articles: Sequence[NewsArticle],
    deps: AgentDeps,
    *,
    goal: str,
    audience: str,
    settings: Settings,
) -> tuple[list[ArticleSummary], int, int, int]:
    """Single batched LLM call; falls back to per-article on failure.

    Returns ``(summaries, input_tokens, output_tokens, fallback_count)``.
    """
    if not articles:
        return [], 0, 0, 0

    batch_size = max(1, int(getattr(settings, "summarization_batch_size", 8)))
    summaries: list[ArticleSummary] = []
    input_tokens = 0
    output_tokens = 0
    fallback_count = 0

    for start in range(0, len(articles), batch_size):
        batch = list(articles[start : start + batch_size])
        values = {
            "goal": goal,
            "audience": audience,
            "count": len(batch),
            "articles": _as_json(
                [
                    {
                        "title": article.title,
                        "url": str(article.url),
                        "source": article.source,
                        "author": article.author or "",
                        "published_at": (article.published_at or "")[:10],
                        "text": article.content_for_llm(settings.max_content_chars),
                    }
                    for article in batch
                ]
            ),
            "min_sentences": SUM_MIN_SENTENCES,
            "max_sentences": SUM_MAX_SENTENCES,
            "min_points": SUM_MIN_POINTS,
            "max_points": SUM_MAX_POINTS,
        }
        parsed, error = _invoke_prompt(deps, ArticleSummaryBatch, "batch_summarization", values)
        usage = getattr(deps.llm, "last_token_usage", {}) or {}
        input_tokens += int(usage.get("input", 0))
        output_tokens += int(usage.get("output", 0))
        if parsed is not None and len(parsed.summaries) == len(batch):
            summaries.extend(parsed.summaries)
            continue

        # Batch failed or returned wrong count: fall back to per-article calls
        if error:
            batch_results, batch_in, batch_out, batch_fallbacks = _per_article_summaries(
                batch, deps, goal, audience, settings, error=error
            )
        else:
            batch_results, batch_in, batch_out, batch_fallbacks = _per_article_summaries(
                batch, deps, goal, audience, settings
            )
        summaries.extend(batch_results)
        input_tokens += batch_in
        output_tokens += batch_out
        fallback_count += batch_fallbacks

    return summaries, input_tokens, output_tokens, fallback_count


def _per_article_summaries(
    articles: Sequence[NewsArticle],
    deps: AgentDeps,
    goal: str,
    audience: str,
    settings: Settings,
    *,
    error: str = "batch result mismatched count",
) -> tuple[list[ArticleSummary], int, int, int]:
    """Generate summaries one-by-one (used only when the batch call failed).

    Returns ``(summaries, input_tokens, output_tokens, fallback_count)``.
    """
    results: list[ArticleSummary] = []
    input_tokens = 0
    output_tokens = 0
    fallback_count = 0
    errors: list[str] = []
    for article in articles:
        values = {
            "title": article.title,
            "url": str(article.url),
            "source": article.source,
            "author": article.author or "",
            "published_at": (article.published_at or "")[:10],
            "text": article.content_for_llm(settings.max_content_chars),
            "audience": audience,
            "goal": goal,
            "min_sentences": SUM_MIN_SENTENCES,
            "max_sentences": SUM_MAX_SENTENCES,
            "min_points": SUM_MIN_POINTS,
            "max_points": SUM_MAX_POINTS,
        }
        summary, summary_error = _invoke_prompt(deps, ArticleSummary, "summarization", values)
        usage = getattr(deps.llm, "last_token_usage", {}) or {}
        input_tokens += int(usage.get("input", 0))
        output_tokens += int(usage.get("output", 0))
        if summary is None:
            errors.append(f"summarization '{article.title[:60]}': {summary_error}")
            summary = _fallback_summary(article)
            fallback_count += 1
        results.append(summary)
    if errors:
        logger.warning("Per-article fallback used for %d article(s): %s", len(errors), error)
    return results, input_tokens, output_tokens, fallback_count


def generate(state: AgentState, *, deps: AgentDeps) -> dict[str, Any]:
    """Draft the newsletter from the validated summaries, then enforce the hard rules."""
    summaries = list(state.get("summaries") or [])
    if not summaries:
        return node_update(state, newsletter=None, log=["Generation skipped: no summaries available."])
    spec = state.get("goal_spec") or fallback_goal_spec(state.get("goal", ""))
    plan = state.get("plan")
    target = _requested_articles(state, deps)
    values = {
        "goal": state.get("goal", ""),
        "topic": getattr(spec, "topic", "technology news"),
        "audience": getattr(spec, "audience", None) or "a technical audience",
        "frequency": getattr(spec, "frequency", "weekly"),
        "requirements": _as_json(plan.newsletter_requirements if plan else []),
        "min_sections": min(GEN_MIN_SECTIONS, len(summaries)),
        "max_sections": max(1, min(GEN_MAX_SECTIONS, target)),
        "min_sentences": GEN_MIN_SENTENCES,
        "max_sentences": GEN_MAX_SENTENCES,
        "min_section_sentences": GEN_MIN_SECTION_SENTENCES,
        "max_section_sentences": GEN_MAX_SECTION_SENTENCES,
        "summaries": _as_json(summaries),
    }
    newsletter, error = _invoke_prompt(deps, Newsletter, "generation", values)
    usage = getattr(deps.llm, "last_token_usage", {}) or {}
    tokens = [f"input_tokens={int(usage.get('input', 0))}", f"output_tokens={int(usage.get('output', 0))}"]
    if newsletter is None:
        errors = [f"generation: {error}"] if error else []
        try:
            newsletter = _fallback_newsletter(summaries, spec)
        except Exception as exc:  # defensive: even the fallback must respect the schema
            logger.exception("Deterministic newsletter fallback failed")
            return node_update(
                state,
                newsletter=None,
                log=["Generation failed and the fallback newsletter was invalid."],
                errors=[*errors, f"generation fallback: {type(exc).__name__}: {exc}"],
            )
        log = [f"Generation used the deterministic fallback ({error or 'LLM unavailable'})."]
    else:
        errors = []
        log = [f"Drafted '{newsletter.subject}' with {newsletter.section_count} section(s)."]
    cleaned, issues = sanitize_newsletter(newsletter, summaries, max_sections=target)
    if issues:
        log.extend(issues)
    return node_update(state, newsletter=cleaned, log=log, errors=[*errors, *issues], _progress_tokens=tokens)

def critique(state: AgentState, *, deps: AgentDeps) -> dict[str, Any]:
    """Independent review; the deterministic audit can veto the LLM verdict."""
    newsletter = state.get("newsletter")
    if newsletter is None:
        return node_update(state, critique=None, log=["Nothing to critique (generation failed)."])
    summaries = list(state.get("summaries") or [])
    audit = audit_newsletter(newsletter, summaries)
    plan = state.get("plan")
    values = {
        "goal": state.get("goal", ""),
        "requirements": _as_json(plan.newsletter_requirements if plan else []),
        "min_sections": min(GEN_MIN_SECTIONS, max(1, len(summaries))),
        "summaries": _as_json(summaries),
        "draft": _as_json(newsletter),
        "audit": _as_json(audit) if audit else "No deterministic issues found.",
    }
    result, error = _invoke_prompt(deps, CritiqueResult, "critique", values)
    usage = getattr(deps.llm, "last_token_usage", {}) or {}
    tokens = [f"input_tokens={int(usage.get('input', 0))}", f"output_tokens={int(usage.get('output', 0))}"]
    if result is None:
        result = CritiqueResult(approved=not audit, improvement_instructions=list(audit))
        log = [f"Critique used the deterministic fallback ({error or 'LLM unavailable'}): {len(audit)} audit issue(s)."]
        errors = [f"critique: {error}"] if error else []
    else:
        log = [f"Critic verdict: {'approved' if result.approved else 'changes requested'} ({result.issue_count} issue(s))."]
        errors = []
    if audit:
        if result.approved:
            result = result.model_copy(update={"approved": False})
            log.append("The deterministic audit vetoed the critic's approval.")
        result = result.model_copy(update={"factual_issues": [*result.factual_issues, *audit]})
    return node_update(
        state,
        critique=result,
        critique_history=[*state.get("critique_history", []), result],
        log=log,
        errors=errors,
        _progress_tokens=tokens,
    )


def revise_newsletter(state: AgentState, *, deps: AgentDeps) -> dict[str, Any]:
    """Rewrite the draft using the critic's (and any human reviewer's) feedback."""
    draft = state.get("newsletter")
    critique = state.get("critique")
    summaries = list(state.get("summaries") or [])
    if draft is None or critique is None:
        return node_update(state, log=["Revision skipped: nothing to revise."])
    plan = state.get("plan")
    revision = int(state.get("revision_count") or 0) + 1
    limit = int(state.get("max_revisions") or MAX_REVISIONS)
    target = _requested_articles(state, deps)
    values = {
        "goal": state.get("goal", ""),
        "requirements": _as_json(plan.newsletter_requirements if plan else []),
        "draft": _as_json(draft),
        "summaries": _as_json(summaries),
        "critique": _as_json(critique),
        "feedback": state.get("human_feedback") or "",
        "revision": revision,
        "max_revisions": limit,
        "min_sections": min(GEN_MIN_SECTIONS, max(1, len(summaries))),
        "max_sections": max(1, min(GEN_MAX_SECTIONS, target)),
        "min_section_sentences": GEN_MIN_SECTION_SENTENCES,
        "max_section_sentences": GEN_MAX_SECTION_SENTENCES,
    }
    newsletter, error = _invoke_prompt(deps, Newsletter, "revision", values)
    usage = getattr(deps.llm, "last_token_usage", {}) or {}
    tokens = [f"input_tokens={int(usage.get('input', 0))}", f"output_tokens={int(usage.get('output', 0))}"]
    if newsletter is None:
        newsletter = draft  # keep the previous draft; the budget is spent regardless
        log = [f"Revision {revision} failed ({error or 'LLM unavailable'}); keeping the previous draft."]
        errors = [f"revision {revision}: {error}"] if error else []
    else:
        log = [f"Revision {revision} produced '{newsletter.subject}' ({newsletter.section_count} section(s))."]
        errors = []
    cleaned, issues = sanitize_newsletter(newsletter, summaries, max_sections=target)
    if issues:
        log.extend(issues)
    return node_update(state, newsletter=cleaned, revision_count=revision, log=log, errors=[*errors, *issues], _progress_tokens=tokens)


def _save_context(state: AgentState, deps: AgentDeps, newsletter: Newsletter) -> dict[str, Any]:
    """Template variables for the final render (every value validated run data)."""
    return build_context(
        newsletter,
        goal=state.get("goal", ""),
        goal_spec=state.get("goal_spec"),
        plan=state.get("plan"),
        summaries=list(state.get("summaries") or []),
        selected_articles=list(state.get("selected_articles") or []),
        candidate_count=len(state.get("raw_articles") or []),
        critique=state.get("critique"),
        critique_history=list(state.get("critique_history") or []),
        revision_count=int(state.get("revision_count") or 0),
        mode=state.get("mode"),
        research_stats=state.get("research_stats"),
        model_name=deps.settings.openrouter_model,
        log_lines=list(state.get("execution_log") or []),
    )


def human_review(state: AgentState, *, deps: AgentDeps) -> dict[str, Any]:
    """Publish gate: a real ``interrupt`` in human-in-the-loop mode, auto-approval otherwise."""
    newsletter = state.get("newsletter")
    mode = state.get("mode") or AgentMode.FULLY_AUTONOMOUS
    if mode is AgentMode.HUMAN_IN_LOOP and newsletter is not None:
        payload = {
            "subject": newsletter.subject,
            "section_count": newsletter.section_count,
            "question": "Approve this newsletter for publication?",
        }
        try:
            decision = interrupt(payload)
        except Exception as exc:  # no checkpointer configured -> degrade gracefully
            logger.warning("interrupt() unavailable (%s); auto-approving.", exc)
            decision = {"approved": True, "feedback": None, "reviewer": "auto (no checkpointer)"}
        if isinstance(decision, dict):
            approval = HumanApproval.model_validate(decision)
        else:
            approval = HumanApproval(approved=bool(decision))
        log = [f"Human review: {approval.action} ({approval.reviewer})."]
    else:
        approval = HumanApproval(approved=True, reviewer="autonomous_mode")
        log = ["Autonomous mode: auto-approved."]
    return node_update(state, human_approval=approval, human_feedback=approval.feedback, log=log)

def save_output(state: AgentState, *, deps: AgentDeps) -> dict[str, Any]:
    """Render, persist and finalise the run result."""
    newsletter = state.get("newsletter")
    spec = state.get("goal_spec")
    plan = state.get("plan")
    summaries = list(state.get("summaries") or [])
    selected = list(state.get("selected_articles") or [])
    mode = state.get("mode") or AgentMode.FULLY_AUTONOMOUS
    log_lines = list(state.get("execution_log") or [])
    errors = list(state.get("errors") or [])
    save_intermediate = state.get("save_intermediate", True)

    # Each run gets its own dated folder: outputs/YYYYMMDD-HHMMSS-<runid>/
    import uuid as _uuid
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    run_stamp = _dt.now(_tz.utc).strftime("%Y%m%d-%H%M%S")
    run_id = _uuid.uuid4().hex[:8]
    run_folder = f"{run_stamp}-{run_id}"
    base_dir = deps.settings.outputs_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_dir / run_folder
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = deps.settings.with_overrides(outputs_dir=run_dir)

    html_content = markdown_content = html_path = markdown_path = None
    articles_xlsx = None
    summaries_path = None
    if newsletter is not None:
        try:
            artifacts = save_newsletter(
                newsletter,
                context=_save_context(state, deps, newsletter),
                output_format=getattr(spec, "output_format", "html+markdown"),
                settings=settings,
            )
        except Exception as exc:
            errors.append(f"save_output: rendering failed: {type(exc).__name__}: {redact_secrets(exc)}")
            log_lines.append(errors[-1])
        else:
            html_content = artifacts.html_content
            markdown_content = artifacts.markdown_content
            html_path = artifacts.html_path
            markdown_path = artifacts.markdown_path
        if save_intermediate:
            try:
                articles_xlsx = dump_articles_excel(
                    list(state.get("evaluated_articles") or []),
                    settings=settings,
                    stem=Path(artifacts.html_path or artifacts.markdown_path).stem,
                )
            except Exception as exc:
                articles_xlsx = f"# dump failed ({type(exc).__name__}: {redact_secrets(exc)})"
            try:
                import json as _json
                summaries_path = _write_json(
                    Path(artifacts.html_path or artifacts.markdown_path).parent / (
                        Path(artifacts.html_path or artifacts.markdown_path).stem + "-summaries"
                    ),
                    _json.dumps(
                        [s.model_dump(mode="json") for s in (summaries or [])],
                        indent=2,
                    ),
                )
            except Exception as exc:
                summaries_path = f"# dump failed ({type(exc).__name__}: {redact_secrets(exc)})"
        log_lines.append(f"Saved newsletter artifacts to {run_dir}.")

    success = newsletter is not None and bool(html_content or markdown_content)
    result = AgentResult(
        success=success,
        goal=state.get("goal", ""),
        selected_articles=selected,
        summaries=summaries,
        newsletter=newsletter,
        html_path=html_path,
        markdown_path=markdown_path,
        critique=state.get("critique"),
        execution_log=log_lines,
        error=None if success else ("; ".join(errors[-3:]) or "The run did not complete."),
        mode=mode,
        goal_spec=spec,
        plan=plan,
        research_articles=list(state.get("raw_articles") or []),
        evaluated_articles=list(state.get("evaluated_articles") or []),
        critique_history=list(state.get("critique_history") or []),
        revision_count=int(state.get("revision_count") or 0),
        awaiting_human_approval=False,
        html_content=html_content,
        markdown_content=markdown_content,
        articles_xlsx_path=articles_xlsx,
        summaries_json_path=summaries_path,
        run_json_path=None,
        run_id=run_id,
        llm_call_count=getattr(deps.llm, "call_count", 0),
    )
    if save_intermediate:
        try:
            run_json_path = write_run_artifact(result, settings=settings)
            result = result.model_copy(
                update={"run_json_path": run_json_path, "execution_log": [*log_lines, f"Run artifact written to {run_json_path}."]}
            )
        except Exception as exc:
            logger.exception("Run artifact could not be written")
            result = result.model_copy(
                update={"execution_log": [*log_lines, f"Run artifact could not be written: {type(exc).__name__}: {redact_secrets(exc)}."]}
            )
    return node_update(
        state,
        html_content=html_content,
        markdown_content=markdown_content,
        html_path=html_path,
        markdown_path=markdown_path,
        final_output=result,
    )

