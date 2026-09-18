"""Deterministic validation, scoring and fallback helpers.

Nothing in this module talks to an LLM. It either (a) pre-computes facts that
must never come from a model (recency, URL sanity, section/summary alignment) or
(b) provides a *logged* deterministic fallback so that a structured-output
failure degrades the run instead of producing nothing.

Every fallback is recorded in the execution log by the node that uses it - the
degradation is never silent.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from schemas import (
    AgentMode,
    AgentPlan,
    ArticleEvaluation,
    ArticleSummary,
    EvaluatedArticle,
    GoalSpec,
    NewsArticle,
    Newsletter,
    NewsletterSection,
)
from tools.deduplication import canonicalize_url

# Ordered heuristics for topic inference (first match wins).
_TOPIC_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("AI agents", ("agent", "agentic", "langgraph", "tool calling", "multi-agent", "mcp", "copilot")),
    ("LLM research", ("research", "paper", "benchmark", "reasoning", "arxiv", "study")),
    ("AI infrastructure", ("infrastructure", "gpu", "inference", "chip", "datacenter", "compute", "nvidia")),
    ("AI products", ("product", "launch", "app", "feature", "release", "assistant", "chatbot")),
    ("AI policy", ("policy", "regulation", "law", "eu", "government", "safety", "risk")),
    ("AI startups", ("startup", "funding", "investment", "series", "valuation", "acquisition")),
    ("open source AI", ("open source", "open-source", "llama", "weights", "hugging face")),
)

_GENERIC_GOAL_WORDS = {
    "create", "write", "generate", "make", "build", "curate", "summarize", "summarise",
    "newsletter", "weekly", "daily", "monthly", "please", "about", "with", "that", "latest",
    "articles", "stories", "issues", "issue", "report", "roundup", "professional",
}

_FREQUENCY_HINTS = (("daily", "daily"), ("weekly", "weekly"), ("monthly", "monthly"), ("biweekly", "biweekly"))


def clamp(value: Any, low: int, high: int) -> int:
    """Integer clamp that tolerates strings coming from a model."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = low
    return max(low, min(high, number))


def clamp_float(value: Any, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = low
    return max(low, min(high, number))


def infer_topic(goal: str) -> str:
    """Pick a short topic label from the goal text (deterministic fallback)."""
    text = (goal or "").lower()
    for topic, keywords in _TOPIC_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return topic
    words = [
        word
        for word in re.findall(r"[A-Za-z][A-Za-z0-9+#.-]{2,}", goal or "")
        if word.lower() not in _GENERIC_GOAL_WORDS
    ]
    return " ".join(words[:4]).title() or "technology news"


def fallback_goal_spec(
    goal: str,
    *,
    requested_articles: int = 6,
    mode: AgentMode | str = AgentMode.FULLY_AUTONOMOUS,
) -> GoalSpec:
    """Parse the goal without an LLM (only used when the LLM call failed)."""
    text = (goal or "").strip()
    if len(text) < 5:
        text = f"AI news roundup: {text}".strip()
    lower = text.lower()

    count_match = re.search(r"(\d{1,2})\s*(?:articles|stories|items|pieces)", lower)
    count = clamp(count_match.group(1), 1, 12) if count_match else clamp(requested_articles, 1, 12)

    frequency = "weekly"
    for hint, label in _FREQUENCY_HINTS:
        if hint in lower:
            frequency = label
            break

    audience_match = re.search(r"\bfor ([A-Za-z0-9 ,&/\-]{3,60})", lower)
    audience = audience_match.group(1).strip(" .") if audience_match else None

    if "markdown" in lower and "html" not in lower:
        output_format = "markdown"
    elif "html" in lower and "markdown" not in lower:
        output_format = "html"
    else:
        output_format = "html+markdown"

    delivery = AgentMode.from_value(mode)
    if any(hint in lower for hint in ("human in the loop", "human-in-the-loop", "approve", "approval", "review")):
        delivery = AgentMode.HUMAN_IN_LOOP

    return GoalSpec(
        goal=text,
        topic=infer_topic(text),
        frequency=frequency,
        audience=audience,
        requested_article_count=count,
        output_format=output_format,
        delivery_mode=delivery,
    )


def fallback_plan(
    goal_spec: GoalSpec,
    *,
    available_sources: Sequence[str] = (),
    max_articles_to_collect: int = 24,
) -> AgentPlan:
    """Plan without an LLM (only used when the planner call failed)."""
    topic = goal_spec.topic
    queries = [
        f"{topic} news this week",
        f"{topic} product announcements",
        f"{topic} research results",
        f"{topic} industry analysis",
    ]
    return AgentPlan(
        objective=(
            f"Produce a {goal_spec.frequency} newsletter about {topic} "
            f"for {goal_spec.audience or 'a technical audience'}."
        ),
        research_queries=queries,
        sources=list(available_sources)[:6],
        max_articles_to_collect=clamp(max_articles_to_collect, 5, 100),
        target_articles=clamp(goal_spec.requested_article_count, 1, 12),
        newsletter_requirements=[
            "Lead with the most consequential story.",
            "Summarise each story in 2-3 sentences using only the given source material.",
            "Explain why each story matters to the reader.",
            "Keep a neutral, professional tone and avoid hype.",
        ],
    )


def deterministic_recency_score(
    article: NewsArticle,
    *,
    window_days: int,
    now: datetime | None = None,
) -> float:
    """Freshness in ``[0, 1]`` computed from the publication date, not the LLM."""
    if not article.published_at:
        return 0.5  # unknown date: stay neutral instead of guessing
    try:
        published = datetime.fromisoformat(str(article.published_at))
    except ValueError:
        return 0.5
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    reference = now or datetime.now(timezone.utc)
    age_days = max(0.0, (reference - published).total_seconds() / 86400)
    horizon = window_days if window_days > 0 else 30
    return round(max(0.0, min(1.0, 1.0 - age_days / horizon)), 3)


def keyword_relevance(article: NewsArticle, terms: Sequence[str]) -> float:
    """Keyword overlap in ``[0, 1]`` (title hits count double)."""
    if not terms:
        return 0.5
    title = article.title.lower()
    body = f"{article.description or ''} {article.content or ''}".lower()
    hits = 0.0
    for term in terms:
        if term in title:
            hits += 2.0
        elif term in body:
            hits += 1.0
    return round(min(1.0, hits / (2.0 * min(len(terms), 5))), 3)


def deterministic_evaluation(
    article: NewsArticle,
    *,
    terms: Sequence[str],
    window_days: int,
    source_weight: float = 0.6,
    duplicate: bool = False,
) -> EvaluatedArticle:
    """Score an article without an LLM (used when the evaluator call failed)."""
    relevance = keyword_relevance(article, terms)
    quality = clamp_float(source_weight, 0.0, 1.0)
    if article.content and len(article.content) > 800:
        quality = min(1.0, quality + 0.2)
    evaluation = ArticleEvaluation(
        relevance_score=relevance,
        quality_score=round(quality, 3),
        recency_score=deterministic_recency_score(article, window_days=window_days),
        duplicate=duplicate,
        selected=relevance >= 0.35,
        reason=(
            "Deterministic fallback score (keyword overlap, source weight and publication date); "
            "the LLM evaluator was unavailable for this article."
        ),
    )
    return EvaluatedArticle(article=article, evaluation=evaluation, evaluation_source="deterministic_fallback")


def sanitize_newsletter(
    newsletter: Newsletter | None,
    summaries: Sequence[ArticleSummary],
    *,
    max_sections: int | None = None,
) -> tuple[Newsletter | None, list[str]]:
    """Enforce the hard rules the LLM cannot be trusted with.

    * every ``article_url`` must exist in the validated summaries (no invented
      links - a section with an unknown URL is dropped),
    * no article appears twice,
    * no more sections than the requested article count,
    * attribution (``source``) is taken from the research data, not the model.
    """
    if newsletter is None:
        return None, ["No newsletter was generated."]

    allowed = {canonicalize_url(summary.url): summary for summary in summaries}
    issues: list[str] = []
    kept: list[NewsletterSection] = []
    seen: set[str] = set()

    for section in newsletter.sections:
        key = canonicalize_url(section.article_url)
        if key not in allowed:
            issues.append(
                f"Removed section '{section.headline[:70]}' because its URL is not part of the research results."
            )
            continue
        if key in seen:
            issues.append(f"Removed duplicate section '{section.headline[:70]}'.")
            continue
        seen.add(key)
        summary = allowed[key]
        if summary.source and section.source != summary.source:
            issues.append(f"Corrected the attribution for '{section.headline[:70]}'.")
            section = section.model_copy(update={"source": summary.source})
        kept.append(section)

    limit = max(1, max_sections if max_sections is not None else len(summaries))
    if len(kept) > limit:
        issues.append(f"Dropped {len(kept) - limit} section(s) beyond the requested article count.")
        kept = kept[:limit]

    if not kept:
        return None, issues + ["The newsletter had no section backed by a research result."]

    if len(kept) != newsletter.section_count or kept != list(newsletter.sections):
        newsletter = newsletter.model_copy(update={"sections": kept})
    return newsletter, issues


def audit_newsletter(newsletter: Newsletter | None, summaries: Sequence[ArticleSummary]) -> list[str]:
    """Deterministic issue list that the critic LLM cannot overrule."""
    if newsletter is None:
        return ["No newsletter was generated."]

    issues: list[str] = []
    allowed = {canonicalize_url(summary.url) for summary in summaries}

    urls = [canonicalize_url(section.article_url) for section in newsletter.sections]
    for section, key in zip(newsletter.sections, urls):
        if key not in allowed:
            issues.append(f"The URL of '{section.headline[:70]}' is not part of the research results.")
    if len(set(urls)) != len(urls):
        issues.append("The same article appears in more than one section.")

    if len(summaries) >= 5 and newsletter.section_count < 5:
        issues.append(
            f"Only {newsletter.section_count} of {len(summaries)} available articles are covered; "
            "the brief asked for about 5-7 sections."
        )

    thin = [section.headline for section in newsletter.sections if len(section.summary) < 80]
    if thin:
        issues.append(f"{len(thin)} section summary/summaries are too short to be useful: {', '.join(thin[:3])}.")
    return issues


def select_articles(
    evaluated: Sequence[EvaluatedArticle],
    *,
    target: int,
    min_relevance: float = 0.35,
) -> list[EvaluatedArticle]:
    """Choose the best articles for the newsletter.

    Articles the LLM marked as ``selected`` come first, then the highest scoring
    remaining ones above ``min_relevance``. ``target`` is an upper bound only -
    no article is ever invented in order to reach it.
    """
    usable = [item for item in evaluated if not item.evaluation.duplicate]
    preferred = sorted(
        (item for item in usable if item.evaluation.selected),
        key=lambda item: (-item.score, item.article.title),
    )
    rest = sorted(
        (
            item
            for item in usable
            if not item.evaluation.selected and item.evaluation.relevance_score >= min_relevance
        ),
        key=lambda item: (-item.score, item.article.title),
    )

    chosen: list[EvaluatedArticle] = []
    seen: set[str] = set()
    for item in [*preferred, *rest]:
        key = item.article.canonical_url or item.article.url
        if key in seen:
            continue
        seen.add(key)
        chosen.append(item)
        if len(chosen) >= max(0, target):
            break
    return chosen