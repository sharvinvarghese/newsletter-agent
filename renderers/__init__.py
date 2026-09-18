"""Rendering of the final artifacts (HTML, Markdown, run JSON).

Everything visual lives in ``config/templates/*.j2``; this package only builds the
context, renders and writes the files. Two Jinja environments are used so that the
HTML output is auto-escaped while the Markdown stays literal.

Nothing here talks to an LLM: the renderer receives validated Pydantic objects and
turns them into files, which keeps the "generate" and "save" steps independently
testable.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import DEFAULT_OUTPUTS_DIR, Settings, get_settings
from config.tuning import Tuning, get_tuning
from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    TemplateError,
    select_autoescape,
)
from schemas import (
    AgentMode,
    AgentResult,
    ArticleSummary,
    CritiqueResult,
    NewsArticle,
    Newsletter,
)
from tools.deduplication import canonicalize_url

logger = logging.getLogger("newsletter_agent.renderers")

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_ENV_CACHE: dict[tuple[str, bool], Environment] = {}


class RenderError(RuntimeError):
    """Raised when a template is missing or cannot be rendered."""


@dataclass(frozen=True)
class SavedArtifacts:
    """Paths and content produced by :func:`save_newsletter`."""

    stem: str
    html_content: str | None = None
    markdown_content: str | None = None
    html_path: str | None = None
    markdown_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def slugify(text: str, *, limit: int = 48) -> str:
    """Filesystem safe slug (falls back to ``issue`` when nothing survives)."""
    cleaned = _SLUG_RE.sub("-", (text or "").lower()).strip("-")
    return cleaned[:limit].strip("-") or "issue"


def environment(tuning: Tuning | None = None, *, html: bool = True) -> Environment:
    """Cached Jinja environment for the configured template directory."""
    active = tuning or get_tuning()
    directory = active.rendering.template_dir()
    key = (str(directory), html)
    cached = _ENV_CACHE.get(key)
    if cached is None:
        cached = Environment(
            loader=FileSystemLoader(str(directory)),
            autoescape=select_autoescape(enabled_extensions=("html", "xml"), default=html),
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )
        _ENV_CACHE[key] = cached
    return cached


def critique_status(critique: CritiqueResult | None) -> str:
    """Short, human readable review verdict used in the artifact metadata."""
    if critique is None:
        return ""
    verdict = "approved" if critique.approved else "not approved"
    if critique.issue_count:
        return f"{verdict} - {critique.issue_count} issue(s)"
    return verdict


def _news_article(item: Any) -> NewsArticle | None:
    """Return the :class:`NewsArticle` behind ``item``, if there is one.

    Callers pass whatever the state holds: plain news articles, evaluated
    articles (article + scores) or the LLM summaries. Only real articles carry a
    publication date and author, so everything else is ignored.
    """
    if isinstance(item, NewsArticle):
        return item
    inner = getattr(item, "article", None)
    return inner if isinstance(inner, NewsArticle) else None


def section_view(
    newsletter: Newsletter,
    summaries: Sequence[ArticleSummary] = (),
    articles: Sequence[Any] = (),
) -> list[dict[str, Any]]:
    """Template friendly view of the sections, enriched from the source data.

    ``NewsletterSection`` deliberately holds editorial text only. The concrete
    facts (publication date, author, key points) live on the articles and on the
    validated summaries the sections were written from, so templates render this
    view instead. Every value is a plain string/list, and anything unknown
    degrades to an empty string so the templates never see ``Undefined``.
    """
    summaries_by_url: dict[str, ArticleSummary] = {}
    for summary in summaries:
        summaries_by_url.setdefault(canonicalize_url(summary.url), summary)

    articles_by_url: dict[str, NewsArticle] = {}
    for item in articles:
        article = _news_article(item)
        if article is not None:
            articles_by_url.setdefault(canonicalize_url(article.url), article)

    view: list[dict[str, Any]] = []
    for section in newsletter.sections:
        key = canonicalize_url(section.article_url)
        summary = summaries_by_url.get(key)
        article = articles_by_url.get(key)
        view.append(
            {
                "headline": section.headline,
                "article_url": section.article_url,
                "source": section.source,
                "summary": section.summary,
                "why_it_matters": section.why_it_matters,
                "published_at": ((article.published_at or "") if article else "")[:10],
                "author": (article.author or "") if article else "",
                "key_points": list(summary.key_points) if summary else [],
                "article_title": (
                    (article.title if article else None)
                    or (summary.title if summary else None)
                    or section.headline
                ),
            }
        )
    return view


def build_context(
    newsletter: Newsletter,
    *,
    goal: str = "",
    goal_spec: Any = None,
    plan: Any = None,
    summaries: Sequence[ArticleSummary] = (),
    selected_articles: Sequence[Any] = (),
    candidate_count: int = 0,
    critique: CritiqueResult | None = None,
    critique_history: Sequence[CritiqueResult] = (),
    revision_count: int = 0,
    mode: AgentMode | str | None = None,
    research_stats: Any = None,
    generated_at: str | None = None,
    model_name: str = "",
    language: str = "en",
    log_lines: Sequence[str] = (),
    badges: Sequence[str] = (),
    output_note: str | None = None,
    tuning: Tuning | None = None,
) -> dict[str, Any]:
    """Assemble the template variables (all values are plain strings/numbers)."""
    active = tuning or get_tuning()
    resolved_mode = AgentMode.from_value(mode) if mode is not None else AgentMode.FULLY_AUTONOMOUS
    marks = list(dict.fromkeys(badges))
    if not marks:
        marks = [resolved_mode.label]
    if revision_count:
        marks.append(f"revised {revision_count}x")

    return {
        "newsletter": newsletter,
        "sections": section_view(newsletter, summaries, selected_articles),
        "goal": goal or (getattr(goal_spec, "goal", "") or ""),
        "newsletter_topic": getattr(goal_spec, "topic", "") or "",
        "audience": getattr(goal_spec, "audience", "") or "",
        "frequency": getattr(goal_spec, "frequency", "") or "",
        "plan_objective": getattr(plan, "objective", "") or "",
        "summaries": list(summaries),
        "selected_articles": list(selected_articles),
        "article_count": len(selected_articles) or newsletter.section_count,
        "candidate_count": candidate_count,
        "critique": critique,
        "critique_status": critique_status(critique),
        "critique_history": list(critique_history),
        "revision_count": revision_count,
        "research_stats": research_stats,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_name": model_name,
        "language": language,
        "log_lines": list(log_lines),
        "badges": marks,
        "attribution": active.rendering.footer.attribution,
        "disclaimer": active.rendering.footer.disclaimer,
        "simulated_delivery_note": (
            active.rendering.simulated_delivery_note if output_note is None else output_note
        ),
    }


def render_html(context: Mapping[str, Any], *, tuning: Tuning | None = None) -> str:
    """Render the HTML template for ``context``."""
    active = tuning or get_tuning()
    try:
        template = environment(active, html=True).get_template(active.rendering.html_template)
        return template.render(**context)
    except TemplateError as exc:
        raise RenderError(f"Could not render {active.rendering.html_template}: {exc}") from exc


def render_markdown(context: Mapping[str, Any], *, tuning: Tuning | None = None) -> str:
    """Render the Markdown template for ``context``."""
    active = tuning or get_tuning()
    try:
        template = environment(active, html=False).get_template(active.rendering.markdown_template)
        return template.render(**context)
    except TemplateError as exc:
        raise RenderError(f"Could not render {active.rendering.markdown_template}: {exc}") from exc


def outputs_dir(settings: Settings | None = None) -> Path:
    """Directory that receives every artifact (created on demand)."""
    active = settings or get_settings()
    directory = Path(active.outputs_dir) if active.outputs_dir else DEFAULT_OUTPUTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def artifact_stem(
    newsletter: Newsletter | None,
    *,
    fallback: str = "newsletter",
    when: datetime | None = None,
    tuning: Tuning | None = None,
) -> str:
    """``<prefix>-<subject slug>-<UTC timestamp>`` file stem (config driven)."""
    rules = (tuning or get_tuning()).rendering.filename
    subject = newsletter.subject if newsletter is not None else fallback
    stamp = (when or datetime.now(timezone.utc)).strftime(rules.timestamp_format)
    return f"{rules.prefix}-{slugify(subject, limit=rules.subject_slug_chars)}-{stamp}"


def _write_text(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    logger.info("Wrote %s (%d characters)", path, len(content))
    return str(path)


def save_newsletter(
    newsletter: Newsletter,
    *,
    context: Mapping[str, Any] | None = None,
    output_format: str | None = None,
    directory: str | Path | None = None,
    settings: Settings | None = None,
    tuning: Tuning | None = None,
    stem: str | None = None,
    context_values: Mapping[str, Any] | None = None,
) -> SavedArtifacts:
    """Render and write the newsletter, returning the paths and the content.

    ``output_format`` must be one of ``newsletter.output_formats`` in
    ``tuning.yaml`` (defaults to the first entry). ``directory`` defaults to the
    configured outputs directory.
    """
    active = tuning or get_tuning()
    allowed = active.newsletter.output_formats
    chosen = (output_format or active.newsletter.default_format).strip().lower()
    if chosen not in allowed:
        logger.warning("Unknown output format '%s'; falling back to '%s'.", chosen, active.newsletter.default_format)
        chosen = active.newsletter.default_format

    payload = dict(context) if context is not None else build_context(
        newsletter, tuning=active, **dict(context_values or {})
    )
    payload.setdefault("newsletter", newsletter)

    target_dir = Path(directory) if directory else outputs_dir(settings)
    target_dir.mkdir(parents=True, exist_ok=True)
    resolved_stem = stem or artifact_stem(newsletter, tuning=active)

    html_content = markdown_content = None
    html_path = markdown_path = None
    if "html" in chosen:
        html_content = render_html(payload, tuning=active)
        html_path = _write_text(target_dir / f"{resolved_stem}.html", html_content)
    if "markdown" in chosen:
        markdown_content = render_markdown(payload, tuning=active)
        markdown_path = _write_text(target_dir / f"{resolved_stem}.md", markdown_content)

    return SavedArtifacts(
        stem=resolved_stem,
        html_content=html_content,
        markdown_content=markdown_content,
        html_path=html_path,
        markdown_path=markdown_path,
    )


def write_run_artifact(
    result: AgentResult,
    *,
    directory: str | Path | None = None,
    settings: Settings | None = None,
    tuning: Tuning | None = None,
    stem: str | None = None,
) -> str:
    """Persist the complete run result as JSON (replayable audit trail)."""
    active = tuning or get_tuning()
    resolved_stem = stem or artifact_stem(result.newsletter, fallback=result.goal, tuning=active)
    target = Path(directory) if directory else outputs_dir(settings)
    target.mkdir(parents=True, exist_ok=True)
    payload = result.model_dump(mode="json")
    return _write_text(
        target / f"{resolved_stem}-run.json",
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
    )


def _write_json(path: Path, content: str) -> str:
    return _write_text(path.with_suffix(".json"), content)


def dump_articles_excel(
    articles: Sequence[Any],
    *,
    directory: str | Path | None = None,
    stem: str | None = None,
    settings: Settings | None = None,
    tuning: Tuning | None = None,
) -> str:
    """Write the supplied articles (raw or `EvaluatedArticle`) to a timestamped
    ``.xlsx`` file and return its path.

    Falls back to CSV when openpyxl is not installed, so the artefact dump never
    hard-fails a run.
    """
    try:
        import openpyxl  # noqa: F401
    except Exception:  # pragma: no cover - optional dependency
        return _dump_articles_csv(articles, directory=directory, stem=stem,
                                  settings=settings, tuning=tuning)

    from openpyxl import Workbook

    active = tuning or get_tuning()
    resolved_stem = stem or artifact_stem(None, fallback="newsletter", tuning=active)
    target_dir = Path(directory) if directory else outputs_dir(settings)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{resolved_stem}-articles.xlsx"

    def _value(article: Any) -> dict[str, str]:
        # Unwrap EvaluatedArticle -> NewsArticle
        art = getattr(article, "article", article) if hasattr(article, "article") else article
        eval_ = getattr(article, "evaluation", None) if hasattr(article, "article") else None
        return {
            "title": getattr(art, "title", "") or "",
            "url": str(getattr(art, "url", "") or ""),
            "source": getattr(art, "source", "") or "",
            "author": getattr(art, "author", "") or "",
            "published_at": str(getattr(art, "published_at", "") or ""),
            "category": getattr(art, "category", "") or "",
            "description": getattr(art, "description", "") or "",
            "content": (getattr(art, "content", "") or "")[:5000],
            "relevance_score": _fmt_score(getattr(eval_, "relevance_score", None)),
            "quality_score": _fmt_score(getattr(eval_, "quality_score", None)),
            "recency_score": _fmt_score(getattr(eval_, "recency_score", None)),
            "selected": "true" if (eval_ and getattr(eval_, "selected", False)) else "",
            "duplicate": "true" if (eval_ and getattr(eval_, "duplicate", False)) else "",
            "reason": getattr(eval_, "reason", "") or "",
        }

    wb = Workbook()
    ws = wb.active
    ws.title = "articles"
    headers = ["title", "url", "source", "author", "published_at", "category",
               "description", "content", "relevance_score", "quality_score",
               "recency_score", "selected", "duplicate", "reason"]
    ws.append(headers)
    for article in articles:
        row = _value(article)
        ws.append([row[h] for h in headers])
    wb.save(path)
    logger.info("Wrote %s (%d articles)", path, len(articles))
    return str(path)


def _fmt_score(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _dump_articles_csv(articles, *, directory=None, stem=None, settings=None, tuning=None) -> str:
    import csv

    active = tuning or get_tuning()
    resolved_stem = stem or artifact_stem(None, fallback="newsletter", tuning=active)
    target_dir = Path(directory) if directory else outputs_dir(settings)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{resolved_stem}-articles.csv"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["title", "url", "source", "author", "published_at", "category", "description"])
        for article in articles:
            art = getattr(article, "article", article) if hasattr(article, "article") else article
            writer.writerow([
                getattr(art, "title", "") or "",
                str(getattr(art, "url", "") or "") if hasattr(article, "article") else "",
            ])
    return str(path)


def newest_artifacts(directory: str | Path | None = None, *, suffix: str = ".html") -> list[Path]:
    """Most recent artifacts first (used by the CLI ``--list`` flag)."""
    target = Path(directory) if directory else outputs_dir()
    if not target.exists():
        return []
    return sorted(
        (path for path in target.glob(f"*{suffix}") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


__all__ = [
    "RenderError",
    "SavedArtifacts",
    "artifact_stem",
    "build_context",
    "environment",
    "newest_artifacts",
    "outputs_dir",
    "render_html",
    "render_markdown",
    "save_newsletter",
    "slugify",
    "write_run_artifact",
]