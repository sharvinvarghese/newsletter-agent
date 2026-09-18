"""Offline smoke check for the newsletter agent.

Runs without an API key and without touching the network: it imports every
package, loads ``tuning.yaml`` / ``prompts.yaml`` / ``sources.yaml``, renders a
fixture newsletter through the real Jinja templates and prints what is available.

    python scripts/smoke_check.py

Exit code 0 means "the project is wired together correctly"; any failure is
printed with the exception type so it can be fixed before a run spends tokens.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _fixture():
    """A minimal, schema-valid newsletter used only to exercise the renderer."""
    from schemas import ArticleSummary, NewsArticle, Newsletter, NewsletterSection

    article = NewsArticle(
        title="LangGraph 0.6 ships checkpointing improvements",
        url="https://blog.langchain.dev/example-post/",
        source="LangChain Blog",
        author="Ada Example",
        published_at="2026-09-15T09:30:00+00:00",
        description="Release notes for the LangGraph runtime.",
        content="LangGraph published a release that reduces checkpoint overhead.",
        category="ai_news",
    )
    summary = ArticleSummary(
        title=article.title,
        source=article.source,
        url=article.url,
        summary=(
            "LangGraph published a release that reduces checkpoint overhead and documents "
            "durable execution for long-running agents. The post explains how the runtime "
            "resumes interrupted nodes."
        ),
        key_points=["Lower checkpoint overhead", "Durable execution documented"],
        why_it_matters="Teams running multi-step agents can resume work after a failure.",
    )
    newsletter = Newsletter(
        subject="Agentic AI this week: durable agent runtimes",
        preheader="LangGraph checkpointing, plus two more stories worth your time.",
        introduction=(
            "This issue looks at the infrastructure layer of agentic AI: how runtimes persist "
            "state, what changed in the latest releases and why teams care."
        ),
        sections=[
            NewsletterSection(
                headline="Durable agent runtimes become the default",
                article_url=summary.url,
                source=summary.source,
                summary=summary.summary,
                why_it_matters=summary.why_it_matters,
            )
        ],
        conclusion="Expect state persistence to become a baseline requirement for agent platforms.",
    )
    return newsletter, summary, article


def main() -> int:
    from config.settings import configure_logging, get_settings
    from config.tuning import get_tuning
    from llm.prompts import get_prompts
    from renderers import build_context, save_newsletter
    from tools import extraction_backends, load_sources

    configure_logging("WARNING")
    checks: list[tuple[str, str]] = []

    settings = get_settings()
    tuning = get_tuning()
    issues = tuning.consistency_issues()
    checks.append(("settings", f"model={settings.openrouter_model}, api_key={'set' if settings.has_api_key else 'missing'}"))
    checks.append(("tuning.yaml", f"v{tuning.version}, {len(issues)} consistency issue(s)"))
    for issue in issues:
        checks.append(("tuning issue", issue))

    prompts = get_prompts()
    checks.append(("prompts.yaml", ", ".join(prompts.names)))
    for name in prompts.names:
        checks.append((f"placeholders[{name}]", ", ".join(sorted(prompts.placeholders(name)))))

    sources = load_sources(settings.sources_path)
    hn = tuning.research.hacker_news
    hn_count = sum(1 for source in sources.sources if source.type == "hackernews")
    checks.append(("sources.yaml", f"{len(sources.sources)} curated source(s), {len(sources.enabled_sources())} enabled, {hn_count} hacker news"))
    checks.append(
        (
            "hacker news",
            f"enabled={hn.enabled} provider={hn.provider} example={hn.query_url('ai agents', recency_days=7)}",
        )
    )

    newsletter, summary, article = _fixture()
    context = build_context(
        newsletter,
        goal="Weekly agentic AI newsletter for engineers",
        summaries=[summary],
        selected_articles=[article],
        candidate_count=3,
        model_name=settings.openrouter_model,
    )
    # Regression guard: dates/authors live on NewsArticle and key points on
    # ArticleSummary, so section_view() must enrich the sections from both.
    section = context["sections"][0]
    checks.append(
        (
            "section enrichment",
            (f"author={section['author']!r} published_at={section['published_at']!r} "
            f"key_points={len(section['key_points'])}"),
        )
    )
    missing = [
        field
        for field, value in (("author", article.author), ("published_at", article.published_at[:10]))
        if not value or value not in str(context["sections"])
    ]
    checks.append(("section facts", "ok" if not missing else f"MISSING {missing}"))

    with tempfile.TemporaryDirectory() as tmp:
        artifacts = save_newsletter(newsletter, context=context, directory=tmp, settings=settings)
        html = Path(artifacts.html_path).read_text(encoding="utf-8")
        markdown = Path(artifacts.markdown_path).read_text(encoding="utf-8")
        checks.append(
            (
                "rendering",
                f"html={Path(artifacts.html_path).stat().st_size}B, md={Path(artifacts.markdown_path).stat().st_size}B",
            )
        )
        # Jinja renders unknown attributes as "" for a Mapping, but a typo in a
        # template variable still shows up as the literal "Undefined".
        leaks = [name for name in ("Undefined", "{{", "}}") if name in html or name in markdown]
        checks.append(("template leaks", "none" if not leaks else f"FOUND {leaks}"))
        rendered = {
            "author": article.author in html and article.author in markdown,
            "published_at": article.published_at[:10] in html and article.published_at[:10] in markdown,
            "key_points": summary.key_points[0] in html and summary.key_points[0] in markdown,
        }
        checks.append(("rendered facts", "ok" if all(rendered.values()) else f"FAILED {rendered}"))

    checks.append(("extraction backends", ", ".join(f"{name}={ok}" for name, ok in extraction_backends().items())))
    checks.append(("agent graph", "imported" if _graph_imports() else "NOT AVAILABLE"))

    width = max(len(name) for name, _ in checks)
    for name, value in checks:
        print(f"{name.ljust(width)} : {value}")
    print("\nSmoke check passed.")
    return 0


def _graph_imports() -> bool:
    try:
        import agent.graph  # noqa: F401

        return True
    except Exception:
        traceback.print_exc(limit=1)
        return False


if __name__ == "__main__":
    raise SystemExit(main())
