"""Dependency container for the graph.

Nodes never build their own LLM client or research tool: they receive an
:class:`AgentDeps` instance. That keeps every node unit-testable with fakes -
no API key, no network - and keeps the Groq wiring in exactly one place.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from config.settings import Settings, get_settings
from llm.structured import StructuredInvoker, StructuredLLM
from schemas import ResearchOutcome
from tools.news_research import NewsResearchTool


class ResearchProvider(Protocol):
    """The part of the research tool the graph depends on."""

    def research(
        self,
        *,
        queries: Sequence[str],
        sources: Sequence[str] | None = None,
        max_articles: int | None = None,
        recency_window_days: int | None = None,
        enrich: bool | None = None,
    ) -> ResearchOutcome: ...


@dataclass(slots=True)
class AgentDeps:
    """Everything the nodes need to do their work."""

    settings: Settings
    llm: StructuredInvoker
    research: ResearchProvider
    on_progress: Callable[[str, str, str], None] | None = None

    def describe(self) -> dict[str, Any]:
        """Non-secret description used by the UI and the run artifact."""
        return {
            "settings": self.settings.safe_snapshot(),
            "llm": {
                "primary_model": getattr(self.llm, "primary_model", self.settings.groq_model),
                "models": list(getattr(self.llm, "models", []) or [self.settings.groq_model]),
            },
            "research_tool": type(self.research).__name__,
        }

    def cache_key(self) -> str:
        """Stable key so compiled graphs (and checkpoints) can be reused."""
        models = ",".join(getattr(self.llm, "models", []) or [self.settings.groq_model])
        return "|".join(
            [
                models,
                str(self.settings.sources_path),
                str(self.settings.outputs_dir),
                str(self.settings.max_revisions),
                str(self.settings.research_max_sources),
                str(self.settings.max_articles_to_collect),
            ]
        )


def build_deps(
    settings: Settings | None = None,
    *,
    model: str | None = None,
    llm: StructuredInvoker | None = None,
    research: ResearchProvider | None = None,
    **overrides: Any,
) -> AgentDeps:
    """Create the default dependencies (or accept injected fakes for tests)."""
    active = (settings or get_settings()).with_overrides(**overrides)
    return AgentDeps(
        settings=active,
        llm=llm or StructuredLLM(settings=active, model=model),
        research=research or NewsResearchTool(settings=active),
        on_progress=None,
    )


def build_deps(
    settings: Settings | None = None,
    *,
    model: str | None = None,
    llm: StructuredInvoker | None = None,
    research: ResearchProvider | None = None,
    **overrides: Any,
) -> AgentDeps:
    """Create the default dependencies (or accept injected fakes for tests)."""
    active = (settings or get_settings()).with_overrides(**overrides)
    return AgentDeps(
        settings=active,
        llm=llm or StructuredLLM(settings=active, model=model),
        research=research or NewsResearchTool(settings=active),
        on_progress=None,
    )