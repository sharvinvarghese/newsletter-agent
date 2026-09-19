"""Top level run artifacts handed back to the caller (Streamlit / CLI / tests)."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import Field

from schemas.articles import ArticleSummary, EvaluatedArticle, NewsArticle
from schemas.base import AgentMode, AgentModel
from schemas.critique import CritiqueResult
from schemas.goal import AgentPlan, GoalSpec
from schemas.newsletter import Newsletter


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AgentResult(AgentModel):
    """The single object the whole workflow is judged by.

    All assignment-required fields are present; the fields below the divider add
    run context that the Streamlit UI displays (and that makes a run replayable
    from the saved JSON artifact).
    """

    success: bool = Field(description="True when a newsletter was generated and saved.")
    goal: str = Field(description="The goal the user submitted.")
    selected_articles: list[NewsArticle] = Field(default_factory=list)
    summaries: list[ArticleSummary] = Field(default_factory=list)
    newsletter: Newsletter | None = None
    html_path: str | None = None
    markdown_path: str | None = None
    critique: CritiqueResult | None = None
    execution_log: list[str] = Field(default_factory=list)
    error: str | None = None

    # --- additional run context ---------------------------------------------
    mode: AgentMode = AgentMode.FULLY_AUTONOMOUS
    goal_spec: GoalSpec | None = None
    plan: AgentPlan | None = None
    research_articles: list[NewsArticle] = Field(default_factory=list)
    evaluated_articles: list[EvaluatedArticle] = Field(default_factory=list)
    critique_history: list[CritiqueResult] = Field(default_factory=list)
    revision_count: int = 0
    thread_id: str | None = None
    awaiting_human_approval: bool = False
    html_content: str | None = None
    markdown_content: str | None = None
    run_json_path: str | None = None
    articles_xlsx_path: str | None = None
    summaries_json_path: str | None = None
    generated_at: str = Field(default_factory=_utc_now)
    run_id: str | None = Field(default=None, description="Short unique ID for the run (folder name fragment).")
    llm_call_count: int = Field(default=0, description="Number of LLM calls made during this run.")

    @property
    def selected_count(self) -> int:
        return len(self.selected_articles)

    def log_text(self) -> str:
        """Execution log as a single string (handy for the UI and bug reports)."""
        return "\n".join(self.execution_log)


class AgentEvent(AgentModel):
    """Progress event emitted while the graph streams (consumed by Streamlit)."""

    node: str = Field(description="Name of the LangGraph node the event belongs to.")
    message: str = Field(default="", description="Human readable one-liner for the UI.")
    log_lines: list[str] = Field(default_factory=list, description="New execution log entries.")
    is_final: bool = Field(default=False, description="True for the last event of a run.")
    result: AgentResult | None = Field(
        default=None,
        description="Filled on the final event: the complete run result.",
    )