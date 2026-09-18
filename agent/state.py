"""LangGraph state definition for the newsletter workflow.

The graph is linear (exactly one node runs at a time), so plain fields are used
instead of reducers: every node returns the complete new list it owns and stays
trivially unit-testable.
"""

from __future__ import annotations

from typing import Any, TypedDict

from schemas import (
    AgentMode,
    AgentPlan,
    AgentResult,
    ArticleSummary,
    CritiqueResult,
    EvaluatedArticle,
    GoalSpec,
    HumanApproval,
    NewsArticle,
    Newsletter,
    ResearchStats,
)


class AgentState(TypedDict, total=False):
    """Everything that flows through the graph."""

    # --- inputs -------------------------------------------------------------
    goal: str
    mode: AgentMode
    thread_id: str | None
    # UI/CLI overrides; None means "let the planner decide"
    target_articles: int | None
    max_articles_to_collect: int | None
    recency_window_days: int | None
    save_intermediate: bool | None

    # --- planning -----------------------------------------------------------
    goal_spec: GoalSpec | None
    plan: AgentPlan | None
    search_queries: list[str]

    # --- research -----------------------------------------------------------
    raw_articles: list[NewsArticle]
    candidate_articles: list[NewsArticle]
    research_stats: ResearchStats | None

    # --- evaluation & selection ---------------------------------------------
    evaluated_articles: list[EvaluatedArticle]
    selected_articles: list[NewsArticle]

    # --- generation ---------------------------------------------------------
    summaries: list[ArticleSummary]
    newsletter: Newsletter | None

    # --- review loop --------------------------------------------------------
    critique: CritiqueResult | None
    critique_history: list[CritiqueResult]
    revision_count: int
    max_revisions: int
    human_approval: HumanApproval | None
    human_feedback: str | None

    # --- output -------------------------------------------------------------
    html_content: str | None
    markdown_content: str | None
    html_path: str | None
    markdown_path: str | None
    run_json_path: str | None
    final_output: AgentResult | None

    # --- bookkeeping --------------------------------------------------------
    execution_log: list[str]
    errors: list[str]


def initial_state(
    goal: str,
    mode: AgentMode | str = AgentMode.FULLY_AUTONOMOUS,
    *,
    thread_id: str | None = None,
    **overrides: Any,
) -> AgentState:
    """Build the starting state, optionally overriding a few fields."""
    resolved_mode = AgentMode.from_value(mode)
    state: AgentState = {
        "goal": (goal or "").strip(),
        "mode": resolved_mode,
        "thread_id": thread_id,
        "target_articles": None,
        "max_articles_to_collect": None,
        "recency_window_days": None,
        "save_intermediate": None,
        "goal_spec": None,
        "plan": None,
        "search_queries": [],
        "raw_articles": [],
        "candidate_articles": [],
        "research_stats": None,
        "evaluated_articles": [],
        "selected_articles": [],
        "summaries": [],
        "newsletter": None,
        "critique_history": [],
        "revision_count": 0,
        "human_approval": None,
        "human_feedback": None,
        "html_content": None,
        "markdown_content": None,
        "html_path": None,
        "markdown_path": None,
        "run_json_path": None,
        "final_output": None,
        "execution_log": [f"Run started in {resolved_mode.label} mode."],
        "errors": [],
    }
    for key, value in overrides.items():
        if key in AgentState.__annotations__:
            state[key] = value  # type: ignore[literal-required]
    return state


def append_log(state: AgentState, *messages: str) -> list[str]:
    """Full log list including the new lines (used as a node return value)."""
    return [*state.get("execution_log", []), *(message for message in messages if message)]


def append_errors(state: AgentState, *messages: str) -> list[str]:
    """Full error list including the new entries."""
    return [*state.get("errors", []), *(message for message in messages if message)]


def node_update(
    state: AgentState,
    *,
    log: tuple[str, ...] | list[str] = (),
    errors: tuple[str, ...] | list[str] = (),
    **values: Any,
) -> dict[str, Any]:
    """Assemble the dict a node returns, merging log/error lines automatically."""
    update: dict[str, Any] = dict(values)
    if log:
        update["execution_log"] = append_log(state, *log)
    if errors:
        update["errors"] = append_errors(state, *errors)
    if "_progress_tokens" in update:
        tokens = update.pop("_progress_tokens")
        if isinstance(tokens, (list, tuple)):
            for token in tokens:
                if isinstance(token, str) and "=" in token:
                    key, _, value = token.partition("=")
                    if key in ("raw_articles", "articles") or key in ("input_tokens", "output_tokens"):
                        try:
                            update[f"_{key}"] = int(value)
                        except (TypeError, ValueError):
                            pass
    return update