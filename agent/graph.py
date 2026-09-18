"""LangGraph wiring for the newsletter workflow.

The flow is linear with one review loop::

    parse_goal -> plan_research -> research -> evaluate -> select ->
    summarize -> generate -> critique -+-> revise_newsletter -> critique
                                       +-> human_review -+-> revise_newsletter
                                                         +-> save_output -> END

Nodes are bound to one :class:`AgentDeps` instance with ``functools.partial``,
so the compiled graph is reusable and every node stays trivially unit-testable
(:mod:`agent.routing` owns every conditional edge decision).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from langgraph.graph import END, StateGraph
from schemas import AgentMode, AgentResult

from agent import nodes
from agent.deps import AgentDeps, build_deps
from agent.routing import route_after_critique, route_after_human_review
from agent.state import AgentState, initial_state

#: LangGraph node name -> UI progress span label.
#: Mirrors the order used by the Streamlit progress panel.
SPAN_BY_NODE: dict[str, str] = {
    "parse_goal": "planning",
    "plan_research": "planning",
    "research": "research",
    "evaluate": "evaluation",
    "select": "evaluation",
    "summarize": "summarization",
    "generate": "generation",
    "critique": "critique",
    "revise_newsletter": "critique",
    "human_review": "human_review",
    "save_output": "save",
    "_research_detail": "research",
    "_llm_status": "generation",
}

#: Ordered list of node names that the Streamlit stepper renders.
NODE_ORDER: list[str] = [
    "parse_goal",
    "plan_research",
    "research",
    "evaluate",
    "select",
    "summarize",
    "generate",
    "critique",
    "human_review",
    "save_output",
]


def run_with_progress(
    compiled_graph,
    state: AgentState,
    *,
    on_progress: Callable[[str, str, str], None] | None = None,
    config: dict[str, Any] | None = None,
) -> AgentState:
    """Run a compiled LangGraph with per-node progress callbacks.

    ``on_progress(node, status, detail)`` is called after every node execution.
    With ``stream_mode='updates'`` the node has already finished when the delta
    arrives, so the emitted status is ``"done"``.
    """

    def _emit(node: str, status: str, detail: str = "") -> None:
        if on_progress is None:
            return
        on_progress(node, status, detail)

    last_state = state
    for event in compiled_graph.stream(state, config=config or {}, stream_mode="updates"):
        for node_name, node_update in event.items():
            if isinstance(node_update, dict):
                detail = ""
                tokens = []
                for key in ("_raw_articles", "_articles", "_input_tokens", "_output_tokens"):
                    if key in node_update:
                        tokens.append(f"{key[1:]}={node_update[key]}")
                detail = " ".join(tokens)
                _emit(node_name, "done", detail)
                last_state = {**last_state, **node_update}
            else:
                _emit(node_name, "done", "")
                last_state = node_update if isinstance(node_update, dict) else last_state
    return last_state


def build_graph(deps: AgentDeps | None = None, **overrides: Any):
    """Compile the newsletter graph around ``deps`` (built on first use)."""
    active = deps if deps is not None else build_deps(**overrides)
    builder: StateGraph = StateGraph(AgentState)
    builder.add_node("parse_goal", partial(nodes.parse_goal, deps=active))
    builder.add_node("plan_research", partial(nodes.plan_research, deps=active))
    builder.add_node("research", partial(nodes.research, deps=active))
    builder.add_node("evaluate", partial(nodes.evaluate, deps=active))
    builder.add_node("select", partial(nodes.select, deps=active))
    builder.add_node("summarize", partial(nodes.summarize, deps=active))
    builder.add_node("generate", partial(nodes.generate, deps=active))
    builder.add_node("critique", partial(nodes.critique, deps=active))
    builder.add_node("revise_newsletter", partial(nodes.revise_newsletter, deps=active))
    builder.add_node("human_review", partial(nodes.human_review, deps=active))
    builder.add_node("save_output", partial(nodes.save_output, deps=active))

    builder.set_entry_point("parse_goal")
    builder.add_edge("parse_goal", "plan_research")
    builder.add_edge("plan_research", "research")
    builder.add_edge("research", "evaluate")
    builder.add_edge("evaluate", "select")
    builder.add_edge("select", "summarize")
    builder.add_edge("summarize", "generate")
    builder.add_edge("generate", "critique")
    builder.add_conditional_edges(
        "critique",
        route_after_critique,
        {
            "revise_newsletter": "revise_newsletter",
            "human_review": "human_review",
            "save_output": "save_output",
        },
    )
    builder.add_edge("revise_newsletter", "critique")
    builder.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {"revise_newsletter": "revise_newsletter", "save_output": "save_output"},
    )
    builder.add_edge("save_output", END)
    return builder.compile()


def _resolved_mode(mode: AgentMode | str | None) -> AgentMode:
    """Accept an :class:`AgentMode` or any of its string values."""
    if isinstance(mode, AgentMode):
        return mode
    return AgentMode.from_value(mode)


def run_newsletter_state(
    goal: str,
    *,
    mode: AgentMode | str = AgentMode.FULLY_AUTONOMOUS,
    model: str | None = None,
    thread_id: str | None = None,
    overrides: dict[str, Any] | None = None,
    on_progress: Callable[[str, str, str], None] | None = None,
) -> AgentState:
    """Run the full workflow and return the final state (logs, errors, result)."""
    extra = {key: value for key, value in (overrides or {}).items() if key != "model"}
    deps = build_deps(model=model, **extra)
    graph = build_graph(deps)
    # Reserved keys are already resolved (mode/thread_id as explicit kwargs,
    # max_revisions via deps.settings after ``build_deps`` applied the UI
    # overrides) - spreading them again would raise "multiple values".
    reserved = {"mode", "thread_id", "max_revisions", "model"}
    state_overrides = {
        key: value
        for key, value in extra.items()
        if key in AgentState.__annotations__ and key not in reserved
    }
    state = initial_state(
        goal,
        mode=_resolved_mode(mode),
        thread_id=thread_id,
        max_revisions=deps.settings.max_revisions,
        **state_overrides,
    )
    return run_with_progress(
        graph,
        state,
        on_progress=on_progress,
        config={"recursion_limit": 64},
    )


def run_newsletter(goal: str, **kwargs: Any) -> AgentResult:
    """Convenience wrapper: run the workflow and return the final result."""
    final = run_newsletter_state(goal, **kwargs)
    result = final.get("final_output")
    if isinstance(result, AgentResult):
        return result
    return AgentResult(
        success=False,
        goal=goal,
        execution_log=list(final.get("execution_log") or []),
        error="\n".join(final.get("errors") or []) or "The graph finished without producing a result.",
        mode=_resolved_mode(kwargs.get("mode", AgentMode.FULLY_AUTONOMOUS)),
    )
