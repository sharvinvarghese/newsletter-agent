"""Live progress bookkeeping for the Streamlit newsletter run.

The graph runner calls the callback produced by :func:`make_callback` after every
LangGraph node, so the Streamlit progress panel can reflect real-time status and
article counts without blocking the run thread.
"""

from __future__ import annotations

import time as _time
from collections.abc import Callable
from dataclasses import dataclass, field

#: Ordered nodes rendered in the Streamlit "nodes of lights" stepper.
NODE_ORDER = [
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

#: Ordered spans used by the progress bar (coarse-grained).
SPAN_ORDER = [
    "planning",
    "research",
    "evaluation",
    "summarization",
    "generation",
    "critique",
    "human_review",
    "save",
]

#: Map a node name to its coarse span label.
NODE_TO_SPAN = {
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
}


@dataclass
class RunProgress:
    """Thread-safe-ish progress snapshot updated by the graph callback."""

    span: str = "starting"
    span_index: int = -1
    status: str = ""
    started_at: float = field(default_factory=_time.time)
    finished_at: float | None = None
    raw_article_count: int = 0
    article_count: int = 0
    completed_nodes: list[str] = field(default_factory=list)
    failed_nodes: list[str] = field(default_factory=list)
    last_node: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    def done(self) -> None:
        self.finished_at = _time.time()

    @property
    def elapsed(self) -> float:
        base = self.finished_at if self.finished_at is not None else _time.time()
        return base - self.started_at

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def node_status(self) -> dict[str, str]:
        """Quick lookup: node name -> "done" | "error" | "" (not started)."""
        return {name: ("error" if name in self.failed_nodes else "done" if name in self.completed_nodes else "") for name in NODE_ORDER}


def make_callback(*, progress: RunProgress) -> Callable[[str, str, str], None]:
    """Return an ``on_progress(node, status, detail)`` closure that updates
    ``progress`` in place. Artifacts like article counts come through the
    ``detail`` field as ``key=value`` tokens (parsed by the UI)."""

    def _on_progress(node: str, status: str, detail: str = "") -> None:
        progress.last_node = node
        progress.span = NODE_TO_SPAN.get(node, node)
        if progress.span in SPAN_ORDER:
            progress.span_index = SPAN_ORDER.index(progress.span)
        progress.status = status
        if status == "done" and node in NODE_ORDER:
            progress.completed_nodes.append(node)
        for token in detail.split():
            if "=" not in token:
                continue
            key, _, value = token.partition("=")
            if key == "raw_articles":
                try:
                    progress.raw_article_count = int(value)
                except ValueError:
                    pass
            elif key == "articles":
                try:
                    progress.article_count = int(value)
                except ValueError:
                    pass
            elif key == "input_tokens":
                try:
                    progress.input_tokens += int(value)
                except ValueError:
                    pass
            elif key == "output_tokens":
                try:
                    progress.output_tokens += int(value)
                except ValueError:
                    pass

    return _on_progress
