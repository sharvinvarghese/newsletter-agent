"""Conditional edges of the newsletter workflow.

The routing decisions live in small pure functions so the critic/revision loop
can be tested without executing the graph.
"""

from __future__ import annotations

from typing import Literal

from agent.state import AgentState

#: Hard cap for the critique -> revise loop. The effective value comes from
#: ``Settings.max_revisions`` (``MAX_REVISIONS`` env var); this constant is the
#: documented fallback when a state was built without settings.
MAX_REVISIONS = 2

CritiqueRoute = Literal["revise_newsletter", "human_review", "save_output"]
ReviewRoute = Literal["revise_newsletter", "save_output"]


def revision_limit(state: AgentState) -> int:
    """Configured maximum number of revisions."""
    try:
        return max(0, int(state.get("max_revisions") or MAX_REVISIONS))
    except (TypeError, ValueError):
        return MAX_REVISIONS


def revisions_left(state: AgentState) -> int:
    """How many revision cycles are still allowed."""
    used = state.get("revision_count") or 0
    try:
        used_count = max(0, int(used))
    except (TypeError, ValueError):
        used_count = 0
    return max(0, revision_limit(state) - used_count)


def should_revise(state: AgentState) -> bool:
    """True when the critic rejected the draft and the budget allows a revision."""
    critique = state.get("critique")
    return bool(critique and not critique.approved) and revisions_left(state) > 0


def route_after_critique(state: AgentState) -> CritiqueRoute:
    """Decide what happens after the critic reviewed the newsletter.

    * no newsletter -> ``save_output`` (generation failed, nothing to review)
    * critic unavailable -> ``human_review`` (a human/auto review is still useful)
    * approved -> ``human_review`` (publish path continues)
    * rejected with budget left -> ``revise_newsletter``
    * rejected and out of budget -> ``human_review`` so the decision is surfaced
    """
    if state.get("newsletter") is None:
        return "save_output"
    critique = state.get("critique")
    if critique is None:
        return "human_review"
    if critique.approved:
        return "human_review"
    return "revise_newsletter" if revisions_left(state) > 0 else "human_review"


def route_after_human_review(state: AgentState) -> ReviewRoute:
    """Decide what happens after the human (or autonomous) review."""
    approval = state.get("human_approval")
    if approval is None or approval.approved:
        return "save_output"
    if revisions_left(state) > 0:
        return "revise_newsletter"
    return "save_output"