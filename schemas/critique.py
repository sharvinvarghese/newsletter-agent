"""Critic and human-approval schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import Field, field_validator

from schemas.base import AgentModel, as_string_list, clean_text


class CritiqueResult(AgentModel):
    """Independent review of a generated newsletter.

    The critic only compares the newsletter against the validated article
    summaries it was given - it never browses the web for new facts.
    """

    approved: bool = Field(
        description="True only when the newsletter is publishable as-is.",
    )
    factual_issues: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Statements that contradict or are unsupported by the summaries.",
    )
    relevance_issues: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Content that does not serve the user's goal or audience.",
    )
    readability_issues: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Tone, clarity, length and structure problems.",
    )
    missing_information: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Important information from the summaries that is absent.",
    )
    improvement_instructions: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Concrete, actionable instructions for the revision node.",
    )

    @field_validator(
        "factual_issues",
        "relevance_issues",
        "readability_issues",
        "missing_information",
        "improvement_instructions",
        mode="before",
    )
    @classmethod
    def _clean_issues(cls, value: Any) -> Any:
        return as_string_list(value, limit=12)

    @property
    def issue_count(self) -> int:
        return (
            len(self.factual_issues)
            + len(self.relevance_issues)
            + len(self.readability_issues)
            + len(self.missing_information)
        )

    @property
    def hard_issues(self) -> list[str]:
        """Issues that must block publication regardless of the model verdict."""
        return list(self.factual_issues) + list(self.missing_information)


class HumanApproval(AgentModel):
    """Decision returned by the human reviewer (or by autonomous mode)."""

    approved: bool = Field(description="True to publish the newsletter as-is.")
    feedback: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional reviewer feedback forwarded to the revision node.",
    )
    reviewer: str = Field(default="human_reviewer", max_length=120)
    decided_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"),
        description="ISO-8601 timestamp of the decision.",
    )

    @field_validator("feedback", "reviewer", mode="before")
    @classmethod
    def _clean(cls, value: Any) -> Any:
        cleaned = clean_text(value)
        return cleaned or None if isinstance(cleaned, str) else cleaned

    @property
    def action(self) -> str:
        return "approve" if self.approved else "revise"