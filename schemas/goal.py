"""Goal and plan schemas: the *what* and the *how* of a run."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, field_validator

from schemas.base import AgentMode, AgentModel, as_string_list, clean_text

_FORMAT_ALIASES = {
    "html": "html",
    "markdown": "markdown",
    "md": "markdown",
    "html+markdown": "html+markdown",
    "htmlmarkdown": "html+markdown",
    "both": "html+markdown",
}


class GoalSpec(AgentModel):
    """Structured interpretation of the user's plain-English goal."""

    goal: str = Field(
        min_length=5,
        max_length=2000,
        description="The user's goal, copied verbatim from the request.",
    )
    topic: str = Field(
        min_length=2,
        max_length=200,
        description="Short topic label, e.g. 'AI agents' or 'AI infrastructure'.",
    )
    frequency: str = Field(
        default="weekly",
        min_length=2,
        max_length=50,
        description="Publication cadence mentioned in the goal (daily, weekly, ...).",
    )
    audience: str | None = Field(
        default=None,
        max_length=200,
        description="Intended readers, e.g. 'AI engineers'. Null when unspecified.",
    )
    requested_article_count: int = Field(
        default=6,
        ge=1,
        le=12,
        description="How many articles the newsletter should contain (aim for 5-7).",
    )
    output_format: Literal["html+markdown", "html", "markdown"] = Field(
        default="html+markdown",
        description="Requested output format: 'html+markdown', 'html' or 'markdown'.",
    )
    delivery_mode: AgentMode = Field(
        default=AgentMode.FULLY_AUTONOMOUS,
        description="'fully_autonomous' or 'human_in_loop'.",
    )

    @field_validator("output_format", mode="before")
    @classmethod
    def _normalise_format(cls, value: object) -> object:
        if isinstance(value, str):
            key = re.sub(r"[^a-z+]+", "", value.strip().lower().replace("and", "+"))
            key = re.sub(r"\++", "+", key)
            if key in _FORMAT_ALIASES:
                return _FORMAT_ALIASES[key]
        return value

    @field_validator("delivery_mode", mode="before")
    @classmethod
    def _normalise_mode(cls, value: object) -> object:
        if value is None:
            return AgentMode.FULLY_AUTONOMOUS
        if isinstance(value, AgentMode):  # already resolved - never stringify an enum
            return value
        return AgentMode.from_value(str(value))

    @field_validator("frequency", mode="before")
    @classmethod
    def _normalise_frequency(cls, value: object) -> object:
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
        return "weekly"


class AgentPlan(AgentModel):
    """The plan the planner node produced for this run."""

    objective: str = Field(
        min_length=10,
        max_length=1000,
        description="One sentence describing what this run must achieve.",
    )
    research_queries: list[str] = Field(
        min_length=1,
        max_length=12,
        description="Search queries used to select and score news sources.",
    )
    sources: list[str] = Field(
        default_factory=list,
        max_length=30,
        description="Ids or names of configured sources to query, best first.",
    )
    max_articles_to_collect: int = Field(
        default=24,
        ge=5,
        le=100,
        description="Upper bound of candidate articles to collect before filtering.",
    )
    target_articles: int = Field(
        default=6,
        ge=1,
        le=12,
        description="Number of articles the newsletter should contain.",
    )
    newsletter_requirements: list[str] = Field(
        min_length=1,
        max_length=12,
        description="Concrete editorial requirements for the newsletter.",
    )

    @field_validator("research_queries", mode="before")
    @classmethod
    def _clean_queries(cls, value: object) -> object:
        return as_string_list(value, limit=12)

    @field_validator("sources", mode="before")
    @classmethod
    def _clean_sources(cls, value: object) -> object:
        return as_string_list(value, limit=30)

    @field_validator("newsletter_requirements", mode="before")
    @classmethod
    def _clean_requirements(cls, value: object) -> object:
        return as_string_list(value, limit=12)

    @field_validator("objective", mode="before")
    @classmethod
    def _clean_objective(cls, value: object) -> object:
        return clean_text(value) if isinstance(value, str) else value
