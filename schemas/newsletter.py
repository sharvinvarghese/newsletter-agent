"""Newsletter schemas: the final deliverable produced by the LLM."""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from schemas.base import AgentModel, HttpUrlStr, clean_text


class NewsletterSection(AgentModel):
    """One article inside the newsletter."""

    headline: str = Field(
        min_length=5,
        max_length=200,
        description="Short, informative headline (not a copy of the article title).",
    )
    article_url: HttpUrlStr = Field(
        description="URL copied verbatim from the supplied summaries - never invented.",
    )
    source: str = Field(min_length=1, max_length=200, description="Publisher of the article.")
    summary: str = Field(
        min_length=30,
        max_length=1200,
        description="2-3 sentence summary written for the newsletter audience.",
    )
    why_it_matters: str = Field(
        min_length=20,
        max_length=800,
        description="Editorial takeaway for the reader.",
    )

    @field_validator("headline", "summary", "why_it_matters", "source", mode="before")
    @classmethod
    def _clean(cls, value: Any) -> Any:
        return clean_text(value)


class Newsletter(AgentModel):
    """The structured newsletter. HTML/Markdown are rendered from this object."""

    subject: str = Field(min_length=5, max_length=160, description="Email subject line.")
    preheader: str = Field(
        min_length=5,
        max_length=200,
        description="Preview text shown next to the subject in inboxes.",
    )
    introduction: str = Field(
        min_length=40,
        max_length=2000,
        description="Short opening that sets up this issue for the audience.",
    )
    sections: list[NewsletterSection] = Field(
        min_length=1,
        max_length=12,
        description="One section per selected article (typically 5-7).",
    )
    conclusion: str = Field(
        min_length=20,
        max_length=2000,
        description="Closing paragraph with a forward-looking takeaway.",
    )

    def article_urls(self) -> set[str]:
        """All article URLs referenced by this newsletter."""
        return {section.article_url for section in self.sections}

    @property
    def section_count(self) -> int:
        return len(self.sections)