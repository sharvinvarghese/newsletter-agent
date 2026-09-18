"""Article schemas: raw articles, evaluations, summaries and research stats."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from schemas.base import (
    AgentModel,
    HttpUrlStr,
    IsoDateTime,
    UnitScore,
    as_string_list,
    clean_text,
)


class NewsArticle(AgentModel):
    """A single article as produced by the research/extraction layer.

    URLs inside this model always come from the crawler - the LLM never invents
    them, it only ranks and summarises articles it was given.
    """

    title: str = Field(min_length=3, max_length=500, description="Article headline.")
    url: HttpUrlStr = Field(description="Absolute article URL taken from the news source.")
    source: str = Field(min_length=1, max_length=200, description="Publisher / feed name.")
    author: str | None = Field(default=None, max_length=300, description="Author names, or null.")
    published_at: IsoDateTime = Field(
        default=None,
        description="ISO-8601 publication timestamp, or null when the feed has none.",
    )
    description: str | None = Field(default=None, max_length=4000, description="Feed summary/teaser.")
    content: str | None = Field(
        default=None,
        description="Full article text extracted with news-please (may be truncated).",
    )
    category: str | None = Field(default=None, max_length=100, description="Source category, e.g. 'ai_news'.")

    @field_validator("title", "source", "author", "description", "content", "category", mode="before")
    @classmethod
    def _clean_strings(cls, value: Any) -> Any:
        return clean_text(value)

    @property
    def canonical_url(self) -> str:
        """Tracking-free, comparable form of :attr:`url` (see tools.deduplication)."""
        # Imported lazily: tools.* depends on schemas.*, so a module level import
        # here would create a circular import.
        from tools.deduplication import canonicalize_url

        return canonicalize_url(self.url)

    @property
    def normalized_title(self) -> str:
        """Title without punctuation/casing noise, used for fuzzy de-duplication."""
        from tools.deduplication import normalize_title

        return normalize_title(self.title)

    def content_for_llm(self, max_chars: int = 6000) -> str:
        """Best available text for the LLM: full text, else teaser, else title."""
        text = (self.content or "").strip() or (self.description or "").strip() or self.title
        if max_chars > 0 and len(text) > max_chars:
            return text[:max_chars].rstrip() + " …[truncated]"
        return text


class ArticleEvaluation(AgentModel):
    """Structured judgement of one article, produced by the evaluator node."""

    relevance_score: UnitScore = Field(description="0.0-1.0 relevance to the newsletter goal and queries.")
    quality_score: UnitScore = Field(description="0.0-1.0 information depth and source quality.")
    recency_score: UnitScore = Field(description="0.0-1.0 freshness (1.0 = published today).")
    duplicate: bool = Field(
        default=False,
        description="True when the article covers the same story as another candidate.",
    )
    selected: bool = Field(default=False, description="True when it should be part of the newsletter.")
    reason: str = Field(min_length=3, max_length=800, description="One or two sentences justifying the scores.")

    @property
    def composite_score(self) -> float:
        """Weighted score using ``scoring.weights`` from ``config/tuning.yaml``."""
        from config.tuning import get_tuning

        return get_tuning().scoring.combined(
            relevance=self.relevance_score,
            quality=self.quality_score,
            recency=self.recency_score,
        )


class ArticleEvaluationItem(ArticleEvaluation):
    """An evaluation bound to the article URL it belongs to."""

    url: HttpUrlStr = Field(description="URL copied verbatim from the candidate list.")


class ArticleEvaluationBatch(AgentModel):
    """Batch response shape for the evaluator (one LLM call per small batch)."""

    evaluations: list[ArticleEvaluationItem] = Field(
        min_length=1,
        max_length=25,
        description="One evaluation per supplied candidate article.",
    )


class EvaluatedArticle(AgentModel):
    """A candidate article together with its evaluation."""

    article: NewsArticle
    evaluation: ArticleEvaluation
    evaluation_source: Literal["llm", "deterministic_fallback"] = Field(
        default="llm",
        description="Where the evaluation came from; fallbacks are used when the LLM fails.",
    )

    @property
    def score(self) -> float:
        return self.evaluation.composite_score

    @property
    def url(self) -> str:
        return self.article.url


class ArticleSummary(AgentModel):
    """LLM summary of a selected article (the only article data used downstream)."""

    title: str = Field(min_length=3, max_length=500, description="Headline of the summarised article.")
    source: str = Field(min_length=1, max_length=200, description="Publisher name.")
    url: HttpUrlStr = Field(description="URL copied verbatim from the supplied article.")
    summary: str = Field(min_length=30, max_length=1500, description="Neutral 2-4 sentence summary.")
    key_points: list[str] = Field(
        min_length=1,
        max_length=6,
        description="2-5 concrete facts, names or numbers taken from the article.",
    )
    why_it_matters: str = Field(
        min_length=20,
        max_length=800,
        description="Why this matters to the newsletter audience.",
    )

    @field_validator("key_points", mode="before")
    @classmethod
    def _clean_points(cls, value: Any) -> Any:
        return as_string_list(value, limit=6)


class ArticleSummaryBatch(AgentModel):
    """Batch response shape for summarisation (one LLM call for multiple articles)."""

    summaries: list[ArticleSummary] = Field(
        min_length=1,
        max_length=25,
        description="One summary per supplied article, in the same order as the input.",
    )


class DuplicateRecord(AgentModel):
    """A deterministic duplicate finding (no LLM involved)."""

    article: NewsArticle
    duplicate_of_url: HttpUrlStr = Field(description="URL of the article that was kept instead.")
    reason: Literal["canonical_url", "normalized_title", "fuzzy_title"] = Field(
        description="Why the article was classified as a duplicate."
    )
    similarity: float | None = Field(default=None, ge=0.0, le=1.0, description="Title similarity, 0-1.")


class DeduplicationResult(AgentModel):
    """Outcome of the deterministic de-duplication step."""

    kept_articles: list[NewsArticle] = Field(default_factory=list)
    removed_articles: list[DuplicateRecord] = Field(default_factory=list)

    @property
    def removed_count(self) -> int:
        return len(self.removed_articles)

    @property
    def input_count(self) -> int:
        return len(self.kept_articles) + len(self.removed_articles)


class ResearchStats(AgentModel):
    """Counters that explain what the research tool actually did."""

    sources_configured: int = 0
    sources_attempted: int = 0
    sources_succeeded: int = 0
    items_read: int = 0
    raw_articles: int = 0
    after_recency_filter: int = 0
    after_keyword_filter: int = 0
    after_url_dedup: int = 0
    enriched_articles: int = 0
    failed_sources: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ResearchOutcome(AgentModel):
    """Articles returned by the research tool plus its statistics."""

    articles: list[NewsArticle] = Field(default_factory=list)
    stats: ResearchStats = Field(default_factory=ResearchStats)