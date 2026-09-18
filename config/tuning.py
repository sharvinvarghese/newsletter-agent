"""Typed access to ``config/tuning.yaml``.

Nothing behavioural is hardcoded in the Python modules: weights, thresholds,
keyword heuristics, prompt budgets, filename rules and the dynamic search-feed
URL template all come from this file. Swapping the YAML changes the agent's
behaviour without touching a single line of code.

``extra="forbid"`` is deliberate - a typo in the YAML raises immediately instead
of silently falling back to a default.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("newsletter_agent.config.tuning")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TUNING_PATH = PROJECT_ROOT / "config" / "tuning.yaml"


class TuningError(RuntimeError):
    """Raised when ``tuning.yaml`` is missing or invalid."""


class _Block(BaseModel):
    """Base class for every tuning block."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AuditTuning(_Block):
    """Deterministic checks the critic cannot overrule."""

    coverage_trigger: int = Field(default=5, ge=0)
    min_covered_sections: int = Field(default=5, ge=1)
    min_summary_chars: int = Field(default=80, ge=1)


class NewsletterTuning(_Block):
    target_articles: int = Field(default=6, ge=1)
    min_articles: int = Field(default=5, ge=1)
    max_articles: int = Field(default=7, ge=1)
    hard_min_articles: int = Field(default=1, ge=1)
    hard_max_articles: int = Field(default=12, ge=1)
    min_sections: int = Field(default=1, ge=1)
    max_sections: int = Field(default=12, ge=1)
    output_formats: list[str] = Field(default_factory=lambda: ["html+markdown", "html", "markdown"])
    audit: AuditTuning = Field(default_factory=AuditTuning)
    fallback_requirements: list[str] = Field(default_factory=list)

    @property
    def default_format(self) -> str:
        return self.output_formats[0] if self.output_formats else "html+markdown"

    def clamp(self, value: Any) -> int:
        """Clamp any requested article count into the legal range."""
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = self.target_articles
        return max(self.hard_min_articles, min(self.hard_max_articles, number))

    def clamp_collection(self, value: Any, *, ceiling: int | None = None) -> int:
        """Clamp the number of collected candidates."""
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = self.target_articles * 4
        upper = min(self.hard_max_articles * 10, ceiling or self.hard_max_articles * 10)
        return max(self.min_articles, min(upper, number))


class ScoringTuning(_Block):
    min_score: float = Field(default=0.0)
    max_score: float = Field(default=1.0)
    weights: dict[str, float] = Field(default_factory=dict)
    min_relevance: float = Field(default=0.35, ge=0.0, le=1.0)
    min_content_chars: int = Field(default=400, ge=0)
    unknown_date_recency: float = Field(default=0.5, ge=0.0, le=1.0)
    fallback_horizon_days: int = Field(default=30, ge=1)
    title_hit_weight: float = Field(default=2.0)
    body_hit_weight: float = Field(default=1.0)
    keyword_hits_cap: int = Field(default=3, ge=1)
    source_weight_default: float = Field(default=0.6, ge=0.0, le=1.0)
    quality_boosters: list[str] = Field(default_factory=list)
    quality_penalties: list[str] = Field(default_factory=list)

    def weight(self, name: str, default: float = 0.0) -> float:
        return float(self.weights.get(name, default))

    def combined(self, *, relevance: float, quality: float, recency: float) -> float:
        """Weighted final score of one evaluated article."""
        total = (
            relevance * self.weight("relevance")
            + quality * self.weight("quality")
            + recency * self.weight("recency")
        )
        return round(max(0.0, min(1.0, total)), 4)

    def normalised(self, value: float, *, default: float = 0.5) -> float:
        """Map a raw value onto the configured 0..1 score scale."""
        span = max(1e-9, self.max_score - self.min_score)
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return round(max(0.0, min(1.0, (number - self.min_score) / span)), 4)


class LlmTuning(_Block):
    """Budgets and temperatures per pipeline stage."""

    text_chars: dict[str, int] = Field(default_factory=dict)
    temperatures: dict[str, float] = Field(default_factory=dict)
    evaluation_max_selected: int = Field(default=7, ge=1)

    def text_budget(self, stage: str, default: int = 2000) -> int:
        """How many characters of an article are sent to ``stage``."""
        return int(self.text_chars.get(stage, default))

    def temperature(self, stage: str) -> float | None:
        """Sampling temperature for ``stage`` (``None`` = use the global setting)."""
        value = self.temperatures.get(stage)
        return None if value is None else float(value)


class PlanningTuning(_Block):
    min_queries: int = Field(default=4, ge=1)
    max_queries: int = Field(default=8, ge=1)
    min_requirements: int = Field(default=3, ge=1)
    max_requirements: int = Field(default=6, ge=1)
    min_collect: int = Field(default=10, ge=1)
    max_collect: int = Field(default=100, ge=1)


class WritingTuning(_Block):
    max_topics: int = Field(default=3, ge=1)
    summary_min_points: int = Field(default=2, ge=1)
    summary_max_points: int = Field(default=4, ge=1)
    summary_min_sentences: int = Field(default=2, ge=1)
    summary_max_sentences: int = Field(default=3, ge=1)
    section_min_sentences: int = Field(default=2, ge=1)
    section_max_sentences: int = Field(default=3, ge=1)
    intro_min_sentences: int = Field(default=2, ge=1)
    intro_max_sentences: int = Field(default=3, ge=1)


class HeuristicsTuning(_Block):
    """Only used by the deterministic fallbacks (when the LLM path fails)."""

    topic_keywords: dict[str, list[str]] = Field(default_factory=dict)
    generic_goal_words: list[str] = Field(default_factory=list)
    fallback_topic: str = "technology news"
    min_goal_chars: int = Field(default=5, ge=1)
    frequency_hints: dict[str, list[str]] = Field(default_factory=dict)
    default_frequency: str = "weekly"
    audience_patterns: list[str] = Field(default_factory=list)
    article_count_patterns: list[str] = Field(default_factory=list)
    human_in_loop_hints: list[str] = Field(default_factory=list)
    format_hints: dict[str, list[str]] = Field(default_factory=dict)

    @property
    def topic_rules(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Ordered ``(label, keywords)`` pairs - the YAML order defines priority."""
        return tuple((label, tuple(terms)) for label, terms in self.topic_keywords.items())

    @property
    def generic_word_set(self) -> frozenset[str]:
        return frozenset(word.lower() for word in self.generic_goal_words)

    @property
    def frequency_rules(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return tuple((label, tuple(hints)) for label, hints in self.frequency_hints.items())

    @property
    def frequency_labels(self) -> list[str]:
        return list(self.frequency_hints) or [self.default_frequency]

    def frequency_label(self, text: str) -> str:
        """Cadence mentioned in ``text`` (default when nothing matches)."""
        lowered = (text or "").lower()
        for label, hints in self.frequency_rules:
            if any(hint in lowered for hint in hints):
                return label
        return self.default_frequency

    def format_hint(self, text: str) -> str:
        """Detect a requested output format from free text."""
        lowered = (text or "").lower()
        wanted = [
            name
            for name, hints in self.format_hints.items()
            if any(hint in lowered for hint in hints)
        ]
        if not wanted:
            return ""
        return "+".join(dict.fromkeys(wanted))


class HackerNewsTuning(_Block):
    """Dynamic Hacker News search via the free Algolia API (no key, no limits)."""

    enabled: bool = True
    provider: str = "Hacker News"
    display_name: str = "{provider}"
    endpoint: str = "https://hn.algolia.com/api/v1/search"
    tags: str = "story"
    max_queries: int = Field(default=4, ge=1)
    max_items: int = Field(default=12, ge=1)
    weight: float = Field(default=0.9, ge=0.0, le=2.0)
    category: str = "community"
    min_points: int = Field(default=0, ge=0)

    def query_url(self, query: str, *, recency_days: int | None = None) -> str:
        """Build the Algolia search URL for one planner query."""
        from datetime import datetime, timezone
        from urllib.parse import urlencode

        params = [
            ("query", (query or "").strip()),
            ("tags", self.tags),
            ("hitsPerPage", str(self.max_items)),
        ]
        if recency_days and recency_days > 0:
            cutoff = int(datetime.now(timezone.utc).timestamp()) - int(recency_days) * 86400
            params.append(("numericFilters", f"created_at_i>{cutoff}"))
        return f"{self.endpoint.rstrip('/')}?{urlencode(params)}"

    def label(self, query: str) -> str:
        """Human readable source name for one query."""
        base = self.display_name.replace("{provider}", self.provider)
        return f"{base}: {query}".strip()


class ResearchTuning(_Block):
    stopwords: list[str] = Field(default_factory=list)
    min_term_length: int = Field(default=3, ge=1)
    max_terms: int = Field(default=60, ge=1)
    sitemap_child_limit: int = Field(default=2, ge=0)
    sitemap_skip_extensions: list[str] = Field(default_factory=list)
    url_skip_fragments: list[str] = Field(default_factory=list)
    hacker_news: HackerNewsTuning = Field(default_factory=HackerNewsTuning)

    @property
    def stopword_set(self) -> frozenset[str]:
        return frozenset(word.lower() for word in self.stopwords)


class FilenameTuning(_Block):
    prefix: str = "newsletter"
    subject_slug_chars: int = Field(default=48, ge=8)
    timestamp_format: str = "%Y%m%d-%H%M%S"


class FooterTuning(_Block):
    attribution: str = ""
    disclaimer: str = ""


class RenderingTuning(_Block):
    """Where the output templates live and how the artifacts are named."""

    templates_dir: Path = Path("config/templates")
    html_template: str = "newsletter.html.j2"
    markdown_template: str = "newsletter.md.j2"
    filename: FilenameTuning = Field(default_factory=FilenameTuning)
    footer: FooterTuning = Field(default_factory=FooterTuning)
    simulated_delivery_note: str = ""

    def template_dir(self) -> Path:
        """Absolute templates directory (relative paths resolve from the project)."""
        return self.templates_dir if self.templates_dir.is_absolute() else PROJECT_ROOT / self.templates_dir


class DeduplicationTuning(_Block):
    """Tables used by the deterministic de-duplication step."""

    title_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    jaccard_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    tracking_params: list[str] = Field(default_factory=list)
    host_prefixes: list[str] = Field(default_factory=list)
    path_suffixes: list[str] = Field(default_factory=list)
    title_entities: dict[str, str] = Field(default_factory=dict)
    filler_words: list[str] = Field(default_factory=list)

    @property
    def tracking_param_set(self) -> frozenset[str]:
        return frozenset(param.lower() for param in self.tracking_params)

    @property
    def filler_word_set(self) -> frozenset[str]:
        return frozenset(word.lower() for word in self.filler_words)


class Tuning(_Block):
    """The complete tuning document."""

    version: int = 1
    newsletter: NewsletterTuning = Field(default_factory=NewsletterTuning)
    scoring: ScoringTuning = Field(default_factory=ScoringTuning)
    llm: LlmTuning = Field(default_factory=LlmTuning)
    planning: PlanningTuning = Field(default_factory=PlanningTuning)
    writing: WritingTuning = Field(default_factory=WritingTuning)
    heuristics: HeuristicsTuning = Field(default_factory=HeuristicsTuning)
    research: ResearchTuning = Field(default_factory=ResearchTuning)
    rendering: RenderingTuning = Field(default_factory=RenderingTuning)
    deduplication: DeduplicationTuning = Field(default_factory=DeduplicationTuning)

    def temperature(self, stage: str) -> float | None:
        return self.llm.temperature(stage)

    def text_budget(self, stage: str, default: int = 2000) -> int:
        return self.llm.text_budget(stage, default)

    def consistency_issues(self) -> list[str]:
        """Cross-field checks, run right after loading."""
        issues: list[str] = []
        news, planning = self.newsletter, self.planning
        if news.min_articles > news.max_articles:
            issues.append("newsletter.min_articles is larger than newsletter.max_articles")
        if news.hard_max_articles < news.max_articles:
            issues.append("newsletter.hard_max_articles is smaller than newsletter.max_articles")
        if news.hard_min_articles > news.min_articles:
            issues.append("newsletter.hard_min_articles is larger than newsletter.min_articles")
        if news.min_sections > news.max_sections:
            issues.append("newsletter.min_sections is larger than newsletter.max_sections")
        if not news.output_formats:
            issues.append("newsletter.output_formats is empty")
        if planning.min_collect > planning.max_collect:
            issues.append("planning.min_collect is larger than planning.max_collect")
        if planning.min_queries > planning.max_queries:
            issues.append("planning.min_queries is larger than planning.max_queries")
        weight_sum = sum(self.scoring.weights.get(key, 0.0) for key in ("relevance", "quality", "recency"))
        if abs(weight_sum - 1.0) > 1e-6:
            issues.append(f"scoring.weights must sum to 1.0 (currently {weight_sum:.3f})")
        hn = self.research.hacker_news
        if hn.enabled and not str(hn.endpoint).startswith("https://"):
            issues.append("research.hacker_news.enabled is true but endpoint is not https")
        return issues


def load_tuning(path: str | Path | None = None) -> Tuning:
    """Read, validate and consistency-check ``tuning.yaml``."""
    config_path = Path(path) if path else DEFAULT_TUNING_PATH
    if not config_path.exists():
        raise TuningError(f"Tuning configuration not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise TuningError(f"{config_path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise TuningError(f"{config_path} must contain a YAML mapping")
    try:
        tuning = Tuning.model_validate(raw)
    except Exception as exc:
        raise TuningError(f"{config_path} is invalid: {exc}") from exc
    for issue in tuning.consistency_issues():
        logger.warning("Tuning issue in %s: %s", config_path.name, issue)
    return tuning


_TUNING: Tuning | None = None


def get_tuning(refresh: bool = False) -> Tuning:
    """Process wide tuning singleton (``refresh=True`` re-reads the YAML)."""
    global _TUNING
    if _TUNING is None or refresh:
        _TUNING = load_tuning()
    return _TUNING
