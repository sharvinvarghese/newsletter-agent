"""Deterministic de-duplication utilities.

This module is deliberately 100% LLM free: two outlets syndicating the same story
must be collapsed *before* tokens are spent ranking them, and the result has to be
reproducible. Every table it uses (tracking parameters, host prefixes, AMP/print
suffixes, HTML entities, filler words, similarity thresholds) comes from the
``deduplication`` block of ``config/tuning.yaml`` - adjust behaviour there.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from config.tuning import DeduplicationTuning, get_tuning
from schemas.articles import DeduplicationResult, DuplicateRecord, NewsArticle

#: Algorithmic patterns (not tuning data): they define what the comparison is.
_TITLE_NOISE_RE = re.compile(r"[^a-z0-9]+")
_TRAILING_SOURCE_RE = re.compile(r"\s*[-|–—:]\s*[a-z0-9 .]{2,40}$")


def _rules() -> DeduplicationTuning:
    """The de-duplication tuning block (loaded once, cached globally)."""
    return get_tuning().deduplication


def default_title_threshold() -> float:
    """Similarity above which two headlines are considered the same story."""
    return _rules().title_threshold


def default_jaccard_weight() -> float:
    """Weight of the token-level Jaccard index inside :func:`title_similarity`."""
    return _rules().jaccard_weight


def canonicalize_url(url: str) -> str:
    """Return a tracking-free, comparable form of ``url``.

    Lower-cases the host, drops the configured prefixes (``www.``, ``amp.``, ...),
    removes tracking query parameters, strips AMP/print/feed suffixes and
    normalises the scheme.
    """
    if not isinstance(url, str) or not url.strip():
        return ""
    parsed = urlparse(url.strip())
    if not parsed.netloc:
        return ""

    rules = _rules()
    scheme = parsed.scheme.lower() or "https"
    host = (parsed.hostname or "").lower()
    for prefix in rules.host_prefixes:
        if host.startswith(prefix) and len(host) > len(prefix) + 3:
            host = host[len(prefix) :]
    port = f":{parsed.port}" if parsed.port and parsed.port not in (80, 443) else ""

    path = parsed.path or "/"
    lowered = path.lower()
    for suffix in rules.path_suffixes:
        if lowered.endswith(suffix.lower()):
            path = path[: -len(suffix)] or "/"
            lowered = path.lower()
    path = path.rstrip("/") or "/"

    tracking = rules.tracking_param_set
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=False)
        if key.lower() not in tracking and not key.lower().startswith("utm_")
    ]
    query = urlencode(sorted(query_items))
    return urlunparse((scheme, f"{host}{port}", path, "", query, ""))


def normalize_title(title: str) -> str:
    """Title with casing, punctuation, entities and publisher suffixes removed."""
    if not isinstance(title, str) or not title.strip():
        return ""
    rules = _rules()
    text = title.strip().lower()
    for entity, replacement in rules.title_entities.items():
        text = text.replace(entity, replacement)
    text = _TRAILING_SOURCE_RE.sub("", text)
    text = _TITLE_NOISE_RE.sub(" ", text)
    words = text.split()
    if not words:
        return ""
    filler = rules.filler_word_set
    filtered = [word for word in words if word not in filler]
    return " ".join(filtered or words)


def title_similarity(left: str, right: str) -> float:
    """Similarity of two headlines in ``[0, 1]``.

    Combines a character level ratio (catches typos/punctuation) with a token
    level Jaccard index (catches reordered wording) plus full containment of a
    long headline in another one. Deliberately conservative so that distinct
    stories from the same publisher stay below the duplicate threshold.
    """
    norm_left, norm_right = normalize_title(left), normalize_title(right)
    if not norm_left or not norm_right:
        return 0.0
    if norm_left == norm_right:
        return 1.0

    ratio = SequenceMatcher(None, norm_left, norm_right).ratio()
    left_tokens, right_tokens = set(norm_left.split()), set(norm_right.split())
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0

    shorter, longer = (norm_left, norm_right) if len(norm_left) <= len(norm_right) else (norm_right, norm_left)
    containment = 1.0 if len(shorter) >= 40 and shorter in longer else 0.0

    weight = default_jaccard_weight()
    blended = ratio * (1.0 - weight) + jaccard * weight
    return round(max(blended, containment), 4)


def find_duplicate_pairs(
    articles: Sequence[NewsArticle],
    *,
    title_threshold: float | None = None,
) -> list[tuple[int, int, str, float | None]]:
    """Return ``(kept_index, duplicate_index, reason, similarity)`` tuples."""
    threshold = default_title_threshold() if title_threshold is None else title_threshold
    pairs: list[tuple[int, int, str, float | None]] = []
    for index, article in enumerate(articles):
        for earlier in range(index):
            reason: str | None = None
            similarity: float | None = None
            canonical, earlier_canonical = article.canonical_url, articles[earlier].canonical_url
            if canonical and canonical == earlier_canonical:
                reason, similarity = "canonical_url", None
            else:
                similarity = title_similarity(article.title, articles[earlier].title)
                if similarity >= 1.0:
                    reason = "normalized_title"
                elif similarity >= threshold:
                    reason = "fuzzy_title"
            if reason:
                pairs.append((earlier, index, reason, similarity))
                break
    return pairs


def deduplicate_articles(
    articles: Iterable[NewsArticle],
    *,
    title_threshold: float | None = None,
) -> DeduplicationResult:
    """Collapse duplicates, keeping the first (best ranked) occurrence.

    Three checks run in this order per article:

    1. identical canonical URL (tracking parameters, AMP variants, ...)
    2. identical normalised title
    3. fuzzy title match above ``title_threshold``
    """
    threshold = default_title_threshold() if title_threshold is None else title_threshold
    kept: list[NewsArticle] = []
    kept_urls: dict[str, str] = {}
    kept_titles: list[tuple[str, str]] = []
    removed: list[DuplicateRecord] = []

    for article in articles:
        canonical = article.canonical_url
        if not canonical:
            removed.append(
                DuplicateRecord(
                    article=article,
                    duplicate_of_url=article.url,
                    reason="canonical_url",
                    similarity=None,
                )
            )
            continue
        if canonical in kept_urls:
            removed.append(
                DuplicateRecord(
                    article=article,
                    duplicate_of_url=kept_urls[canonical],
                    reason="canonical_url",
                    similarity=None,
                )
            )
            continue

        normalized = article.normalized_title
        match_reason: str | None = None
        match_url = ""
        match_similarity: float | None = None
        for kept_title, kept_url in kept_titles:
            if normalized and normalized == kept_title:
                match_reason, match_url, match_similarity = "normalized_title", kept_url, 1.0
                break
            similarity = title_similarity(article.title, kept_title)
            if similarity >= threshold:
                match_reason, match_url, match_similarity = "fuzzy_title", kept_url, similarity
                break

        if match_reason is not None:
            removed.append(
                DuplicateRecord(
                    article=article,
                    duplicate_of_url=match_url,
                    reason=match_reason,  # type: ignore[arg-type]
                    similarity=match_similarity,
                )
            )
            continue

        kept.append(article)
        kept_urls[canonical] = article.url
        kept_titles.append((normalized, article.url))

    return DeduplicationResult(kept_articles=kept, removed_articles=removed)


def __getattr__(name: str) -> Any:
    """Backwards compatible access to the thresholds as module constants."""
    if name == "DEFAULT_TITLE_THRESHOLD":
        return default_title_threshold()
    if name == "DEFAULT_JACCARD_WEIGHT":
        return default_jaccard_weight()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "canonicalize_url",
    "deduplicate_articles",
    "default_jaccard_weight",
    "default_title_threshold",
    "find_duplicate_pairs",
    "normalize_title",
    "title_similarity",
]