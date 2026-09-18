"""Controlled news research: curated feeds plus dynamic Hacker News search.

Two kinds of source are read, both free and open:

* the curated RSS/sitemap list in ``config/sources.yaml`` (reproducible, curated
  by the operator), and
* dynamic Hacker News queries built from the planner's queries via the free
  Algolia API (``research.hacker_news`` in ``tuning.yaml`` - JSON, no API key,
  no rate limits, and it returns the *real publisher URL* instead of an
  aggregator redirect). This is what keeps a run from being limited to a
  hardcoded list of outlets.

Why feedparser instead of news-please's ``RssCrawler``? news-please's crawler
classes are scrapy spiders that need a Twisted reactor and (by default) a
MongoDB/Elasticsearch publisher - a poor fit for a Streamlit process. This module
therefore reads the feeds itself and uses news-please (plus trafilatura) where
they are strongest: metadata and boilerplate-free full-text extraction, see
:mod:`tools.article_extractor`.
"""

from __future__ import annotations

import html as html_lib
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import feedparser
import requests
import yaml
from config.settings import DEFAULT_SOURCES_PATH, Settings, get_settings
from config.tuning import HackerNewsTuning, get_tuning
from pydantic import Field, field_validator
from schemas.articles import NewsArticle, ResearchOutcome, ResearchStats
from schemas.base import AgentModel, HttpUrlStr, as_string_list

from tools.article_extractor import DEFAULT_USER_AGENT, extract_many

logger = logging.getLogger("newsletter_agent.tools.research")

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9+.#\-]{2,}")
_TAG_RE = re.compile(r"<[^>]+>")
_LOC_RE = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>", re.IGNORECASE)


def _hn_rules() -> HackerNewsTuning:
    """Dynamic Hacker News search rules from ``config/tuning.yaml``."""
    return get_tuning().research.hacker_news


def extract_terms(queries: Iterable[str], *, limit: int | None = None) -> list[str]:
    """Turn research queries into comparable lowercase search terms.

    The stopword list, minimum word length and term budget come from
    ``research.*`` in ``config/tuning.yaml``.
    """
    rules = get_tuning().research
    budget = rules.max_terms if limit is None else limit
    minimum = rules.min_term_length
    stopwords = rules.stopword_set
    terms: list[str] = []
    for query in queries:
        for word in _WORD_RE.findall((query or "").lower()):
            if len(word) < minimum or word in stopwords or word in terms:
                continue
            terms.append(word)
            if len(terms) >= budget:
                return terms
    return terms


class SourceConfigError(RuntimeError):
    """Raised when ``sources.yaml`` is missing or invalid."""


class NewsSource(AgentModel):
    """One configured feed or sitemap."""

    id: str = Field(min_length=2, max_length=60)
    name: str = Field(min_length=2, max_length=120)
    url: HttpUrlStr
    type: Literal["rss", "atom", "sitemap", "hackernews"] = "rss"
    category: str | None = Field(default=None, max_length=80)
    topics: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    weight: float = Field(default=0.8, ge=0.0, le=2.0)
    enabled: bool = True
    max_items: int | None = Field(default=None, ge=1, le=100)

    @field_validator("topics", "keywords", mode="before")
    @classmethod
    def _clean_lists(cls, value: Any) -> Any:
        return as_string_list(value, limit=40)

    @property
    def is_feed(self) -> bool:
        return self.type in {"rss", "atom"}

    @property
    def search_terms(self) -> list[str]:
        """Keywords plus the words of the topic labels, lower-cased."""
        terms = [term.lower() for term in self.keywords]
        for topic in self.topics:
            terms.extend(part for part in topic.lower().replace("_", " ").split() if part)
        return list(dict.fromkeys(terms))


def strip_html(text: str | None) -> str:
    """Remove tags/entities from feed summaries."""
    if not text:
        return ""
    return html_lib.unescape(_TAG_RE.sub(" ", str(text))).strip()


class HttpConfig(AgentModel):
    """Knobs from the ``http`` section of ``sources.yaml``."""

    user_agent: str = DEFAULT_USER_AGENT
    timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    max_items_per_source: int = Field(default=25, ge=1, le=100)
    default_recency_window_days: int = Field(default=14, ge=0, le=365)
    max_workers: int = Field(default=4, ge=1, le=16)
    enrich_full_text: bool = True
    enrich_limit: int = Field(default=12, ge=0, le=50)


class SourcesConfig(AgentModel):
    """Validated content of ``config/sources.yaml``."""

    version: int = 1
    http: HttpConfig = Field(default_factory=HttpConfig)
    sources: list[NewsSource] = Field(min_length=1)

    def enabled_sources(self) -> list[NewsSource]:
        return [source for source in self.sources if source.enabled]

    def source_index(self) -> dict[str, NewsSource]:
        """Lookup table for planner supplied identifiers (id, name, category)."""
        index: dict[str, NewsSource] = {}
        for source in self.sources:
            keys = {source.id.lower(), source.name.lower()}
            if source.category:
                keys.add(source.category.lower())
            for key in keys:
                index.setdefault(key, source)
                index.setdefault(key.replace("-", "_").replace(" ", "_"), source)
        return index

    def resolve(self, identifiers: Sequence[str]) -> tuple[list[NewsSource], list[str]]:
        """Map planner identifiers onto configured sources.

        Returns the resolved sources (input order, de-duplicated) plus the
        identifiers that could not be matched.
        """
        index = self.source_index()
        resolved: list[NewsSource] = []
        unknown: list[str] = []
        for identifier in identifiers:
            key = str(identifier).strip().lower()
            source = index.get(key) or index.get(key.replace("-", "_").replace(" ", "_"))
            if source is None:
                unknown.append(str(identifier))
                continue
            if source not in resolved:
                resolved.append(source)
        return resolved, unknown


def load_sources(path: str | Path | None = None) -> SourcesConfig:
    """Read and validate the source configuration file."""
    config_path = Path(path) if path else DEFAULT_SOURCES_PATH
    if not config_path.exists():
        raise SourceConfigError(f"Source configuration not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise SourceConfigError(f"{config_path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise SourceConfigError(f"{config_path} must contain a YAML mapping")
    try:
        return SourcesConfig.model_validate(raw)
    except Exception as exc:
        raise SourceConfigError(f"{config_path} is invalid: {exc}") from exc


class NewsResearchTool:
    """Gather candidate articles from the configured sources.

    The planner decides *what* to look for (queries and preferred sources), this
    tool decides *how* to fetch it. It never invents articles: when the feeds
    return nothing the run simply has fewer candidates.

    Feeds are read sequentially because a feed is small and this keeps the run
    deterministic; the (much slower) per-article full-text extraction is
    parallelised inside :func:`tools.article_extractor.extract_many`.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        sources_config: SourcesConfig | None = None,
        sources_path: str | Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._sources_path = sources_path
        self._config = sources_config

    # --- configuration ------------------------------------------------------
    @property
    def config(self) -> SourcesConfig:
        """The validated source configuration (loaded on first use)."""
        if self._config is None:
            self._config = load_sources(self._sources_path or self.settings.sources_path)
        return self._config

    def reload_sources(self) -> SourcesConfig:
        """Drop the cached configuration and read the file again."""
        self._config = None
        return self.config

    @property
    def http(self) -> HttpConfig:
        return self.config.http

    def _headers(self, accept: str) -> dict[str, str]:
        return {
            "User-Agent": self.http.user_agent,
            "Accept": accept,
            "Accept-Language": "en-US,en;q=0.9",
        }

    # --- source selection ---------------------------------------------------
    def score_source(self, source: NewsSource, terms: Sequence[str]) -> float:
        """Deterministic relevance of a source for the given research terms."""
        if not terms:
            return source.weight
        haystack = " ".join([source.name.lower(), source.category or "", *source.search_terms])
        hits = sum(1 for term in terms if term in haystack)
        return round(source.weight * (1.0 + hits), 4)

    def select_sources(
        self,
        queries: Sequence[str],
        *,
        requested: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[NewsSource]:
        """Sources to query: planner hints first, then the best matching ones."""
        config = self.config
        terms = extract_terms(queries)
        max_sources = max(1, limit or self.settings.research_max_sources)

        selected: list[NewsSource] = []
        if requested:
            resolved, unknown = config.resolve(requested)
            if unknown:
                logger.info("Plan referenced %d unknown source(s): %s", len(unknown), ", ".join(unknown))
            selected.extend(resolved)

        ranked = sorted(
            config.enabled_sources(),
            key=lambda source: (-self.score_source(source, terms), source.name.lower()),
        )
        for source in ranked:
            if len(selected) >= max_sources:
                break
            if source not in selected:
                selected.append(source)
        return selected[:max_sources]

    # --- dynamic Hacker News queries (config driven) -------------------------
    def search_sources(self, queries: Sequence[str], *, recency_days: int | None = None) -> list[NewsSource]:
        """One Hacker News query source per research query.

        The rules live in ``research.hacker_news`` of ``tuning.yaml`` (endpoint,
        tags, limits, weight). Because these sources are derived from the
        planner's queries, a run is not limited to the curated list in
        ``sources.yaml`` - and every hit returns the *real publisher URL*, so no
        redirect decoding is involved. ``recency_days`` is accepted for
        signature compatibility; the strict recency filter happens in
        :meth:`research` after collection.
        """
        rules = _hn_rules()
        if not rules.enabled:
            return []
        unique: list[str] = []
        seen: set[str] = set()
        for query in queries:
            text = (query or "").strip()
            if text and text.lower() not in seen:
                seen.add(text.lower())
                unique.append(text)

        sources: list[NewsSource] = []
        for index, query in enumerate(unique[: rules.max_queries]):
            try:
                sources.append(
                    NewsSource(
                        id=f"hn_{index + 1}",
                        name=rules.label(query)[:120],
                        url=rules.endpoint,
                        type="hackernews",
                        category=rules.category or None,
                        keywords=[query],
                        weight=rules.weight,
                        max_items=rules.max_items,
                    )
                )
            except Exception as exc:  # one bad query must not stop the run
                logger.warning("Could not build a Hacker News source for %r: %s", query, exc)
        return sources

    def _fetch_hackernews(self, source: NewsSource) -> tuple[list[NewsArticle], list[str]]:
        """Query the free Hacker News (Algolia) API for this source's search term.

        The response is plain JSON and every hit carries the *real publisher
        URL*, so no redirect decoding is involved. For the dynamic sources built
        by :meth:`search_sources` the single keyword is the planner's research
        query; curated sources in ``sources.yaml`` use their first keyword.
        """
        rules = _hn_rules()
        if not rules.enabled:
            return [], [f"{source.id}: hacker news search is disabled in tuning.yaml"]
        query = (source.keywords[0] if source.keywords else source.name).strip()
        limit = min(source.max_items or rules.max_items, rules.max_items)

        params: dict[str, str] = {"query": query, "tags": rules.tags, "hitsPerPage": str(limit)}
        filters: list[str] = []
        window = self.http.default_recency_window_days
        if window > 0:
            cutoff = int(datetime.now(timezone.utc).timestamp()) - window * 86400
            filters.append(f"created_at_i>{cutoff}")
        if rules.min_points > 0:
            filters.append(f"points>={rules.min_points}")
        if filters:
            params["numericFilters"] = ",".join(filters)

        try:
            response = requests.get(
                rules.endpoint,
                params=params,
                timeout=self.http.timeout_seconds,
                headers=self._headers("application/json"),
            )
        except requests.RequestException as exc:
            return [], [f"{source.id}: request failed ({type(exc).__name__})"]
        if response.status_code >= 400:
            return [], [f"{source.id}: HTTP {response.status_code}"]
        try:
            hits = (response.json() or {}).get("hits") or []
        except ValueError:
            return [], [f"{source.id}: response was not JSON"]

        articles = [article for article in (self._hn_hit_to_article(hit, source) for hit in hits) if article]
        problems: list[str] = []
        if not hits:
            problems.append(f"{source.id}: hacker news query {query!r} returned no hits")
        elif not articles:
            problems.append(f"{source.id}: {len(hits)} hits but none usable")
        return articles, problems

    def _hn_hit_to_article(self, hit: Mapping[str, Any], source: NewsSource) -> NewsArticle | None:
        """Convert one Algolia hit into a validated :class:`NewsArticle`."""
        title = strip_html(hit.get("title") or hit.get("story_title"))
        link = str(hit.get("url") or "").strip()
        if not link:  # Ask HN / discussions: fall back to the thread itself.
            object_id = str(hit.get("objectID") or "").strip()
            link = f"https://news.ycombinator.com/item?id={object_id}" if object_id else ""
        if len(title) < 3 or not link:
            return None

        published_at: str | None = None
        created = hit.get("created_at_i")
        if created:
            try:
                published_at = datetime.fromtimestamp(int(created), tz=timezone.utc).isoformat()
            except (TypeError, ValueError, OverflowError, OSError):
                published_at = None
        try:
            return NewsArticle(
                title=title[:500],
                url=link,
                source=source.name,
                author=strip_html(hit.get("author")) or None,
                published_at=published_at,
                description=strip_html(hit.get("story_text"))[:2000] or None,
                category=source.category,
            )
        except Exception as exc:
            logger.debug("Skipping unusable Hacker News hit from %s: %s", source.id, exc)
            return None

    # --- main entry point ---------------------------------------------------
    def research(
        self,
        *,
        queries: Sequence[str],
        sources: Sequence[str] | None = None,
        max_articles: int | None = None,
        recency_window_days: int | None = None,
        enrich: bool | None = None,
    ) -> ResearchOutcome:
        """Collect, filter, rank and (optionally) enrich candidate articles."""
        settings = self.settings
        terms = extract_terms(queries)
        try:
            config = self.config
        except SourceConfigError as exc:
            logger.error("Research is unavailable: %s", exc)
            return ResearchOutcome(articles=[], stats=ResearchStats(notes=[str(exc)]))

        http = config.http
        window = http.default_recency_window_days if recency_window_days is None else recency_window_days
        budget = max(1, max_articles or settings.max_articles_to_collect)

        selected = self.select_sources(queries, requested=sources, limit=settings.research_max_sources)
        dynamic = self.search_sources(queries, recency_days=window)
        stats = ResearchStats(
            sources_configured=len(config.enabled_sources()) + len(dynamic),
            sources_attempted=len(selected) + len(dynamic),
        )
        queue = [*selected, *dynamic]
        if not queue:
            stats.notes.append("No sources were selected for this run.")
            return ResearchOutcome(articles=[], stats=stats)

        collected: list[NewsArticle] = []
        for source in queue:
            try:
                articles, problems = self.fetch_source(source, terms=terms)
            except Exception as exc:  # one broken source must never kill the run
                articles, problems = [], [f"{source.id}: {type(exc).__name__}: {exc}"]
            if problems:
                stats.failed_sources.append(source.id)
                stats.notes.extend(problems[:2])
                logger.warning("Source %s had problems: %s", source.id, "; ".join(problems[:2]))
            else:
                stats.sources_succeeded += 1
            stats.items_read += len(articles)
            collected.extend(articles)

        stats.raw_articles = len(collected)
        recent = [article for article in collected if self.is_recent(article, window)]
        stats.after_recency_filter = len(recent)

        unique: list[NewsArticle] = []
        seen: set[str] = set()
        for article in recent:
            key = article.canonical_url
            if key and key in seen:
                continue
            seen.add(key)
            unique.append(article)
        stats.after_url_dedup = len(unique)

        scored = [(self.keyword_score(article, terms), article) for article in unique]
        stats.after_keyword_filter = sum(1 for score, _ in scored if score > 0)
        ranked = [article for _, article in sorted(scored, key=lambda item: (-item[0], item[1].title))]

        shortlist = ranked[:budget]
        should_enrich = (http.enrich_full_text if enrich is None else enrich) and http.enrich_limit > 0
        if should_enrich and shortlist:
            shortlist, enriched_count = self._enrich(shortlist, http.enrich_limit)
            stats.enriched_articles = enriched_count

        stats.notes.append(
            f"Collected {len(collected)} articles from {stats.sources_succeeded}/{len(selected)} sources; "
            f"kept {len(shortlist)} after recency/keyword filtering."
        )
        return ResearchOutcome(articles=shortlist, stats=stats)

    # --- enrichment & scoring helpers ---------------------------------------
    def _enrich(self, articles: Sequence[NewsArticle], limit: int) -> tuple[list[NewsArticle], int]:
        """Fetch full text with news-please for the best candidates."""
        http = self.http
        targets = [article for article in articles if not article.content][:limit]
        if not targets:
            return list(articles), 0
        extracted = extract_many(
            [article.url for article in targets],
            limit=limit,
            workers=http.max_workers,
            timeout=http.timeout_seconds,
            user_agent=http.user_agent,
            source_names={article.url: article.source for article in targets},
        )
        if not extracted:
            return list(articles), 0
        by_url = {article.canonical_url: article for article in extracted}
        merged = [
            self._merge(article, by_url[article.canonical_url])
            if article.canonical_url in by_url
            else article
            for article in articles
        ]
        return merged, len(extracted)

    @staticmethod
    def _merge(feed_article: NewsArticle, extracted: NewsArticle) -> NewsArticle:
        """Use extracted fields only where the feed had nothing better."""
        updates: dict[str, Any] = {}
        if extracted.content:
            updates["content"] = extracted.content
        if extracted.description and not feed_article.description:
            updates["description"] = extracted.description
        if extracted.author and not feed_article.author:
            updates["author"] = extracted.author
        if extracted.published_at and not feed_article.published_at:
            updates["published_at"] = extracted.published_at
        return feed_article.model_copy(update=updates) if updates else feed_article

    @staticmethod
    def is_recent(article: NewsArticle, window_days: int) -> bool:
        """Recency filter. Articles without a date are kept (they may still be new)."""
        if window_days <= 0 or not article.published_at:
            return True
        try:
            published = datetime.fromisoformat(str(article.published_at))
        except ValueError:
            return True
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        return published >= datetime.now(timezone.utc) - timedelta(days=window_days)

    @staticmethod
    def keyword_score(article: NewsArticle, terms: Sequence[str]) -> float:
        """Keyword overlap score: title hits count double, body hits single."""
        if not terms:
            return 0.0
        title = article.title.lower()
        body = f"{article.description or ''} {article.content or ''}".lower()
        score = 0.0
        for term in terms:
            if term in title:
                score += 2.0
            elif term in body:
                score += 1.0
        return round(score, 3)

    # --- fetching -----------------------------------------------------------
    def fetch_source(self, source: NewsSource, *, terms: Sequence[str] | None = None) -> tuple[list[NewsArticle], list[str]]:
        """Fetch one source. Returns ``(articles, problems)``."""
        if source.type == "hackernews":
            return self._fetch_hackernews(source)
        if source.is_feed:
            return self._fetch_feed(source)
        return self._fetch_sitemap(source, terms=terms or [])

    def _fetch_feed(self, source: NewsSource) -> tuple[list[NewsArticle], list[str]]:
        """Read one RSS/Atom feed with feedparser."""
        http = self.http
        limit = source.max_items or http.max_items_per_source
        try:
            response = requests.get(
                source.url,
                timeout=http.timeout_seconds,
                headers=self._headers(
                    "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.8"
                ),
            )
        except requests.RequestException as exc:
            return [], [f"{source.id}: request failed ({type(exc).__name__})"]
        if response.status_code >= 400:
            return [], [f"{source.id}: HTTP {response.status_code}"]

        parsed = feedparser.parse(response.content)
        entries = list(getattr(parsed, "entries", None) or [])[:limit]
        articles = [article for article in (self._entry_to_article(entry, source) for entry in entries) if article]

        problems: list[str] = []
        if not entries:
            problems.append(f"{source.id}: feed returned no entries")
        elif not articles:
            problems.append(f"{source.id}: {len(entries)} entries but none usable")
        if not articles and getattr(parsed, "bozo", 0):
            problems.append(f"{source.id}: feed parse warning ({getattr(parsed, 'bozo_exception', 'unknown')})")
        return articles, problems

    def _entry_to_article(self, entry: Mapping[str, Any], source: NewsSource) -> NewsArticle | None:
        """Convert one feedparser entry into a validated :class:`NewsArticle`."""
        title = strip_html(entry.get("title"))
        link = str(entry.get("link") or "").strip()
        if not link:
            identifier = str(entry.get("id") or "").strip()
            link = identifier if identifier.startswith("http") else ""
        if len(title) < 3 or not link:
            return None

        published = entry.get("published_parsed") or entry.get("updated_parsed")
        published_at: str | None = None
        if published:
            try:
                published_at = datetime(*published[:6], tzinfo=timezone.utc).isoformat()
            except (TypeError, ValueError):
                published_at = None

        description = strip_html(entry.get("summary") or entry.get("description"))[:2000]
        author = strip_html(entry.get("author")) or None
        try:
            return NewsArticle(
                title=title[:500],
                url=link,
                source=source.name,
                author=author,
                published_at=published_at,
                description=description or None,
                category=source.category,
            )
        except Exception as exc:
            logger.debug("Skipping unusable feed entry from %s: %s", source.id, exc)
            return None

    # --- sitemap sources (opt-in, bounded) ----------------------------------
    def _fetch_sitemap(self, source: NewsSource, *, terms: Sequence[str]) -> tuple[list[NewsArticle], list[str]]:
        """Read a bounded number of URLs from a sitemap and extract them."""
        http = self.http
        limit = source.max_items or http.max_items_per_source
        try:
            response = requests.get(
                source.url,
                timeout=http.timeout_seconds,
                headers=self._headers("application/xml, text/xml;q=0.9, */*;q=0.8"),
            )
        except requests.RequestException as exc:
            return [], [f"{source.id}: sitemap request failed ({type(exc).__name__})"]
        if response.status_code >= 400:
            return [], [f"{source.id}: sitemap HTTP {response.status_code}"]

        urls = self._sitemap_urls(response.text, limit=limit, terms=terms)
        if not urls:
            return [], [f"{source.id}: sitemap contained no matching URLs"]

        articles = extract_many(
            urls,
            limit=limit,
            workers=http.max_workers,
            timeout=http.timeout_seconds,
            user_agent=http.user_agent,
            source_names={url: source.name for url in urls},
        )
        if source.category:
            articles = [
                article if article.category else article.model_copy(update={"category": source.category})
                for article in articles
            ]
        return articles, []

    def _sitemap_urls(self, xml: str, *, limit: int, terms: Sequence[str]) -> list[str]:
        """Extract article URLs from a sitemap (index), filtered by keywords."""
        http = self.http
        locations = _LOC_RE.findall(xml or "")
        child_sitemaps = [url for url in locations if url.lower().endswith((".xml", ".xml.gz"))]
        page_urls = [url for url in locations if url not in child_sitemaps]

        # A sitemap index points at child sitemaps; follow a bounded number of them.
        for child in child_sitemaps[: get_tuning().research.sitemap_child_limit]:
            try:
                child_response = requests.get(
                    child,
                    timeout=http.timeout_seconds,
                    headers=self._headers("application/xml, text/xml;q=0.9, */*;q=0.8"),
                )
            except requests.RequestException as exc:
                logger.warning("Could not read child sitemap %s: %s", child, exc)
                continue
            if child_response.status_code < 400:
                page_urls.extend(_LOC_RE.findall(child_response.text))

        rules = get_tuning().research
        skip_extensions = tuple(extension.lower() for extension in rules.sitemap_skip_extensions)
        skip_fragments = tuple(fragment.lower() for fragment in rules.url_skip_fragments)
        candidates = [
            url
            for url in dict.fromkeys(page_urls)
            if not url.lower().endswith(skip_extensions)
            and not any(fragment in url.lower() for fragment in skip_fragments)
        ]
        if terms:
            matching = [url for url in candidates if any(term in url.lower() for term in terms)]
            candidates = matching or candidates
        return candidates[:limit]