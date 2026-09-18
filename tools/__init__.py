"""Tools the agent uses to gather news.

* :mod:`tools.news_research` - reads the configured RSS/sitemap sources and the
  dynamic news-search feeds the planner asked for.
* :mod:`tools.article_extractor` - article extraction with news-please and
  trafilatura (both open source, no hand written scraping).
* :mod:`tools.deduplication` - deterministic duplicate detection.

None of these modules talks to an LLM: everything here is reproducible and
testable without an API key.
"""

from tools.article_extractor import (
    article_from_news_please,
    article_from_trafilatura,
    extract_article,
    extract_many,
    extract_with_backend,
    extraction_backends,
    fetch_html,
    news_please_available,
    trafilatura_available,
)
from tools.deduplication import (
    canonicalize_url,
    deduplicate_articles,
    default_title_threshold,
    find_duplicate_pairs,
    normalize_title,
    title_similarity,
)
from tools.news_research import NewsResearchTool, NewsSource, load_sources

__all__ = [
    # research
    "NewsResearchTool",
    "NewsSource",
    "article_from_news_please",
    "article_from_trafilatura",
    # de-duplication
    "canonicalize_url",
    "deduplicate_articles",
    "default_title_threshold",
    # extraction
    "extract_article",
    "extract_many",
    "extract_with_backend",
    "extraction_backends",
    "fetch_html",
    "find_duplicate_pairs",
    "load_sources",
    "news_please_available",
    "normalize_title",
    "title_similarity",
    "trafilatura_available",
]