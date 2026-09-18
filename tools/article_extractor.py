"""Article extraction: news-please first, trafilatura as the open-source fallback.

Both backends are free, open source Python libraries - there is no hand written
HTML scraping anywhere in this project:

* **news-please** (github.com/fhamborg/news-please) is the primary extractor: it
  returns title, lead, main text, authors and the publication date for a wide
  range of news layouts.
* **trafilatura** (github.com/adbar/trafilatura) is the safety net: a different
  extraction algorithm for the pages news-please returns nothing usable for.

The HTTP layer is our own ``requests`` call so that timeouts, the user agent and
a maximum page size can be enforced (``NewsPlease.from_url`` brings its own
network stack and stays available via ``use_news_please_network``).

When neither backend produces text, the caller keeps the feed metadata
(title + teaser) instead of guessing - see :meth:`NewsArticle.content_for_llm`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import requests
from schemas.articles import NewsArticle

logger = logging.getLogger("newsletter_agent.tools.extractor")

#: Last-resort user agent; ``sources.yaml -> http.user_agent`` wins when present.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; AINewsletterAgent/1.0; +https://github.com/your-user/ai-newsletter-agent)"
)
DEFAULT_TIMEOUT = 20.0
MAX_HTML_BYTES = 3_000_000


def _news_please() -> Any:
    """Return the :class:`newsplease.NewsPlease` class, or ``None``."""
    try:
        from newsplease import NewsPlease
    except Exception:  # pragma: no cover - depends on the environment
        return None
    return NewsPlease


def _trafilatura() -> Any:
    """Return the ``trafilatura`` module, or ``None``."""
    try:
        import trafilatura
    except Exception:  # pragma: no cover - depends on the environment
        return None
    return trafilatura


def news_please_available() -> bool:
    """True when news-please can be imported (used by the UI and tests)."""
    return _news_please() is not None


def trafilatura_available() -> bool:
    """True when trafilatura can be imported (used by the UI and tests)."""
    return _trafilatura() is not None


def extraction_backends() -> dict[str, bool]:
    """Availability of every extraction backend (surfaced in the UI sidebar)."""
    return {"news-please": news_please_available(), "trafilatura": trafilatura_available()}


def fetch_html(url: str, *, timeout: float = DEFAULT_TIMEOUT, user_agent: str = DEFAULT_USER_AGENT) -> str | None:
    """Download a page with a hard timeout. Returns ``None`` on any failure."""
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": user_agent, "Accept-Language": "en-US,en;q=0.9"},
            allow_redirects=True,
        )
    except requests.RequestException as exc:
        logger.warning("HTML download failed for %s: %s", url, exc)
        return None
    if response.status_code >= 400:
        logger.warning("HTML download for %s returned HTTP %s", url, response.status_code)
        return None
    if len(response.content) > MAX_HTML_BYTES:
        logger.warning("HTML download for %s exceeds the size limit; truncating", url)
    return response.text


def _iso_datetime(value: Any) -> str | None:
    """Best-effort ISO-8601 string from whatever a backend returned."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return text


def _clean_source(source_name: str | None, *candidates: Any, url: str = "") -> str:
    """First non-empty of: source name, backend values, then the URL host."""
    for candidate in (source_name, *candidates):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return urlparse(url).netloc or "unknown"


def article_from_news_please(raw: Any, url: str, source_name: str | None = None) -> NewsArticle | None:
    """Map a ``newsplease.Article`` onto our validated :class:`NewsArticle`."""
    if raw is None:
        return None
    title = (getattr(raw, "title", None) or getattr(raw, "headline", None) or "").strip()
    main_text = (getattr(raw, "maintext", None) or "").strip()
    if len(title) < 3 and len(main_text) < 3:
        return None

    authors = getattr(raw, "authors", None)
    if isinstance(authors, (list, tuple)):
        author = ", ".join(str(name) for name in authors if name) or None
    else:
        author = str(authors).strip() or None if authors else None

    description = (getattr(raw, "description", None) or "").strip() or None
    try:
        return NewsArticle(
            title=title or main_text[:120],
            url=url,
            source=_clean_source(source_name, getattr(raw, "source_domain", None), url=url),
            author=author,
            published_at=_iso_datetime(getattr(raw, "date_publish", None)),
            description=description,
            content=main_text or None,
        )
    except Exception as exc:  # validation error == "nothing extracted"
        logger.warning("news-please output for %s did not validate: %s", url, exc)
        return None


def article_from_trafilatura(
    html: str,
    url: str,
    source_name: str | None = None,
    *,
    output_format: str = "json",
) -> NewsArticle | None:
    """Map a trafilatura extraction onto our validated :class:`NewsArticle`."""
    trafilatura = _trafilatura()
    if trafilatura is None or not html:
        return None
    try:
        extracted = trafilatura.extract(
            html,
            url=url,
            output_format=output_format,
            with_metadata=True,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
    except Exception as exc:  # pragma: no cover - backend specific
        logger.warning("trafilatura failed for %s: %s", url, exc)
        return None
    if not extracted:
        return None

    try:
        document = json.loads(extracted) if output_format == "json" else {"text": extracted}
    except (TypeError, ValueError) as exc:
        logger.warning("trafilatura returned unreadable output for %s: %s", url, exc)
        return None
    if not isinstance(document, Mapping):
        return None

    title = str(document.get("title") or "").strip()
    text = str(document.get("text") or document.get("raw_text") or "").strip()
    if len(title) < 3 and len(text) < 3:
        return None

    description = str(document.get("description") or document.get("excerpt") or "").strip() or None
    hostname = document.get("sitename") or document.get("source-hostname") or document.get("hostname")
    try:
        return NewsArticle(
            title=(title or text[:120]).strip(),
            url=url,
            source=_clean_source(source_name, hostname, url=url),
            author=str(document.get("author") or "").strip() or None,
            published_at=_iso_datetime(document.get("date")),
            description=description,
            content=text or None,
        )
    except Exception as exc:
        logger.warning("trafilatura output for %s did not validate: %s", url, exc)
        return None


def extract_with_backend(
    html: str,
    url: str,
    *,
    source_name: str | None = None,
    backend: str = "auto",
) -> tuple[NewsArticle | None, str]:
    """Extract with one specific backend. Returns ``(article, backend_used)``.

    ``backend`` is ``"auto"`` (news-please, then trafilatura), ``"news-please"``
    or ``"trafilatura"``. The backend that actually produced the article is
    returned so nodes and tests can assert which path ran.
    """
    if not html:
        return None, "none"

    wanted = (backend or "auto").strip().lower()
    if wanted in {"auto", "news-please"}:
        news_please = _news_please()
        if html and news_please is not None:
            try:
                article = article_from_news_please(news_please.from_html(html, url=url), url, source_name)
            except Exception as exc:
                logger.warning("news-please extraction failed for %s: %s", url, exc)
                article = None
            if article is not None:
                return article, "news-please"
        if wanted == "news-please":
            return None, "news-please"

    if wanted in {"auto", "trafilatura"}:
        article = article_from_trafilatura(html, url, source_name)
        if article is not None:
            logger.debug("Extracted %s with trafilatura", url)
            return article, "trafilatura"
        return None, "trafilatura"

    logger.warning("Unknown extraction backend '%s' for %s", backend, url)
    return None, "none"


def extract_article(
    url: str,
    *,
    source_name: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    user_agent: str = DEFAULT_USER_AGENT,
    use_news_please_network: bool = False,
    backend: str = "auto",
    html: str | None = None,
) -> NewsArticle | None:
    """Extract a single article, preferring news-please over trafilatura.

    ``html`` may be supplied to skip the download (used by tests and by callers
    that already fetched the page).
    """
    page = html if html is not None else fetch_html(url, timeout=timeout, user_agent=user_agent)
    if page:
        article, _ = extract_with_backend(page, url, source_name=source_name, backend=backend)
        if article is not None:
            return article

    if use_news_please_network:
        news_please = _news_please()
        if news_please is not None:
            try:
                return article_from_news_please(news_please.from_url(url), url, source_name)
            except Exception as exc:
                logger.warning("news-please from_url failed for %s: %s", url, exc)
    return None


def extract_many(
    article_urls: Iterable[str],
    *,
    limit: int = 8,
    workers: int = 4,
    timeout: float = DEFAULT_TIMEOUT,
    user_agent: str = DEFAULT_USER_AGENT,
    source_names: Mapping[str, str] | None = None,
    use_news_please_network: bool = False,
    backend: str = "auto",
) -> list[NewsArticle]:
    """Extract several articles in parallel, preserving the input order."""
    selected = list(dict.fromkeys(url for url in article_urls if url))[: max(0, limit)]
    if not selected:
        return []

    lookup = dict(source_names or {})
    extracted: dict[str, NewsArticle] = {}
    pool = ThreadPoolExecutor(max_workers=max(1, min(workers, len(selected))), thread_name_prefix="nl-extract")
    try:
        futures = {
            pool.submit(
                extract_article,
                url,
                source_name=lookup.get(url),
                timeout=timeout,
                user_agent=user_agent,
                use_news_please_network=use_news_please_network,
                backend=backend,
            ): url
            for url in selected
        }
        budget = timeout * len(selected) + 10
        for future in as_completed(futures, timeout=budget):
            url = futures[future]
            try:
                article = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Extraction crashed for %s: %s", url, exc)
                continue
            if article is not None:
                extracted[url] = article
    except TimeoutError:
        logger.warning("Full-text extraction stopped early after %.0fs", timeout * len(selected) + 10)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    return [extracted[url] for url in selected if url in extracted]