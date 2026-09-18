"""Shared building blocks for every schema used by the agent.

The models in this package serve two purposes:

1. They validate *everything* that is produced by the LLM before it is consumed
   by the next node (see :mod:`llm.structured`).
2. They are converted into JSON Schema documents and handed to OpenRouter as
   ``response_format`` payloads, so the model is constrained by the schema
   instead of being asked "please answer in JSON".
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, datetime, timezone
from enum import Enum
from typing import Annotated, Any
from urllib.parse import urlparse

from pydantic import AfterValidator, BaseModel, ConfigDict

_WHITESPACE_RE = re.compile(r"\s+")


class AgentMode(str, Enum):
    """How autonomous the workflow is allowed to be."""

    FULLY_AUTONOMOUS = "fully_autonomous"
    HUMAN_IN_LOOP = "human_in_loop"

    @classmethod
    def from_value(cls, value: AgentMode | str | None) -> AgentMode:
        """Tolerant conversion used by CLI/Streamlit/LLM inputs."""
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.FULLY_AUTONOMOUS
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "autonomous": cls.FULLY_AUTONOMOUS,
            "fully_autonomous": cls.FULLY_AUTONOMOUS,
            "auto": cls.FULLY_AUTONOMOUS,
            "human_in_loop": cls.HUMAN_IN_LOOP,
            "human_in_the_loop": cls.HUMAN_IN_LOOP,
            "human": cls.HUMAN_IN_LOOP,
            "hitl": cls.HUMAN_IN_LOOP,
            "manual": cls.HUMAN_IN_LOOP,
        }
        if normalized in aliases:
            return aliases[normalized]
        raise ValueError(f"Unknown agent mode: {value!r}")

    @property
    def label(self) -> str:
        return "Fully Autonomous" if self is AgentMode.FULLY_AUTONOMOUS else "Human-in-the-Loop"


class AgentModel(BaseModel):
    """Base class for all agent data structures.

    * ``extra="ignore"`` keeps the pipeline resilient: providers that do not
      enforce strict JSON schemas sometimes add keys. Unknown keys are dropped
      instead of crashing a run, every *known* field is still validated.
    * ``str_strip_whitespace`` cleans LLM output without extra boilerplate.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


def clean_text(value: Any) -> Any:
    """Collapse whitespace in strings, leave other values untouched."""
    if isinstance(value, str):
        return _WHITESPACE_RE.sub(" ", value).strip()
    return value


def normalize_unit_score(value: Any) -> float:
    """Normalise a 0..1 score, tolerating models that answer on a 0..10 scale."""
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"score must be numeric, got {value!r}") from exc
    if score > 1.0:
        score = score / 10.0 if score <= 10.0 else 1.0
    return max(0.0, min(1.0, score))


def validate_http_url(value: Any) -> Any:
    """Accept only absolute http(s) URLs."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("url must be a non-empty string")
    url = value.strip()
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"url must be an absolute http(s) URL, got {url!r}")
    return url


def normalize_iso_datetime(value: Any) -> Any:
    """Best effort ISO-8601 normalisation; unknown formats become ``None``."""
    if value is None or isinstance(value, datetime):
        return value.isoformat() if isinstance(value, datetime) else None
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).isoformat()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d %B %Y", "%B %d, %Y", "%a, %d %b %Y %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    return None


def as_string_list(value: Any, *, limit: int = 15) -> Any:
    """Coerce ``str | Iterable[str] | None`` into a deduplicated list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        raw: Iterable[Any] = [part for part in re.split(r"[\n;]+", value)]
    elif isinstance(value, Iterable):
        raw = value
    else:
        raw = [value]

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            item = str(item)
        item = _WHITESPACE_RE.sub(" ", item).strip(" -•\t")
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
        if len(cleaned) >= limit:
            break
    return cleaned


# Reusable annotated types -----------------------------------------------------
CleanStr = Annotated[str, AfterValidator(clean_text)]
HttpUrlStr = Annotated[str, AfterValidator(validate_http_url)]
UnitScore = Annotated[float, AfterValidator(normalize_unit_score)]
IsoDateTime = Annotated[str | None, AfterValidator(normalize_iso_datetime)]
