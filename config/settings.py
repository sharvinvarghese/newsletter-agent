"""Central configuration for the AI Newsletter Agent.

Every knob is read from an environment variable (optionally loaded from ``.env``),
or from Streamlit secrets (for Streamlit Cloud deployment), so no secret and no
<<<<<<< HEAD
tunable value is hardcoded inside the agent logic.
=======
tunable value is hardcoded inside the agent logic. The OpenRouter model defaults
to ``openrouter/free`` but can be swapped with the ``OPENROUTER_MODEL`` variable.
>>>>>>> 6e132963f48a8d8a8f197a3a066d820b5841ba40
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_SOURCES_PATH = CONFIG_DIR / "sources.yaml"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MODEL = "qwen/qwen3.8-27b"
LOGGER_NAME = "newsletter_agent"

# Loaded once; real environment variables always win over the .env file.
load_dotenv(PROJECT_ROOT / ".env", override=True)


def _get_secret(name: str, default: str = "") -> str:
    """Get a secret from environment variables or Streamlit secrets."""
    # Environment variables take priority
    value = os.getenv(name)
    if value is not None and value.strip():
        return value.strip()
    
    # Fall back to Streamlit secrets (only available in Streamlit context)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and name in st.secrets:
            secret_value = st.secrets[name]
            if secret_value is not None and str(secret_value).strip():
                return str(secret_value).strip()
    except Exception:
        pass
    
    return default


def _env_str(name: str, default: str = "") -> str:
    value = _get_secret(name)
    return value if value else default


def _env_int(name: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    raw = _env_str(name)
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    raw = _env_str(name)
    try:
        value = float(raw) if raw else default
    except ValueError:
        value = default
    return max(minimum, value) if minimum is not None else value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env_str(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _env_list(name: str, default: tuple[str, ...] = ()) -> list[str]:
    raw = _env_str(name)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]


class Settings(BaseModel):
    """Runtime configuration (use :meth:`with_overrides` to tweak a copy)."""

    model_config = ConfigDict(extra="forbid")

    # --- Groq ---------------------------------------------------------
    groq_api_key: SecretStr | None = Field(
        default=None,
        description="Bearer token for Groq. Never logged, never rendered.",
    )
    groq_model: str = _env_str("GROQ_MODEL", DEFAULT_MODEL)
    groq_fallback_models: list[str] = Field(
        default_factory=lambda: _env_list("GROQ_FALLBACK_MODELS")
    )
    groq_base_url: str = ""

    # --- LLM behaviour ------------------------------------------------------
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096
    llm_timeout_seconds: float = _env_float("LLM_TIMEOUT_SECONDS", 120.0, minimum=1.0)
    llm_max_attempts: int = 2
    #: Send ``strict: true`` when asking for JSON-Schema structured output.
    #: Set LLM_STRICT_JSON_SCHEMA=false if a provider rejects strict schemas.
    llm_strict_json_schema: bool = True

    # --- Agent behaviour ----------------------------------------------------
    max_revisions: int = 2
    target_articles: int = 6
    max_articles_to_collect: int = 24
    recency_window_days: int = 14
    evaluation_batch_size: int = 6
    summarization_batch_size: int = 5

    # --- Research tool ------------------------------------------------------
    research_max_sources: int = 8
    max_items_per_source: int = 25
    enrich_article_limit: int = 12
    enrich_full_text: bool = True
    research_max_workers: int = 4
    request_timeout_seconds: float = 20.0
    max_content_chars: int = 6000

    # --- Paths / logging ----------------------------------------------------
    sources_path: Path = DEFAULT_SOURCES_PATH
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR
    log_level: str = "INFO"

    @field_validator("groq_model", mode="before")
    @classmethod
    def _model_fallback(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) and value.strip() else DEFAULT_MODEL

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_level(cls, value: Any) -> Any:
        return str(value or "INFO").upper()

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from the process environment (and ``.env``)."""
        groq_key = _env_str("GROQ_API_KEY") or _env_str("groqkey")
        return cls(
            groq_api_key=SecretStr(groq_key) if groq_key else None,
            groq_model=_env_str("GROQ_MODEL", DEFAULT_MODEL),
            groq_fallback_models=_env_list("GROQ_FALLBACK_MODELS"),
            groq_base_url=_env_str("GROQ_BASE_URL", ""),
            llm_temperature=_env_float("LLM_TEMPERATURE", 0.2, minimum=0.0),
            llm_max_tokens=_env_int("LLM_MAX_TOKENS", 4096, minimum=256),
            llm_timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", 120.0, minimum=5.0),
            llm_max_attempts=_env_int("LLM_MAX_ATTEMPTS", 2, minimum=1, maximum=5),
            llm_strict_json_schema=_env_bool("LLM_STRICT_JSON_SCHEMA", True),
            max_revisions=_env_int("MAX_REVISIONS", 2, minimum=0, maximum=5),
            target_articles=_env_int("TARGET_ARTICLES", 6, minimum=1, maximum=12),
            max_articles_to_collect=_env_int("MAX_ARTICLES_TO_COLLECT", 24, minimum=5, maximum=100),
            recency_window_days=_env_int("RECENCY_WINDOW_DAYS", 14, minimum=0, maximum=365),
            evaluation_batch_size=_env_int("EVALUATION_BATCH_SIZE", 6, minimum=1, maximum=25),
            summarization_batch_size=_env_int("SUMMARIZATION_BATCH_SIZE", 5, minimum=1, maximum=25),
            research_max_sources=_env_int("RESEARCH_MAX_SOURCES", 8, minimum=1, maximum=30),
            max_items_per_source=_env_int("MAX_ITEMS_PER_SOURCE", 25, minimum=1, maximum=100),
            enrich_article_limit=_env_int("ENRICH_ARTICLE_LIMIT", 12, minimum=0, maximum=50),
            enrich_full_text=_env_bool("ENRICH_FULL_TEXT", True),
            research_max_workers=_env_int("RESEARCH_MAX_WORKERS", 4, minimum=1, maximum=16),
            request_timeout_seconds=_env_float("REQUEST_TIMEOUT_SECONDS", 20.0, minimum=2.0),
            max_content_chars=_env_int("MAX_CONTENT_CHARS", 6000, minimum=500),
            sources_path=Path(_env_str("SOURCES_PATH", str(DEFAULT_SOURCES_PATH))),
            outputs_dir=Path(_env_str("OUTPUTS_DIR", str(DEFAULT_OUTPUTS_DIR))),
            log_level=_env_str("LOG_LEVEL", "INFO"),
        )

    # --- helpers ------------------------------------------------------------
    @property
    def has_groq(self) -> bool:
        """True when a Groq API key is configured."""
        if self.groq_api_key is None:
            return False
        value = self.groq_api_key.get_secret_value().strip()
        return bool(value)

    def groq_key(self) -> str | None:
        """The Groq key from settings, if one was configured."""
        if self.groq_api_key is None:
            return None
        value = self.groq_api_key.get_secret_value().strip()
        return value or None

    def candidate_models(self, model_override: str | None = None) -> list[str]:
        """Models to try in order: the configured one, then any fallbacks."""
        if model_override:
            return [model_override]
        base = self.groq_model
        fallbacks = self.groq_fallback_models
        models: list[str] = []
        for candidate in [base, *fallbacks]:
            if candidate and candidate.strip() and candidate.strip() not in models:
                models.append(candidate.strip())
        return models

    def active_model(self, model_override: str | None = None) -> str:
        """The model slug that will be used (respects override)."""
        candidates = self.candidate_models(model_override)
        return candidates[0] if candidates else DEFAULT_MODEL

    def with_overrides(self, **overrides: Any) -> Settings:
        """Copy of these settings with ``None`` values ignored."""
        clean = {key: value for key, value in overrides.items() if value is not None}
        return self.model_copy(update=clean) if clean else self

    def safe_snapshot(self) -> dict[str, Any]:
        """Serialisable view that never contains the API key."""
        data = self.model_dump(mode="json", exclude={"groq_api_key"})
        data["groq_key_configured"] = self.has_groq
        return data


_SETTINGS: Settings | None = None


def get_settings(*, refresh: bool = False) -> Settings:
    """Process wide settings singleton (``refresh=True`` re-reads the env)."""
    global _SETTINGS
    if _SETTINGS is None or refresh:
        _SETTINGS = Settings.from_env()
    return _SETTINGS


def configure_logging(level: str | None = None) -> None:
    """Idempotent logging setup used by ``app.py`` and the test suite."""
    active = (level or get_settings().log_level or "INFO").upper()
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=active,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    logging.getLogger(LOGGER_NAME).setLevel(active)