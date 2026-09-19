"""Groq access: key resolution, model selection and model construction.

No secret is hardcoded anywhere. The API key is resolved from the explicit
argument, the process environment (``.env``), the settings object or the
Streamlit secrets store - in that order.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any

from config.settings import Settings, _env_str, get_settings

logger = logging.getLogger("newsletter_agent.llm")

API_KEY_ENV = "GROQ_API_KEY"
_SECRET_PATTERN = None  # Groq keys don't have a standard pattern like OpenRouter


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM stack (package or credentials) cannot be used."""


class MissingAPIKeyError(LLMUnavailableError):
    """Raised when no Groq API key could be resolved."""


def redact_secrets(text: Any) -> str:
    """Strip anything that looks like an API key from a message."""
    # Basic redaction for any key-like string
    import re
    return re.sub(r"(sk-|gsk_)[A-Za-z0-9\-_]+", "***REDACTED***", str(text))


def _streamlit_secret(name: str) -> str | None:
    """Read a Streamlit secret without making Streamlit a hard dependency."""
    try:
        import streamlit as st
    except ImportError:  # pragma: no cover - only when running outside Streamlit
        return None
    try:
        value = st.secrets[name]
    except (KeyError, AttributeError):
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_api_key(settings: Settings | None = None, explicit: str | None = None) -> str | None:
    """Resolve the Groq API key from the available sources."""
    for candidate in (explicit, os.getenv(API_KEY_ENV)):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    active = settings or get_settings()
    from_settings = active.groq_key()
    if from_settings:
        return from_settings
    return _streamlit_secret(API_KEY_ENV)


def require_api_key(settings: Settings | None = None, explicit: str | None = None) -> str:
    """Like :func:`resolve_api_key` but raises a helpful error when missing."""
    key = resolve_api_key(settings, explicit)
    if not key:
        raise MissingAPIKeyError(
            "GROQ_API_KEY is not set. Put it in a .env file (see .env.example), "
            "export it as an environment variable, or add it to the Streamlit secrets store."
        )
    return key


def model_name(settings: Settings | None = None, override: str | None = None) -> str:
    """The model slug that will be used (default: ``qwen/qwen3.8-27b``)."""
    active = settings or get_settings()
    return active.active_model(override)


def candidate_models(settings: Settings | None = None, override: str | None = None) -> list[str]:
    """Models to try in order: the configured one, then any fallbacks."""
    active = settings or get_settings()
    return active.candidate_models(override)


def package_available() -> bool:
    """True when at least one chat-model provider package can be imported."""
    return bool(importlib.util.find_spec("langchain_groq"))


def _groq_key(settings: Settings | None, explicit: str | None) -> str | None:
    """Resolve a Groq API key from explicit arg, settings, or env."""
    if explicit:
        return explicit
    active = settings or get_settings()
    value = active.groq_key()
    if value:
        return value
    return _env_str("GROQ_API_KEY") or _env_str("groqkey") or None


def build_chat_model(
    model: str | None = None,
    *,
    settings: Settings | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    api_key: str | None = None,
) -> Any:
    """Create a chat model instance for Groq.

    When ``settings.provider == "groq"`` a ``ChatGroq`` is built (timeout in
    **seconds**). Optional features are attached when the installed
    library accepts them and skipped otherwise, so a dependency bump cannot
    break the agent.
    """
    active = settings or get_settings()
    resolved_model = model_name(active, model)
    effective_timeout = active.llm_timeout_seconds if timeout is None else timeout

    return _build_groq_model(active, resolved_model, temperature, max_tokens, effective_timeout, api_key)


def _build_groq_model(
    active: Settings,
    model: str,
    temperature: float | None,
    max_tokens: int | None,
    timeout: float,
    api_key: str | None,
) -> Any:
    """Create a ``ChatGroq`` instance."""
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise LLMUnavailableError(
            "langchain-groq is not installed. Install it with "
            "pip install langchain-groq."
        ) from exc

    key = _groq_key(active, api_key)
    if not key:
        raise MissingAPIKeyError(
            "GROQ_API_KEY is not set. Add it to your .env file "
            "(or the legacy groqkey var), export it, or add it to the "
            "Streamlit secrets store."
        )

    kwargs: dict[str, Any] = {
        "model_name": model,
        "groq_api_key": key,
        "temperature": active.llm_temperature if temperature is None else temperature,
        "max_tokens": min(active.llm_max_tokens if max_tokens is None else max_tokens, 8192),
        "request_timeout": timeout,
        "max_retries": active.llm_max_attempts,
    }
    if active.groq_base_url:
        kwargs["groq_api_base"] = active.groq_base_url

    try:
        return ChatGroq(**kwargs)
    except Exception as exc:
        raise LLMUnavailableError(f"Could not create the Groq chat model: {redact_secrets(exc)}") from exc