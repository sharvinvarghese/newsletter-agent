"""OpenRouter access: key resolution, model selection and model construction.

No secret is hardcoded anywhere. The API key is resolved from the explicit
argument, the process environment (``.env``), the settings object or the
Streamlit secrets store - in that order.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
from typing import Any

from config.settings import Settings, _env_str, get_settings

logger = logging.getLogger("newsletter_agent.llm")

API_KEY_ENV = "OPENROUTER_API_KEY"
_SECRET_PATTERN = re.compile(r"sk-or-[A-Za-z0-9\-_]+")


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM stack (package or credentials) cannot be used."""


class MissingAPIKeyError(LLMUnavailableError):
    """Raised when no OpenRouter API key could be resolved."""


def redact_secrets(text: Any) -> str:
    """Strip anything that looks like an OpenRouter key from a message."""
    return _SECRET_PATTERN.sub("sk-or-***", str(text))


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
    """Resolve the OpenRouter API key from the available sources."""
    for candidate in (explicit, os.getenv(API_KEY_ENV)):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    active = settings or get_settings()
    from_settings = active.api_key()
    if from_settings:
        return from_settings
    return _streamlit_secret(API_KEY_ENV)


def require_api_key(settings: Settings | None = None, explicit: str | None = None) -> str:
    """Like :func:`resolve_api_key` but raises a helpful error when missing."""
    key = resolve_api_key(settings, explicit)
    if not key:
        raise MissingAPIKeyError(
            "OPENROUTER_API_KEY is not set. Put it in a .env file (see .env.example), "
            "export it as an environment variable, or add it to the Streamlit secrets store."
        )
    return key


def model_name(settings: Settings | None = None, override: str | None = None) -> str:
    """The model slug that will be used (default: ``openrouter/free``)."""
    active = settings or get_settings()
    return active.active_model(override)


def candidate_models(settings: Settings | None = None, override: str | None = None) -> list[str]:
    """Models to try in order: the configured one, then any fallbacks."""
    active = settings or get_settings()
    return active.candidate_models(override)


def package_available() -> bool:
    """True when at least one chat-model provider package can be imported."""
    if importlib.util.find_spec("langchain_openrouter"):
        return True
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
    """Create a chat model instance for the configured provider.

    When ``settings.provider == "groq"`` a ``ChatGroq`` is built (timeout in
    **seconds**). Otherwise a ``ChatOpenRouter`` is built (timeout in
    **milliseconds**). Optional features are attached when the installed
    library accepts them and skipped otherwise, so a dependency bump cannot
    break the agent.
    """
    active = settings or get_settings()
    resolved_model = model_name(active, model)
    effective_timeout = active.llm_timeout_seconds if timeout is None else timeout

    if active.provider == "groq":
        return _build_groq_model(active, resolved_model, temperature, max_tokens, effective_timeout, api_key)
    return _build_openrouter_model(active, resolved_model, temperature, max_tokens, effective_timeout, api_key)


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
            "pip install langchain-groq, or set PROVIDER=openrouter."
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
        "max_tokens": min(active.llm_max_tokens if max_tokens is None else max_tokens, 512),
        "request_timeout": timeout,
        "max_retries": active.llm_max_attempts,
    }
    if active.groq_base_url:
        kwargs["groq_api_base"] = active.groq_base_url

    try:
        return ChatGroq(**kwargs)
    except Exception as exc:
        raise LLMUnavailableError(f"Could not create the Groq chat model: {redact_secrets(exc)}") from exc


def _build_openrouter_model(
    active: Settings,
    model: str,
    temperature: float | None,
    max_tokens: int | None,
    timeout: float,
    api_key: str | None,
) -> Any:
    """Create a ``ChatOpenRouter`` instance."""
    key = require_api_key(active, api_key)

    try:
        from langchain_openrouter import ChatOpenRouter
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise LLMUnavailableError(
            "langchain-openrouter is not installed. Install the project requirements first "
            "(pip install -r requirements.txt)."
        ) from exc

    base_kwargs: dict[str, Any] = {
        "model": model,
        "openrouter_api_key": key,
        "temperature": active.llm_temperature if temperature is None else temperature,
        "max_tokens": active.llm_max_tokens if max_tokens is None else max_tokens,
        "request_timeout": int(timeout * 1000),
    }

    headers: dict[str, str] = {}
    if active.openrouter_app_url:
        headers["HTTP-Referer"] = active.openrouter_app_url
    if active.openrouter_app_name:
        headers["X-Title"] = active.openrouter_app_name

    optional: dict[str, Any] = {}
    if headers:
        optional["default_headers"] = headers
    if active.openrouter_app_name:
        optional["app_title"] = active.openrouter_app_name
    if active.openrouter_app_url:
        optional["app_url"] = active.openrouter_app_url
    if active.openrouter_require_parameters:
        optional["openrouter_provider"] = {"require_parameters": True}
    if active.openrouter_response_healing:
        optional["plugins"] = [{"id": "response-healing"}]

    try:
        return ChatOpenRouter(**base_kwargs, **optional)
    except Exception as exc:
        logger.debug(
            "Optional OpenRouter settings were rejected (%s); using the minimal setup.",
            redact_secrets(exc),
        )

    try:
        return ChatOpenRouter(**base_kwargs)
    except Exception as exc:
        raise LLMUnavailableError(f"Could not create the OpenRouter chat model: {redact_secrets(exc)}") from exc