"""Structured LLM output on top of OpenRouter.

Three tiers are tried, in order:

1. ``method="json_schema"`` - provider side JSON-Schema enforcement.
2. ``method="function_calling"`` - tool/function calling, for models whose
   endpoints do not implement ``response_format: json_schema``.
3. A manually prompted JSON object, parsed and validated with the *same*
   Pydantic model. This is the only place in the codebase where a raw model
   string is parsed, and its output still has to pass Pydantic before it is
   allowed to leave this module.

Each tier is retried once with a corrective message when Pydantic rejects the
output, and the whole cycle moves on to the next model from
``OPENROUTER_FALLBACK_MODELS`` before giving up. The agent therefore never
receives unvalidated data: this module returns a validated Pydantic model or
raises :class:`StructuredOutputError`.
"""

from __future__ import annotations

import json
import logging
import re
import time as _time
from collections.abc import Sequence
from typing import Any, Protocol, TypeVar, runtime_checkable

from config.settings import Settings, get_settings
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from llm.client import build_chat_model, candidate_models, model_name, redact_secrets

logger = logging.getLogger("newsletter_agent.llm.structured")

ModelT = TypeVar("ModelT", bound=BaseModel)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _is_rate_limit(exc: Exception) -> bool:
    """True when the exception represents a 429 / rate-limit response."""
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "too many requests" in text or "otp" in text

class _TokenUsageHandler(BaseCallbackHandler):
    """Captures input/output token counts from the LangChain response."""

    def __init__(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        usage = getattr(response, "response_metadata", {}) or {}
        if isinstance(usage, dict):
            usage = usage.get("usage", usage)
        if isinstance(usage, dict):
            self.input_tokens += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            self.output_tokens += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)


#: Strategies in the order they are attempted.
DEFAULT_STRATEGIES: tuple[str, ...] = ("json_schema", "function_calling", "json_object")

_JSON_INSTRUCTION = (
    "Return ONLY a single JSON object that validates against this JSON Schema. "
    "Do not wrap it in markdown code fences and do not add commentary.\n"
    "JSON Schema:\n{schema}"
)


class StructuredOutputError(RuntimeError):
    """Raised when no strategy/model produced schema-valid output."""


@runtime_checkable
class StructuredInvoker(Protocol):
    """The interface the graph nodes depend on (keeps them easy to fake)."""

    def invoke(
        self,
        schema: type[ModelT],
        messages: Sequence[Any],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelT: ...


def system_message(text: str) -> SystemMessage:
    """Convenience constructor for a system message."""
    return SystemMessage(content=text)


def user_message(text: str) -> HumanMessage:
    """Convenience constructor for a user message."""
    return HumanMessage(content=text)


def to_messages(messages: Sequence[Any]) -> list[BaseMessage]:
    """Accept ``BaseMessage`` objects or ``{"role": ..., "content": ...}`` dicts."""
    converted: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, BaseMessage):
            converted.append(message)
            continue
        if isinstance(message, dict):
            role = str(message.get("role", "user")).lower()
            content = str(message.get("content", ""))
            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "assistant":
                converted.append(AIMessage(content=content))
            else:
                converted.append(HumanMessage(content=content))
            continue
        converted.append(HumanMessage(content=str(message)))
    return converted


def parse_json_payload(text: Any) -> Any:
    """Extract the first JSON value from a model response (fences tolerated)."""
    if isinstance(text, (dict, list)):
        return text
    if isinstance(text, BaseModel):
        return text.model_dump()

    raw = str(text or "").strip()
    if not raw:
        raise ValueError("the model returned an empty response")
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = _JSON_OBJECT_RE.search(raw)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"the model response did not contain valid JSON: {exc}") from exc
    raise ValueError("the model response did not contain a JSON object")


class StructuredLLM:
    """OpenRouter chat models behind a Pydantic-validated ``invoke``.

    ``last_call`` records which model/strategy produced the most recent
    successful response, so the nodes can report it in the execution log.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        models: Sequence[str] | None = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        strategies: Sequence[str] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.strategies = tuple(strategies or self._default_strategies())
        self.models = list(models) if models else candidate_models(self.settings, model)
        if not self.models:
            self.models = [model_name(self.settings)]
        self.last_call: dict[str, Any] = {}
        self.last_token_usage: dict[str, int] = {}
        self._model_cache: dict[str, Any] = {}

    # --- introspection ------------------------------------------------------
    @property
    def primary_model(self) -> str:
        return self.models[0]

    def describe(self) -> str:
        """Human readable model summary for the UI/execution log."""
        if len(self.models) == 1:
            return self.primary_model
        return f"{self.primary_model} (fallbacks: {', '.join(self.models[1:])})"

    def _default_strategies(self) -> tuple[str, ...]:
        """Pick strategies appropriate for the active provider.

        Groq's ChatGroq only supports ``json_schema`` via ``with_structured_output``;
        the other strategies make extra API calls that always fail, so we limit
        to ``json_schema`` to avoid wasting rate-limit budget. OpenRouter supports
        all three.
        """
        if self.settings.provider == "groq":
            return ("json_schema",)
        return DEFAULT_STRATEGIES

    # --- public API ---------------------------------------------------------
    def invoke(
        self,
        schema: type[ModelT],
        messages: Sequence[Any],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelT:
        """Return a validated ``schema`` instance or raise StructuredOutputError."""
        prepared = to_messages(messages)
        failures: list[str] = []

        for model in self.models:
            for strategy in self.strategies:
                correction = ""
                for attempt in (1, 2):
                    payload = prepared if attempt == 1 else [*prepared, user_message(correction)]
                    try:
                        raw = self._call(model, strategy, schema, payload, temperature, max_tokens)
                    except Exception as exc:
                        detail = redact_secrets(exc).strip().replace("\n", " ")[:300]
                        failures.append(f"{model} / {strategy}: {detail}")
                        logger.warning("Structured call failed (%s / %s): %s", model, strategy, detail)
                        if _is_rate_limit(exc):
                            _time.sleep(1.0)  # brief backoff before next model
                        break

                    validated, error = self._validate(schema, raw)
                    if validated is not None:
                        self.last_call = {
                            "schema": schema.__name__,
                            "model": model,
                            "strategy": strategy,
                            "attempt": attempt,
                        }
                        logger.debug("Structured call ok (%s / %s / attempt %d)", model, strategy, attempt)
                        return validated

                    failures.append(f"{model} / {strategy}: {error[:200]}")
                    logger.warning(
                        "Structured output for %s failed validation (%s / %s): %s",
                        schema.__name__,
                        model,
                        strategy,
                        error[:300],
                    )
                    correction = self._correction_text(schema, error)

        raise StructuredOutputError(
            f"Could not obtain valid {schema.__name__} output. Last attempts: " + " | ".join(failures[-5:])
        )

    # --- strategies ---------------------------------------------------------
    def _call(
        self,
        model: str,
        strategy: str,
        schema: type[ModelT],
        messages: list[BaseMessage],
        temperature: float | None,
        max_tokens: int | None,
    ) -> Any:
        llm = self._model(model, temperature, max_tokens)
        handler = _TokenUsageHandler()
        config = {"callbacks": [handler]}

        if strategy == "json_schema":
            result = self._structured_chain(llm, schema, "json_schema").invoke(messages, config=config)
            self.last_token_usage = {"input": handler.input_tokens, "output": handler.output_tokens}
            return result
        if strategy == "function_calling":
            result = self._structured_chain(llm, schema, "function_calling").invoke(messages, config=config)
            self.last_token_usage = {"input": handler.input_tokens, "output": handler.output_tokens}
            return result
        if strategy == "json_object":
            schema_text = json.dumps(schema.model_json_schema(), ensure_ascii=False)
            bound = llm.bind(response_format={"type": "json_object"})
            response = bound.invoke([*messages, user_message(_JSON_INSTRUCTION.format(schema=schema_text))], config=config)
            self.last_token_usage = {"input": handler.input_tokens, "output": handler.output_tokens}
            return parse_json_payload(getattr(response, "content", response))
        raise ValueError(f"Unknown structured-output strategy: {strategy}")

    def _structured_chain(self, llm: Any, schema: type[ModelT], method: str) -> Any:
        """Build ``with_structured_output`` with strict mode when supported."""
        if method == "json_schema" and self.settings.llm_strict_json_schema:
            try:
                return llm.with_structured_output(schema, method=method, strict=True)
            except TypeError:  # pragma: no cover - older langchain-openrouter
                logger.debug("This langchain-openrouter version does not accept strict=True.")
        return llm.with_structured_output(schema, method=method)

    # --- validation ---------------------------------------------------------
    def _validate(self, schema: type[ModelT], raw: Any) -> tuple[ModelT | None, str]:
        """Validate a raw model response. Returns ``(model, error_message)``."""
        if isinstance(raw, schema):
            return raw, ""

        candidate: Any = raw.model_dump() if isinstance(raw, BaseModel) else raw
        try:
            return schema.model_validate(candidate), ""
        except ValidationError as exc:
            error = f"{exc.error_count()} validation error(s): " + "; ".join(
                f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}" for issue in exc.errors()[:4]
            )
        except Exception as exc:  # pragma: no cover - defensive
            error = f"{type(exc).__name__}: {exc}"

        # Last resort: tolerate provider specific extra keys.
        if isinstance(candidate, dict):
            known = set(schema.model_fields)
            trimmed = {key: value for key, value in candidate.items() if key in known}
            if trimmed and trimmed != candidate:
                try:
                    return schema.model_validate(trimmed), ""
                except ValidationError as exc:
                    error += f" | after dropping unknown keys: {exc.error_count()} validation error(s)"
        return None, error

    @staticmethod
    def _correction_text(schema: type[BaseModel], error: str) -> str:
        """Corrective message appended for the second attempt."""
        fields = ", ".join(schema.model_fields)
        text = (
            f"Your previous answer did not validate against the required schema for {schema.__name__}.\n"
            f"Return ONLY the corrected JSON object.\nRequired fields: {fields}\n"
        )
        return text + (f"Validation error: {error[:400]}\n" if error else "")

    def _model(self, model: str, temperature: float | None, max_tokens: int | None) -> Any:
        """Cached chat model instance for one model/parameter combination."""
        key = f"{model}|{temperature}|{max_tokens}"
        cached = self._model_cache.get(key)
        if cached is None:
            cached = build_chat_model(
                model,
                settings=self.settings,
                api_key=self.api_key,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            )
            self._model_cache[key] = cached
        return cached


def structured_invoke(
    schema: type[ModelT],
    messages: Sequence[Any],
    *,
    llm: StructuredInvoker | None = None,
    settings: Settings | None = None,
    **kwargs: Any,
) -> ModelT:
    """One-shot helper: use ``llm`` when given, otherwise build a default client."""
    invoker: StructuredInvoker = llm if llm is not None else StructuredLLM(settings=settings)
    return invoker.invoke(schema, messages, **kwargs)