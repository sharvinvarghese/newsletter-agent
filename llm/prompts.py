"""Prompt library backed by ``config/prompts.yaml``.

No prompt string lives in Python code: nodes ask for a named prompt and supply
the values the prompt declares as ``$placeholders``. The library validates that
every placeholder is supplied *before* the call, so a typo surfaces as a clear
configuration error instead of a prompt with a literal ``$article_url`` in it.

``string.Template`` (``$name``) is used instead of ``str.format`` on purpose:
prompts contain JSON examples with braces, which ``str.format`` would mangle.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from string import Template
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("newsletter_agent.llm.prompts")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPTS_PATH = PROJECT_ROOT / "config" / "prompts.yaml"


class PromptError(RuntimeError):
    """Raised when ``prompts.yaml`` is missing, invalid or under-supplied."""


class PromptTemplate(BaseModel):
    """One prompt: a system and a user message with ``$placeholders``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    system: str = Field(min_length=1)
    user: str = Field(min_length=1)

    @property
    def placeholders(self) -> frozenset[str]:
        """Every placeholder used by the system or user message."""
        return frozenset(Template(self.system).get_identifiers()) | frozenset(
            Template(self.user).get_identifiers()
        )

    def render(self, name: str, values: Mapping[str, Any]) -> RenderedPrompt:
        missing = sorted(self.placeholders - set(values))
        if missing:
            raise PromptError(
                f"Prompt '{name}' needs the value(s) {', '.join(missing)}. "
                f"Known placeholders: {', '.join(sorted(self.placeholders))}."
            )
        payload = {key: _as_text(value) for key, value in values.items()}
        try:
            return RenderedPrompt(
                name=name,
                system=Template(self.system).substitute(payload).strip(),
                user=Template(self.user).substitute(payload).strip(),
            )
        except (KeyError, ValueError) as exc:
            raise PromptError(f"Prompt '{name}' could not be rendered: {exc}") from exc


class RenderedPrompt(BaseModel):
    """A ready-to-send prompt (plain strings, so it is trivially testable)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    system: str
    user: str

    def as_messages(self) -> list[Any]:
        """LangChain messages for :meth:`StructuredInvoker.invoke`."""
        from llm.structured import (  # local: keeps imports light
            system_message,
            user_message,
        )

        return [system_message(self.system), user_message(self.user)]


def _as_text(value: Any) -> str:
    """Render a prompt value as text (models and lists are pretty printed)."""
    if value is None:
        return ""
    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    if isinstance(value, (list, tuple, dict, set)):
        import json

        try:
            return json.dumps(list(value) if isinstance(value, (set, tuple)) else value, indent=2, default=str)
        except TypeError:  # pragma: no cover - defensive
            return "\n".join(f"- {item}" for item in value)
    return str(value)


class PromptLibrary:
    """Named access to every prompt in ``config/prompts.yaml``."""

    def __init__(self, prompts: Mapping[str, PromptTemplate], *, source: str = "prompts.yaml") -> None:
        if not prompts:
            raise PromptError(f"{source} does not define any prompts")
        self._prompts: dict[str, PromptTemplate] = dict(prompts)
        self.source = source

    def __contains__(self, name: str) -> bool:
        return name in self._prompts

    def __len__(self) -> int:
        return len(self._prompts)

    @property
    def names(self) -> list[str]:
        return sorted(self._prompts)

    def template(self, name: str) -> PromptTemplate:
        try:
            return self._prompts[name]
        except KeyError as exc:
            raise PromptError(
                f"Unknown prompt '{name}'. Defined prompts: {', '.join(self.names)}"
            ) from exc

    def placeholders(self, name: str) -> frozenset[str]:
        return self.template(name).placeholders

    def render(self, name: str, **values: Any) -> RenderedPrompt:
        """Render ``name`` with ``values`` (raises :class:`PromptError` on gaps)."""
        return self.template(name).render(name, values)


def load_prompts(path: str | Path | None = None) -> PromptLibrary:
    """Read and validate the prompt file."""
    config_path = Path(path) if path else DEFAULT_PROMPTS_PATH
    if not config_path.exists():
        raise PromptError(f"Prompt file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise PromptError(f"{config_path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, Mapping) or not isinstance(raw.get("prompts"), Mapping):
        raise PromptError(f"{config_path} must contain a top-level 'prompts' mapping")
    try:
        prompts = {
            str(name): PromptTemplate.model_validate(body) for name, body in raw["prompts"].items()
        }
    except Exception as exc:
        raise PromptError(f"{config_path} is invalid: {exc}") from exc
    library = PromptLibrary(prompts, source=config_path.name)
    logger.debug("Loaded %d prompt(s) from %s", len(library), config_path.name)
    return library


_PROMPTS: PromptLibrary | None = None


def get_prompts(refresh: bool = False) -> PromptLibrary:
    """Process wide prompt library (``refresh=True`` re-reads the YAML)."""
    global _PROMPTS
    if _PROMPTS is None or refresh:
        _PROMPTS = load_prompts()
    return _PROMPTS
