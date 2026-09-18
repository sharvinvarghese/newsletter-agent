"""LLM access layer.

* :mod:`llm.client` - OpenRouter credentials, model selection, model creation.
* :mod:`llm.structured` - Pydantic-validated structured output with retries and
  an isolated JSON fallback.
"""

from llm.client import (
    LLMUnavailableError,
    MissingAPIKeyError,
    build_chat_model,
    candidate_models,
    model_name,
    package_available,
    redact_secrets,
    require_api_key,
    resolve_api_key,
)
from llm.prompts import (
    PromptError,
    PromptLibrary,
    PromptTemplate,
    RenderedPrompt,
    get_prompts,
    load_prompts,
)
from llm.structured import (
    StructuredInvoker,
    StructuredLLM,
    StructuredOutputError,
    parse_json_payload,
    structured_invoke,
    system_message,
    user_message,
)

__all__ = [
    "LLMUnavailableError",
    "MissingAPIKeyError",
    "PromptError",
    # prompts
    "PromptLibrary",
    "PromptTemplate",
    "RenderedPrompt",
    "StructuredInvoker",
    # structured output
    "StructuredLLM",
    "StructuredOutputError",
    # client
    "build_chat_model",
    "candidate_models",
    "get_prompts",
    "load_prompts",
    "model_name",
    "package_available",
    "parse_json_payload",
    "redact_secrets",
    "require_api_key",
    "resolve_api_key",
    "structured_invoke",
    "system_message",
    "user_message",
]