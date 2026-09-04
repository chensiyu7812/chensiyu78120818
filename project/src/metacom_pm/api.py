from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Literal, Mapping, Type, TypeVar
from urllib.parse import quote
import httpx
from pydantic import BaseModel, ValidationError

from .io import canonical_json, sha256_text, utc_now

T = TypeVar("T", bound=BaseModel)
NormalizedFinishReason = Literal[
    "complete", "length", "tool_call", "content_filter", "unknown"
]
ANTHROPIC_STRUCTURED_OUTPUT_TOOL_NAME = "submit_structured_response"
OPENAI_UNSUPPORTED_STRICT_SCHEMA_KEYWORDS = frozenset(
    {
        "allOf",
        "not",
        "dependentRequired",
        "dependentSchemas",
        "if",
        "then",
        "else",
    }
)


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    model: str
    api_key_env: str
    timeout_seconds: float = 180.0
    # A human-declared model family is required for confirmatory runs.  Model
    # aliases and vendor gateways are not reliable indicators of independence.
    family: str | None = None
    # The request protocol is part of endpoint identity. ``auto`` preserves
    # backward compatibility for ordinary OpenAI/Anthropic routes, while
    # providers with multiple API surfaces (notably Gemini) must freeze it.
    transport: str = "auto"

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError(f"Environment variable {self.api_key_env} is not set")
        return key


SUPPORTED_ENDPOINT_TRANSPORTS = frozenset(
    {
        "openai_chat_completions",
        "anthropic_messages",
        "gemini_generate_content",
    }
)


def endpoint_transport(endpoint: Endpoint) -> str:
    """Resolve and validate the exact HTTP protocol used by an endpoint."""

    declared = str(endpoint.transport or "auto").strip()
    if declared != "auto":
        if declared not in SUPPORTED_ENDPOINT_TRANSPORTS:
            raise ValueError(f"unsupported endpoint transport: {declared}")
        return declared
    if "anthropic.com" in endpoint.base_url:
        return "anthropic_messages"
    if "generativelanguage.googleapis.com" in endpoint.base_url:
        if endpoint.base_url.rstrip("/").endswith("/openai"):
            raise ValueError(
                "Gemini's OpenAI-compatibility route is not an approved strict-"
                "schema transport; declare gemini_generate_content and use the "
                "native /v1beta base URL"
            )
        return "gemini_generate_content"
    return "openai_chat_completions"


@dataclass
class CallResult:
    text: str
    raw_response: dict[str, Any]
    usage: dict[str, int]
    latency_ms: float
    request_hash: str
    provider_finish_reason: str | None = None
    normalized_finish_reason: NormalizedFinishReason = "unknown"


def normalize_provider_finish_reason(
    raw_response: Mapping[str, Any],
) -> tuple[str | None, NormalizedFinishReason]:
    """Extract and normalize OpenAI-compatible or Anthropic finish metadata."""

    provider_reason: str | None = None
    choices = raw_response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        raw_reason = choices[0].get("finish_reason")
        if raw_reason is not None:
            provider_reason = str(raw_reason)
    else:
        candidates = raw_response.get("candidates")
        if (
            isinstance(candidates, list)
            and candidates
            and isinstance(candidates[0], Mapping)
            and candidates[0].get("finishReason") is not None
        ):
            provider_reason = str(candidates[0].get("finishReason"))
    if provider_reason is None and raw_response.get("stop_reason") is not None:
        provider_reason = str(raw_response.get("stop_reason"))

    if provider_reason is None or not provider_reason.strip():
        return provider_reason, "unknown"
    normalized = provider_reason.strip().casefold()
    if normalized in {"stop", "end_turn", "stop_sequence"}:
        return provider_reason, "complete"
    if normalized in {"length", "max_tokens", "model_context_window_exceeded"}:
        return provider_reason, "length"
    if normalized in {"tool_calls", "tool_call", "function_call", "tool_use"}:
        return provider_reason, "tool_call"
    if normalized in {
        "content_filter",
        "refusal",
        "safety",
        "recitation",
        "prohibited_content",
        "spii",
        "image_safety",
    }:
        return provider_reason, "content_filter"
    return provider_reason, "unknown"


class ProviderRequestError(RuntimeError):
    """A deterministic provider-side client error that must not be retried."""

    def __init__(
        self,
        *,
        status_code: int,
        detail: str,
        schema_mode: bool,
        request_hash: str | None = None,
        usage: Mapping[str, Any] | None = None,
        response_diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        self.status_code = int(status_code)
        self.detail = str(detail)
        self.schema_mode = bool(schema_mode)
        self.request_hash = request_hash
        self.usage = dict(usage) if usage is not None else None
        self.response_diagnostics = (
            dict(response_diagnostics) if response_diagnostics is not None else None
        )
        mode = "schema mode " if self.schema_mode else ""
        super().__init__(f"{mode}HTTP {self.status_code}: {self.detail}")


RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


def _classify_retryable_exception(exc: BaseException) -> tuple[str, int | None]:
    """Classify a caught exception for a caller's bounded, ledger-visible retry loop.

    Returns (retry_class, status_code). retry_class is one of:
    "rate_limited_429", "request_timeout_408", "http_5xx",
    "network_timeout", "missing_field", "other".
    Only the caller decides whether/how many times to retry each class; this
    function only describes what happened.
    """

    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        if status_code == 429:
            return "rate_limited_429", status_code
        if status_code == 408:
            return "request_timeout_408", status_code
        if status_code in RETRYABLE_HTTP_STATUS_CODES:
            return "http_5xx", status_code
        return "other", status_code
    if isinstance(exc, httpx.HTTPError):
        # Connection errors, timeouts, and similar transport failures carry no
        # HTTP status code but are the same kind of transient infrastructure
        # issue as a 5xx response.
        return "network_timeout", None
    if isinstance(exc, (KeyError, IndexError, TypeError, AttributeError)) or (
        isinstance(exc, ValueError) and "empty model response" in str(exc)
    ):
        return "missing_field", None
    return "other", None


class RetryableProviderError(RuntimeError):
    """Same message as the original exhausted-retries RuntimeError, plus
    structured info about the last attempt so a ledger-visible bounded-retry
    loop (see attempt_ledger.py) can decide whether to reserve another
    physical attempt without parsing the message string."""

    def __init__(
        self,
        message: str,
        *,
        last_retry_class: str,
        last_status_code: int | None,
        attempts_tried: int,
        request_hash: str | None = None,
        usage: Mapping[str, Any] | None = None,
        response_diagnostics: Mapping[str, Any] | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.last_retry_class = last_retry_class
        self.last_status_code = last_status_code
        self.attempts_tried = attempts_tried
        self.request_hash = request_hash
        self.usage = dict(usage) if usage is not None else None
        self.response_diagnostics = (
            dict(response_diagnostics) if response_diagnostics is not None else None
        )
        self.retry_after_seconds = retry_after_seconds


class StructuredOutputValidationError(RuntimeError):
    """Preserve a paid provider response rejected by local semantic validation."""

    def __init__(
        self,
        *,
        call: CallResult,
        parsed_payload: Any,
        response_schema: Type[BaseModel],
        validation_error: ValidationError,
    ) -> None:
        self.call = call
        self.parsed_payload = parsed_payload
        self.response_schema = response_schema
        self.validation_errors = json.loads(
            json.dumps(
                validation_error.errors(
                    include_url=False,
                    include_context=True,
                    include_input=False,
                ),
                default=str,
            )
        )
        super().__init__(
            f"{response_schema.__name__} semantic validation failed after a "
            f"successful provider response: {validation_error}"
        )


def require_reported_usage(
    usage: Mapping[str, Any] | None, *, stage: str
) -> dict[str, int]:
    """Return normalized provider usage or fail closed on absent accounting."""

    raw = dict(usage or {})
    required = ("prompt_tokens", "completion_tokens", "total_tokens")
    try:
        normalized = {key: int(raw[key]) for key in required}
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{stage} provider reported usage is missing or invalid") from exc
    if normalized["prompt_tokens"] <= 0 or normalized["completion_tokens"] <= 0:
        raise RuntimeError(
            f"{stage} provider reported usage must contain positive prompt/completion tokens"
        )
    if normalized["total_tokens"] != (
        normalized["prompt_tokens"] + normalized["completion_tokens"]
    ):
        raise RuntimeError(
            f"{stage} provider reported total_tokens is internally inconsistent"
        )
    return normalized


def openai_strict_json_schema(response_schema: Type[BaseModel]) -> dict[str, Any]:
    """Return a locally validated OpenAI Structured Outputs schema.

    OpenAI strict schemas require every declared object property to be listed
    in ``required`` and every object to forbid additional properties.  Reject
    incompatible Pydantic schemas before an HTTP attempt instead of paying for
    a deterministic provider-side 400.  Post-hoc Pydantic validation remains
    authoritative for value constraints and model validators.
    """

    schema = response_schema.model_json_schema()
    if schema.get("type") != "object" or not isinstance(
        schema.get("properties"), Mapping
    ):
        raise ValueError(
            f"{response_schema.__name__} strict schema root must be an object"
        )

    def visit(node: Any, path: str) -> None:
        if isinstance(node, list):
            for index, item in enumerate(node):
                visit(item, f"{path}[{index}]")
            return
        if not isinstance(node, dict):
            return
        unsupported = sorted(OPENAI_UNSUPPORTED_STRICT_SCHEMA_KEYWORDS & set(node))
        if unsupported:
            raise ValueError(
                f"{response_schema.__name__} strict schema uses unsupported "
                f"keywords at {path}: {unsupported}"
            )
        if "default" in node:
            raise ValueError(
                f"{response_schema.__name__} strict schema contains a default "
                f"at {path}; generated fields must be explicit"
            )
        if node.get("type") == "object" or "properties" in node:
            properties = node.get("properties")
            if not isinstance(properties, dict):
                raise ValueError(
                    f"{response_schema.__name__} strict object at {path} must "
                    "declare properties"
                )
            if node.get("additionalProperties") is not False:
                raise ValueError(
                    f"{response_schema.__name__} strict object at {path} must "
                    "set additionalProperties=false"
                )
            required = node.get("required")
            if (
                not isinstance(required, list)
                or len(required) != len(set(required))
                or set(required) != set(properties)
            ):
                raise ValueError(
                    f"{response_schema.__name__} strict object at {path} must "
                    "list every property exactly once in required"
                )
        for key, value in node.items():
            visit(value, f"{path}.{key}")

    visit(schema, "$")
    return schema


def _bounded_provider_error_parts(
    node: Any, *, depth: int = 0, maximum_parts: int = 12
) -> list[str]:
    """Extract only named error fields from dict- or list-shaped JSON.

    Some compatible endpoints return a top-level list instead of OpenAI's
    ``{"error": ...}`` object.  Recursion is deliberately shallow and only
    whitelisted diagnostic fields are retained, so prompts, responses, and
    credentials cannot leak into an attempt ledger.
    """

    if depth > 5 or maximum_parts <= 0:
        return []
    if isinstance(node, list):
        parts: list[str] = []
        for value in node[:10]:
            parts.extend(
                _bounded_provider_error_parts(
                    value,
                    depth=depth + 1,
                    maximum_parts=maximum_parts - len(parts),
                )
            )
            if len(parts) >= maximum_parts:
                break
        return parts[:maximum_parts]
    if not isinstance(node, Mapping):
        return []
    parts = []
    for key in ("message", "type", "param", "code", "status", "reason"):
        value = node.get(key)
        if isinstance(value, (str, int, float)) and str(value).strip():
            cleaned = " ".join(str(value).split())[:1000]
            parts.append(f"{key}={cleaned}")
    for key in ("error", "errors", "details", "fieldViolations"):
        if key in node and len(parts) < maximum_parts:
            parts.extend(
                _bounded_provider_error_parts(
                    node[key],
                    depth=depth + 1,
                    maximum_parts=maximum_parts - len(parts),
                )
            )
    # Preserve order while suppressing duplicated wrappers.
    return list(dict.fromkeys(parts))[:maximum_parts]


def _provider_error_summary(response: httpx.Response) -> str:
    """Extract a bounded structured provider error without logging prompts."""

    try:
        body = response.json()
    except (ValueError, json.JSONDecodeError):
        return "provider returned no structured error detail"
    parts = _bounded_provider_error_parts(body)
    return "; ".join(parts) or "provider returned no structured error detail"


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse Retry-After without treating a malformed header as retryable data."""

    raw = response.headers.get("retry-after")
    if raw is None or not raw.strip():
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _openai_usage_from_body(body: Any) -> dict[str, int] | None:
    """Preserve valid usage even when the response payload is otherwise malformed."""

    if not isinstance(body, Mapping) or not isinstance(body.get("usage"), Mapping):
        return None
    usage_raw = body["usage"]
    try:
        return {
            "prompt_tokens": int(usage_raw.get("prompt_tokens") or 0),
            "completion_tokens": int(usage_raw.get("completion_tokens") or 0),
            "total_tokens": int(usage_raw.get("total_tokens") or 0),
        }
    except (TypeError, ValueError):
        return None


def _gemini_usage_from_body(body: Any) -> dict[str, int] | None:
    """Normalize native Gemini generateContent usage metadata."""

    if not isinstance(body, Mapping) or not isinstance(
        body.get("usageMetadata"), Mapping
    ):
        return None
    usage_raw = body["usageMetadata"]
    try:
        return {
            "prompt_tokens": int(usage_raw.get("promptTokenCount") or 0),
            "completion_tokens": int(usage_raw.get("candidatesTokenCount") or 0),
            "total_tokens": int(usage_raw.get("totalTokenCount") or 0),
        }
    except (TypeError, ValueError):
        return None


def _provider_response_diagnostics(
    response: httpx.Response, *, body: Any = None
) -> dict[str, Any]:
    """Return bounded response metadata and a body hash, never prompts or secrets."""

    if body is None:
        try:
            body = response.json()
        except (ValueError, json.JSONDecodeError):
            body = None
    try:
        response_text = response.text
    except (httpx.ResponseNotRead, UnicodeError):
        response_text = ""
    diagnostics: dict[str, Any] = {
        "status_code": int(response.status_code),
        "response_body_sha256": sha256_text(response_text),
        "response_json_type": type(body).__name__ if body is not None else None,
        "retry_after_seconds": _retry_after_seconds(response),
    }
    for header in ("x-request-id", "request-id", "nv-request-id"):
        value = response.headers.get(header)
        if value:
            diagnostics["provider_request_id"] = str(value)[:256]
            diagnostics["provider_request_id_header"] = header
            break
    if isinstance(body, Mapping):
        diagnostics["response_json_top_level_keys"] = sorted(
            str(key) for key in body
        )[:100]
        choices = body.get("choices")
        if isinstance(choices, list):
            diagnostics["choices_count"] = len(choices)
            if choices and isinstance(choices[0], Mapping):
                diagnostics["first_choice_keys"] = sorted(
                    str(key) for key in choices[0]
                )[:100]
                message = choices[0].get("message")
                if isinstance(message, Mapping):
                    diagnostics["first_message_keys"] = sorted(
                        str(key) for key in message
                    )[:100]
        candidates = body.get("candidates")
        if isinstance(candidates, list):
            diagnostics["candidates_count"] = len(candidates)
            if candidates and isinstance(candidates[0], Mapping):
                diagnostics["first_candidate_keys"] = sorted(
                    str(key) for key in candidates[0]
                )[:100]
    elif isinstance(body, list):
        diagnostics["response_json_list_length"] = len(body)
        if body and isinstance(body[0], Mapping):
            diagnostics["first_list_item_keys"] = sorted(
                str(key) for key in body[0]
            )[:100]
    return diagnostics


def chat_request_payload(
    endpoint: Endpoint,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    seed: int | None,
    response_schema: Type[BaseModel] | None,
) -> dict[str, Any]:
    """Build the exact initial provider payload used for cost/provenance hashes."""

    transport = endpoint_transport(endpoint)
    if transport == "anthropic_messages":
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]
        payload: dict[str, Any] = {
            "model": endpoint.model,
            "messages": non_system,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if response_schema is not None:
            payload["tools"] = [
                {
                    "name": ANTHROPIC_STRUCTURED_OUTPUT_TOOL_NAME,
                    "description": (
                        "Return the evaluator result using the required strict schema."
                    ),
                    "input_schema": response_schema.model_json_schema(),
                }
            ]
            payload["tool_choice"] = {
                "type": "tool",
                "name": ANTHROPIC_STRUCTURED_OUTPUT_TOOL_NAME,
            }
        return payload
    if transport == "gemini_generate_content":
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        contents: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "")
            if role == "system":
                continue
            if role == "assistant":
                native_role = "model"
            elif role == "user":
                native_role = "user"
            else:
                raise ValueError(f"unsupported Gemini message role: {role}")
            contents.append(
                {
                    "role": native_role,
                    "parts": [{"text": str(message["content"])}],
                }
            )
        if not contents:
            raise ValueError("Gemini request requires at least one non-system message")
        generation_config: dict[str, Any] = {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_tokens),
        }
        if seed is not None:
            generation_config["seed"] = int(seed)
        if response_schema is not None:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseJsonSchema"] = (
                response_schema.model_json_schema()
            )
        payload = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_parts:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_parts)}]
            }
        return payload
    payload = {
        "model": endpoint.model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if response_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_schema.__name__,
                "strict": True,
                "schema": openai_strict_json_schema(response_schema),
            },
        }
    return payload


def request_payload_has_schema(payload: Mapping[str, Any]) -> bool:
    """Recognize OpenAI, Anthropic, or native Gemini schema contracts."""

    response_format = payload.get("response_format")
    if isinstance(response_format, Mapping):
        json_schema = response_format.get("json_schema")
        if (
            response_format.get("type") == "json_schema"
            and isinstance(json_schema, Mapping)
            and isinstance(json_schema.get("schema"), Mapping)
        ):
            return True
    generation_config = payload.get("generationConfig")
    if (
        isinstance(generation_config, Mapping)
        and generation_config.get("responseMimeType") == "application/json"
        and isinstance(generation_config.get("responseJsonSchema"), Mapping)
    ):
        return True
    tools = payload.get("tools")
    tool_choice = payload.get("tool_choice")
    if not isinstance(tools, list) or not isinstance(tool_choice, Mapping):
        return False
    forced_name = str(tool_choice.get("name") or "")
    return bool(
        tool_choice.get("type") == "tool"
        and forced_name
        and any(
            isinstance(tool, Mapping)
            and tool.get("name") == forced_name
            and isinstance(tool.get("input_schema"), Mapping)
            for tool in tools
        )
    )


class OpenAICompatibleClient:
    """Small fail-closed OpenAI-compatible chat client.

    It does not depend on a vendor SDK.  Requested JSON schema mode is
    fail-closed: a provider may not silently downgrade a frozen structured
    request to schema-less JSON.
    """

    def __init__(self, endpoint: Endpoint):
        self.endpoint = endpoint
        self._chat_path = self._resolve_chat_path(endpoint.base_url)
        self._client = httpx.Client(
            base_url=endpoint.base_url.rstrip("/"),
            timeout=endpoint.timeout_seconds,
            headers={
                "Authorization": f"Bearer {endpoint.api_key}",
                "Content-Type": "application/json",
            },
        )

    @staticmethod
    def _resolve_chat_path(base_url: str) -> str:
        """Return the chat-completions path for root or prefixed base URLs.

        OpenAI/NVIDIA often use a root base URL plus /v1/chat/completions.
        Some OpenAI-compatible providers document a prefixed base URL, e.g.
        Gemini's /v1beta/openai, where appending another /v1 would break.
        """
        normalized = base_url.rstrip("/")
        if normalized.endswith("/v1") or normalized.endswith("/openai"):
            return "/chat/completions"
        return "/v1/chat/completions"

    def close(self) -> None:
        self._client.close()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        response_schema: Type[T] | None = None,
        retries: int = 3,
    ) -> tuple[CallResult, T | None]:
        payload = chat_request_payload(
            self.endpoint,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            response_schema=response_schema,
        )
        request_hash = sha256_text(canonical_json(payload))
        if retries < 1:
            raise ValueError("retries must be at least 1")
        errors: list[str] = []
        last_retry_class = "other"
        last_status_code: int | None = None
        last_usage: dict[str, int] | None = None
        last_response_diagnostics: dict[str, Any] | None = None
        last_retry_after_seconds: float | None = None
        attempts_tried = 0
        for attempt in range(1, retries + 1):
            attempts_tried = attempt
            started = time.perf_counter()
            response: httpx.Response | None = None
            body: Any = None
            usage: dict[str, int] | None = None
            try:
                response = self._client.post(self._chat_path, json=payload)
                # Rate limiting and request-timeout responses are transient,
                # but remain visible to the outer durable-attempt ledger when
                # this client is deliberately called with retries=1.
                if response.status_code in {408, 429}:
                    last_status_code = int(response.status_code)
                    last_retry_class = (
                        "rate_limited_429"
                        if response.status_code == 429
                        else "request_timeout_408"
                    )
                    try:
                        body = response.json()
                    except (ValueError, json.JSONDecodeError):
                        body = None
                    last_usage = _openai_usage_from_body(body)
                    last_response_diagnostics = _provider_response_diagnostics(
                        response, body=body
                    )
                    last_retry_after_seconds = _retry_after_seconds(response)
                    wait = max(
                        last_retry_after_seconds or 0.0,
                        float(min(30 * attempt, 120)),
                    )
                    errors.append(
                        f"attempt {attempt}: HTTP {response.status_code} "
                        f"({last_retry_class}), waiting {wait:g}s"
                    )
                    if attempt < retries:
                        time.sleep(wait)
                        continue
                    break
                # Deterministic client errors are not fixed by retries.  Retain
                # the frozen schema payload and surface the provider's bounded
                # structured error instead of silently issuing a schema-less
                # request with a different hash.
                if 400 <= response.status_code < 500:
                    try:
                        body = response.json()
                    except (ValueError, json.JSONDecodeError):
                        body = None
                    raise ProviderRequestError(
                        status_code=response.status_code,
                        detail=_provider_error_summary(response),
                        schema_mode="response_format" in payload,
                        request_hash=request_hash,
                        usage=_openai_usage_from_body(body),
                        response_diagnostics=_provider_response_diagnostics(
                            response, body=body
                        ),
                    )
                response.raise_for_status()
                body = response.json()
                usage = _openai_usage_from_body(body) or {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
                text = body["choices"][0]["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("empty model response")
                provider_finish_reason, normalized_finish_reason = (
                    normalize_provider_finish_reason(body)
                )
                call = CallResult(
                    text=text.strip(),
                    raw_response=body,
                    usage=usage,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    request_hash=request_hash,
                    provider_finish_reason=provider_finish_reason,
                    normalized_finish_reason=normalized_finish_reason,
                )
                if response_schema is None:
                    return call, None
                try:
                    parsed_obj = json.loads(text)
                except json.JSONDecodeError:
                    start, end = text.find("{"), text.rfind("}")
                    if start < 0 or end <= start:
                        raise ValueError("response is not valid JSON")
                    parsed_obj = json.loads(text[start : end + 1])
                try:
                    parsed = response_schema.model_validate(parsed_obj)
                except ValidationError as exc:
                    raise StructuredOutputValidationError(
                        call=call,
                        parsed_payload=parsed_obj,
                        response_schema=response_schema,
                        validation_error=exc,
                    ) from exc
                return call, parsed
            except (
                httpx.HTTPError,
                KeyError,
                IndexError,
                TypeError,
                AttributeError,
                ValueError,
                ValidationError,
                json.JSONDecodeError,
            ) as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                last_retry_class, last_status_code = _classify_retryable_exception(exc)
                if response is not None:
                    last_response_diagnostics = _provider_response_diagnostics(
                        response, body=body
                    )
                    last_retry_after_seconds = _retry_after_seconds(response)
                    last_usage = usage or _openai_usage_from_body(body)
                if attempt == retries:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))
        raise RetryableProviderError(
            "API call failed after strict retries: " + " | ".join(errors),
            last_retry_class=last_retry_class,
            last_status_code=last_status_code,
            attempts_tried=attempts_tried,
            request_hash=request_hash,
            usage=last_usage,
            response_diagnostics=last_response_diagnostics,
            retry_after_seconds=last_retry_after_seconds,
        )


class GeminiNativeClient:
    """Fail-closed client for Gemini's native ``generateContent`` API.

    Google documents structured output on this surface through
    ``generationConfig.responseJsonSchema``.  It is intentionally separate
    from ``OpenAICompatibleClient`` so an OpenAI compatibility-layer quirk
    cannot silently change the frozen schema contract.
    """

    def __init__(self, endpoint: Endpoint):
        if endpoint_transport(endpoint) != "gemini_generate_content":
            raise ValueError("GeminiNativeClient requires gemini_generate_content")
        if "generativelanguage.googleapis.com" not in endpoint.base_url:
            raise ValueError("Gemini native transport requires Google's API host")
        if endpoint.base_url.rstrip("/").endswith("/openai"):
            raise ValueError("Gemini native transport forbids the /openai route")
        self.endpoint = endpoint
        model_name = endpoint.model.removeprefix("models/").strip()
        if not model_name:
            raise ValueError("Gemini native transport requires a model name")
        self._generate_path = (
            f"/models/{quote(model_name, safe='-._')}:generateContent"
        )
        self._client = httpx.Client(
            base_url=endpoint.base_url.rstrip("/"),
            timeout=endpoint.timeout_seconds,
            headers={
                "x-goog-api-key": endpoint.api_key,
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        self._client.close()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        response_schema: Type[T] | None = None,
        retries: int = 3,
    ) -> tuple[CallResult, T | None]:
        payload = chat_request_payload(
            self.endpoint,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            response_schema=response_schema,
        )
        request_hash = sha256_text(
            canonical_json(
                {
                    "transport": "gemini_generate_content",
                    "base_url": self.endpoint.base_url.rstrip("/"),
                    "model": self.endpoint.model,
                    "path": self._generate_path,
                    "payload": payload,
                }
            )
        )
        if retries < 1:
            raise ValueError("retries must be at least 1")
        errors: list[str] = []
        last_retry_class = "other"
        last_status_code: int | None = None
        last_usage: dict[str, int] | None = None
        last_response_diagnostics: dict[str, Any] | None = None
        last_retry_after_seconds: float | None = None
        attempts_tried = 0
        for attempt in range(1, retries + 1):
            attempts_tried = attempt
            started = time.perf_counter()
            response: httpx.Response | None = None
            body: Any = None
            usage: dict[str, int] | None = None
            try:
                response = self._client.post(self._generate_path, json=payload)
                if response.status_code in {408, 429}:
                    last_status_code = int(response.status_code)
                    last_retry_class = (
                        "rate_limited_429"
                        if response.status_code == 429
                        else "request_timeout_408"
                    )
                    try:
                        body = response.json()
                    except (ValueError, json.JSONDecodeError):
                        body = None
                    last_usage = _gemini_usage_from_body(body)
                    last_response_diagnostics = _provider_response_diagnostics(
                        response, body=body
                    )
                    last_retry_after_seconds = _retry_after_seconds(response)
                    wait = max(
                        last_retry_after_seconds or 0.0,
                        float(min(30 * attempt, 120)),
                    )
                    errors.append(
                        f"attempt {attempt}: HTTP {response.status_code} "
                        f"({last_retry_class}), waiting {wait:g}s"
                    )
                    if attempt < retries:
                        time.sleep(wait)
                        continue
                    break
                if 400 <= response.status_code < 500:
                    try:
                        body = response.json()
                    except (ValueError, json.JSONDecodeError):
                        body = None
                    raise ProviderRequestError(
                        status_code=response.status_code,
                        detail=_provider_error_summary(response),
                        schema_mode=request_payload_has_schema(payload),
                        request_hash=request_hash,
                        usage=_gemini_usage_from_body(body),
                        response_diagnostics=_provider_response_diagnostics(
                            response, body=body
                        ),
                    )
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, Mapping):
                    raise TypeError("Gemini response root is not an object")
                usage = _gemini_usage_from_body(body) or {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
                candidates = body["candidates"]
                if not isinstance(candidates, list) or not candidates:
                    raise ValueError("empty model response")
                first = candidates[0]
                if not isinstance(first, Mapping):
                    raise TypeError("Gemini candidate is not an object")
                content = first["content"]
                if not isinstance(content, Mapping):
                    raise TypeError("Gemini candidate content is not an object")
                parts = content["parts"]
                if not isinstance(parts, list):
                    raise TypeError("Gemini candidate parts is not a list")
                text = "".join(
                    str(part["text"])
                    for part in parts
                    if isinstance(part, Mapping)
                    and isinstance(part.get("text"), str)
                ).strip()
                if not text:
                    raise ValueError("empty model response")
                provider_finish_reason, normalized_finish_reason = (
                    normalize_provider_finish_reason(body)
                )
                call = CallResult(
                    text=text,
                    raw_response=dict(body),
                    usage=usage,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    request_hash=request_hash,
                    provider_finish_reason=provider_finish_reason,
                    normalized_finish_reason=normalized_finish_reason,
                )
                if response_schema is None:
                    return call, None
                try:
                    parsed_obj = json.loads(text)
                except json.JSONDecodeError:
                    start, end = text.find("{"), text.rfind("}")
                    if start < 0 or end <= start:
                        raise ValueError("response is not valid JSON")
                    parsed_obj = json.loads(text[start : end + 1])
                try:
                    parsed = response_schema.model_validate(parsed_obj)
                except ValidationError as exc:
                    raise StructuredOutputValidationError(
                        call=call,
                        parsed_payload=parsed_obj,
                        response_schema=response_schema,
                        validation_error=exc,
                    ) from exc
                return call, parsed
            except (
                httpx.HTTPError,
                KeyError,
                IndexError,
                TypeError,
                AttributeError,
                ValueError,
                ValidationError,
                json.JSONDecodeError,
            ) as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                last_retry_class, last_status_code = _classify_retryable_exception(exc)
                if response is not None:
                    last_response_diagnostics = _provider_response_diagnostics(
                        response, body=body
                    )
                    last_retry_after_seconds = _retry_after_seconds(response)
                    last_usage = usage or _gemini_usage_from_body(body)
                if attempt == retries:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))
        raise RetryableProviderError(
            "Gemini API call failed after strict retries: " + " | ".join(errors),
            last_retry_class=last_retry_class,
            last_status_code=last_status_code,
            attempts_tried=attempts_tried,
            request_hash=request_hash,
            usage=last_usage,
            response_diagnostics=last_response_diagnostics,
            retry_after_seconds=last_retry_after_seconds,
        )


class AnthropicClient:
    """Minimal Anthropic Messages API client (claude-* models).

    Anthropic's native API is NOT OpenAI-compatible, so we need a separate
    client.  The interface mirrors OpenAICompatibleClient so callers can swap
    transparently.
    """

    BASE_URL = "https://api.anthropic.com"
    API_VERSION = "2023-06-01"

    def __init__(self, endpoint: Endpoint):
        self.endpoint = endpoint
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            timeout=endpoint.timeout_seconds,
            headers={
                "x-api-key": endpoint.api_key,
                "anthropic-version": self.API_VERSION,
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        self._client.close()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        response_schema: Type[T] | None = None,
        retries: int = 3,
    ) -> tuple[CallResult, T | None]:
        payload = chat_request_payload(
            self.endpoint,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            response_schema=response_schema,
        )
        request_hash = sha256_text(canonical_json(payload))
        errors: list[str] = []
        for attempt in range(1, retries + 1):
            started = time.perf_counter()
            try:
                response = self._client.post("/v1/messages", json=payload)
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    raise ProviderRequestError(
                        status_code=response.status_code,
                        detail=_provider_error_summary(response),
                        schema_mode=response_schema is not None,
                    )
                response.raise_for_status()
                body = response.json()
                content = body["content"]
                parsed_obj = None
                if response_schema is not None:
                    tool_blocks = [
                        block
                        for block in content
                        if block.get("type") == "tool_use"
                        and block.get("name")
                        == ANTHROPIC_STRUCTURED_OUTPUT_TOOL_NAME
                    ]
                    if len(tool_blocks) != 1:
                        raise ValueError(
                            "Anthropic structured response must contain exactly one "
                            "required tool_use block"
                        )
                    parsed_obj = tool_blocks[0].get("input")
                    text = canonical_json(parsed_obj)
                else:
                    text_blocks = [
                        block.get("text")
                        for block in content
                        if block.get("type") == "text"
                    ]
                    text = "\n".join(
                        value for value in text_blocks if isinstance(value, str)
                    ).strip()
                    if not text:
                        raise ValueError("empty model response")
                usage_raw = body.get("usage") or {}
                usage = {
                    "prompt_tokens": int(usage_raw.get("input_tokens") or 0),
                    "completion_tokens": int(usage_raw.get("output_tokens") or 0),
                    "total_tokens": int(
                        (usage_raw.get("input_tokens") or 0)
                        + (usage_raw.get("output_tokens") or 0)
                    ),
                }
                provider_finish_reason, normalized_finish_reason = (
                    normalize_provider_finish_reason(body)
                )
                call = CallResult(
                    text=text.strip(),
                    raw_response=body,
                    usage=usage,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    request_hash=request_hash,
                    provider_finish_reason=provider_finish_reason,
                    normalized_finish_reason=normalized_finish_reason,
                )
                if response_schema is None:
                    return call, None
                try:
                    parsed = response_schema.model_validate(parsed_obj)
                except ValidationError as exc:
                    raise StructuredOutputValidationError(
                        call=call,
                        parsed_payload=parsed_obj,
                        response_schema=response_schema,
                        validation_error=exc,
                    ) from exc
                return call, parsed
            except (httpx.HTTPError, KeyError, IndexError, ValueError, ValidationError, json.JSONDecodeError) as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                if attempt == retries:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError("Anthropic API call failed after retries: " + " | ".join(errors))


def make_client(
    endpoint: Endpoint,
) -> OpenAICompatibleClient | GeminiNativeClient | AnthropicClient:
    """Return the client for the endpoint's frozen transport protocol."""

    transport = endpoint_transport(endpoint)
    if transport == "anthropic_messages":
        return AnthropicClient(endpoint)
    if transport == "gemini_generate_content":
        return GeminiNativeClient(endpoint)
    return OpenAICompatibleClient(endpoint)


def request_log(
    *,
    stage: str,
    endpoint: Endpoint,
    messages: list[dict[str, str]],
    result: CallResult | None,
    parsed: BaseModel | None,
    error: str | None,
    prompt_hash: str,
    record_ids: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp": utc_now(),
        "stage": stage,
        **record_ids,
        "model": endpoint.model,
        "model_family": endpoint.family,
        "base_url": endpoint.base_url,
        "transport": endpoint_transport(endpoint),
        "prompt_hash": prompt_hash,
        "messages_hash": sha256_text(canonical_json(messages)),
        "request_hash": result.request_hash if result else None,
        "raw_text": result.text if result else None,
        "raw_response": result.raw_response if result else None,
        "provider_finish_reason": result.provider_finish_reason if result else None,
        "normalized_finish_reason": (
            result.normalized_finish_reason if result else None
        ),
        "completion_truncated": (
            result.normalized_finish_reason == "length" if result else None
        ),
        "validated": parsed.model_dump(mode="json") if parsed else None,
        "usage": result.usage if result else None,
        "latency_ms": result.latency_ms if result else None,
        "error": error,
    }
