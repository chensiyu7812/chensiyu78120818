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
ANTHROPIC_STRICT_TOOL_SCHEMA_PROJECTION_PROTOCOL = (
    "anthropic-strict-tool-schema-projection-v2-strip-numeric-and-array-bounds"
)
ANTHROPIC_UNSUPPORTED_STRICT_TOOL_SCHEMA_KEYWORDS = frozenset(
    {
        "exclusiveMaximum",
        "exclusiveMinimum",
        "maxItems",
        "maximum",
        "minItems",
        "minimum",
        "multipleOf",
    }
)
GEMINI_USAGE_BREAKDOWN_KEYS = (
    "gemini_prompt_tokens",
    "gemini_candidate_tokens",
    "gemini_thought_tokens",
    "gemini_tool_use_prompt_tokens",
    "gemini_cached_content_tokens",
)
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
    # Declared, frozen endpoint capability, decided before any request is
    # sent -- never a runtime fallback after a rejection (see
    # test_schema_http_400_is_diagnostic_and_never_downgrades). Some OpenAI-
    # compatible providers (DeepSeek's own official API, unlike NVIDIA's
    # gateway hosting the same model) reject the newer strict json_schema
    # response_format with a 400 and only support the older, loose
    # json_object mode. Default True preserves every existing endpoint's
    # behavior unchanged.
    supports_strict_json_schema: bool = True
    # "provider_default" omits the ``thinking`` request field entirely,
    # preserving every existing endpoint's behavior unchanged. deepseek-v4-
    # flash defaults to thinking enabled (confirmed via api-docs.deepseek.com
    # /guides/thinking_mode/) and has no separate reasoning-token budget from
    # max_tokens, so a short judge max_tokens can be entirely consumed by an
    # unrequested chain-of-thought before any JSON content is emitted --
    # exactly the finish_reason=length pattern observed on three real
    # deepseek_official judge calls (completion_tokens landed precisely on
    # the configured ceiling every time). "disabled"/"enabled" send an
    # explicit ``{"thinking": {"type": ...}}`` field.
    thinking_mode: str = "provider_default"
    # OpenAI-compatible Qwen hybrid-thinking endpoints use the top-level
    # ``enable_thinking`` boolean rather than DeepSeek's ``thinking`` object.
    # ``None`` preserves every existing endpoint byte-for-byte.  The field is
    # deliberately part of Endpoint so it is included in content-addressed
    # physical call identities.
    enable_thinking: bool | None = None
    # OpenAI reasoning models can require the provider-default sampling
    # surface. ``explicit`` preserves every historical endpoint byte-for-byte;
    # ``omit`` deliberately leaves temperature out of the HTTP payload.
    temperature_mode: Literal["explicit", "omit"] = "explicit"
    # Newer OpenAI reasoning endpoints use ``max_completion_tokens`` instead
    # of the older Chat Completions ``max_tokens`` field.  Keep the historical
    # default and opt in per endpoint so no unrelated provider changes shape.
    max_output_tokens_parameter: Literal[
        "max_tokens", "max_completion_tokens"
    ] = "max_tokens"
    # Anthropic strict tool use is an endpoint capability, not a runtime
    # fallback.  The historical default remains unchanged.  Qualification
    # stages that require provider-enforced schema conformance opt in and bind
    # this value into their physical-call identity.
    anthropic_strict_tool_use: bool = False
    # Native Gemini 2.5 requests count hidden thoughts against maxOutputTokens.
    # ``None`` preserves every historical request byte-for-byte; an explicit
    # non-negative value is emitted as generationConfig.thinkingConfig.
    gemini_thinking_budget: int | None = None
    # Pre-5.1 OpenAI reasoning models default to medium effort, which can
    # consume an entire max_completion_tokens allowance before emitting the
    # visible strict-JSON answer. ``None`` preserves historical behavior.
    openai_reasoning_effort: Literal[
        "minimal", "low", "medium", "high"
    ] | None = None

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
    # Structured providers occasionally wrap an otherwise valid JSON object in
    # a Markdown fence or a short prose prefix.  When that surface is repaired
    # deterministically, retain an explicit audit record instead of silently
    # pretending that the provider returned exact JSON.
    structured_output_audit: dict[str, Any] | None = None


def parse_audited_json_surface(text: str) -> tuple[Any, dict[str, Any]]:
    """Parse one JSON value with narrowly bounded, fully audited normalization.

    Exact JSON remains the preferred surface.  The only accepted normalization
    removes either one complete Markdown JSON fence or a short prefix/suffix
    surrounding exactly one JSON object.  JSON values are never edited.  A
    malformed object, multiple objects, or a long wrapper remains a provider
    output-format failure and is handled by the outer paid-attempt ledger.
    """

    raw = str(text)
    stripped = raw.strip()
    raw_sha256 = sha256_text(raw)
    parse_error: json.JSONDecodeError | None = None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        parse_error = exc
        parsed = None
    else:
        return parsed, {
            "initially_valid_json": True,
            "normalization_kind": "exact_json",
            "raw_text_sha256": raw_sha256,
            "normalized_json_sha256": sha256_text(canonical_json(parsed)),
            "discarded_prefix_chars": 0,
            "discarded_suffix_chars": 0,
        }

    lines = stripped.splitlines()
    if (
        len(lines) >= 3
        and lines[0].strip().casefold() in {"```", "```json"}
        and lines[-1].strip() == "```"
    ):
        inner = "\n".join(lines[1:-1]).strip()
        try:
            fenced = json.loads(inner)
        except json.JSONDecodeError:
            pass
        else:
            return fenced, {
                "initially_valid_json": False,
                "normalization_kind": "single_markdown_json_fence",
                "raw_text_sha256": raw_sha256,
                "normalized_json_sha256": sha256_text(canonical_json(fenced)),
                "discarded_prefix_chars": len(lines[0]) + 1,
                "discarded_suffix_chars": len(lines[-1]) + 1,
            }

    # Recover the recurring provider shape `We{...}` without accepting an
    # arbitrary essay around JSON.  The wrapper is bounded, may not contain a
    # second JSON delimiter, and exactly one decodable object must exist.
    decoder = json.JSONDecoder()
    candidates: list[tuple[Any, str, int, int]] = []
    for start, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            embedded, end = decoder.raw_decode(stripped, start)
        except json.JSONDecodeError:
            continue
        prefix = stripped[:start]
        suffix = stripped[end:]
        if not isinstance(embedded, dict):
            continue
        if len(prefix) > 80 or len(suffix) > 80:
            continue
        if any(token in prefix + suffix for token in ("{", "}", "[", "]")):
            continue
        candidates.append(
            (embedded, "single_bounded_json_object", len(prefix), len(suffix))
        )

    if len(candidates) != 1:
        if parse_error is None:  # pragma: no cover - exact JSON returned above
            raise ValueError("provider response is not valid JSON")
        raise parse_error
    parsed, kind, prefix_chars, suffix_chars = candidates[0]
    return parsed, {
        "initially_valid_json": False,
        "normalization_kind": kind,
        "raw_text_sha256": raw_sha256,
        "normalized_json_sha256": sha256_text(canonical_json(parsed)),
        "discarded_prefix_chars": int(prefix_chars),
        "discarded_suffix_chars": int(suffix_chars),
    }


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
    "network_timeout", "missing_field", "provider_output_format", "other".
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
    if isinstance(exc, ValueError) and "finish_reason=length" in str(exc):
        # The provider truncated the response because max_tokens ran out
        # (commonly an unrequested reasoning/thinking budget consuming the
        # entire allowance before any content token) -- a contract/config
        # mismatch, not transient noise. Retrying the identical request is
        # unlikely to help, so this is deliberately its own class, outside
        # both RETRYABLE_UP_TO_FULL_BUDGET and BOUNDED_PROVIDER_OUTPUT_RETRY_
        # CLASSES in bounded_retry.py -- it falls through to terminal on the
        # first occurrence rather than burning a retry budget chasing the
        # same ceiling.
        return "output_token_limit", None
    if isinstance(exc, (KeyError, IndexError, TypeError, AttributeError)) or (
        isinstance(exc, ValueError) and "empty model response" in str(exc)
    ):
        return "missing_field", None
    if isinstance(exc, json.JSONDecodeError) or (
        isinstance(exc, ValueError) and "response is not valid JSON" in str(exc)
    ):
        return "provider_output_format", None
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
        provider_text: str | None = None,
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
        self.provider_text = str(provider_text) if provider_text is not None else None


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
    present_gemini_keys = {
        key for key in GEMINI_USAGE_BREAKDOWN_KEYS if key in raw
    }
    if present_gemini_keys:
        if present_gemini_keys != set(GEMINI_USAGE_BREAKDOWN_KEYS):
            raise RuntimeError(
                f"{stage} Gemini usage breakdown is incomplete"
            )
        try:
            gemini = {
                key: int(raw[key]) for key in GEMINI_USAGE_BREAKDOWN_KEYS
            }
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{stage} Gemini usage breakdown is invalid"
            ) from exc
        if any(value < 0 for value in gemini.values()):
            raise RuntimeError(
                f"{stage} Gemini usage breakdown cannot contain negative tokens"
            )
        if gemini["gemini_cached_content_tokens"] > gemini[
            "gemini_prompt_tokens"
        ]:
            raise RuntimeError(
                f"{stage} Gemini cached tokens exceed prompt tokens"
            )
        if normalized["prompt_tokens"] != (
            gemini["gemini_prompt_tokens"]
            + gemini["gemini_tool_use_prompt_tokens"]
        ):
            raise RuntimeError(
                f"{stage} Gemini normalized prompt accounting is inconsistent"
            )
        if normalized["completion_tokens"] != (
            gemini["gemini_candidate_tokens"]
            + gemini["gemini_thought_tokens"]
        ):
            raise RuntimeError(
                f"{stage} Gemini normalized completion accounting is inconsistent"
            )
        normalized.update(gemini)
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


def anthropic_strict_tool_schema(
    response_schema: Type[BaseModel],
) -> dict[str, Any]:
    """Project a canonical local schema onto Anthropic's strict-tool subset.

    Anthropic strict tool use rejects JSON-Schema numeric and array bound
    keywords such as ``minimum``, ``maximum``, ``minItems``, and ``maxItems``.
    The projection is deterministic and only affects the provider-visible tool
    schema; the canonical Pydantic model is still used after the paid response
    and remains authoritative for numeric/array bounds and cross-field
    validators.
    """

    canonical = response_schema.model_json_schema()

    def project(node: Any) -> Any:
        if isinstance(node, list):
            return [project(value) for value in node]
        if not isinstance(node, dict):
            return node
        return {
            key: project(value)
            for key, value in node.items()
            if key not in ANTHROPIC_UNSUPPORTED_STRICT_TOOL_SCHEMA_KEYWORDS
        }

    projected = project(canonical)
    if not isinstance(projected, dict):
        raise TypeError("Anthropic strict tool schema projection is not an object")
    return projected


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
    """Normalize native Gemini usage without dropping billable components.

    Gemini defines totalTokenCount as prompt + candidates + tool-use prompt +
    thoughts. Cached content is already included in promptTokenCount, so it is
    recorded for audit but is not added a second time.
    """

    if not isinstance(body, Mapping) or not isinstance(
        body.get("usageMetadata"), Mapping
    ):
        return None
    usage_raw = body["usageMetadata"]
    try:
        provider_prompt = int(usage_raw.get("promptTokenCount") or 0)
        candidates = int(usage_raw.get("candidatesTokenCount") or 0)
        thoughts = int(usage_raw.get("thoughtsTokenCount") or 0)
        tool_use_prompt = int(usage_raw.get("toolUsePromptTokenCount") or 0)
        cached_content = int(usage_raw.get("cachedContentTokenCount") or 0)
        return {
            "prompt_tokens": provider_prompt + tool_use_prompt,
            "completion_tokens": candidates + thoughts,
            "total_tokens": int(usage_raw.get("totalTokenCount") or 0),
            "gemini_prompt_tokens": provider_prompt,
            "gemini_candidate_tokens": candidates,
            "gemini_thought_tokens": thoughts,
            "gemini_tool_use_prompt_tokens": tool_use_prompt,
            "gemini_cached_content_tokens": cached_content,
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
                finish_reason = choices[0].get("finish_reason")
                if finish_reason is not None:
                    diagnostics["first_choice_finish_reason"] = str(finish_reason)
                message = choices[0].get("message")
                if isinstance(message, Mapping):
                    diagnostics["first_message_keys"] = sorted(
                        str(key) for key in message
                    )[:100]
                    # Bounded lengths only -- never the content itself, which
                    # may include the model's chain-of-thought or judged text.
                    content = message.get("content")
                    if isinstance(content, str):
                        diagnostics["first_message_content_length"] = len(content)
                    reasoning_content = message.get("reasoning_content")
                    if isinstance(reasoning_content, str):
                        diagnostics["first_message_reasoning_content_length"] = len(
                            reasoning_content
                        )
        usage = body.get("usage")
        if isinstance(usage, Mapping):
            completion_details = usage.get("completion_tokens_details")
            if isinstance(completion_details, Mapping):
                reasoning_tokens = completion_details.get("reasoning_tokens")
                if reasoning_tokens is not None:
                    try:
                        diagnostics["completion_reasoning_tokens"] = int(
                            reasoning_tokens
                        )
                    except (TypeError, ValueError):
                        pass
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
        if (
            endpoint.temperature_mode != "explicit"
            or endpoint.max_output_tokens_parameter != "max_tokens"
            or endpoint.gemini_thinking_budget is not None
            or endpoint.openai_reasoning_effort is not None
        ):
            raise ValueError(
                "OpenAI reasoning request capabilities cannot be applied to "
                "the Anthropic transport"
            )
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
            provider_schema = (
                anthropic_strict_tool_schema(response_schema)
                if endpoint.anthropic_strict_tool_use
                else response_schema.model_json_schema()
            )
            tool = {
                "name": ANTHROPIC_STRUCTURED_OUTPUT_TOOL_NAME,
                "description": (
                    "Return the evaluator result using the required strict schema."
                ),
                "input_schema": provider_schema,
            }
            if endpoint.anthropic_strict_tool_use:
                tool["strict"] = True
            payload["tools"] = [tool]
            payload["tool_choice"] = {
                "type": "tool",
                "name": ANTHROPIC_STRUCTURED_OUTPUT_TOOL_NAME,
            }
        return payload
    if transport == "gemini_generate_content":
        if (
            endpoint.temperature_mode != "explicit"
            or endpoint.max_output_tokens_parameter != "max_tokens"
            or endpoint.openai_reasoning_effort is not None
        ):
            raise ValueError(
                "OpenAI reasoning request capabilities cannot be applied to "
                "the Gemini transport"
            )
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
        if endpoint.gemini_thinking_budget is not None:
            thinking_budget = int(endpoint.gemini_thinking_budget)
            if thinking_budget < 0:
                raise ValueError("Gemini thinking budget must be non-negative")
            generation_config["thinkingConfig"] = {
                "thinkingBudget": thinking_budget
            }
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
    }
    if endpoint.gemini_thinking_budget is not None:
        raise ValueError(
            "Gemini thinking budget cannot be applied to OpenAI transport"
        )
    if endpoint.temperature_mode == "explicit":
        payload["temperature"] = float(temperature)
    elif endpoint.temperature_mode != "omit":  # pragma: no cover - Literal
        raise ValueError(
            f"unsupported temperature mode: {endpoint.temperature_mode}"
        )
    output_parameter = endpoint.max_output_tokens_parameter
    if output_parameter not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError(  # pragma: no cover - Literal
            f"unsupported max-output parameter: {output_parameter}"
        )
    payload[output_parameter] = int(max_tokens)
    if seed is not None:
        payload["seed"] = int(seed)
    if endpoint.thinking_mode != "provider_default":
        payload["thinking"] = {"type": endpoint.thinking_mode}
    if endpoint.enable_thinking is not None:
        payload["enable_thinking"] = bool(endpoint.enable_thinking)
    if endpoint.openai_reasoning_effort is not None:
        payload["reasoning_effort"] = endpoint.openai_reasoning_effort
    if response_schema is not None:
        if endpoint.supports_strict_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__,
                    "strict": True,
                    "schema": openai_strict_json_schema(response_schema),
                },
            }
        else:
            # Loose JSON mode: the provider only guarantees syntactically
            # valid JSON, not schema conformance. OpenAICompatibleClient.chat
            # still runs response_schema.model_validate(...) on the parsed
            # result afterward, so schema conformance is enforced locally
            # either way -- a mismatch surfaces as StructuredOutputValidation
            # Error and feeds the existing bounded provider-output retry,
            # exactly as it would under strict mode.
            payload["response_format"] = {"type": "json_object"}
    return payload


def request_payload_has_schema(payload: Mapping[str, Any]) -> bool:
    """Recognize OpenAI, Anthropic, or native Gemini schema contracts.

    Includes the loose ``json_object`` mode used by OpenAI-compatible
    providers that reject the stricter ``json_schema`` response_format
    (declared via Endpoint.supports_strict_json_schema=False) -- it is a
    deliberately weaker, but still intentional, structured-output request,
    not a missing one.
    """

    response_format = payload.get("response_format")
    if isinstance(response_format, Mapping):
        json_schema = response_format.get("json_schema")
        if (
            response_format.get("type") == "json_schema"
            and isinstance(json_schema, Mapping)
            and isinstance(json_schema.get("schema"), Mapping)
        ):
            return True
        if response_format.get("type") == "json_object":
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
        last_provider_text: str | None = None
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
                provider_finish_reason, normalized_finish_reason = (
                    normalize_provider_finish_reason(body)
                )
                text = body["choices"][0]["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    if normalized_finish_reason == "length":
                        # Distinguish "the provider truncated the response
                        # because max_tokens ran out" (e.g. an unrequested
                        # reasoning/thinking budget consumed everything
                        # before any content token) from a genuinely
                        # unexplained empty response -- these need different
                        # retry treatment (see _classify_retryable_exception).
                        raise ValueError(
                            "empty model response (finish_reason=length)"
                        )
                    raise ValueError("empty model response")
                last_provider_text = text
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
                    parsed_obj, surface_audit = parse_audited_json_surface(text)
                except json.JSONDecodeError as exc:
                    if normalized_finish_reason == "length":
                        raise ValueError(
                            "response is not valid JSON (finish_reason=length)"
                        ) from exc
                    raise
                call.structured_output_audit = surface_audit
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
            provider_text=last_provider_text,
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
        last_provider_text: str | None = None
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
                last_provider_text = text
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
                parsed_obj, surface_audit = parse_audited_json_surface(text)
                call.structured_output_audit = surface_audit
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
            provider_text=last_provider_text,
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
                last_usage = usage
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
            except (
                httpx.HTTPError,
                KeyError,
                IndexError,
                ValueError,
                ValidationError,
                json.JSONDecodeError,
            ) as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                last_retry_class, last_status_code = (
                    _classify_retryable_exception(exc)
                )
                if response is not None:
                    last_response_diagnostics = (
                        _provider_response_diagnostics(response, body=body)
                    )
                    last_retry_after_seconds = _retry_after_seconds(response)
                if attempt == retries:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))
        raise RetryableProviderError(
            "Anthropic API call failed after strict retries: "
            + " | ".join(errors),
            last_retry_class=last_retry_class,
            last_status_code=last_status_code,
            attempts_tried=attempts_tried,
            request_hash=request_hash,
            usage=last_usage,
            response_diagnostics=last_response_diagnostics,
            retry_after_seconds=last_retry_after_seconds,
        )


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
        "structured_output_audit": (
            result.structured_output_audit if result else None
        ),
        "validated": parsed.model_dump(mode="json") if parsed else None,
        "usage": result.usage if result else None,
        "latency_ms": result.latency_ms if result else None,
        "error": error,
    }
