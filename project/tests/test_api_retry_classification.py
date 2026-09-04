from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from metacom_pm.api import (
    Endpoint,
    OpenAICompatibleClient,
    ProviderRequestError,
    RetryableProviderError,
)
from metacom_pm.pm_v2_contracts import StrictModel


class _TinyStructuredOutput(StrictModel):
    verdict: str


def _client(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> OpenAICompatibleClient:
    monkeypatch.setenv("TEST_RETRY_API_KEY", "test-only")
    endpoint = Endpoint(
        base_url="https://provider.example",
        model="test-model",
        api_key_env="TEST_RETRY_API_KEY",
        family="test-family",
    )
    client = OpenAICompatibleClient(endpoint)
    client._client.close()
    client._client = httpx.Client(
        base_url=endpoint.base_url,
        transport=httpx.MockTransport(handler),
    )
    return client


def test_real_http_503_is_classified_and_carries_response_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(
        monkeypatch,
        lambda _request: httpx.Response(
            503,
            headers={"nv-request-id": "nv-503-test"},
            json={"error": {"message": "temporarily unavailable"}},
        ),
    )
    try:
        with pytest.raises(RetryableProviderError) as exc_info:
            client.chat([{"role": "user", "content": "test"}], retries=1)
    finally:
        client.close()
    failure = exc_info.value
    assert failure.last_retry_class == "http_5xx"
    assert failure.last_status_code == 503
    assert failure.attempts_tried == 1
    assert len(failure.request_hash or "") == 64
    assert failure.response_diagnostics is not None
    assert failure.response_diagnostics["status_code"] == 503
    assert failure.response_diagnostics["provider_request_id"] == "nv-503-test"
    assert len(failure.response_diagnostics["response_body_sha256"]) == 64


def test_real_http_429_preserves_retry_after_for_the_outer_ledger_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(
        monkeypatch,
        lambda _request: httpx.Response(
            429,
            headers={"retry-after": "17"},
            json={"error": {"message": "slow down"}},
        ),
    )
    try:
        with pytest.raises(RetryableProviderError) as exc_info:
            client.chat([{"role": "user", "content": "test"}], retries=1)
    finally:
        client.close()
    failure = exc_info.value
    assert failure.last_retry_class == "rate_limited_429"
    assert failure.last_status_code == 429
    assert failure.retry_after_seconds == 17.0
    assert failure.response_diagnostics["retry_after_seconds"] == 17.0


def test_real_http_408_is_not_misreported_as_a_5xx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(
        monkeypatch,
        lambda _request: httpx.Response(
            408, json={"error": {"message": "request timeout"}}
        ),
    )
    try:
        with pytest.raises(RetryableProviderError) as exc_info:
            client.chat([{"role": "user", "content": "test"}], retries=1)
    finally:
        client.close()
    assert exc_info.value.last_retry_class == "request_timeout_408"
    assert exc_info.value.last_status_code == 408


def test_2xx_missing_content_preserves_usage_and_safe_body_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(
        monkeypatch,
        lambda _request: httpx.Response(
            200,
            json={
                "id": "paid-but-malformed",
                "choices": [{"message": {"role": "assistant"}}],
                "usage": {
                    "prompt_tokens": 21,
                    "completion_tokens": 3,
                    "total_tokens": 24,
                },
            },
        ),
    )
    try:
        with pytest.raises(RetryableProviderError) as exc_info:
            client.chat([{"role": "user", "content": "test"}], retries=1)
    finally:
        client.close()
    failure = exc_info.value
    assert failure.last_retry_class == "missing_field"
    assert failure.last_status_code is None
    assert failure.usage == {
        "prompt_tokens": 21,
        "completion_tokens": 3,
        "total_tokens": 24,
    }
    assert failure.response_diagnostics["status_code"] == 200
    assert failure.response_diagnostics["first_message_keys"] == ["role"]
    assert "paid-but-malformed" not in str(failure.response_diagnostics)


def test_empty_content_with_finish_reason_length_is_output_token_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A response cut off by max_tokens (e.g. an unrequested reasoning budget
    consuming the whole allowance before any content token, the real pattern
    observed on deepseek-v4-flash) is a contract/config mismatch, not random
    provider noise -- it must be classified distinctly from missing_field so
    the caller does not burn a bounded retry budget re-sending the identical
    request into the identical wall."""

    client = _client(
        monkeypatch,
        lambda _request: httpx.Response(
            200,
            json={
                "id": "truncated-by-length",
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "x" * 500,
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 859,
                    "completion_tokens": 600,
                    "total_tokens": 1459,
                    "completion_tokens_details": {"reasoning_tokens": 600},
                },
            },
        ),
    )
    try:
        with pytest.raises(RetryableProviderError) as exc_info:
            client.chat([{"role": "user", "content": "test"}], retries=1)
    finally:
        client.close()
    failure = exc_info.value
    assert failure.last_retry_class == "output_token_limit"
    assert "finish_reason=length" in str(failure)
    diagnostics = failure.response_diagnostics
    assert diagnostics["first_choice_finish_reason"] == "length"
    assert diagnostics["first_message_content_length"] == 0
    assert diagnostics["first_message_reasoning_content_length"] == 500
    assert diagnostics["completion_reasoning_tokens"] == 600


def test_truncated_json_with_finish_reason_length_is_output_token_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-empty but truncated-mid-string content under finish_reason=length
    must not be misclassified as ordinary provider_output_format noise
    either -- same underlying cause (ran out of max_tokens) as the empty-
    content case above, so the same distinct, non-blindly-retried class."""

    client = _client(
        monkeypatch,
        lambda _request: httpx.Response(
            200,
            json={
                "id": "truncated-json-by-length",
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": '{"emotional_support": 4, "rationale": "cut off mid',
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 938,
                    "completion_tokens": 700,
                    "total_tokens": 1638,
                },
            },
        ),
    )
    try:
        with pytest.raises(RetryableProviderError) as exc_info:
            client.chat(
                [{"role": "user", "content": "test"}],
                response_schema=_TinyStructuredOutput,
                retries=1,
            )
    finally:
        client.close()
    failure = exc_info.value
    assert failure.last_retry_class == "output_token_limit"
    assert "finish_reason=length" in str(failure)


def test_empty_content_without_length_finish_reason_stays_missing_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Genuinely unexplained empty content (no length signal at all) keeps
    the existing missing_field classification and its existing bounded
    retry treatment -- this fix must not reclassify every empty response."""

    client = _client(
        monkeypatch,
        lambda _request: httpx.Response(
            200,
            json={
                "id": "empty-no-length-signal",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": ""},
                    }
                ],
                "usage": {
                    "prompt_tokens": 21,
                    "completion_tokens": 0,
                    "total_tokens": 21,
                },
            },
        ),
    )
    try:
        with pytest.raises(RetryableProviderError) as exc_info:
            client.chat([{"role": "user", "content": "test"}], retries=1)
    finally:
        client.close()
    assert exc_info.value.last_retry_class == "missing_field"


def test_schema_mode_audits_bounded_prefix_without_changing_json_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(
        monkeypatch,
        lambda _request: httpx.Response(
            200,
            json={
                "id": "paid-prefixed-json",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": 'We{"verdict":"supported"}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 21,
                    "completion_tokens": 5,
                    "total_tokens": 26,
                },
            },
        ),
    )
    try:
        call, parsed = client.chat(
            [{"role": "user", "content": "test"}],
            response_schema=_TinyStructuredOutput,
            retries=1,
        )
    finally:
        client.close()
    assert parsed is not None and parsed.verdict == "supported"
    assert call.usage == {
        "prompt_tokens": 21,
        "completion_tokens": 5,
        "total_tokens": 26,
    }
    assert call.structured_output_audit == {
        "initially_valid_json": False,
        "normalization_kind": "single_bounded_json_object",
        "raw_text_sha256": call.structured_output_audit["raw_text_sha256"],
        "normalized_json_sha256": call.structured_output_audit[
            "normalized_json_sha256"
        ],
        "discarded_prefix_chars": 2,
        "discarded_suffix_chars": 0,
    }


def test_real_http_403_is_deterministic_and_carries_request_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(
        monkeypatch,
        lambda _request: httpx.Response(
            403,
            json={"error": {"message": "forbidden", "code": "bad_key"}},
        ),
    )
    try:
        with pytest.raises(ProviderRequestError) as exc_info:
            client.chat([{"role": "user", "content": "test"}], retries=3)
    finally:
        client.close()
    failure = exc_info.value
    assert failure.status_code == 403
    assert len(failure.request_hash or "") == 64
    assert failure.response_diagnostics["status_code"] == 403


def test_transport_timeout_is_classified_without_fabricating_http_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("injected timeout", request=request)

    client = _client(monkeypatch, timeout)
    try:
        with pytest.raises(RetryableProviderError) as exc_info:
            client.chat([{"role": "user", "content": "test"}], retries=1)
    finally:
        client.close()
    failure = exc_info.value
    assert failure.last_retry_class == "network_timeout"
    assert failure.last_status_code is None
    assert failure.response_diagnostics is None
    assert len(failure.request_hash or "") == 64
