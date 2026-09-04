from __future__ import annotations

from typing import Any

import httpx
import pytest
from pydantic import model_validator

from metacom_pm.api import (
    CallResult,
    Endpoint,
    OpenAICompatibleClient,
    ProviderRequestError,
    StructuredOutputValidationError,
    chat_request_payload,
    normalize_provider_finish_reason,
    openai_strict_json_schema,
    request_log,
)
from metacom_pm.pm_v2_data import GeneratedBundleDraft, GeneratedUserBundle
from metacom_pm.pm_v2_contracts import StrictModel
from metacom_pm.pm_v2_forced_swap import ForcedSwapJudgeOutput
from metacom_pm.pm_v2_judging import ResponseJudgeOutput, RiskJudgeOutput
from metacom_pm.synthetic_generation import GeneratedLongitudinalUser


PROVIDER_SCHEMAS = (
    GeneratedBundleDraft,
    GeneratedLongitudinalUser,
    ResponseJudgeOutput,
    RiskJudgeOutput,
    ForcedSwapJudgeOutput,
)


def _object_nodes(node: Any):
    if isinstance(node, list):
        for item in node:
            yield from _object_nodes(item)
        return
    if not isinstance(node, dict):
        return
    if node.get("type") == "object" or "properties" in node:
        yield node
    for value in node.values():
        yield from _object_nodes(value)


@pytest.mark.parametrize("schema_model", PROVIDER_SCHEMAS)
def test_every_provider_schema_is_openai_strict_compatible(schema_model) -> None:
    schema = openai_strict_json_schema(schema_model)
    for node in _object_nodes(schema):
        assert node["additionalProperties"] is False
        assert set(node["required"]) == set(node["properties"])


def test_generation_schema_requires_item_risk_labels_and_excludes_provenance() -> None:
    schema = openai_strict_json_schema(GeneratedUserBundle)
    assert "provenance" not in schema["properties"]
    memory_schema = schema["$defs"]["GeneratedMemory"]
    assert {
        "stale",
        "conflicts_with_current_state",
        "private_sensitivity",
    } <= set(memory_schema["required"])


def test_provider_generation_schema_has_exact_named_role_slots() -> None:
    schema = openai_strict_json_schema(GeneratedBundleDraft)
    assert set(schema["properties"]) == {
        "profile_summary",
        "stable_preferences",
        "boundaries",
        "context_only",
        "profile_needed",
        "summary_needed",
        "event_needed",
        "multi_source_needed",
        "memory_harmful",
        "strategy_helpful",
        "strategy_harmful",
        "ambiguous",
    }
    harmful_case = schema["$defs"]["GeneratedMemoryHarmfulCaseDraft"]
    assert {
        "profile_source",
        "summary_source",
        "event_source",
    } <= set(harmful_case["required"])


def test_non_strict_pydantic_schema_is_rejected_before_network() -> None:
    class OptionalOutput(StrictModel):
        value: int = 1

    with pytest.raises(ValueError, match="must list every property"):
        openai_strict_json_schema(OptionalOutput)


def test_schema_http_400_is_diagnostic_and_never_downgrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_OPENAI_KEY", "test-only")
    endpoint = Endpoint(
        base_url="https://api.openai.com",
        model="gpt-4o-mini",
        api_key_env="TEST_OPENAI_KEY",
    )
    client = OpenAICompatibleClient(endpoint)
    client._client.close()

    class RejectingTransport:
        def __init__(self) -> None:
            self.payloads: list[dict[str, Any]] = []

        def post(self, path: str, *, json: dict[str, Any]) -> httpx.Response:
            self.payloads.append(json)
            return httpx.Response(
                400,
                request=httpx.Request("POST", f"https://api.openai.com{path}"),
                json={
                    "error": {
                        "message": "Invalid schema for response_format",
                        "type": "invalid_request_error",
                        "param": "response_format",
                        "code": "invalid_json_schema",
                    }
                },
            )

        def close(self) -> None:
            pass

    transport = RejectingTransport()
    client._client = transport  # type: ignore[assignment]
    with pytest.raises(ProviderRequestError) as exc_info:
        client.chat(
            [{"role": "user", "content": "Return scores."}],
            response_schema=ResponseJudgeOutput,
            retries=3,
        )
    error = str(exc_info.value)
    assert "schema mode HTTP 400" in error
    assert "Invalid schema for response_format" in error
    assert "invalid_json_schema" in error
    assert len(transport.payloads) == 1
    assert "response_format" in transport.payloads[0]


def test_paid_structured_response_survives_local_semantic_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SemanticOutput(StrictModel):
        value: int

        @model_validator(mode="after")
        def reject_value(self):
            if self.value == 1:
                raise ValueError("injected semantic failure")
            return self

    monkeypatch.setenv("TEST_OPENAI_KEY", "test-only")
    endpoint = Endpoint(
        base_url="https://api.openai.com",
        model="gpt-4o-mini",
        api_key_env="TEST_OPENAI_KEY",
    )
    client = OpenAICompatibleClient(endpoint)
    client._client.close()

    class SuccessfulTransport:
        def post(self, path: str, *, json: dict[str, Any]) -> httpx.Response:
            return httpx.Response(
                200,
                request=httpx.Request("POST", f"https://api.openai.com{path}"),
                json={
                    "id": "paid-response-id",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": '{"value":1}'},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 4,
                        "total_tokens": 16,
                    },
                },
            )

        def close(self) -> None:
            pass

    client._client = SuccessfulTransport()  # type: ignore[assignment]
    with pytest.raises(StructuredOutputValidationError) as exc_info:
        client.chat(
            [{"role": "user", "content": "Return one value."}],
            response_schema=SemanticOutput,
            retries=1,
        )
    failure = exc_info.value
    assert failure.parsed_payload == {"value": 1}
    assert failure.call.raw_response["id"] == "paid-response-id"
    assert failure.call.usage == {
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "total_tokens": 16,
    }
    assert failure.call.provider_finish_reason == "stop"
    assert failure.call.normalized_finish_reason == "complete"
    assert failure.validation_errors[0]["loc"] == []


def test_chat_payload_binds_the_locally_validated_schema() -> None:
    endpoint = Endpoint(
        base_url="https://api.openai.com",
        model="gpt-4o-mini",
        api_key_env="IGNORED",
    )
    payload = chat_request_payload(
        endpoint,
        [{"role": "user", "content": "Generate."}],
        temperature=0.0,
        max_tokens=100,
        seed=7,
        response_schema=GeneratedBundleDraft,
    )
    assert payload["response_format"]["json_schema"]["schema"] == (
        openai_strict_json_schema(GeneratedBundleDraft)
    )


@pytest.mark.parametrize(
    ("raw_response", "provider_reason", "normalized_reason"),
    (
        ({"choices": [{"finish_reason": "stop"}]}, "stop", "complete"),
        ({"choices": [{"finish_reason": "length"}]}, "length", "length"),
        ({"choices": [{"finish_reason": "tool_calls"}]}, "tool_calls", "tool_call"),
        (
            {"choices": [{"finish_reason": "content_filter"}]},
            "content_filter",
            "content_filter",
        ),
        ({"stop_reason": "end_turn"}, "end_turn", "complete"),
        ({"stop_reason": "max_tokens"}, "max_tokens", "length"),
        ({"stop_reason": "tool_use"}, "tool_use", "tool_call"),
        (
            {"stop_reason": "future_provider_reason"},
            "future_provider_reason",
            "unknown",
        ),
        ({"choices": [{}]}, None, "unknown"),
        ({"content": []}, None, "unknown"),
    ),
)
def test_provider_finish_reasons_are_normalized_without_losing_raw_value(
    raw_response: dict[str, Any],
    provider_reason: str | None,
    normalized_reason: str,
) -> None:
    assert normalize_provider_finish_reason(raw_response) == (
        provider_reason,
        normalized_reason,
    )


def test_request_log_exposes_truncation_as_first_class_metadata() -> None:
    endpoint = Endpoint(
        base_url="https://api.openai.com",
        model="gpt-4o-mini",
        api_key_env="IGNORED",
    )
    result = CallResult(
        text="Incomplete response",
        raw_response={"choices": [{"finish_reason": "length"}]},
        usage={"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        latency_ms=10.0,
        request_hash="request-hash",
        provider_finish_reason="length",
        normalized_finish_reason="length",
    )

    log = request_log(
        stage="supporter_generation",
        endpoint=endpoint,
        messages=[{"role": "user", "content": "Generate."}],
        result=result,
        parsed=None,
        error=None,
        prompt_hash="prompt-hash",
        record_ids={"state_id": "state-001"},
    )

    assert log["provider_finish_reason"] == "length"
    assert log["normalized_finish_reason"] == "length"
    assert log["completion_truncated"] is True
