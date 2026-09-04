from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from metacom_pm.api import Endpoint, chat_request_payload
from metacom_pm.io import canonical_json, sha256_file
from metacom_pm.v1_5_strict_judge_bakeoff import (
    StrictBakeoffPairwiseOutput,
    bakeoff_contract_record,
    build_call_plan,
    build_cost_estimate,
    endpoint_from_record,
    strict_bakeoff_carry_forward,
    validate_endpoint_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads(
        (
            ROOT / "configs/pm_v1_5_strict_judge_bakeoff_v1.json"
        ).read_text(encoding="utf-8")
    )


def _ordered_rows() -> list[dict]:
    rows = []
    for index in range(12):
        for order in (0, 1):
            rows.append(
                {
                    "pair_id": f"pair_{index:02d}",
                    "state_id": f"state_{index:02d}",
                    "order_variant": order,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Judge two replies.",
                        },
                        {
                            "role": "user",
                            "content": "Return exactly one JSON object.",
                        },
                    ],
                }
            )
    return rows


def test_frozen_candidate_models_prices_and_capabilities():
    record = _config()
    validate_endpoint_contract(record)
    assert set(record["candidates"]) == {
        "openai_gpt_5_mini",
        "anthropic_claude_haiku_4_5",
        "google_gemini_2_5_flash",
    }
    gpt = endpoint_from_record(record["candidates"]["openai_gpt_5_mini"])
    payload = chat_request_payload(
        gpt,
        [{"role": "user", "content": "Return JSON."}],
        temperature=0.0,
        max_tokens=700,
        seed=12091,
        response_schema=StrictBakeoffPairwiseOutput,
    )
    assert payload["model"] == "gpt-5-mini-2025-08-07"
    assert payload["max_completion_tokens"] == 700
    assert "max_tokens" not in payload
    assert "temperature" not in payload
    assert payload["reasoning_effort"] == "minimal"
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True

    claude = endpoint_from_record(
        record["candidates"]["anthropic_claude_haiku_4_5"]
    )
    payload = chat_request_payload(
        claude,
        [{"role": "user", "content": "Return JSON."}],
        temperature=0.0,
        max_tokens=700,
        seed=12091,
        response_schema=StrictBakeoffPairwiseOutput,
    )
    assert payload["tools"][0]["strict"] is True
    assert payload["tool_choice"]["name"] == "submit_structured_response"

    gemini = endpoint_from_record(
        record["candidates"]["google_gemini_2_5_flash"]
    )
    payload = chat_request_payload(
        gemini,
        [{"role": "user", "content": "Return JSON."}],
        temperature=0.0,
        max_tokens=700,
        seed=12091,
        response_schema=StrictBakeoffPairwiseOutput,
    )
    assert payload["generationConfig"]["responseMimeType"] == "application/json"
    assert isinstance(
        payload["generationConfig"]["responseJsonSchema"], dict
    )
    assert payload["generationConfig"]["thinkingConfig"] == {
        "thinkingBudget": 0
    }


def test_reasoning_controls_are_opt_in_and_transport_scoped():
    messages = [{"role": "user", "content": "Return JSON."}]
    legacy_openai = Endpoint(
        base_url="https://api.openai.com/v1",
        model="legacy",
        api_key_env="OPENAI_API_KEY",
        transport="openai_chat_completions",
    )
    payload = chat_request_payload(
        legacy_openai,
        messages,
        temperature=0.0,
        max_tokens=50,
        seed=1,
        response_schema=None,
    )
    assert "reasoning_effort" not in payload
    legacy_gemini = Endpoint(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        model="gemini-2.5-flash",
        api_key_env="GEMINI_API_KEY",
        transport="gemini_generate_content",
    )
    payload = chat_request_payload(
        legacy_gemini,
        messages,
        temperature=0.0,
        max_tokens=50,
        seed=1,
        response_schema=None,
    )
    assert "thinkingConfig" not in payload["generationConfig"]


@pytest.mark.parametrize(
    ("candidate", "field", "replacement"),
    [
        ("openai_gpt_5_mini", "model", "gpt-5-mini"),
        ("openai_gpt_5_mini", "input_usd_per_million_tokens", 0.1),
        (
            "openai_gpt_5_mini",
            "max_output_tokens_parameter",
            "max_tokens",
        ),
        (
            "openai_gpt_5_mini",
            "openai_reasoning_effort",
            "medium",
        ),
        (
            "openai_gpt_5_mini",
            "provider_max_output_tokens",
            700,
        ),
        (
            "anthropic_claude_haiku_4_5",
            "anthropic_strict_tool_use",
            False,
        ),
        (
            "google_gemini_2_5_flash",
            "transport",
            "openai_chat_completions",
        ),
        (
            "google_gemini_2_5_flash",
            "gemini_thinking_budget",
            None,
        ),
    ],
)
def test_candidate_or_price_drift_fails_closed(
    candidate: str, field: str, replacement: object
):
    record = _config()
    record["candidates"][candidate][field] = replacement
    with pytest.raises(RuntimeError, match="drifted"):
        validate_endpoint_contract(record)


def test_call_plan_is_same_24_ordered_items_for_all_three_candidates():
    plan = build_call_plan(
        ordered_pairwise_rows=_ordered_rows(),
        endpoint_contract=_config(),
    )
    assert len(plan) == 72
    by_candidate = {}
    for row in plan:
        by_candidate.setdefault(row["candidate_key"], []).append(
            (
                row["record_ids"]["pair_id"],
                row["record_ids"]["order_variant"],
                row["prompt_sha256"],
            )
        )
    assert {key: len(rows) for key, rows in by_candidate.items()} == {
        "anthropic_claude_haiku_4_5": 24,
        "google_gemini_2_5_flash": 24,
        "openai_gpt_5_mini": 24,
    }
    normalized = [
        [(pair, order, prompt) for pair, order, prompt in rows]
        for rows in by_candidate.values()
    ]
    assert normalized[0] == normalized[1] == normalized[2]
    assert len({row["physical_call_key"] for row in plan}) == 72
    assert all(
        row["request_parameters"]["input_token_bound_protocol"]
        == "complete_provider_payload_including_schema_x1.5_v1"
        for row in plan
    )
    assert all(
        len(row["request_parameters"]["provider_request_payload_sha256"])
        == 64
        for row in plan
    )
    max_tokens = {
        key: {
            row["max_output_tokens"]
            for row in plan
            if row["candidate_key"] == key
        }
        for key in by_candidate
    }
    assert max_tokens == {
        "anthropic_claude_haiku_4_5": {700},
        "google_gemini_2_5_flash": {700},
        "openai_gpt_5_mini": {1400},
    }
    assert {
        row["request_parameters"]["gemini_thinking_budget"]
        for row in plan
        if row["candidate_key"] == "google_gemini_2_5_flash"
    } == {0}
    assert {
        row["request_parameters"]["openai_reasoning_effort"]
        for row in plan
        if row["candidate_key"] == "openai_gpt_5_mini"
    } == {"minimal"}


def test_reason_length_is_not_provider_schema_gate_but_blank_is_rejected():
    schema_json = canonical_json(
        StrictBakeoffPairwiseOutput.model_json_schema()
    )
    assert "maxLength" not in schema_json
    assert "minLength" not in schema_json
    long_reason = "reason " * 500
    parsed = StrictBakeoffPairwiseOutput.model_validate(
        {
            "overall_preference": "A",
            "support_quality_preference": "A",
            "evidence_handling_preference": "tie",
            "safety_preference": "tie",
            "overall_reason": long_reason,
            "support_quality_reason": "supported",
            "evidence_handling_reason": "equivalent",
            "safety_reason": "equivalent",
        },
        strict=True,
    )
    assert parsed.overall_reason == long_reason
    with pytest.raises(ValueError, match="reason must not be blank"):
        StrictBakeoffPairwiseOutput.model_validate(
            {
                **parsed.model_dump(),
                "overall_reason": "   ",
            },
            strict=True,
        )


def test_contract_drift_changes_identity_and_never_creates_labels():
    base = bakeoff_contract_record(
        endpoint_contract=_config(),
        packet_files={"packet": "a"},
        human_anchor={"binding_sha256": "h"},
        gpt_anchor={"selected_results_sha256": "g"},
    )
    changed_packet = bakeoff_contract_record(
        endpoint_contract=_config(),
        packet_files={"packet": "b"},
        human_anchor={"binding_sha256": "h"},
        gpt_anchor={"selected_results_sha256": "g"},
    )
    changed_human = bakeoff_contract_record(
        endpoint_contract=_config(),
        packet_files={"packet": "a"},
        human_anchor={"binding_sha256": "other"},
        gpt_anchor={"selected_results_sha256": "g"},
    )
    changed_gpt = bakeoff_contract_record(
        endpoint_contract=_config(),
        packet_files={"packet": "a"},
        human_anchor={"binding_sha256": "h"},
        gpt_anchor={"selected_results_sha256": "other"},
    )
    assert len(
        {
            base["contract_sha256"],
            changed_packet["contract_sha256"],
            changed_human["contract_sha256"],
            changed_gpt["contract_sha256"],
        }
    ) == 4
    assert base["planned_new_logical_calls"] == 72
    assert base["training_labels_created"] is False
    assert base["decision_boundary"]["majority_vote_is_gold"] is False


def test_anthropic_strict_capability_changes_call_identity():
    record = _config()
    strict_plan = build_call_plan(
        ordered_pairwise_rows=_ordered_rows(),
        endpoint_contract=record,
    )
    changed = copy.deepcopy(record)
    changed["candidates"]["anthropic_claude_haiku_4_5"][
        "anthropic_strict_tool_use"
    ] = False
    # Bypass the frozen validator only to prove the identity layer includes
    # the capability.  The public builder correctly rejects this drift.
    with pytest.raises(RuntimeError, match="drifted"):
        build_call_plan(
            ordered_pairwise_rows=_ordered_rows(),
            endpoint_contract=changed,
        )
    strict_keys = {
        row["physical_call_key"]
        for row in strict_plan
        if row["candidate_key"] == "anthropic_claude_haiku_4_5"
    }
    assert len(strict_keys) == 24


def test_exact_carry_forward_only_reuses_unchanged_succeeded_calls(
    tmp_path: Path,
):
    plan = build_call_plan(
        ordered_pairwise_rows=_ordered_rows(),
        endpoint_contract=_config(),
    )
    row = next(
        item
        for item in plan
        if item["candidate_key"] == "anthropic_claude_haiku_4_5"
    )
    source = tmp_path / "source"
    source.mkdir()
    ledger_rows = [
        {
            "event": "STARTED",
            "call_key": row["physical_call_key"],
            "attempt_index": 1,
            "record_ids": row["record_ids"],
        },
        {
            "event": "SUCCEEDED",
            "call_key": row["physical_call_key"],
            "attempt_index": 1,
            "record_ids": row["record_ids"],
            "result": {"parsed": {"overall_preference": "A"}},
        },
    ]
    (source / "physical_attempt_ledger.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in ledger_rows),
        encoding="utf-8",
    )
    (source / "qualification_results.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    (source / "artifact_attestation.json").write_text(
        json.dumps(
            {
                "protocol": "pm-v1.5-strict-judge-bakeoff-attestation-v1",
                "stage": (
                    "longitudinal_train_only_strict_pairwise_"
                    "judge_bakeoff_v1"
                ),
                "physical_attempt_ledger_sha256": sha256_file(
                    source / "physical_attempt_ledger.jsonl"
                ),
                "qualification_results_sha256": sha256_file(
                    source / "qualification_results.jsonl"
                ),
                "training_labels_created": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    record, selected = strict_bakeoff_carry_forward(
        source_dir=source,
        plan=plan,
        root=tmp_path,
    )
    assert record["carried_logical_calls"] == 1
    assert record["carried_by_candidate"] == {
        "anthropic_claude_haiku_4_5": 1
    }
    assert selected == ledger_rows
    estimate = build_cost_estimate(
        plan=plan,
        contract={"contract_sha256": "c" * 64},
        code_manifest={},
        maximum_attempts=2,
        max_api_calls=200,
        max_estimated_usd=2.0,
        max_input_tokens_per_call=12000,
        carry_forward=record,
    )
    assert estimate["logical_calls"] == 72
    assert estimate["carried_forward_logical_calls"] == 1
    assert estimate["new_logical_calls"] == 71
    assert estimate["maximum_physical_attempts"] == 142
