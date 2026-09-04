from __future__ import annotations

import json
from pathlib import Path

from metacom_pm.io import iter_jsonl, read_json, sha256_file
from metacom_pm.v1_5_external_qa_adapter_v2 import (
    endpoint_compatibility_errors,
    memory_payload_from_v1_messages,
    qa_messages_v2,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v2_adapter_preserves_payload_but_removes_metaprompt_example() -> None:
    source = [
        {"role": "system", "content": "old task with an example"},
        {
            "role": "user",
            "content": "Question: When?\nRelevant Memory:\n1. [date] fact",
        },
    ]
    payload = memory_payload_from_v1_messages(source)
    assert payload == "1. [date] fact"
    messages = qa_messages_v2(question="When?", memory_payload=payload)
    blob = str(messages)
    assert "old task with an example" not in blob
    assert "1. [date] fact" in blob
    assert "When?" in blob


def test_compatibility_gate_checks_shape_not_correctness() -> None:
    assert endpoint_compatibility_errors("Answer: any unsupported text", finish_reason="complete") == []
    errors = endpoint_compatibility_errors(
        "## Task Description Relevant Memory: copied", finish_reason="length"
    )
    assert "NON_COMPLETE_FINISH:length" in errors
    assert "MISSING_ANSWER_PREFIX" in errors
    assert any(item.startswith("PROMPT_ECHO:") for item in errors)


def test_e4_v2_changes_only_qa_adapter_and_preserves_frozen_routes() -> None:
    old = list(
        iter_jsonl(
            ROOT
            / "outputs/pm_v1_5_v5_2_external_e3_retrieval_v1"
            / "qa_final_call_plan_private.jsonl"
        )
    )
    new = list(
        iter_jsonl(
            ROOT
            / "outputs/pm_v1_5_v5_2_external_e4_plan_v2_qa_adapter"
            / "qa_call_plan_v2_private.jsonl"
        )
    )
    assert len(old) == len(new) == 2090
    route_fields = {
        "call_id",
        "question_id",
        "condition",
        "selected_session_ids",
        "selected_session_scores",
        "selected_typed_components",
        "seed",
        "output_token_cap",
    }
    for before, after in zip(old, new, strict=True):
        assert {key: before[key] for key in route_fields} == {
            key: after[key] for key in route_fields
        }
        assert after["pm_or_route_changed"] is False
        assert after["gold_or_outcome_read"] is False
        assert "## Task Description" not in json.dumps(after["messages"])


def test_e4_v2_seal_reuses_response_plans_and_requires_passed_pilot() -> None:
    old_seal = read_json(
        ROOT / "outputs/pm_v1_5_v5_2_external_e4_plan_v1/execution_seal.json"
    )
    new_seal = read_json(
        ROOT
        / "outputs/pm_v1_5_v5_2_external_e4_plan_v2_qa_adapter"
        / "execution_seal.json"
    )
    for relative in (
        "outputs/pm_v1_5_v5_2_external_e2_plan_v1/response_core_call_plan_private.jsonl",
        "outputs/pm_v1_5_v5_2_external_e3_retrieval_v1/response_raw_call_plan_private.jsonl",
    ):
        assert new_seal["source_sha256"][relative] == old_seal["source_sha256"][relative]
        assert sha256_file(ROOT / relative) == old_seal["source_sha256"][relative]
    assert read_json(
        ROOT
        / "outputs/pm_v1_5_v5_2_external_qa_adapter_pilot_v1_execution"
        / "summary.json"
    )["status"] == "PASS_READY_TO_VERSION_E4_QA_ADAPTER"
    assert new_seal["pm_or_route_changed"] is False


def test_e4_continuation_carries_exact_successes_and_only_paces_transport() -> None:
    base = list(
        iter_jsonl(
            ROOT
            / "outputs/pm_v1_5_v5_2_external_e4_plan_v2_qa_adapter"
            / "physical_call_plan_private.jsonl"
        )
    )
    plan = ROOT / "outputs/pm_v1_5_v5_2_external_e4_continuation_plan_v1"
    carried = list(iter_jsonl(plan / "carried_outcomes_private.jsonl"))
    remaining = list(iter_jsonl(plan / "remaining_physical_call_plan_private.jsonl"))
    assert len(base) == 4218
    assert len(carried) == 38
    assert len(remaining) == 4180
    carried_ids = {row["call_id"] for row in carried}
    remaining_ids = {row["call_id"] for row in remaining}
    assert not carried_ids & remaining_ids
    assert carried_ids | remaining_ids == {row["call_id"] for row in base}
    base_by_id = {row["call_id"]: row for row in base}
    for row in remaining:
        before = base_by_id[row["call_id"]]
        assert row["messages_sha256"] == before["messages_sha256"]
        assert row["request_parameters"] == before["request_parameters"]
        assert row["source_plan_relative_path"] == before["source_plan_relative_path"]
    cost = read_json(plan / "cost_estimate.json")
    assert cost["minimum_inter_call_seconds"] == 2.0
    assert cost["rate_limit_backoff_seconds"] == [60.0, 60.0]
    assert cost["provider_visible_messages_or_parameters_changed"] is False
    assert cost["transport_only_change"] is True
    assert cost["pm_or_route_changed"] is False
