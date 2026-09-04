from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import pytest

from metacom_pm.api import Endpoint, chat_request_payload
from metacom_pm.attempt_ledger import PersistentAttemptLedger, physical_call_key
from metacom_pm.io import canonical_json, write_jsonl
from metacom_pm.v1_5_judge_qualification import (
    REGIME_ACTION_PAIRS,
    REGIME_SAMPLE_COUNTS,
    CombinedQualityRiskOutput,
    QualificationPairwiseOutput,
    build_combined_messages,
    qualification_contract_record,
    select_qualification_pairs,
)


ROOT = Path(__file__).resolve().parents[1]


def _runner_module():
    path = (
        ROOT
        / "scripts/v1_5/21r_run_low_budget_judge_qualification_v1_5.py"
    )
    spec = importlib.util.spec_from_file_location("judge_qualification_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _preparer_module():
    path = (
        ROOT
        / "scripts/v1_5/21q_prepare_low_budget_judge_qualification_v1_5.py"
    )
    spec = importlib.util.spec_from_file_location(
        "judge_qualification_preparer", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_prefix_reader_never_parses_later_split_rows(tmp_path: Path):
    path = tmp_path / "combined.jsonl"
    path.write_text('{"state_id":"train"}\nnot-json-from-later-split\n')
    rows, provenance = _preparer_module()._read_exact_prefix(
        path, expected_rows=1
    )
    assert rows == [{"state_id": "train"}]
    assert provenance["rows_consumed"] == 1
    with pytest.raises(json.JSONDecodeError):
        _preparer_module()._read_exact_prefix(path, expected_rows=2)


def test_qwen_enable_thinking_is_explicit_and_content_addressed():
    base = dict(
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        model="qwen3.7-max-2026-06-08",
        api_key_env="DASHSCOPE_API_KEY",
        family="qwen3_7_max",
        transport="openai_chat_completions",
        supports_strict_json_schema=False,
    )
    off = Endpoint(**base, enable_thinking=False)
    on = Endpoint(**base, enable_thinking=True)
    default = Endpoint(**base)
    messages = [{"role": "user", "content": "Return JSON."}]
    assert chat_request_payload(
        off,
        messages,
        temperature=0.0,
        max_tokens=50,
        seed=1,
        response_schema=QualificationPairwiseOutput,
    )["enable_thinking"] is False
    assert chat_request_payload(
        on,
        messages,
        temperature=0.0,
        max_tokens=50,
        seed=1,
        response_schema=QualificationPairwiseOutput,
    )["enable_thinking"] is True
    assert "enable_thinking" not in chat_request_payload(
        default,
        messages,
        temperature=0.0,
        max_tokens=50,
        seed=1,
        response_schema=QualificationPairwiseOutput,
    )
    request_parameters = {"max_tokens": 50}
    key_off = physical_call_key(
        stage="test",
        record_ids={"x": 1},
        prompt_sha256="a" * 64,
        endpoint=off,
        request_parameters=request_parameters,
    )
    key_on = physical_call_key(
        stage="test",
        record_ids={"x": 1},
        prompt_sha256="a" * 64,
        endpoint=on,
        request_parameters=request_parameters,
    )
    assert key_off != key_on


def test_v2_endpoints_freeze_qwen_plus_and_gpt_reasoning_payload():
    contract = json.loads(
        (ROOT / "configs/pm_v1_5_judge_qualification_v2.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["qwen"] == {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "family": "qwen3_7_plus",
        "input_usd_per_million_tokens": 0.4,
        "model": "qwen3.7-plus-2026-05-26",
        "output_usd_per_million_tokens": 1.6,
        "supports_strict_json_schema": False,
        "transport": "openai_chat_completions",
    }
    runner = _runner_module()
    endpoints, _ = runner._endpoint_records(contract)
    gpt = endpoints["gpt_anchor"]
    payload = chat_request_payload(
        gpt,
        [{"role": "user", "content": "Return JSON."}],
        temperature=0.0,
        max_tokens=700,
        seed=9051,
        response_schema=QualificationPairwiseOutput,
    )
    assert payload["model"] == "gpt-5.6-sol"
    assert payload["max_completion_tokens"] == 700
    assert "max_tokens" not in payload
    assert "temperature" not in payload
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True


def test_reasoning_request_capabilities_change_physical_identity():
    base = dict(
        base_url="https://api.openai.com/v1",
        model="gpt-5.6-sol",
        api_key_env="OPENAI_API_KEY",
        family="openai_gpt_5_6_sol",
        transport="openai_chat_completions",
    )
    legacy = Endpoint(**base)
    reasoning = Endpoint(
        **base,
        temperature_mode="omit",
        max_output_tokens_parameter="max_completion_tokens",
    )
    parameters = {"max_output_tokens": 700}
    legacy_key = physical_call_key(
        stage="qualification-test",
        record_ids={"pair": 1},
        prompt_sha256="a" * 64,
        endpoint=legacy,
        request_parameters=parameters,
    )
    reasoning_key = physical_call_key(
        stage="qualification-test",
        record_ids={"pair": 1},
        prompt_sha256="a" * 64,
        endpoint=reasoning,
        request_parameters=parameters,
    )
    assert legacy_key != reasoning_key
    different_snapshot_key = physical_call_key(
        stage="qualification-test",
        record_ids={"pair": 1},
        prompt_sha256="a" * 64,
        endpoint=Endpoint(
            **{**base, "model": "gpt-5.6-sol-different-snapshot"},
            temperature_mode="omit",
            max_output_tokens_parameter="max_completion_tokens",
        ),
        request_parameters=parameters,
    )
    assert reasoning_key != different_snapshot_key
    legacy_payload = chat_request_payload(
        legacy,
        [{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=20,
        seed=1,
        response_schema=None,
    )
    assert legacy_payload["temperature"] == 0.0
    assert legacy_payload["max_tokens"] == 20
    assert "max_completion_tokens" not in legacy_payload


def _fixture_rows():
    states = []
    evaluators = []
    outcomes = []
    users = [f"u{index:02d}" for index in range(24)]
    regimes = list(REGIME_ACTION_PAIRS)
    for user_index, user_id in enumerate(users):
        for regime_index, regime in enumerate(regimes):
            state_id = f"state_{user_index:02d}_{regime_index:02d}"
            states.append(
                {
                    "state_id": state_id,
                    "user_id": user_id,
                    "split": "train",
                    "current_user_text": f"current {state_id}",
                    "current_session_history": [
                        {"role": "user", "content": "history"}
                    ],
                    "current_session_summary": "summary",
                    "allowed_actions": sorted(
                        {action for pair in REGIME_ACTION_PAIRS.values() for action in pair[:2]}
                    ),
                }
            )
            evaluators.append(
                {
                    "state_id": state_id,
                    "regime": regime,
                    "authorized_user_context": "authorized",
                }
            )
            for action in states[-1]["allowed_actions"]:
                outcomes.append(
                    {
                        "state_id": state_id,
                        "action_id": action,
                        "response": f"response {state_id} {action}",
                        "candidate_memory_view": [],
                        "candidate_strategy_view": [],
                    }
                )
    return states, evaluators, outcomes


def test_selection_is_train_only_unique_and_regime_balanced():
    states, evaluators, outcomes = _fixture_rows()
    pairs = select_qualification_pairs(
        state_rows=states,
        evaluator_rows=evaluators,
        outcome_rows=outcomes,
    )
    assert len(pairs) == 12
    assert len({row["user_id"] for row in pairs}) == 12
    assert {
        regime: sum(row["regime"] == regime for row in pairs)
        for regime in REGIME_SAMPLE_COUNTS
    } == REGIME_SAMPLE_COUNTS
    states[0]["split"] = "internal_test"
    selected_after = select_qualification_pairs(
        state_rows=states,
        evaluator_rows=evaluators,
        outcome_rows=outcomes,
    )
    assert states[0]["state_id"] not in {
        row["state_id"] for row in selected_after
    }


def test_contract_freezes_qualification_not_training_labels():
    record = qualification_contract_record(
        source_lineage={"x": "y"},
        endpoint_contract={"protocol": "test"},
        human_anchor={"protocol": "human-test"},
    )
    assert record["planned_logical_calls"] == 91
    assert record["training_labels_created"] is False
    assert record["decision_boundary"]["majority_vote_is_gold"] is False
    assert (
        record["pre_outcome_qualification_thresholds"][
            "human_anchor_is_required_before_promoting_a_bulk_labeler"
        ]
        is True
    )
    assert record["structured_output_instruction"] == {
        "protocol": "explicit-json-literal-in-every-model-prompt-v1",
        "required_literal_case_insensitive": "json",
        "applies_to_qwen_loose_json_object_mode": True,
        "applies_to_gpt_strict_json_schema_mode": True,
        "purpose": "provider compatibility and equivalent output formatting only",
        "loose_json_schema_bridge": {
            "protocol": "provider-visible-full-pydantic-json-schema-v1",
            "complete_schema_is_injected_for_every_loose_endpoint": True,
            "field_names_required_set_types_enums_and_lengths_visible": True,
            "schema_serialization": "canonical_json",
            "strict_schema_endpoints_receive_constraints_out_of_band": True,
        },
    }
    assert record["pre_outcome_qualification_thresholds"][
        "loose_schema_compatibility_pilot"
    ] == {
        "required_logical_calls": 5,
        "required_valid_outputs": 5,
        "maximum_schema_validation_failures": 0,
        "transport_retries_do_not_change_compatibility": True,
        "creates_scientific_qualification_verdict": False,
    }


def test_combined_prompt_explicitly_satisfies_dashscope_json_object_contract():
    messages = build_combined_messages(
        {
            "visible_state": {
                "current_user_text": "I feel stuck.",
                "recent_dialogue": [],
                "current_session_summary": "",
            },
            "authorized_user_context": "No additional facts.",
            "candidate_b": {
                "selected_context": {"memory": [], "strategy": []},
                "response": "That sounds difficult.",
            },
        }
    )
    assert "json" in canonical_json(messages).casefold()


def test_runner_plan_has_exact_matrix_and_retry_aware_budget():
    runner = _runner_module()
    endpoint_record = {
        "qwen": {
            "base_url": "https://qwen.invalid/v1",
            "model": "qwen",
            "api_key_env": "QWEN_KEY",
            "family": "qwen",
            "transport": "openai_chat_completions",
            "supports_strict_json_schema": False,
            "input_usd_per_million_tokens": 1.0,
            "output_usd_per_million_tokens": 1.0,
        },
        "gpt_anchor": {
            "base_url": "https://gpt.invalid/v1",
            "model": "gpt",
            "api_key_env": "OPENAI_API_KEY",
            "family": "gpt",
            "transport": "openai_chat_completions",
            "supports_strict_json_schema": True,
            "input_usd_per_million_tokens": 2.0,
            "output_usd_per_million_tokens": 3.0,
        },
    }
    endpoints, prices = runner._endpoint_records(endpoint_record)
    pairwise_rows = []
    for pair_index in range(12):
        for order in (0, 1):
            pairwise_rows.append(
                {
                    "pair_id": f"p{pair_index}",
                    "state_id": f"s{pair_index}",
                    "order_variant": order,
                    "messages": [
                        {"role": "user", "content": "compare; return JSON"}
                    ],
                }
            )
    equivalence_rows = [
        {
            "equivalence_id": f"e{index}",
            "state_id": f"s{index}",
            "regime": f"r{index}",
            "combined_messages": [
                {"role": "user", "content": "combined; return JSON"}
            ],
            "quality_messages": [
                {"role": "user", "content": "quality; return JSON"}
            ],
            "risk_messages": [
                {"role": "user", "content": "risk; return JSON"}
            ],
        }
        for index in range(9)
    ]
    plan = runner.build_call_plan(
        pairwise_rows=pairwise_rows,
        equivalence_rows=equivalence_rows,
        gpt_anchor_pair_ids={f"p{index}" for index in range(8)},
        endpoints=endpoints,
        prices=prices,
        seed=1,
    )
    assert len(plan) == 91
    assert sum(
        row["condition"] == "qwen_nonthinking_pairwise" for row in plan
    ) == 24
    assert sum(
        row["condition"] == "qwen_thinking_pairwise" for row in plan
    ) == 24
    assert sum(
        row["condition"] == "gpt_high_quality_pairwise_anchor" for row in plan
    ) == 16
    assert len(
        {
            row["physical_call_key"]
            for row in plan
            if row["condition"].startswith("qwen_")
        }
    ) == 75
    for row in plan:
        visible_prompt = canonical_json(row["messages"])
        visible_content = row["messages"][-1]["content"]
        if row["endpoint_key"].startswith("qwen_"):
            assert "LOOSE JSON SCHEMA CONTRACT" in visible_prompt
            schema = runner.SCHEMAS[row["schema_kind"]].model_json_schema()
            assert canonical_json(schema) in visible_content
            for required_field in schema["required"]:
                assert required_field in visible_prompt
        else:
            assert "LOOSE JSON SCHEMA CONTRACT" not in visible_prompt
        if (
            row["endpoint_key"].startswith("qwen_")
            and row["schema_kind"] == "pairwise"
        ):
            assert '"maxLength":400' in visible_content
            assert '"enum":["A","B","tie","insufficient"]' in visible_content
            assert "support_quality_reason" in visible_prompt
            assert "evidence_handling_reason" in visible_prompt
            assert "safety_reason" in visible_prompt
        if row["schema_kind"] == "combined":
            assert '"maxLength":500' in visible_content
    schemas = {row["schema_kind"] for row in plan}
    assert schemas == {"pairwise", "combined", "quality", "risk"}
    by_equivalence: dict[str, list[dict]] = {}
    for row in plan:
        equivalence_id = row["record_ids"].get("equivalence_id")
        if equivalence_id:
            by_equivalence.setdefault(equivalence_id, []).append(row)
    assert len(by_equivalence) == 9
    for rows in by_equivalence.values():
        assert {row["schema_kind"] for row in rows} == {
            "combined",
            "quality",
            "risk",
        }
        assert len({row["record_ids"]["state_id"] for row in rows}) == 1

    repriced = {
        endpoint_key: dict(price)
        for endpoint_key, price in prices.items()
    }
    repriced["qwen_nonthinking"]["input"] += 0.01
    repriced_plan = runner.build_call_plan(
        pairwise_rows=pairwise_rows,
        equivalence_rows=equivalence_rows,
        gpt_anchor_pair_ids={f"p{index}" for index in range(8)},
        endpoints=endpoints,
        prices=repriced,
        seed=1,
    )
    assert repriced_plan != plan
    assert sum(
        float(row["maximum_single_attempt_cost_usd"])
        for row in repriced_plan
    ) != sum(
        float(row["maximum_single_attempt_cost_usd"]) for row in plan
    )


def test_qwen_loose_json_plan_adds_provider_visible_schema_bridge():
    runner = _runner_module()
    endpoint_record = {
        "qwen": {
            "base_url": "https://qwen.invalid/v1",
            "model": "qwen",
            "api_key_env": "QWEN_KEY",
            "family": "qwen",
            "transport": "openai_chat_completions",
            "supports_strict_json_schema": False,
            "input_usd_per_million_tokens": 1.0,
            "output_usd_per_million_tokens": 1.0,
        },
        "gpt_anchor": {
            "base_url": "https://gpt.invalid/v1",
            "model": "gpt",
            "api_key_env": "OPENAI_API_KEY",
            "family": "gpt",
            "transport": "openai_chat_completions",
            "supports_strict_json_schema": True,
            "input_usd_per_million_tokens": 2.0,
            "output_usd_per_million_tokens": 3.0,
        },
    }
    endpoints, prices = runner._endpoint_records(endpoint_record)
    plan: list[dict] = []
    runner._add_call(
        plan,
        condition="qwen_nonthinking_combined",
        endpoint_key="qwen_nonthinking",
        endpoint=endpoints["qwen_nonthinking"],
        price=prices["qwen_nonthinking"],
        schema_kind="combined",
        messages=[{"role": "user", "content": "Return the fields."}],
        record_ids={"state_id": "s"},
        max_output_tokens=100,
        seed=1,
        input_token_safety_factor=1.25,
    )
    visible_prompt = canonical_json(plan[0]["messages"])
    visible_content = plan[0]["messages"][-1]["content"]
    assert "JSON" in visible_prompt
    assert "quality_rationale" in visible_prompt
    assert "risk_rationale" in visible_prompt
    assert '"maxLength":500' in visible_content
    assert '"required":' in visible_content


def test_loose_schema_compatibility_pilot_covers_every_qwen_shape():
    runner = _runner_module()
    plan = [
        {
            "physical_call_key": f"k{index}",
            "endpoint_key": endpoint_key,
            "schema_kind": schema_kind,
        }
        for index, (endpoint_key, schema_kind) in enumerate(
            (
                ("qwen_nonthinking", "pairwise"),
                ("qwen_thinking", "pairwise"),
                ("qwen_nonthinking", "combined"),
                ("qwen_nonthinking", "quality"),
                ("qwen_nonthinking", "risk"),
                ("gpt_anchor", "pairwise"),
            )
        )
    ]
    selected = runner.select_loose_schema_compatibility_pilot(plan)
    assert [
        (row["endpoint_key"], row["schema_kind"]) for row in selected
    ] == [
        ("qwen_nonthinking", "pairwise"),
        ("qwen_thinking", "pairwise"),
        ("qwen_nonthinking", "combined"),
        ("qwen_nonthinking", "quality"),
        ("qwen_nonthinking", "risk"),
    ]


def test_runner_isolates_schema_failure_but_keeps_matrix_nonreportable():
    source = inspect.getsource(_runner_module().main)
    assert "except StructuredOutputValidationError" in source
    assert "isolated_schema_failures.append(call_key)" in source
    assert "INCOMPLETE_LOOSE_SCHEMA_COMPATIBILITY_PILOT_NO_VERDICT" in source
    assert '"qualification matrix is incomplete"' in source


def test_delta_carry_forward_reuses_only_exact_real_success(
    tmp_path: Path,
):
    runner = _runner_module()
    source = tmp_path / "source"
    source.mkdir()
    endpoint = Endpoint(
        base_url="https://gpt.invalid/v1",
        model="gpt",
        api_key_env="OPENAI_API_KEY",
        family="gpt",
        transport="openai_chat_completions",
        supports_strict_json_schema=True,
    )
    plan: list[dict] = []
    runner._add_call(
        plan,
        condition="gpt_high_quality_pairwise_anchor",
        endpoint_key="gpt_anchor",
        endpoint=endpoint,
        price={"input": 1.0, "output": 1.0},
        schema_kind="pairwise",
        messages=[{"role": "user", "content": "Return JSON."}],
        record_ids={"pair_id": "p"},
        max_output_tokens=100,
        seed=1,
        input_token_safety_factor=1.25,
    )
    write_jsonl(source / "call_plan.jsonl", plan)
    call_key = plan[0]["physical_call_key"]
    ledger = PersistentAttemptLedger(
        source / "physical_attempt_ledger.jsonl",
        stage=runner.STAGE,
        expected_calls={call_key: 2},
        maximum_total_attempts=2,
    )
    reservation = ledger.reserve(
        call_key,
        record_ids={"pair_id": "p"},
        prompt_sha256=plan[0]["prompt_sha256"],
    )
    ledger.finish(
        reservation,
        succeeded=True,
        request_hash="r" * 64,
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        error=None,
        result={
            "parsed": {
                "overall_preference": "A",
                "support_quality_preference": "A",
                "evidence_handling_preference": "A",
                "safety_preference": "A",
                "overall_reason": "x",
                "support_quality_reason": "x",
                "evidence_handling_reason": "x",
                "safety_reason": "x",
            }
        },
    )
    carried = runner._load_delta_carry_forward(
        source_dir=source,
        plan=plan,
        maximum_attempts=2,
    )
    assert carried["carried_call_keys"] == {call_key}
    assert carried["source_ledger_sha256"]

    changed: list[dict] = []
    runner._add_call(
        changed,
        condition="gpt_high_quality_pairwise_anchor",
        endpoint_key="gpt_anchor",
        endpoint=endpoint,
        price={"input": 1.0, "output": 1.0},
        schema_kind="pairwise",
        messages=[{"role": "user", "content": "Return different JSON."}],
        record_ids={"pair_id": "p"},
        max_output_tokens=100,
        seed=1,
        input_token_safety_factor=1.25,
    )
    not_carried = runner._load_delta_carry_forward(
        source_dir=source,
        plan=changed,
        maximum_attempts=2,
    )
    assert not not_carried["carried_call_keys"]


def test_combined_schema_keeps_all_thirteen_dimensions():
    fields = set(CombinedQualityRiskOutput.model_fields)
    assert len(fields - {"quality_rationale", "risk_rationale"}) == 13
