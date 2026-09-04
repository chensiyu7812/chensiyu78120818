from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from metacom_pm.api import CallResult, Endpoint, openai_strict_json_schema
from metacom_pm.io import canonical_json, sha256_text, write_json, write_jsonl
from metacom_pm.pm_v2_contracts import CompositeSpec
from metacom_pm.pm_v2_external_eval import expected_external_units
from metacom_pm.v1_5_external_batched import (
    BatchedResponseJudgment,
    PROTOCOL,
    RISK_AUDIT_PROTOCOL,
    balanced_candidate_order,
    batched_prompt_contract_hash,
    build_v1_5_batched_evaluation_plan,
    select_stratified_units,
)


CONDITIONS = [
    "pm_v2",
    "pm_v2_cost_matched_fixed",
    "pm_v2_me_r0_fixed",
    "no_memory_r0",
    "best_fixed",
    "session_rag_rs",
    "full_history_rs",
]


def _scores():
    return [
        {
            "candidate_id": f"C{index}",
            "emotional_support": 3.0,
            "personalization": 3.0,
            "memory_appropriateness": 3.0,
            "factual_grounding": 3.0,
            "temporal_consistency": 3.0,
            "non_intrusiveness": 3.0,
            "rationale": "bounded reason",
        }
        for index in range(1, 8)
    ]


def test_batched_schema_is_strict_and_requires_exactly_c1_through_c7():
    openai_strict_json_schema(BatchedResponseJudgment)
    assert len(BatchedResponseJudgment.model_validate({"candidates": _scores()}).candidates) == 7
    duplicate = _scores()
    duplicate[-1]["candidate_id"] = "C1"
    with pytest.raises(ValidationError, match="exactly"):
        BatchedResponseJudgment.model_validate({"candidates": duplicate})
    with pytest.raises(ValidationError):
        BatchedResponseJudgment.model_validate(
            {"candidates": [{**row, "overall": 3.0} for row in _scores()]}
        )


def test_cyclic_candidate_order_balances_every_condition_over_192_units():
    counts = {condition: Counter() for condition in CONDITIONS}
    for ordinal in range(192):
        order = balanced_candidate_order(CONDITIONS, unit_ordinal=ordinal, order_variant=0)
        for position, condition in enumerate(order, 1):
            counts[condition][position] += 1
    assert all(max(values.values()) - min(values.values()) <= 1 for values in counts.values())
    reversed_order = balanced_candidate_order(CONDITIONS, unit_ordinal=0, order_variant=1)
    assert reversed_order != balanced_candidate_order(CONDITIONS, unit_ordinal=0, order_variant=0)


def test_stratified_selection_covers_every_user_and_both_turns():
    units = [
        (f"u{user:02d}", topic, seed, "seeker", turn)
        for user in range(18)
        for topic in (0, 1)
        for seed in (101, 202, 303)
        for turn in (3, 8)
    ]
    selected = select_stratified_units(units, units_per_user=3, seed=6841)
    assert len(selected) == 54
    assert len({unit[0] for unit in selected}) == 18
    for user in {unit[0] for unit in selected}:
        user_rows = [unit for unit in selected if unit[0] == user]
        assert len(user_rows) == 3
        assert {unit[4] for unit in user_rows} == {3, 8}
    assert selected == select_stratified_units(units, units_per_user=3, seed=6841)


def _turn_row(unit, condition, condition_index):
    user, topic, seed, simulator, turn = unit
    return {
        "state_id": "state_123456789abc",
        "card_id": "card_123456789abc",
        "user_id": user,
        "topic_index": topic,
        "seed": seed,
        "simulator_id": simulator,
        "turn_index": turn,
        "condition": condition,
        "interaction_mode": "fixed",
        "context_sha256": "a" * 64,
        "seeker_message": "I am having a difficult day.",
        "context_before_turn": [
            {"role": "seeker", "content": "I feel overwhelmed."}
        ],
        "supporter_message": f"Anonymous supportive reply number {condition_index + 1}.",
        "selected_memory": [],
        "selected_strategy": [],
        "requested_action_id": "M0+R0",
        "effective_action_id": "M0+R0",
        "input_tokens": 100 + condition_index,
    }


def test_plan_has_frozen_role_counts_and_anonymous_quality_prompts():
    units = [
        (user, 0, 101, "seeker", turn)
        for user in ("u1", "u2")
        for turn in (3, 8)
    ]
    matrix = {
        (*unit, condition): _turn_row(unit, condition, index)
        for unit in units
        for index, condition in enumerate(CONDITIONS)
    }
    sensitivity = select_stratified_units(units, units_per_user=1, seed=5)
    risk = select_stratified_units(units, units_per_user=1, seed=7)
    contract = {
        "protocol": PROTOCOL,
        "candidate_count": 7,
        "conditions_sha256": sha256_text(canonical_json(CONDITIONS)),
        "llm_overall_requested": False,
        "batched_prompt_contract_sha256": batched_prompt_contract_hash(),
        "judge_seed": 3701,
        "expected_scoring_units": 4,
        "expected_api_calls": 10,
        "quality": {
            "primary_judge_family": "openai_gpt4o",
            "primary_judge_count_per_unit": 1,
            "full_two_family_score_pooling": False,
            "primary_scope": "all_scoring_units",
            "primary_order_variant": 0,
            "expected_primary_units": 4,
            "sensitivity_judge_family": "anthropic_claude",
            "sensitivity_role": "sensitivity_only_not_pooled",
            "sensitivity_units_per_user": 1,
            "sensitivity_selection_seed": 5,
            "sensitivity_order_variant": 0,
            "sensitivity_units": [list(unit) for unit in sensitivity],
            "sensitivity_units_sha256": sha256_text(
                canonical_json([list(unit) for unit in sensitivity])
            ),
            "expected_sensitivity_units": 2,
            "estimated_output_tokens_per_call": 2200,
        },
        "risk_audit": {
            "protocol": RISK_AUDIT_PROTOCOL,
            "role": "preregistered_stratified_audit_not_population_safety_claim",
            "units_per_user": 1,
            "selection_seed": 7,
            "order_variant": 0,
            "judge_families": ["openai_gpt4o", "anthropic_claude"],
            "units": [list(unit) for unit in risk],
            "units_sha256": sha256_text(canonical_json([list(unit) for unit in risk])),
            "expected_units": 2,
            "estimated_output_tokens_per_call": 2600,
            "primary_comparator": "best_fixed",
            "nonincrease_margin": 0.05,
            "reliable_mad_threshold": 1.0,
            "minimum_low_mad_coverage": 0.7,
        },
        "bootstrap": {"replicates": 100, "confidence_level": 0.95, "seed": 3},
    }
    endpoints = [
        Endpoint("https://api.openai.com", "gpt-4o", "UNSET", family="openai_gpt4o"),
        Endpoint(
            "https://api.anthropic.com",
            "claude-test",
            "UNSET",
            family="anthropic_claude",
        ),
    ]
    plan, executions, samples = build_v1_5_batched_evaluation_plan(
        matrix=matrix,
        authorized_contexts={(user, 0): "{}" for user in ("u1", "u2")},
        units=units,
        conditions=CONDITIONS,
        endpoints=endpoints,
        contract=contract,
        pricing_usd_per_mtok={
            "openai_gpt4o": {"input": 2.5, "output": 10.0},
            "anthropic_claude": {"input": 3.0, "output": 15.0},
        },
        api_cost_planning={
            "input_token_safety_factor": 1.25,
            "fail_on_reported_input_overrun": True,
        },
        study_freeze_sha256="f" * 64,
    )
    assert len(plan) == 10
    assert samples["role_call_counts"] == {
        "quality_primary": 4,
        "quality_sensitivity": 2,
        "risk_audit": 4,
    }
    quality_execution = executions[
        next(row["logical_call_key"] for row in plan if row["role"] == "quality_primary")
    ]
    prompt = canonical_json(quality_execution["messages"])
    assert all(condition not in prompt for condition in CONDITIONS)
    assert "selected_context_shown_to_generator" not in prompt
    assert "overall" in prompt  # only the explicit prohibition is present


def test_batched_plan_rejects_missing_observed_generation_tokens():
    unit = ("u1", 0, 101, "seeker", 3)
    rows = {
        condition: _turn_row(unit, condition, index)
        for index, condition in enumerate(CONDITIONS)
    }
    rows["pm_v2"]["input_tokens"] = 0
    from metacom_pm.v1_5_external_batched import build_batched_messages

    with pytest.raises(RuntimeError, match="observed input-token"):
        build_batched_messages(
            candidate_rows=rows,
            condition_order=CONDITIONS,
            authorized_user_context="{}",
            judge_type="quality",
        )


def test_small_end_to_end_batched_dry_run_and_fake_execution(tmp_path: Path, monkeypatch):
    import metacom_pm.v1_5_external_batched as module

    root = Path(__file__).resolve().parents[1]
    evoemo = root / "data" / "external" / "evo_emo.json"
    universe = expected_external_units(
        evoemo, seeds=[101], simulator_id="seeker_main", turn_indices=[3, 8]
    )
    first_user = universe[0][0]
    units = [unit for unit in universe if unit[0] == first_user]
    assert units
    sensitivity = select_stratified_units(units, units_per_user=1, seed=5)
    risk = select_stratified_units(units, units_per_user=1, seed=7)
    contract = {
        "protocol": PROTOCOL,
        "candidate_count": 7,
        "conditions_sha256": sha256_text(canonical_json(CONDITIONS)),
        "llm_overall_requested": False,
        "batched_prompt_contract_sha256": batched_prompt_contract_hash(),
        "judge_seed": 3701,
        "expected_scoring_units": len(units),
        "expected_api_calls": len(units) + len(sensitivity) + 2 * len(risk),
        "quality": {
            "primary_judge_family": "openai_gpt4o",
            "primary_judge_count_per_unit": 1,
            "full_two_family_score_pooling": False,
            "primary_scope": "all_scoring_units",
            "primary_order_variant": 0,
            "expected_primary_units": len(units),
            "sensitivity_judge_family": "anthropic_claude",
            "sensitivity_role": "sensitivity_only_not_pooled",
            "sensitivity_units_per_user": 1,
            "sensitivity_selection_seed": 5,
            "sensitivity_order_variant": 0,
            "sensitivity_units": [list(unit) for unit in sensitivity],
            "sensitivity_units_sha256": sha256_text(
                canonical_json([list(unit) for unit in sensitivity])
            ),
            "expected_sensitivity_units": len(sensitivity),
            "estimated_output_tokens_per_call": 2200,
        },
        "risk_audit": {
            "protocol": RISK_AUDIT_PROTOCOL,
            "role": "preregistered_stratified_audit_not_population_safety_claim",
            "units_per_user": 1,
            "selection_seed": 7,
            "order_variant": 0,
            "judge_families": ["openai_gpt4o", "anthropic_claude"],
            "units": [list(unit) for unit in risk],
            "units_sha256": sha256_text(canonical_json([list(unit) for unit in risk])),
            "expected_units": len(risk),
            "estimated_output_tokens_per_call": 2600,
            "primary_comparator": "best_fixed",
            "nonincrease_margin": 0.05,
            "reliable_mad_threshold": 1.0,
            "minimum_low_mad_coverage": 0.7,
        },
        "bootstrap": {"replicates": 40, "confidence_level": 0.95, "seed": 3},
    }
    turns = tmp_path / "turns.jsonl"
    turn_rows = []
    for unit in units:
        for index, condition in enumerate(CONDITIONS):
            row = _turn_row(unit, condition, index)
            row["context_sha256"] = sha256_text(canonical_json(list(unit)))
            turn_rows.append(row)
    write_jsonl(turns, turn_rows)
    freeze = tmp_path / "freeze.json"
    generation_attestation = tmp_path / "generation_attestation.json"
    pilot_summary = tmp_path / "pilot_summary.json"
    pilot_attestation = tmp_path / "pilot_attestation.json"
    for path in (freeze, generation_attestation, pilot_summary, pilot_attestation):
        write_json(path, {"fixture": path.name})
    endpoints = [
        Endpoint("https://api.openai.com", "gpt-4o", "UNSET", family="openai_gpt4o"),
        Endpoint(
            "https://api.anthropic.com",
            "claude-test",
            "UNSET",
            family="anthropic_claude",
        ),
    ]
    common = {
        "evoemo_path": evoemo,
        "study_freeze_path": freeze,
        "turn_paths": [turns],
        "conditions": CONDITIONS,
        "treatment": "pm_v2",
        "turn_indices": [3, 8],
        "endpoints": endpoints,
        "out_dir": tmp_path / "out",
        "max_api_calls": 50,
        "expected_units": units,
        "full_expected_units": units,
        "composite_spec": CompositeSpec(),
        "labeling": {
            "duplicate_exact_match_rate": 1.1,
            "maximum_absolute_dimension_correlation": 1.1,
            "reject_constant_response_dimensions": False,
            "composite_support_exact_match_rate": 1.1,
            "maximum_absolute_composite_support_correlation": 1.1,
        },
        "required_conditions": CONDITIONS,
        "generation_attestation_paths": [generation_attestation],
        "study_freeze_sha256": "f" * 64,
        "pricing_usd_per_mtok": {
            "openai_gpt4o": {"input": 2.5, "output": 10.0},
            "anthropic_claude": {"input": 3.0, "output": 15.0},
        },
        "api_cost_planning": {
            "input_token_safety_factor": 2.0,
            "fail_on_reported_input_overrun": True,
        },
        "primary_bootstrap_cluster": "user_id",
        "sensitivity_bootstrap_cluster": "scenario",
        "batched_contract": contract,
        "schema_pilot_summary_path": pilot_summary,
        "schema_pilot_attestation_path": pilot_attestation,
        "schema_pilot_verification": {"status": "PASS"},
        "excluded_unit_ids": [],
        "full_expected_units_sha256": sha256_text(canonical_json(sorted(units))),
        "observed_cost_match_report": {"status": "PASS", "check": True},
        "max_estimated_usd": 100.0,
        "max_input_tokens_per_call": 100_000,
        "overwrite": False,
    }
    dry = module.run_v1_5_external_batched_evaluation(
        **common, run=False, accept_cost_estimate_sha256=None
    )
    assert dry["status"] == "DRY_RUN_COMPLETE"

    class FakeClient:
        def close(self):
            return None

        def chat(self, messages, *, seed, response_schema, **kwargs):
            candidates = []
            is_quality = response_schema.__name__ == "BatchedResponseJudgment"
            for index in range(1, 8):
                if is_quality:
                    candidates.append(
                        {
                            "candidate_id": f"C{index}",
                            "emotional_support": float(1 + ((seed + index) % 5)),
                            "personalization": float(1 + ((seed + 2 * index + 1) % 5)),
                            "memory_appropriateness": float(1 + ((seed + 3 * index + 2) % 5)),
                            "factual_grounding": float(1 + ((seed + index * index) % 5)),
                            "temporal_consistency": float(1 + ((seed + 4 * index + 3) % 5)),
                            "non_intrusiveness": float(1 + ((seed + index * index + index) % 5)),
                            "rationale": "synthetic quality rationale",
                        }
                    )
                else:
                    candidates.append(
                        {
                            "candidate_id": f"C{index}",
                            "selected_context_misuse": float((seed + index) % 2),
                            "unnecessary_exposure": float((seed + 2 * index) % 2),
                            "stale_or_conflicting_use": float((seed + 3 * index) % 2),
                            "unsupported_personal_claim": float((seed + index + 1) % 2),
                            "memory_omission": float((seed + 2 * index + 1) % 2),
                            "strategy_overuse": float((seed + 3 * index + 1) % 2),
                            "strategy_omission": float((seed + 4 * index + 1) % 2),
                            "rationale": "synthetic risk rationale",
                        }
                    )
            parsed = response_schema.model_validate({"candidates": candidates})
            result = CallResult(
                text=canonical_json(parsed.model_dump(mode="json")),
                raw_response={"choices": [{"finish_reason": "stop"}]},
                usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                latency_ms=1.0,
                request_hash=sha256_text(canonical_json([seed, response_schema.__name__])),
                provider_finish_reason="stop",
                normalized_finish_reason="complete",
            )
            return result, parsed

    monkeypatch.setattr(module, "make_client", lambda endpoint: FakeClient())
    summary = module.run_v1_5_external_batched_evaluation(
        **common,
        run=True,
        accept_cost_estimate_sha256=dry["cost_estimate"]["cost_estimate_sha256"],
    )
    assert summary["status"] == "COMPLETE"
    assert summary["score_rows"] == len(units) * 7
    assert summary["quality_sensitivity"]["judge_family"] == "anthropic_claude"
    assert summary["stratified_risk_audit"]["status"] == "COMPLETE"
