from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from metacom_pm.io import canonical_json, read_json, sha256_text
from metacom_pm.pm_v2_contracts import PMV2Split
from metacom_pm.v1_5_factorized_weak_learning import (
    action_components,
    build_factorized_signal_report,
    select_human_anchor_rows,
)


WEIGHTS = {
    "emotional_support": 0.3,
    "personalization": 0.2,
    "memory_appropriateness": 0.15,
    "factual_grounding": 0.15,
    "temporal_consistency": 0.1,
    "non_intrusiveness": 0.1,
}
ROOT = Path(__file__).resolve().parents[1]


def _response(score):
    return {field: score for field in WEIGHTS}


def test_action_components_factorize_memory_and_strategy():
    assert action_components("M0+R0") == frozenset()
    assert action_components("MPMSME+RS") == frozenset(
        {"MP", "MS", "ME", "RS"}
    )
    with pytest.raises(ValueError):
        action_components("MP+BAD")


def test_factorized_report_retains_families_and_repeated_backgrounds():
    state = SimpleNamespace(
        state_id="s1",
        user_id="u1",
        split=PMV2Split.TRAIN,
        allowed_actions=["M0+R0", "MP+R0", "M0+RS", "MP+RS"],
    )
    labels = [
        {
            "state_id": "s1",
            "action_id": action,
            "provenance": {"prompt_equivalence_id": action},
        }
        for action in state.allowed_actions
    ]
    scores = {
        "M0+R0": 3.0,
        "MP+R0": 4.0,
        "M0+RS": 2.0,
        "MP+RS": 3.0,
    }
    raw = []
    for action, score in scores.items():
        for family in ("a", "b"):
            raw.append(
                {
                    "status": "SUCCESS",
                    "schema_success": True,
                    "state_id": "s1",
                    "action_id": action,
                    "judge_family": family,
                    "response": _response(score),
                }
            )
    report = build_factorized_signal_report(
        states=[state],
        weak_labels=labels,
        raw_judge_rows=raw,
        response_weights=WEIGHTS,
        expected_judge_families=["a", "b"],
        evaluator_contexts={
            "s1": {
                "needed_memory_sources": ["MP"],
                "strategy_resource_target": "harmful",
            }
        },
    )
    rows = {
        row["component"]: row for row in report["effect_rows"]
    }
    assert rows["MP"]["factorial_background_count"] == 2
    assert rows["MP"]["consensus"] == "positive"
    assert rows["MP"]["structural_target"] == "positive"
    assert rows["RS"]["consensus"] == "negative"
    assert rows["RS"]["structural_target"] == "negative"
    assert report["by_component"]["MP"]["structural_target_alignment_rate"] == 1.0


def test_factorized_report_rejects_internal_state():
    state = SimpleNamespace(
        state_id="internal",
        user_id="u",
        split=PMV2Split.INTERNAL_TEST,
        allowed_actions=["M0+R0"],
    )
    with pytest.raises(RuntimeError, match="train-only"):
        build_factorized_signal_report(
            states=[state],
            weak_labels=[],
            raw_judge_rows=[],
            response_weights=WEIGHTS,
            expected_judge_families=["a", "b"],
        )


def test_factorized_small_sample_contract_is_self_bound_and_outcome_blind():
    path = (
        ROOT
        / "data"
        / "pm_v1_5_contracts"
        / "factorized_small_sample_weak_learning_v1.json"
    )
    contract = read_json(path)
    expected = contract.pop("contract_sha256")
    assert expected == sha256_text(canonical_json(contract))
    assert contract["scope"]["internal_test_outcomes_opened"] is False
    assert contract["scope"]["external_outcomes_opened"] is False
    assert contract["estimand"]["direct_16_action_absolute_score_argmax_forbidden"]
    assert contract["weak_label_model"]["hard_majority_vote_forbidden"]


def test_human_anchor_selection_is_balanced_and_state_disjoint():
    rows = []
    for component in ("MP", "MS", "ME", "RS"):
        for stratum in ("positive", "negative", "opposite", "uncertain"):
            for index in range(3):
                rows.append(
                    {
                        "state_id": f"{component}_{stratum}_{index}",
                        "component": component,
                        "consensus": stratum,
                    }
                )
    first = select_human_anchor_rows(rows, seed=7)
    second = select_human_anchor_rows(list(reversed(rows)), seed=7)
    assert first == second
    assert len(first) == 32
    assert len({row["state_id"] for row in first}) == 32
