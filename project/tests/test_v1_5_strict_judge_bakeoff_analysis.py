from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from metacom_pm.io import canonical_json, sha256_text
from metacom_pm.v1_5_strict_judge_bakeoff import EXPECTED_CANDIDATES
from metacom_pm.v1_5_strict_judge_bakeoff_analysis import (
    aggregate_strict_bakeoff_qualification,
)


ROOT = Path(__file__).resolve().parents[1]


def _parsed(value: str) -> dict:
    return {
        "overall_preference": value,
        "support_quality_preference": value,
        "evidence_handling_preference": value,
        "safety_preference": value,
        "overall_reason": "reason",
        "support_quality_reason": "reason",
        "evidence_handling_reason": "reason",
        "safety_reason": "reason",
    }


def _fixture() -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    results: list[dict] = []
    plan: list[dict] = []
    for candidate in EXPECTED_CANDIDATES:
        for pair_index in range(12):
            for order in (0, 1):
                record_ids = {
                    "candidate_key": candidate,
                    "pair_id": f"p{pair_index}",
                    "state_id": f"s{pair_index}",
                    "order_variant": order,
                }
                plan.append(
                    {
                        "candidate_key": candidate,
                        "condition": f"{candidate}_pairwise",
                        "judge_family": candidate,
                        "judge_model": f"{candidate}_model",
                        "record_ids": record_ids,
                    }
                )
                results.append(
                    {
                        "candidate_key": candidate,
                        "judge_family": candidate,
                        "judge_model": f"{candidate}_model",
                        "record_ids": record_ids,
                        "request_hash": "a" * 64,
                        "parsed": _parsed("A" if order == 0 else "B"),
                    }
                )
    gpt: list[dict] = []
    for pair_index in range(8):
        for order in (0, 1):
            gpt.append(
                {
                    "condition": "gpt_high_quality_pairwise_anchor",
                    "judge_family": "openai_gpt_5_6_sol",
                    "judge_model": "gpt-5.6-sol",
                    "record_ids": {
                        "pair_id": f"p{pair_index}",
                        "state_id": f"s{pair_index}",
                        "order_variant": order,
                    },
                    "parsed": _parsed("A" if order == 0 else "B"),
                }
            )
    human = [
        {
            "pair_id": f"p{pair_index}",
            "confidence": 3 + (pair_index % 3),
            "canonical_overall_preference": "A",
            "canonical_support_quality_preference": "A",
            "canonical_evidence_handling_preference": "A",
            "canonical_safety_preference": "A",
        }
        for pair_index in range(12)
    ]
    contract = {
        "protocol": "pm-v1.5-train-only-strict-pairwise-judge-bakeoff-v1",
        "candidate_keys": list(EXPECTED_CANDIDATES),
        "pair_count": 12,
        "pre_outcome_criteria": {
            "required_schema_valid_rate": 1.0,
            "minimum_ab_ba_order_consistency": 0.8,
            "minimum_pairwise_informative_rate": 0.5,
            "minimum_gpt_anchor_direction_agreement_secondary": 0.7,
        },
    }
    return results, plan, gpt, human, contract


def test_pass_requires_researcher_signoff_and_creates_no_labels():
    results, plan, gpt, human, contract = _fixture()
    report = aggregate_strict_bakeoff_qualification(
        result_rows=results,
        call_plan_rows=plan,
        gpt_anchor_rows=gpt,
        human_normalized_rows=human,
        contract=contract,
    )
    assert report["status"] == (
        "AUTOMATIC_CRITERIA_PASS_REQUIRES_RESEARCHER_SIGNOFF"
    )
    assert report["automatically_supported_candidates"] == list(
        EXPECTED_CANDIDATES
    )
    assert report["automatic_promotion"] is False
    assert report["training_labels_created"] is False
    assert report["human_anchor_agreement"][
        "openai_gpt_5_mini"
    ]["automatic_gold"] is False


def test_order_effect_fails_only_the_affected_candidate():
    results, plan, gpt, human, contract = _fixture()
    candidate = "google_gemini_2_5_flash"
    changed = 0
    for row in results:
        if (
            row["candidate_key"] == candidate
            and row["record_ids"]["order_variant"] == 1
            and changed < 3
        ):
            row["parsed"] = _parsed("A")
            changed += 1
    report = aggregate_strict_bakeoff_qualification(
        result_rows=results,
        call_plan_rows=plan,
        gpt_anchor_rows=gpt,
        human_normalized_rows=human,
        contract=contract,
    )
    assert report["automatic_checks"][candidate][
        "all_dimension_ab_ba_order_consistency"
    ]["pass"] is False
    assert candidate not in report["automatically_supported_candidates"]
    assert "openai_gpt_5_mini" in report["automatically_supported_candidates"]


def test_anchor_direction_failure_is_fail_closed():
    results, plan, gpt, human, contract = _fixture()
    candidate = "anthropic_claude_haiku_4_5"
    for row in results:
        if (
            row["candidate_key"] == candidate
            and int(row["record_ids"]["pair_id"][1:]) < 8
        ):
            row["parsed"] = _parsed(
                "B" if row["record_ids"]["order_variant"] == 0 else "A"
            )
    report = aggregate_strict_bakeoff_qualification(
        result_rows=results,
        call_plan_rows=plan,
        gpt_anchor_rows=gpt,
        human_normalized_rows=human,
        contract=contract,
    )
    assert report["automatic_checks"][candidate][
        "overall_gpt_anchor_direction_agreement_secondary"
    ]["pass"] is False
    assert candidate not in report["automatically_supported_candidates"]


def test_incomplete_or_drifted_matrix_is_rejected():
    results, plan, gpt, human, contract = _fixture()
    with pytest.raises(RuntimeError, match="incomplete"):
        aggregate_strict_bakeoff_qualification(
            result_rows=results[:-1],
            call_plan_rows=plan,
            gpt_anchor_rows=gpt,
            human_normalized_rows=human,
            contract=contract,
        )
    drifted = copy.deepcopy(results)
    drifted[0]["judge_model"] = "wrong"
    with pytest.raises(RuntimeError, match="judge identity drifted"):
        aggregate_strict_bakeoff_qualification(
            result_rows=drifted,
            call_plan_rows=plan,
            gpt_anchor_rows=gpt,
            human_normalized_rows=human,
            contract=contract,
        )


def test_repository_no_go_freeze_is_content_addressed_and_nonpromoting():
    path = (
        ROOT
        / "data/pm_v1_5_contracts/"
        "longitudinal_strict_judge_bakeoff_freeze_v1.json"
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    binding = record.pop("binding_sha256")
    assert binding == sha256_text(canonical_json(record))
    assert (
        record["status"]
        == "LONGITUDINAL_STRICT_JUDGE_BAKEOFF_NOT_SUPPORTED"
    )
    assert record["frozen_interpretation"][
        "automatic_supported_candidates"
    ] == []
    assert record["stop_rule"]["candidate_auto_promotion"] is False
    assert record["stop_rule"]["bulk_label_generation_authorized"] is False
    assert record["stop_rule"]["training_labels_created"] is False
