from __future__ import annotations

from collections import Counter
from copy import deepcopy

import pytest

from metacom_pm.v1_5_mvp_human_audit import (
    MVP_HUMAN_AUDIT_PROTOCOL,
    build_human_audit_packet,
    select_human_audit_pairs,
)
from metacom_pm.v1_5_mvp_human_audit_results import analyze_human_audit
from metacom_pm.v1_5_mvp_judge import RISK_IDS


def _pair(index: int, cue: str, family: str) -> dict:
    pair_id = f"pair-{index}"
    return {
        "pair_id": pair_id,
        "state_id": f"state-{index}",
        "user_id": f"user-{index}",
        "boundary_cue": cue,
        "visible_dialogue": {"current_user_text": "Please help."},
        "arms": {
            "R0": {"response": "R0 response"},
            "RS": {
                "response": "RS response",
                "selected_strategy_family": family,
            },
        },
        "risk_applicability": {
            risk_id: {
                "applicable": risk_id != "stale_or_conflicting_evidence_use",
                "reason": "test",
            }
            for risk_id in RISK_IDS
        },
    }


def _fixture() -> tuple[dict, list[dict], list[dict]]:
    pairs = [
        _pair(
            i,
            "listen_only" if i < 2 else "advice_welcome",
            "Reflection of feelings"
            if i % 2 == 0
            else "Providing Suggestions",
        )
        for i in range(4)
    ]
    built = build_human_audit_packet(pairs, pair_count=4, seed=7)
    annotations = []
    for item in built["packet"]:
        findings = []
        for label in ("A", "B"):
            findings.append(
                {
                    "blind_response_id": item["blind_response_ids"][label],
                    "material_risk_findings": {
                        risk_id: {
                            "judgment": (
                                "no"
                                if item["risk_applicability"][risk_id]["applicable"]
                                else "not_applicable"
                            ),
                            "evidence_excerpt": "",
                            "notes": "",
                        }
                        for risk_id in RISK_IDS
                    },
                }
            )
        annotations.append(
            {
                "protocol": MVP_HUMAN_AUDIT_PROTOCOL,
                "blind_item_id": item["blind_item_id"],
                "quality_preference": "A",
                "quality_decisive_criterion": "request_fit",
                "quality_notes": "",
                "responses": findings,
                "annotator_id": "",
            }
        )
    llm_quality = [
        {
            "pair_id": row["pair_id"],
            "preference": "tie",
            "orders_consistent": True,
        }
        for row in built["private_key"]
    ]
    llm_risk = [
        {
            "pair_id": row["pair_id"],
            "arm": arm,
            "risk_id": risk_id,
            "material_event": False,
        }
        for row in built["private_key"]
        for arm in ("R0", "RS")
        for risk_id in RISK_IDS
    ]
    return built, annotations, llm_quality, llm_risk


def test_completed_human_audit_is_unblinded_but_not_train_authorized() -> None:
    built, annotations, llm_quality, llm_risk = _fixture()
    result = analyze_human_audit(
        annotations=annotations,
        packet=built["packet"],
        private_key=built["private_key"],
        llm_quality_rows=llm_quality,
        llm_risk_rows=llm_risk,
    )
    analysis = result["analysis"]
    assert analysis["data_quality"]["complete"] is True
    assert analysis["quality"]["human_arm_preferences"]["RS"] == 2
    assert analysis["quality"]["human_arm_preferences"]["R0"] == 2
    assert analysis["training_readiness"]["authorized"] is False
    assert len(result["unblinded_risk_rows"]) == 4 * 2 * len(RISK_IDS)


def test_missing_blind_item_is_rejected() -> None:
    built, annotations, llm_quality, llm_risk = _fixture()
    with pytest.raises(ValueError, match="coverage mismatch"):
        analyze_human_audit(
            annotations=annotations[:-1],
            packet=built["packet"],
            private_key=built["private_key"],
            llm_quality_rows=llm_quality,
            llm_risk_rows=llm_risk,
        )


def test_material_yes_requires_literal_excerpt() -> None:
    built, annotations, llm_quality, llm_risk = _fixture()
    broken = deepcopy(annotations)
    first = broken[0]["responses"][0]["material_risk_findings"]
    first["excessive_directiveness"]["judgment"] = "yes"
    with pytest.raises(ValueError, match="yes requires an evidence excerpt"):
        analyze_human_audit(
            annotations=broken,
            packet=built["packet"],
            private_key=built["private_key"],
            llm_quality_rows=llm_quality,
            llm_risk_rows=llm_risk,
        )


def test_confirmation_selection_can_exclude_the_discovery_sample() -> None:
    pairs = [
        _pair(
            i,
            "listen_only" if i < 8 else "advice_welcome",
            "Reflection of feelings"
            if i % 2 == 0
            else "Providing Suggestions",
        )
        for i in range(16)
    ]
    discovery = select_human_audit_pairs(pairs, pair_count=4, seed=7)
    excluded = {row["pair_id"] for row in discovery}
    confirmation = select_human_audit_pairs(
        pairs,
        pair_count=4,
        seed=11,
        excluded_pair_ids=excluded,
    )
    assert excluded.isdisjoint({row["pair_id"] for row in confirmation})
    assert Counter(row["boundary_cue"] for row in confirmation) == {
        "listen_only": 2,
        "advice_welcome": 2,
    }
