from __future__ import annotations

import copy
from pathlib import Path

from metacom_pm.io import read_json
from metacom_pm.v1_5_learnable_transfer_readiness import (
    BLOCKED_STATUS,
    PASS_STATUS,
    REQUIRED_EVIDENCE_CHECKS,
    audit_failure_ledger,
    build_formal_fit_readiness,
    external_support_blockers,
    validate_completed_human_annotations,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "data/pm_v1_5_contracts/learnable_transfer_training_data_v2.json"
)


def _annotations():
    return [
        {
            "blind_item_id": f"human_{index}",
            "overall_preference": "A",
            "support_quality_preference": "A",
            "evidence_handling_preference": "B",
            "safety_preference": "tie",
            "confidence": 4,
        }
        for index in range(32)
    ]


def _evidence(name: str, protocol: str, contract_sha256: str):
    return {
        "protocol": protocol,
        "contract_sha256": contract_sha256,
        "status": "PASS",
        "scope": {
            "train_only": True,
            "internal_outcomes_opened": False,
            "external_outcomes_opened": False,
        },
        "checks": {
            check: True for check in REQUIRED_EVIDENCE_CHECKS[name]
        },
    }


def _external(esconv=0.05, evoemo=0.05, contract_sha256=None):
    return {
        "training_support_contract_sha256": contract_sha256,
        "merged_synthetic_plus_esconv_auxiliary": {
            "esconv_test": {"severe_rate": esconv},
            "evoemo_formal_turn_states": {
                "formal_result": True,
                "severe_rate": evoemo,
            },
        }
    }


def test_failure_ledger_has_unique_ids_and_v2_root_causes():
    audit = audit_failure_ledger(
        ROOT / "docs/PM_V1_TO_V1_5_GLOBAL_FAILURE_LEDGER_ZH.md"
    )
    assert audit["issue_count"] == audit["unique_issue_count"]
    assert audit["declared_issue_count"] == audit["unique_issue_count"]
    assert audit["declared_issue_count_matches"]
    assert audit["duplicate_issue_ids"] == []
    assert audit["required_v2_issue_ids_present"]


def test_human_annotations_must_be_complete_but_are_not_gold():
    assert validate_completed_human_annotations(_annotations()) == []
    broken = _annotations()
    broken[0]["overall_preference"] = None
    assert any(
        "incomplete_preference_rows_1" in value
        for value in validate_completed_human_annotations(broken)
    )


def test_external_domains_pass_and_fail_separately():
    assert external_support_blockers(
        _external(), maximum_severe_ood_rate=0.1
    ) == []
    blockers = external_support_blockers(
        _external(evoemo=1.0), maximum_severe_ood_rate=0.1
    )
    assert blockers == ["external_support_evoemo_severe_ood_above_limit"]


def test_complete_evidence_allows_exactly_one_formal_fit():
    contract = read_json(CONTRACT)
    sha = contract["contract_sha256"]
    root = {
        "status": "TRAINING_DATA_REDESIGN_REQUIRED_BEFORE_REFIT",
        "required_next_contract": {"contract_sha256": sha},
        "scope": {
            "internal_outcomes_opened": False,
            "external_outcomes_opened": False,
        },
    }
    ledger = {
        "duplicate_issue_ids": [],
        "required_v2_issue_ids_present": True,
    }
    reports = {
        "clean_contrast_report": _evidence(
            "clean_contrast", "pm-v1.5-clean-contrast-integrity-v1", sha
        ),
        "mechanism_uptake_report": _evidence(
            "mechanism_uptake", "pm-v1.5-clean-mechanism-uptake-v1", sha
        ),
        "measurement_report": _evidence(
            "measurement", "pm-v1.5-mechanism-matched-measurement-v1", sha
        ),
        "learnability_report": _evidence(
            "learnability", "pm-v1.5-train-only-component-learnability-v1", sha
        ),
    }
    ready = build_formal_fit_readiness(
        contract_path=CONTRACT,
        root_cause_report=root,
        ledger_audit=ledger,
        legacy_human_annotations=_annotations(),
        external_support_report=_external(contract_sha256=sha),
        **reports,
    )
    assert ready["status"] == PASS_STATUS
    assert ready["formal_fit_authorized"]

    drifted = copy.deepcopy(reports)
    drifted["measurement_report"]["contract_sha256"] = "wrong"
    blocked = build_formal_fit_readiness(
        contract_path=CONTRACT,
        root_cause_report=root,
        ledger_audit=ledger,
        legacy_human_annotations=_annotations(),
        external_support_report=_external(contract_sha256=sha),
        **drifted,
    )
    assert blocked["status"] == BLOCKED_STATUS
    assert "measurement_contract_hash_mismatch" in blocked["blockers"]
