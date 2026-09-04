"""Fail-closed readiness checks for the redesigned PM-v1.5 training route.

The checker does not claim that a scientific result is guaranteed.  It prevents
formal fitting while any known prerequisite is absent, stale, contaminated, or
unsupported.  Small train-only mechanism pilots remain allowed while readiness
is BLOCKED.
"""

from __future__ import annotations

import collections
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import read_json, sha256_file
from .v1_5_training_data_root_cause import (
    require_learnable_transfer_training_contract,
)


READINESS_PROTOCOL = "pm-v1.5-learnable-transfer-formal-fit-readiness-v1"
PASS_STATUS = "READY_FOR_ONE_PREREGISTERED_FORMAL_FIT"
BLOCKED_STATUS = "BLOCKED_BEFORE_FORMAL_FIT"

_PREFERENCE_FIELDS = (
    "overall_preference",
    "support_quality_preference",
    "evidence_handling_preference",
    "safety_preference",
)
_PREFERENCE_VALUES = {"A", "B", "tie"}

_EVIDENCE_PROTOCOLS = {
    "clean_contrast": "pm-v1.5-clean-contrast-integrity-v1",
    "mechanism_uptake": "pm-v1.5-clean-mechanism-uptake-v1",
    "measurement": "pm-v1.5-mechanism-matched-measurement-v1",
    "learnability": "pm-v1.5-train-only-component-learnability-v1",
}
REQUIRED_EVIDENCE_CHECKS = {
    "clean_contrast": frozenset(
        {
            "exact_lineage",
            "positive_zero_nonhelpful_contamination",
            "control_helpful_placebo_harm_complete",
            "same_treatment_except_evidence",
            "no_domain_feature",
            "unified_legal_action_mask",
        }
    ),
    "mechanism_uptake": frozenset(
        {
            "all_MP_MS_ME_RS_components_covered",
            "RS_memory_unavailable_and_available_covered",
            "helpful_above_frozen_margin",
            "irrelevant_placebo_nonpositive",
            "harmful_nonpositive_or_safely_ignored",
            "source_source_interactions_audited",
            "source_RS_interactions_audited",
        }
    ),
    "measurement": frozenset(
        {
            "mechanism_matched_human_anchor_complete",
            "human_anchor_not_bulk_gold",
            "judge_families_retained_separately",
            "disagreement_and_low_confidence_abstain",
            "evidence_quote_validity_pass",
            "risk_action_applicability_mask_pass",
        }
    ),
    "learnability": frozenset(
        {
            "user_or_dialogue_group_disjoint_cv",
            "effective_group_counts_sufficient",
            "beats_M0_R0_baseline",
            "beats_transparent_rule_baseline",
            "component_direction_recovery_pass",
            "uncertainty_fallback_to_M0_R0_verified",
            "one_candidate_frozen_before_calibration",
            "calibration_and_internal_outcomes_not_read",
        }
    ),
}


def audit_failure_ledger(path: str | Path) -> dict[str, Any]:
    """Check unique issue IDs and reconcile the reader-facing total."""

    text = Path(path).read_text(encoding="utf-8")
    ids = re.findall(r"\| (V15-[A-Z]+-\d+) \|", text)
    declared_match = re.search(
        r"本文共有\s*(\d+)\s*个互不重复的 V1\.5 issue ID", text
    )
    declared_issue_count = (
        int(declared_match.group(1)) if declared_match else None
    )
    duplicates = sorted(
        issue_id
        for issue_id, count in collections.Counter(ids).items()
        if count > 1
    )
    categories = collections.Counter(issue_id.split("-")[1] for issue_id in ids)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "issue_count": len(ids),
        "unique_issue_count": len(set(ids)),
        "declared_issue_count": declared_issue_count,
        "declared_issue_count_matches": declared_issue_count == len(set(ids)),
        "duplicate_issue_ids": duplicates,
        "category_counts": dict(sorted(categories.items())),
        "required_v2_issue_ids_present": all(
            issue_id in ids
            for issue_id in (
                "V15-DATA-17",
                "V15-DATA-19",
                "V15-DATA-20",
                "V15-DATA-21",
            )
        ),
    }


def validate_completed_human_annotations(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_rows: int = 32,
) -> list[str]:
    """Return human-anchor blockers without treating anchors as automatic gold."""

    blockers: list[str] = []
    if len(rows) != expected_rows:
        blockers.append(
            f"legacy_human_diagnostic_expected_{expected_rows}_rows_got_{len(rows)}"
        )
    ids = [str(row.get("blind_item_id") or "") for row in rows]
    if "" in ids or len(ids) != len(set(ids)):
        blockers.append("legacy_human_diagnostic_ids_missing_or_duplicated")
    incomplete_preference_rows = 0
    invalid_confidence_rows = 0
    for row in rows:
        if any(row.get(field) not in _PREFERENCE_VALUES for field in _PREFERENCE_FIELDS):
            incomplete_preference_rows += 1
        confidence = row.get("confidence")
        if not isinstance(confidence, int) or not 1 <= confidence <= 5:
            invalid_confidence_rows += 1
    if incomplete_preference_rows:
        blockers.append(
            "legacy_human_diagnostic_incomplete_preference_rows_"
            f"{incomplete_preference_rows}"
        )
    if invalid_confidence_rows:
        blockers.append(
            "legacy_human_diagnostic_invalid_confidence_rows_"
            f"{invalid_confidence_rows}"
        )
    return blockers


def _require_future_evidence(
    *,
    name: str,
    report: Mapping[str, Any] | None,
    contract_sha256: str,
) -> list[str]:
    blockers: list[str] = []
    if report is None:
        return [f"{name}_report_missing"]
    if report.get("protocol") != _EVIDENCE_PROTOCOLS[name]:
        blockers.append(f"{name}_protocol_mismatch")
    if report.get("contract_sha256") != contract_sha256:
        blockers.append(f"{name}_contract_hash_mismatch")
    if report.get("status") != "PASS":
        blockers.append(f"{name}_status_not_pass")
    scope = dict(report.get("scope") or {})
    if scope.get("train_only") is not True:
        blockers.append(f"{name}_not_train_only")
    if scope.get("internal_outcomes_opened") is not False:
        blockers.append(f"{name}_internal_outcome_boundary_broken")
    if scope.get("external_outcomes_opened") is not False:
        blockers.append(f"{name}_external_outcome_boundary_broken")
    checks = dict(report.get("checks") or {})
    required_checks = REQUIRED_EVIDENCE_CHECKS[name]
    if set(checks) != required_checks:
        blockers.append(f"{name}_check_set_mismatch")
    if any(checks.get(check) is not True for check in required_checks):
        blockers.append(f"{name}_checks_incomplete")
    return blockers


def external_support_blockers(
    report: Mapping[str, Any],
    *,
    maximum_severe_ood_rate: float,
    contract_sha256: str | None = None,
) -> list[str]:
    """Validate ESConv and EvoEmo observable support separately."""

    merged = dict(report.get("merged_synthetic_plus_esconv_auxiliary") or {})
    esconv = dict(merged.get("esconv_test") or {})
    evoemo = dict(merged.get("evoemo_formal_turn_states") or {})
    blockers: list[str] = []
    if (
        contract_sha256 is not None
        and report.get("training_support_contract_sha256") != contract_sha256
    ):
        blockers.append("external_support_contract_hash_mismatch")
    if not esconv:
        blockers.append("external_support_esconv_missing")
    elif float(esconv.get("severe_rate", 1.0)) > maximum_severe_ood_rate:
        blockers.append("external_support_esconv_severe_ood_above_limit")
    if not evoemo:
        blockers.append("external_support_evoemo_missing")
    else:
        if evoemo.get("formal_result") is not True:
            blockers.append("external_support_evoemo_formal_coverage_incomplete")
        if float(evoemo.get("severe_rate", 1.0)) > maximum_severe_ood_rate:
            blockers.append("external_support_evoemo_severe_ood_above_limit")
    return blockers


def build_formal_fit_readiness(
    *,
    contract_path: str | Path,
    root_cause_report: Mapping[str, Any],
    ledger_audit: Mapping[str, Any],
    legacy_human_annotations: Sequence[Mapping[str, Any]],
    external_support_report: Mapping[str, Any],
    clean_contrast_report: Mapping[str, Any] | None = None,
    mechanism_uptake_report: Mapping[str, Any] | None = None,
    measurement_report: Mapping[str, Any] | None = None,
    learnability_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic readiness decision for one formal fit."""

    contract = require_learnable_transfer_training_contract(contract_path)
    contract_sha256 = str(contract["contract_sha256"])
    blockers: list[str] = []

    if ledger_audit.get("duplicate_issue_ids"):
        blockers.append("failure_ledger_has_duplicate_issue_ids")
    if ledger_audit.get("required_v2_issue_ids_present") is not True:
        blockers.append("failure_ledger_missing_required_v2_issues")

    if (
        root_cause_report.get("status")
        != "TRAINING_DATA_REDESIGN_REQUIRED_BEFORE_REFIT"
    ):
        blockers.append("root_cause_report_status_mismatch")
    required = dict(root_cause_report.get("required_next_contract") or {})
    if required.get("contract_sha256") != contract_sha256:
        blockers.append("root_cause_report_contract_hash_mismatch")
    scope = dict(root_cause_report.get("scope") or {})
    if scope.get("internal_outcomes_opened") is not False:
        blockers.append("root_cause_report_opened_internal_outcomes")
    if scope.get("external_outcomes_opened") is not False:
        blockers.append("root_cause_report_opened_external_outcomes")

    blockers.extend(validate_completed_human_annotations(legacy_human_annotations))
    blockers.extend(
        _require_future_evidence(
            name="clean_contrast",
            report=clean_contrast_report,
            contract_sha256=contract_sha256,
        )
    )
    blockers.extend(
        _require_future_evidence(
            name="mechanism_uptake",
            report=mechanism_uptake_report,
            contract_sha256=contract_sha256,
        )
    )
    blockers.extend(
        _require_future_evidence(
            name="measurement",
            report=measurement_report,
            contract_sha256=contract_sha256,
        )
    )
    blockers.extend(
        _require_future_evidence(
            name="learnability",
            report=learnability_report,
            contract_sha256=contract_sha256,
        )
    )
    maximum_ood = float(
        contract["external_transfer_support"]["pre_freeze_support_gate"][
            "maximum_severe_ood_rate_per_external_domain"
        ]
    )
    blockers.extend(
        external_support_blockers(
            external_support_report,
            maximum_severe_ood_rate=maximum_ood,
            contract_sha256=contract_sha256,
        )
    )
    blockers = sorted(set(blockers))
    return {
        "protocol": READINESS_PROTOCOL,
        "status": PASS_STATUS if not blockers else BLOCKED_STATUS,
        "api_calls_made": 0,
        "formal_fit_authorized": not blockers,
        "small_train_only_pilots_remain_allowed": True,
        "contract_sha256": contract_sha256,
        "failure_ledger": dict(ledger_audit),
        "blockers": blockers,
        "known_risk_guarantee_boundary": (
            "READY guarantees that all preregistered known-pitfall gates passed; "
            "it does not guarantee calibration, internal, or external success."
        ),
    }
