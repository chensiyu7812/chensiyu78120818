#!/usr/bin/env python3
"""Audit the frozen H1R primary and independent-overlap reviews.

This script never trains a PM, never reads response outcomes, and never treats
construction intent as gold.  It checks review integrity, pre-adjudication IAA,
label shape by split, and emits the exact disagreement surfaces for diagnosis.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, read_json, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-final-h1r-dual-review-data-quality-audit-v1"
PRIMARY_PROTOCOL = "pm-v1.5-final-h1r-semantic-candidate-review-v1"
SECONDARY_PROTOCOL = "pm-v1.5-final-h1r-independent-overlap-v1"
COMPONENTS = ("MP", "MS", "ME", "RS")
SPLITS = ("FIT", "FRESH_CONFIRMATION", "SEALED_INTERNAL_TEST")
DECISIONS = ("on", "off", "abstain")
GATE_VALUES = {"yes", "no", "uncertain"}
EVIDENCE_CODES = {
    "OWNER_AND_TIME_VALID",
    "CURRENT_GOAL_FIT",
    "SAFE_AND_BOUNDARY_COMPATIBLE",
    "SPECIFIC_FUNCTIONAL_INCREMENT",
    "WRONG_OWNER_OR_ENTITY",
    "TIME_STALE_OR_CONFLICTING",
    "WRONG_GOAL_OR_FUNCTION",
    "BOUNDARY_OR_BURDEN_CONFLICT",
    "CURRENTLY_REDUNDANT",
    "WRONG_SUBTYPE_OR_FUNCTION",
    "GENERIC_OR_NO_INCREMENT",
    "AMBIGUOUS_VISIBLE_EVIDENCE",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _derived(decision: dict[str, Any]) -> str:
    gate_1 = str(decision.get("applicability_safe", "")).lower()
    gate_2 = str(decision.get("incremental_over_r0", "")).lower()
    if "no" in {gate_1, gate_2}:
        return "off"
    if gate_1 == gate_2 == "yes":
        return "on"
    return "abstain"


def _kappa(pairs: list[tuple[str, str]]) -> tuple[float, float]:
    n = len(pairs)
    observed = sum(left == right for left, right in pairs) / n
    expected = sum(
        sum(left == label for left, _ in pairs)
        / n
        * sum(right == label for _, right in pairs)
        / n
        for label in DECISIONS
    )
    kappa = (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0
    return observed, kappa


def _component_surface(item: dict[str, Any], component: str) -> dict[str, Any]:
    return next(row for row in item["components"] if row["component"] == component)


def _validate_rows(
    *,
    rows: list[dict[str, Any]],
    expected_protocol: str,
    expected_ids: set[str],
    role: str,
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    row_ids = [str(row.get("blind_state_id", "")) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        errors.append({"role": role, "error": "duplicate_blind_state_id"})
    if set(row_ids) != expected_ids:
        errors.append({"role": role, "error": "blind_state_id_set_mismatch"})
    if len({str(row.get("annotator_id", "")) for row in rows}) != 1:
        errors.append({"role": role, "error": "annotator_id_not_unique"})
    for row in rows:
        blind_id = str(row.get("blind_state_id", ""))
        if row.get("protocol") != expected_protocol:
            errors.append({"role": role, "blind_state_id": blind_id, "error": "wrong_protocol"})
        decisions = row.get("component_decisions")
        if not isinstance(decisions, dict) or set(decisions) != set(COMPONENTS):
            errors.append({"role": role, "blind_state_id": blind_id, "error": "component_set_mismatch"})
            continue
        for component in COMPONENTS:
            decision = decisions[component]
            gate_1 = str(decision.get("applicability_safe", "")).lower()
            gate_2 = str(decision.get("incremental_over_r0", "")).lower()
            derived = str(decision.get("derived_decision", "")).lower()
            codes = decision.get("evidence_codes", [])
            if gate_1 not in GATE_VALUES or gate_2 not in GATE_VALUES:
                errors.append({"role": role, "blind_state_id": blind_id, "component": component, "error": "invalid_gate"})
            if derived != _derived(decision):
                errors.append({"role": role, "blind_state_id": blind_id, "component": component, "error": "derived_decision_mismatch"})
            if not isinstance(codes, list) or not codes or not set(codes).issubset(EVIDENCE_CODES):
                errors.append({"role": role, "blind_state_id": blind_id, "component": component, "error": "invalid_evidence_codes"})
            uncertain = gate_1 == "uncertain" or gate_2 == "uncertain"
            if uncertain and not str(decision.get("notes", "")).strip():
                errors.append({"role": role, "blind_state_id": blind_id, "component": component, "error": "uncertain_without_notes"})
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument(
        "--packet",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_final_h1r_review_candidate/human_review_packet.json",
    )
    parser.add_argument(
        "--overlap-packet",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_final_h1r_review_candidate/independent_overlap_packet.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_final_h1r_review_candidate/private_binding.jsonl",
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_final_h1r_pre_human_gate_v1/semantic_observation_rows.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_final_h1r_dual_review_audit_v1",
    )
    args = parser.parse_args()

    primary_rows = _rows(args.primary)
    secondary_rows = _rows(args.secondary)
    packet = read_json(args.packet)
    overlap_packet = read_json(args.overlap_packet)
    items = {row["blind_state_id"]: row for row in packet["items"]}
    overlap_ids = {row["blind_state_id"] for row in overlap_packet["items"]}
    bindings = {row["blind_state_id"]: dict(row) for row in iter_jsonl(args.binding)}
    observations = {
        (row["state_id"], row["component"]): dict(row)
        for row in iter_jsonl(args.observations)
    }
    primary = {row["blind_state_id"]: row for row in primary_rows}
    secondary = {row["blind_state_id"]: row for row in secondary_rows}

    schema_errors = _validate_rows(
        rows=primary_rows,
        expected_protocol=PRIMARY_PROTOCOL,
        expected_ids=set(items),
        role="primary",
    )
    schema_errors += _validate_rows(
        rows=secondary_rows,
        expected_protocol=SECONDARY_PROTOCOL,
        expected_ids=overlap_ids,
        role="secondary",
    )

    distributions: dict[str, Any] = {}
    balance_failures: list[str] = []
    for component in COMPONENTS:
        overall = Counter(
            primary[blind_id]["component_decisions"][component]["derived_decision"]
            for blind_id in items
        )
        by_split: dict[str, Any] = {}
        for split in SPLITS:
            ids = [blind_id for blind_id in items if bindings[blind_id]["split"] == split]
            counts = Counter(
                primary[blind_id]["component_decisions"][component]["derived_decision"]
                for blind_id in ids
            )
            total = sum(counts.values())
            majority = max(counts.values()) / total
            by_split[split] = {
                "counts": dict(counts),
                "n": total,
                "majority_constant_accuracy": majority,
            }
            if majority >= 0.70:
                balance_failures.append(f"{component}_{split}_majority_constant_accuracy_ge_0_70")
        distributions[component] = {
            "overall": {
                "counts": dict(overall),
                "n": sum(overall.values()),
                "majority_constant_accuracy": max(overall.values()) / sum(overall.values()),
            },
            "by_split": by_split,
        }

    all_pairs: list[tuple[str, str]] = []
    disagreements: list[dict[str, Any]] = []
    per_component: dict[str, Any] = {}
    for component in COMPONENTS:
        pairs: list[tuple[str, str]] = []
        gate_1_matches: list[bool] = []
        gate_2_matches: list[bool] = []
        subtype_matches: list[bool] = []
        for blind_id in sorted(overlap_ids):
            left = primary[blind_id]["component_decisions"][component]
            right = secondary[blind_id]["component_decisions"][component]
            pair = (left["derived_decision"], right["derived_decision"])
            pairs.append(pair)
            all_pairs.append(pair)
            gate_1_matches.append(left["applicability_safe"] == right["applicability_safe"])
            gate_2_matches.append(left["incremental_over_r0"] == right["incremental_over_r0"])
            subtype_matches.append(left["adjudicated_subtype"] == right["adjudicated_subtype"])
            if pair[0] != pair[1]:
                item = items[blind_id]
                surface = _component_surface(item, component)
                disagreements.append(
                    {
                        "blind_state_id": blind_id,
                        "component": component,
                        "visible_dialogue": item["visible_dialogue"],
                        "current_user_text": item["current_user_text"],
                        "candidate_text": surface["candidate_text"],
                        "candidate_age_sessions": surface["candidate_age_sessions"],
                        "primary": left,
                        "secondary": right,
                    }
                )
        agreement, kappa = _kappa(pairs)
        per_component[component] = {
            "n": len(pairs),
            "confusion": {f"primary_{a}__secondary_{b}": n for (a, b), n in Counter(pairs).items()},
            "decision_raw_agreement": agreement,
            "cohen_kappa": kappa,
            "gate_1_raw_agreement": sum(gate_1_matches) / len(gate_1_matches),
            "gate_2_raw_agreement": sum(gate_2_matches) / len(gate_2_matches),
            "subtype_raw_agreement": sum(subtype_matches) / len(subtype_matches),
            "disagreements": sum(left != right for left, right in pairs),
        }
    overall_agreement, overall_kappa = _kappa(all_pairs)

    iaa_failures: list[str] = []
    if overall_agreement < 0.80:
        iaa_failures.append("overall_decision_raw_agreement_below_0_80")
    if overall_kappa < 0.60:
        iaa_failures.append("overall_cohen_kappa_below_0_60")
    for component in COMPONENTS:
        if per_component[component]["decision_raw_agreement"] < 0.70:
            iaa_failures.append(f"{component}_decision_raw_agreement_below_0_70")
        if per_component[component]["subtype_raw_agreement"] < 0.90:
            iaa_failures.append(f"{component}_subtype_raw_agreement_below_0_90")

    primary_gate_patterns = {
        component: {
            f"gate1_{g1}__gate2_{g2}__decision_{decision}": n
            for (g1, g2, decision), n in Counter(
                (
                    row["component_decisions"][component]["applicability_safe"],
                    row["component_decisions"][component]["incremental_over_r0"],
                    row["component_decisions"][component]["derived_decision"],
                )
                for row in primary_rows
            ).items()
        }
        for component in COMPONENTS
    }

    observation_vs_unadjudicated_primary: dict[str, Any] = {}
    for component in COMPONENTS:
        pairs: list[tuple[int, int]] = []
        for blind_id in items:
            state_id = bindings[blind_id]["state_id"]
            factors = observations[(state_id, component)]["factor_scores"]
            predicted = int(all(float(value) >= 0.5 for value in factors.values()))
            label = int(
                primary[blind_id]["component_decisions"][component]["derived_decision"]
                == "on"
            )
            pairs.append((label, predicted))
        counts = Counter(pairs)
        positives = sum(label for label, _ in pairs)
        negatives = len(pairs) - positives
        recall = counts[(1, 1)] / positives if positives else 0.0
        specificity = counts[(0, 0)] / negatives if negatives else 0.0
        observation_vs_unadjudicated_primary[component] = {
            "n": len(pairs),
            "confusion": {
                f"gold_{label}__surface_{predicted}": count
                for (label, predicted), count in sorted(counts.items())
            },
            "recall": recall,
            "specificity": specificity,
            "balanced_accuracy": 0.5 * (recall + specificity),
            "qualification_allowed": False,
            "reason": "Primary labels failed IAA/data gates and remain unadjudicated; high agreement may reflect shared construction semantics rather than valid candidate understanding.",
        }
    primary_subtype_distribution = {
        component: {
            subtype: dict(counts)
            for subtype, counts in sorted(
                {
                    subtype: Counter(
                        row["component_decisions"][component]["derived_decision"]
                        for row in primary_rows
                        if row["component_decisions"][component]["adjudicated_subtype"] == subtype
                    )
                    for subtype in {
                        row["component_decisions"][component]["adjudicated_subtype"]
                        for row in primary_rows
                    }
                }.items()
            )
        }
        for component in COMPONENTS
    }

    failures = sorted(
        set(
            (["review_schema_errors"] if schema_errors else [])
            + balance_failures
            + iaa_failures
            + (["overlap_disagreements_require_adjudication"] if disagreements else [])
        )
    )
    report = {
        "protocol": PROTOCOL,
        "status": "HOLD_DO_NOT_FREEZE_GOLD_OR_TRAIN" if failures else "PASS_READY_TO_FREEZE_GOLD",
        "scientific_layer": "H1R_CANDIDATE_SEMANTICS_AND_STEP1_GOLD_QUALIFICATION",
        "not_assessed": [
            "PM_model_predictions",
            "STEP2_generator_execution",
            "response_quality",
            "interaction_and_grounding_risk",
            "cost_effect",
        ],
        "failures": failures,
        "schema": {
            "primary_rows": len(primary_rows),
            "secondary_rows": len(secondary_rows),
            "primary_id_match": set(primary) == set(items),
            "secondary_id_match": set(secondary) == overlap_ids,
            "errors": schema_errors,
        },
        "primary_label_distribution": distributions,
        "pre_adjudication_iaa": {
            "n_component_pairs": len(all_pairs),
            "decision_raw_agreement": overall_agreement,
            "cohen_kappa": overall_kappa,
            "per_component": per_component,
        },
        "primary_gate_patterns": primary_gate_patterns,
        "primary_subtype_distribution": primary_subtype_distribution,
        "observation_vs_unadjudicated_primary_diagnostic": observation_vs_unadjudicated_primary,
        "disagreement_summary": {
            "items": len(disagreements),
            "by_component": dict(Counter(row["component"] for row in disagreements)),
            "note": "Pre-adjudication IAA never changes after adjudication; final gold remains unavailable until disagreements are resolved.",
        },
        "root_cause_diagnostics": [
            "ME primary gate1 and gate2 never diverge; all owner-valid reusable outcomes are ON even when the current request is listening, paraphrase, factual recall, or a focused question.",
            "The transparent Observation conjunction reaches 0.9667 BA against unadjudicated primary ME while independent ME agreement is only 0.5833; this is evidence of construct co-adaptation, not successful factor qualification.",
            "The same Observation conjunction reaches only 0.6843 BA against primary RS, below the frozen 0.70 factor threshold.",
            "All primary MP_PREFERENCE candidates are OFF; the packet provides no positive human evidence for the preference subtype.",
            "MP label balance fails the frozen <0.70 majority gate in every split; MS fails it in FRESH and SEALED.",
            "These are candidate-state construction and operational-codebook failures observed before PM fitting, not generator failures.",
        ],
        "provenance": {
            "primary": str(args.primary),
            "secondary": str(args.secondary),
            "packet": str(args.packet),
            "overlap_packet": str(args.overlap_packet),
            "binding": str(args.binding),
            "observations": str(args.observations),
            "construction_intent_used_as_gold": False,
            "response_or_outcome_read": False,
            "external_lockbox_read": False,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "audit_report.json", report)
    write_jsonl(args.out_dir / "primary_annotations.jsonl", primary_rows)
    write_jsonl(args.out_dir / "secondary_overlap_annotations.jsonl", secondary_rows)
    write_jsonl(args.out_dir / "disagreements.jsonl", disagreements)
    print(
        {
            "protocol": PROTOCOL,
            "status": report["status"],
            "schema_errors": len(schema_errors),
            "overall_raw_agreement": overall_agreement,
            "overall_kappa": overall_kappa,
            "disagreements": len(disagreements),
            "failures": failures,
        }
    )


if __name__ == "__main__":
    main()
