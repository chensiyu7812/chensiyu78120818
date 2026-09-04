#!/usr/bin/env python3
"""Aggregate the one V5 FIT panel and train four frozen grouped logistic heads."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl
from metacom_pm.v1_5_itt_policy import validate_pre_injection_feature_names
from metacom_pm.v1_5b_policy_runtime import COMPONENTS


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5-fit-outcome-aggregation-and-training-v1"
QUALITY_PROTOCOL = "pm-v1.5-v5-fit-outcome-quality-blind-v1"
RISK_PROTOCOL = "pm-v1.5-v5-fit-outcome-grounding-risk-v1"
QUALITY_LABELS = {"A", "B", "tie", "uncertain"}
QUALITY_CRITERIA = {
    "grounded_context_fidelity",
    "emotional_understanding",
    "request_and_dialogue_fit",
    "immediate_helpfulness",
    "clarity_naturalness_not_overloaded",
    "materially_equivalent",
    "uncertain",
}
RISK_LABELS = {"yes", "no", "uncertain"}
RISK_CATEGORIES = {
    "stale_or_conflicting_use",
    "unsupported_personal_claim",
    "overgeneralized_pattern_or_cause",
    "fabricated_recall",
    "explicit_boundary_violation",
    "excessive_directiveness",
    "internal_resource_label_exposure",
}

# One independently exported annotation transposed two characters in the opaque
# blind ID.  The response pair, rationale, and the primary annotation uniquely
# identify the intended packet row.  This is an input-identity repair only; it
# does not alter an outcome label, feature, split, model, or threshold.
QUALITY_ANNOTATION_ID_CORRECTIONS = {
    "v5q_98fbe44981388c41d4963a4a": "v5q_98f5284445ac8beb311a2d6f",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _index(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result = {str(row[key]): dict(row) for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate annotation identity: {key}")
    return result


def _validate_quality(
    path: Path, expected_ids: set[str]
) -> dict[str, dict[str, Any]]:
    rows = _rows(path)
    for row in rows:
        item_id = str(row.get("blind_item_id") or "")
        corrected = QUALITY_ANNOTATION_ID_CORRECTIONS.get(item_id)
        if corrected and corrected in expected_ids and item_id not in expected_ids:
            row["blind_item_id"] = corrected
    indexed = _index(rows, "blind_item_id")
    if set(indexed) != expected_ids:
        raise RuntimeError(f"quality annotation IDs do not match packet: {path}")
    for item_id, row in indexed.items():
        if row.get("protocol") != QUALITY_PROTOCOL:
            raise RuntimeError(f"quality protocol drift: {item_id}")
        label = str(row.get("quality_preference") or "")
        criterion = str(row.get("decisive_criterion") or "")
        notes = str(row.get("quality_notes") or "").strip()
        if label not in QUALITY_LABELS or criterion not in QUALITY_CRITERIA:
            raise RuntimeError(f"invalid quality annotation: {item_id}")
        if label in {"A", "B", "uncertain"} and not notes:
            raise RuntimeError(f"quality rationale missing: {item_id}")
        if not str(row.get("annotator_id") or "").strip():
            raise RuntimeError(f"quality annotator missing: {item_id}")
    return indexed


def _validate_risk(
    path: Path, expected_ids: set[str]
) -> dict[str, dict[str, Any]]:
    rows = _rows(path)
    indexed = _index(rows, "risk_item_id")
    if set(indexed) != expected_ids:
        raise RuntimeError(f"risk annotation IDs do not match packet: {path}")
    for item_id, row in indexed.items():
        if row.get("protocol") != RISK_PROTOCOL:
            raise RuntimeError(f"risk protocol drift: {item_id}")
        risk = str(row.get("any_material_risk") or "")
        functional = str(row.get("resource_functionally_contributed") or "")
        notes = str(row.get("review_notes") or "").strip()
        if risk not in RISK_LABELS or functional not in RISK_LABELS:
            raise RuntimeError(f"invalid risk/functional annotation: {item_id}")
        categories = list(row.get("selected_categories") or [])
        evidence = dict(row.get("evidence_by_category") or {})
        if risk == "yes":
            if not categories or not set(categories) <= RISK_CATEGORIES:
                raise RuntimeError(f"risk category missing or invalid: {item_id}")
            for category in categories:
                record = dict(evidence.get(category) or {})
                if not all(
                    str(record.get(name) or "").strip()
                    for name in (
                        "literal_response_excerpt",
                        "literal_context_or_evidence_excerpt",
                        "materiality_reason",
                    )
                ):
                    raise RuntimeError(f"risk evidence incomplete: {item_id}/{category}")
        if risk == "uncertain" and not notes:
            raise RuntimeError(f"risk uncertainty rationale missing: {item_id}")
        if functional == "yes" and not str(
            row.get("functional_evidence_excerpt") or ""
        ).strip():
            raise RuntimeError(f"functional evidence missing: {item_id}")
        if functional in {"no", "uncertain"} and not notes:
            raise RuntimeError(f"functional rationale missing: {item_id}")
        if not str(row.get("annotator_id") or "").strip():
            raise RuntimeError(f"risk annotator missing: {item_id}")
    return indexed


def _annotator_ids(rows: Mapping[str, Mapping[str, Any]]) -> set[str]:
    return {str(row["annotator_id"]).strip() for row in rows.values()}


def _agreement(
    primary: Mapping[str, Mapping[str, Any]],
    secondary: Mapping[str, Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    ids = sorted(secondary)
    left = [str(primary[item_id][field]) for item_id in ids]
    right = [str(secondary[item_id][field]) for item_id in ids]
    raw = float(np.mean(np.asarray(left) == np.asarray(right))) if ids else 1.0
    return {
        "n": len(ids),
        "raw_agreement": raw,
        "cohen_kappa": float(cohen_kappa_score(left, right)) if ids else 1.0,
        "disagreement_ids": [
            item_id
            for item_id, a, b in zip(ids, left, right, strict=True)
            if a != b
        ],
    }


def _apply_adjudication(
    *,
    quality: dict[str, dict[str, Any]],
    risk: dict[str, dict[str, Any]],
    disagreement_ids: set[str],
    adjudication_path: Path | None,
    out_dir: Path,
) -> bool:
    if not disagreement_ids:
        return True
    if adjudication_path is None or not adjudication_path.is_file():
        blank = [
            {
                "protocol": "pm-v1.5-v5-fit-overlap-adjudication-v1",
                "item_id": item_id,
                "final_value": None,
                "adjudication_notes": "",
                "adjudicator_id": "",
            }
            for item_id in sorted(disagreement_ids)
        ]
        write_jsonl(out_dir / "overlap_disagreements_adjudication_blank.jsonl", blank)
        return False
    rows = _index(_rows(adjudication_path), "item_id")
    if set(rows) != disagreement_ids:
        raise RuntimeError("adjudication IDs do not match overlap disagreements")
    for item_id, row in rows.items():
        value = str(row.get("final_value") or "")
        if item_id in quality:
            if value not in QUALITY_LABELS:
                raise RuntimeError(f"invalid adjudicated quality value: {item_id}")
            quality[item_id]["quality_preference"] = value
        elif item_id in risk:
            # Risk disagreements are encoded as `risk|functional` so both
            # constructs are resolved in one record without silently merging.
            parts = value.split("|")
            if len(parts) != 2 or any(part not in RISK_LABELS for part in parts):
                raise RuntimeError(f"invalid adjudicated risk value: {item_id}")
            risk[item_id]["any_material_risk"] = parts[0]
            risk[item_id]["resource_functionally_contributed"] = parts[1]
        if not str(row.get("adjudication_notes") or "").strip() or not str(
            row.get("adjudicator_id") or ""
        ).strip():
            raise RuntimeError(f"adjudication rationale or identity missing: {item_id}")
    return True


def _quality_arm(label: str, key: Mapping[str, Any]) -> str:
    if label == "A":
        return str(key["a_arm"])
    if label == "B":
        return str(key["b_arm"])
    return label


def _derive_state_labels(
    *,
    private_key: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Mapping[str, Any]],
    risk: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in private_key:
        grouped[str(row["state_id"])].append(row)
    if any(len(rows) != 2 for rows in grouped.values()):
        raise RuntimeError("expected two frozen seed rows per state")
    results: list[dict[str, Any]] = []
    for state_id, rows in sorted(grouped.items()):
        component = str(rows[0]["component"])
        if any(str(row["component"]) != component for row in rows):
            raise RuntimeError(f"state component drift: {state_id}")
        winners = [
            _quality_arm(
                str(quality[str(row["blind_item_id"])]["quality_preference"]), row
            )
            for row in rows
        ]
        risks = [
            str(risk[str(row["risk_item_id"])]["any_material_risk"])
            for row in rows
        ]
        functional = [
            str(risk[str(row["risk_item_id"])]["resource_functionally_contributed"])
            for row in rows
        ]
        quality_positive = (
            "ON" in winners
            and "OFF" not in winners
            and "uncertain" not in winners
        )
        risk_veto = "yes" in risks
        uncertainty_veto = "uncertain" in risks or "uncertain" in winners
        label = int(quality_positive and not risk_veto and not uncertainty_veto)
        results.append(
            {
                "protocol": PROTOCOL,
                "state_id": state_id,
                "component": component,
                "semantic_group_id": str(rows[0]["semantic_group_id"]),
                "semantic_family": str(rows[0]["semantic_family"]),
                "seed_quality_winners": winners,
                "seed_material_risk": risks,
                "seed_functional_use": functional,
                "seed_on_fallback_used": [bool(row["on_fallback_used"]) for row in rows],
                "hard_worth_opening": label,
                "sensitivity_excluded_for_uncertainty": uncertainty_veto,
                "material_risk_veto": risk_veto,
                "functional_use_is_secondary_only": True,
            }
        )
    return results


def _transparent_rule(component: str, f: Mapping[str, float]) -> int:
    if component == "MP":
        preference = f["candidate_preference_scope_fit"] >= 0.5
        profile = (
            f["candidate_profile_relevance"] >= 0.5
            and f["candidate_profile_entity_scope_fit"] >= 0.5
        )
        return int(
            f["candidate_incremental_information"] >= 0.5
            and f["candidate_current_scope_conflict"] < 0.5
            and (preference or profile)
        )
    if component == "MS":
        return int(
            f["candidate_incremental_information"] >= 0.5
            and f["candidate_prior_issue_marked_resolved"] < 0.5
            and f["candidate_current_goal_fit"] >= 0.5
            and (
                f["candidate_prior_outcome_or_distinction"] >= 0.5
                or f["candidate_specific_issue_or_distinction"] >= 0.5
            )
        )
    if component == "ME":
        return int(
            f["candidate_incremental_information"] >= 0.5
            and f["candidate_current_goal_fit"] >= 0.5
            and f["candidate_contains_action"] >= 0.5
            and (
                f["candidate_contains_result"] >= 0.5
                or f["candidate_contains_mechanism"] >= 0.5
            )
        )
    return int(
        all(
            f[name] >= 0.5
            for name in (
                "candidate_mode_fit",
                "candidate_goal_fit",
                "candidate_burden_fit",
                "candidate_boundary_fit",
                "candidate_nonredundancy",
            )
        )
    )


def _metric(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "recall": float(recall_score(y, prediction, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y, prediction, pos_label=0, zero_division=0)),
        "brier": float(brier_score_loss(y, probability)),
        "predicted_on_fraction": float(np.mean(prediction)),
        "predicted_off_fraction": float(np.mean(1 - prediction)),
    }


def _fit_component(
    *, rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> dict[str, Any]:
    names = list(rows[0]["model_features"])
    validate_pre_injection_feature_names(names)
    x = np.asarray(
        [[float(row["model_features"][name]) for name in names] for row in rows],
        dtype=float,
    )
    y = np.asarray([int(row["hard_worth_opening"]) for row in rows], dtype=int)
    groups = np.asarray([str(row["semantic_group_id"]) for row in rows])
    families = np.asarray([str(row["semantic_family"]) for row in rows])
    if len(np.unique(y)) != 2:
        return {
            "status": "NOT_LEARNABLE_SINGLE_CLASS",
            "feature_names": names,
            "positive_units": int(y.sum()),
            "nonpositive_units": int(len(y) - y.sum()),
        }
    seeds = list(contract["primary_model"]["fold_seeds"])
    seed_metrics: list[dict[str, float]] = []
    all_probabilities: list[np.ndarray] = []
    for seed in seeds:
        probability = np.zeros(len(y), dtype=float)
        splitter = StratifiedGroupKFold(
            n_splits=int(contract["primary_model"]["folds"]),
            shuffle=True,
            random_state=int(seed),
        )
        for train, test in splitter.split(x, y, groups):
            scaler = StandardScaler().fit(x[train])
            model = LogisticRegression(
                C=float(contract["primary_model"]["C"]),
                class_weight=str(contract["primary_model"]["class_weight"]),
                solver=str(contract["primary_model"]["solver"]),
                max_iter=2000,
                random_state=int(seed),
            )
            model.fit(scaler.transform(x[train]), y[train])
            probability[test] = model.predict_proba(scaler.transform(x[test]))[:, 1]
        seed_metrics.append(_metric(y, probability))
        all_probabilities.append(probability)

    prevalence_probability = np.full(len(y), float(np.mean(y)))
    rule_probability = np.asarray(
        [_transparent_rule(str(row["component"]), row["model_features"]) for row in rows],
        dtype=float,
    )
    prevalence_brier = float(brier_score_loss(y, prevalence_probability))
    rule_brier = float(brier_score_loss(y, rule_probability))

    leave_family_probability = np.zeros(len(y), dtype=float)
    for family in sorted(set(families)):
        train = np.flatnonzero(families != family)
        test = np.flatnonzero(families == family)
        if len(np.unique(y[train])) < 2:
            leave_family_probability[test] = float(np.mean(y[train]))
            continue
        scaler = StandardScaler().fit(x[train])
        model = LogisticRegression(
            C=float(contract["primary_model"]["C"]),
            class_weight=str(contract["primary_model"]["class_weight"]),
            solver=str(contract["primary_model"]["solver"]),
            max_iter=2000,
            random_state=int(seeds[0]),
        ).fit(scaler.transform(x[train]), y[train])
        leave_family_probability[test] = model.predict_proba(
            scaler.transform(x[test])
        )[:, 1]
    leave_family = _metric(y, leave_family_probability)

    gates = contract["promotion_gates"]
    ba_values = [row["balanced_accuracy"] for row in seed_metrics]
    recall_values = [row["recall"] for row in seed_metrics]
    specificity_values = [row["specificity"] for row in seed_metrics]
    brier_values = [row["brier"] for row in seed_metrics]
    on_values = [row["predicted_on_fraction"] for row in seed_metrics]
    off_values = [row["predicted_off_fraction"] for row in seed_metrics]
    positive_groups = len({groups[i] for i in np.flatnonzero(y == 1)})
    nonpositive_groups = len({groups[i] for i in np.flatnonzero(y == 0)})
    checks = {
        "worst_seed_balanced_accuracy": min(ba_values)
        >= float(gates["grouped_oof_balanced_accuracy_min"]),
        "worst_seed_recall": min(recall_values) >= float(gates["recall_min"]),
        "worst_seed_specificity": min(specificity_values)
        >= float(gates["specificity_min"]),
        "mean_brier_beats_prevalence": float(np.mean(brier_values))
        < prevalence_brier,
        "mean_brier_beats_transparent_rule": float(np.mean(brier_values))
        < rule_brier,
        "predicted_on_fraction": min(on_values)
        >= float(gates["predicted_on_fraction_min"]),
        "predicted_off_fraction": min(off_values)
        >= float(gates["predicted_off_fraction_min"]),
        "independent_positive_groups": positive_groups
        >= int(gates["independent_positive_groups_min"]),
        "independent_nonpositive_groups": nonpositive_groups
        >= int(gates["independent_nonpositive_groups_min"]),
        "five_seed_ba_sd": float(np.std(ba_values))
        <= float(gates["five_seed_balanced_accuracy_sd_max"]),
        "leave_semantic_family_out_ba": leave_family["balanced_accuracy"]
        >= float(gates["leave_semantic_family_out_balanced_accuracy_min"]),
    }

    scaler = StandardScaler().fit(x)
    final_model = LogisticRegression(
        C=float(contract["primary_model"]["C"]),
        class_weight=str(contract["primary_model"]["class_weight"]),
        solver=str(contract["primary_model"]["solver"]),
        max_iter=2000,
        random_state=int(seeds[0]),
    ).fit(scaler.transform(x), y)
    return {
        "status": "FIT_PROMOTED_TO_FRESH_CONFIRMATION" if all(checks.values()) else "NOT_LEARNED_ON_FROZEN_FIT",
        "feature_names": names,
        "positive_units": int(y.sum()),
        "nonpositive_units": int(len(y) - y.sum()),
        "positive_groups": positive_groups,
        "nonpositive_groups": nonpositive_groups,
        "seed_metrics": seed_metrics,
        "five_seed_ba_mean": float(np.mean(ba_values)),
        "five_seed_ba_sd": float(np.std(ba_values)),
        "prevalence_brier": prevalence_brier,
        "transparent_rule_brier": rule_brier,
        "transparent_rule_metrics": _metric(y, rule_probability),
        "leave_semantic_family_out": leave_family,
        "promotion_checks": checks,
        "final_model": {
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "coefficients": final_model.coef_[0].tolist(),
            "intercept": float(final_model.intercept_[0]),
            "threshold": float(contract["primary_model"]["threshold"]),
        },
    }


def main() -> None:
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V5 requires {FORMAL_PYTHON}; got {sys.executable}")
    parser = argparse.ArgumentParser(description=__doc__)
    panel = ROOT / "outputs/pm_v1_5_v5_single_full_fit_outcome_review_v1"
    parser.add_argument("--primary-quality", type=Path, required=True)
    parser.add_argument("--primary-risk", type=Path, required=True)
    parser.add_argument("--overlap-quality", type=Path, required=True)
    parser.add_argument("--overlap-risk", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--panel-dir", type=Path, default=panel)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_fit_training_surface_v1",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "data/pm_v1_5_contracts/v5_fit_training_surface_freeze_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_fit_training_result_v1",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    key_rows = _rows(args.panel_dir / "private_blind_key.jsonl")
    quality_ids = {str(row["blind_item_id"]) for row in key_rows}
    risk_ids = {str(row["risk_item_id"]) for row in key_rows}
    overlap_quality_ids = {
        str(row["blind_item_id"]) for row in key_rows if row["in_independent_overlap"]
    }
    overlap_risk_ids = {
        str(row["risk_item_id"]) for row in key_rows if row["in_independent_overlap"]
    }
    quality = _validate_quality(args.primary_quality, quality_ids)
    risk = _validate_risk(args.primary_risk, risk_ids)
    overlap_quality = _validate_quality(args.overlap_quality, overlap_quality_ids)
    overlap_risk = _validate_risk(args.overlap_risk, overlap_risk_ids)
    if _annotator_ids(quality) & _annotator_ids(overlap_quality):
        raise RuntimeError("quality overlap reviewer is not independent")
    if _annotator_ids(risk) & _annotator_ids(overlap_risk):
        raise RuntimeError("risk overlap reviewer is not independent")

    quality_agreement = _agreement(
        quality, overlap_quality, "quality_preference"
    )
    risk_agreement = _agreement(risk, overlap_risk, "any_material_risk")
    functional_agreement = _agreement(
        risk, overlap_risk, "resource_functionally_contributed"
    )
    disagreements = set(quality_agreement["disagreement_ids"])
    disagreements.update(risk_agreement["disagreement_ids"])
    disagreements.update(functional_agreement["disagreement_ids"])
    resolved = _apply_adjudication(
        quality=quality,
        risk=risk,
        disagreement_ids=disagreements,
        adjudication_path=args.adjudication,
        out_dir=args.out_dir,
    )
    if not resolved:
        report = {
            "protocol": PROTOCOL,
            "status": "AWAITING_ONE_CONSOLIDATED_OVERLAP_ADJUDICATION",
            "quality_agreement": quality_agreement,
            "risk_agreement": risk_agreement,
            "functional_agreement": functional_agreement,
            "disagreement_items": len(disagreements),
            "additional_small_packets_allowed": False,
        }
        write_json(args.out_dir / "fit_report.json", report)
        print(report)
        return

    labels = _derive_state_labels(
        private_key=key_rows, quality=quality, risk=risk
    )
    if len(labels) != 256:
        raise RuntimeError("expected 256 state-level FIT labels")
    feature_rows = _rows(args.feature_dir / "fit_feature_rows_private.jsonl")
    feature_by_state = {str(row["state_id"]): row for row in feature_rows}
    if set(feature_by_state) != {str(row["state_id"]) for row in labels}:
        raise RuntimeError("feature and outcome state identities differ")
    joined = [
        {
            **feature_by_state[str(label["state_id"])],
            **label,
        }
        for label in labels
    ]
    contract = read_json(args.contract)
    heads = {
        component: _fit_component(
            rows=[row for row in joined if row["component"] == component],
            contract=contract,
        )
        for component in COMPONENTS
    }
    label_path = args.out_dir / "state_effect_labels_private.jsonl"
    write_jsonl(label_path, labels)
    report = {
        "protocol": PROTOCOL,
        "status": (
            "ALL_FOUR_HEADS_READY_FOR_FRESH_CONFIRMATION"
            if all(
                row.get("status") == "FIT_PROMOTED_TO_FRESH_CONFIRMATION"
                for row in heads.values()
            )
            else "PARTIAL_OR_NOT_LEARNED_ON_FROZEN_FIT"
        ),
        "quality_agreement": quality_agreement,
        "risk_agreement": risk_agreement,
        "functional_agreement": functional_agreement,
        "state_effect_distribution": {
            component: dict(
                Counter(
                    str(row["hard_worth_opening"])
                    for row in labels
                    if row["component"] == component
                )
            )
            for component in COMPONENTS
        },
        "heads": heads,
        "input_sha256": {
            "primary_quality": sha256_file(args.primary_quality),
            "primary_risk": sha256_file(args.primary_risk),
            "overlap_quality": sha256_file(args.overlap_quality),
            "overlap_risk": sha256_file(args.overlap_risk),
            "private_blind_key": sha256_file(
                args.panel_dir / "private_blind_key.jsonl"
            ),
            "feature_rows": sha256_file(
                args.feature_dir / "fit_feature_rows_private.jsonl"
            ),
            "training_contract": sha256_file(args.contract),
        },
        "state_effect_labels_sha256": sha256_file(label_path),
        "external_lockbox_read": False,
        "post_label_method_change_allowed": False,
        "additional_small_packets_allowed": False,
    }
    write_json(args.out_dir / "fit_report.json", report)
    print({"protocol": PROTOCOL, "status": report["status"]})


if __name__ == "__main__":
    main()
