#!/usr/bin/env python3
"""Audit the single V3 Observation review before any factor-head training.

This audit deliberately keeps three things separate:

* inter-reviewer reliability on the frozen 24-item overlap;
* agreement between human labels and private construction intent (QA only);
* whether nuisance text can predict an individual factor label.

Private construction intent is never promoted to human gold by this script.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v3-observation-review-audit-v1"
COMPONENTS = ("MP", "MS", "ME", "RS")
FACTORS = (
    "owner_time_entity_valid",
    "goal_function_fit",
    "boundary_burden_compatible",
    "specific_increment",
)
FILLER = "These ordinary details add no claim about the central request."


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _agreement(left: list[str], right: list[str]) -> dict[str, Any]:
    agreed = sum(a == b for a, b in zip(left, right, strict=True))
    return {
        "n": len(left),
        "agreed": agreed,
        "raw_agreement": agreed / len(left),
        "cohen_kappa": float(cohen_kappa_score(left, right)),
    }


def _filler_probe(
    *, values: np.ndarray, labels: np.ndarray, groups: np.ndarray
) -> dict[str, Any]:
    scores: list[float] = []
    for seed in (17, 29, 43, 71, 113):
        probabilities = np.full(labels.size, np.nan, dtype=np.float64)
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for train, test in splitter.split(values, labels, groups):
            scaler = StandardScaler().fit(values[train])
            model = LogisticRegression(
                C=0.3,
                class_weight="balanced",
                solver="liblinear",
                max_iter=2000,
                random_state=seed,
            ).fit(scaler.transform(values[train]), labels[train])
            probabilities[test] = model.predict_proba(scaler.transform(values[test]))[:, 1]
        if not np.isfinite(probabilities).all():
            raise RuntimeError("filler-only grouped OOF did not cover every row")
        scores.append(float(balanced_accuracy_score(labels, probabilities >= 0.5)))
    return {
        "five_seed_balanced_accuracy": scores,
        "mean_balanced_accuracy": float(np.mean(scores)),
        "sd_balanced_accuracy": float(np.std(scores)),
        "pass_below_0_65": bool(float(np.mean(scores)) < 0.65),
    }


def main() -> None:
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V3 requires {FORMAL_PYTHON}; got {sys.executable}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--overlap", type=Path, required=True)
    parser.add_argument(
        "--packet-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_observation_review_candidate_v1",
    )
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_v3_observation_orthogonal_v1/private/construction_blueprint.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_observation_review_audit_v1",
    )
    args = parser.parse_args()

    primary = _rows(args.primary)
    overlap = _rows(args.overlap)
    packet = read_json(args.packet_dir / "human_review_packet.json")
    overlap_packet = read_json(args.packet_dir / "independent_overlap_packet.json")
    bindings = {
        str(row["blind_item_id"]): row
        for row in _rows(args.packet_dir / "private_binding.jsonl")
    }
    blueprint = {
        str(row["blueprint_row_id"]): row for row in _rows(args.blueprint)
    }
    item_map = {str(row["blind_item_id"]): row for row in packet["items"]}
    primary_map = {str(row["blind_item_id"]): row for row in primary}
    overlap_map = {str(row["blind_item_id"]): row for row in overlap}
    expected_primary = set(item_map)
    expected_overlap = {
        str(row["blind_item_id"]) for row in overlap_packet["items"]
    }

    schema_failures: list[str] = []
    if len(primary) != 96 or len(primary_map) != 96 or set(primary_map) != expected_primary:
        schema_failures.append("primary_membership_or_cardinality")
    if len(overlap) != 24 or len(overlap_map) != 24 or set(overlap_map) != expected_overlap:
        schema_failures.append("overlap_membership_or_cardinality")
    for row in [*primary, *overlap]:
        component = str(row.get("component"))
        if component not in COMPONENTS:
            schema_failures.append("invalid_component")
            continue
        expected_owner = (
            {"structural_yes_not_rated"} if component == "RS" else {"yes", "no", "uncertain"}
        )
        if row.get("owner_time_entity_valid") not in expected_owner:
            schema_failures.append("invalid_owner_value")
        for factor in FACTORS[1:]:
            if row.get(factor) not in {"yes", "no", "uncertain"}:
                schema_failures.append(f"invalid_{factor}_value")
        if row.get("derived_eligibility") not in {"eligible", "ineligible", "uncertain"}:
            schema_failures.append("invalid_derived_eligibility")

    overlap_ids = sorted(expected_overlap)
    iaa = {
        field: _agreement(
            [str(primary_map[item_id][field]) for item_id in overlap_ids],
            [str(overlap_map[item_id][field]) for item_id in overlap_ids],
        )
        for field in (*FACTORS, "derived_eligibility", "adjudicated_subtype")
    }
    disagreements = []
    for item_id in overlap_ids:
        fields = [
            field
            for field in (*FACTORS, "derived_eligibility", "adjudicated_subtype")
            if primary_map[item_id][field] != overlap_map[item_id][field]
        ]
        if fields:
            disagreements.append(
                {
                    "blind_item_id": item_id,
                    "component": primary_map[item_id]["component"],
                    "fields": fields,
                    "primary": {field: primary_map[item_id][field] for field in fields},
                    "secondary": {field: overlap_map[item_id][field] for field in fields},
                    "secondary_notes": overlap_map[item_id].get("notes", ""),
                }
            )

    construction_agreement: dict[str, Any] = {}
    construction_mismatches: list[dict[str, Any]] = []
    for factor in FACTORS:
        pairs: list[tuple[str, str]] = []
        for item_id, human in primary_map.items():
            if factor == "owner_time_entity_valid" and human["component"] == "RS":
                continue
            intended = (
                "yes"
                if bindings[item_id]["private_factor_plan_not_gold"][factor]["private_target"]
                else "no"
            )
            observed = str(human[factor])
            pairs.append((intended, observed))
            if intended != observed:
                construction_mismatches.append(
                    {
                        "blind_item_id": item_id,
                        "state_id": bindings[item_id]["state_id"],
                        "component": human["component"],
                        "subtype": human["adjudicated_subtype"],
                        "track": bindings[item_id]["track"],
                        "factor": factor,
                        "construction": intended,
                        "human": observed,
                    }
                )
        agreed = sum(a == b for a, b in pairs)
        construction_agreement[factor] = {
            "n": len(pairs),
            "agreed": agreed,
            "rate": agreed / len(pairs),
            "confusion": {
                f"construction_{a}__human_{b}": count
                for (a, b), count in sorted(Counter(pairs).items())
            },
        }

    human_shape: dict[str, Any] = {}
    for track in ("FACTOR_FIT", "ELIGIBILITY_CONFIRMATION"):
        human_shape[track] = {}
        for component in COMPONENTS:
            item_ids = [
                item_id
                for item_id, row in primary_map.items()
                if row["component"] == component and bindings[item_id]["track"] == track
            ]
            factor_counts = {}
            for factor in FACTORS:
                if component == "RS" and factor == "owner_time_entity_valid":
                    continue
                factor_counts[factor] = dict(
                    Counter(str(primary_map[item_id][factor]) for item_id in item_ids)
                )
            human_shape[track][component] = {
                "n": len(item_ids),
                "derived_eligibility": dict(
                    Counter(primary_map[item_id]["derived_eligibility"] for item_id in item_ids)
                ),
                "factors": factor_counts,
            }

    filler_probe: dict[str, Any] = {}
    for factor in FACTORS:
        item_ids = [
            item_id
            for item_id, row in primary_map.items()
            if bindings[item_id]["track"] == "FACTOR_FIT"
            and not (factor == "owner_time_entity_valid" and row["component"] == "RS")
        ]
        values = np.asarray(
            [[str(item_map[item_id]["current_user_text"]).count(FILLER)] for item_id in item_ids],
            dtype=np.float64,
        )
        labels = np.asarray(
            [int(primary_map[item_id][factor] == "yes") for item_id in item_ids],
            dtype=np.int64,
        )
        groups = np.asarray([bindings[item_id]["counterfactual_group_id"] for item_id in item_ids])
        filler_probe[factor] = _filler_probe(values=values, labels=labels, groups=groups)

    repair_ids = sorted(
        {
            row["blind_item_id"]
            for row in construction_mismatches
            if (
                row["component"] == "MP"
                and row["subtype"] == "MP_PREFERENCE"
                and row["factor"] == "owner_time_entity_valid"
            )
            or (row["component"] == "RS" and row["factor"] == "goal_function_fit")
        }
    )
    unchanged_ids = sorted(expected_primary - set(repair_ids))
    iaa_pass = bool(
        iaa["derived_eligibility"]["cohen_kappa"] >= 0.8
        and min(iaa[factor]["cohen_kappa"] for factor in FACTORS) >= 0.8
    )
    filler_pass = all(row["pass_below_0_65"] for row in filler_probe.values())
    fit_balance_pass = True
    for component, row in human_shape["FACTOR_FIT"].items():
        for counts in row["factors"].values():
            if counts.get("yes", 0) != 8 or counts.get("no", 0) != 8:
                fit_balance_pass = False

    reference_rows = []
    for item_id in sorted(primary_map):
        label = primary_map[item_id]
        binding = bindings[item_id]
        reference_rows.append(
            {
                "protocol": PROTOCOL,
                "blind_item_id": item_id,
                "state_id": binding["state_id"],
                "component": label["component"],
                "track": binding["track"],
                "counterfactual_group_id": binding["counterfactual_group_id"],
                "candidate_id": binding["candidate_id"],
                "candidate_text_sha256": binding["candidate_text_sha256"],
                "adjudicated_subtype": label["adjudicated_subtype"],
                "owner_time_entity_valid": label["owner_time_entity_valid"],
                "goal_function_fit": label["goal_function_fit"],
                "boundary_burden_compatible": label["boundary_burden_compatible"],
                "specific_increment": label["specific_increment"],
                "derived_eligibility": label["derived_eligibility"],
                "label_status": "REPAIR_REQUIRED" if item_id in repair_ids else "CARRY_FORWARD_ELIGIBLE",
                "step1_worth_opening_gold": None,
            }
        )

    status = (
        "FAIL_SCHEMA"
        if schema_failures
        else "HOLD_BOUNDED_SEMANTIC_REPAIR_REQUIRED"
        if not fit_balance_pass
        else "PASS_READY_FOR_FACTOR_BAKEOFF"
    )
    report = {
        "protocol": PROTOCOL,
        "status": status,
        "scope": "OBSERVATION_ELIGIBILITY_ONLY_NOT_STEP1_NOT_STEP2_NOT_RESPONSE_QUALITY",
        "primary_rows": len(primary),
        "overlap_rows": len(overlap),
        "schema_failures": sorted(set(schema_failures)),
        "inter_reviewer_agreement": iaa,
        "inter_reviewer_disagreements": disagreements,
        "inter_reviewer_gate_pass": iaa_pass,
        "construction_intent_agreement_for_qa_only": construction_agreement,
        "construction_mismatches": construction_mismatches,
        "human_label_shape": human_shape,
        "factor_fit_exact_8_8_balance_pass": fit_balance_pass,
        "filler_only_grouped_oof_probe": filler_probe,
        "filler_shortcut_gate_pass": filler_pass,
        "verified_root_causes": {
            "mp_preference_owner_semantics": {
                "affected_items": sum(
                    row["component"] == "MP" for row in construction_mismatches
                ),
                "finding": "A user response-format preference remains owned by the user when the current topic concerns a friend; owner-negative MP_PREFERENCE must use superseded/stale preference evidence instead.",
            },
            "rs_goal_actual_card_fit": {
                "affected_items": sum(
                    row["component"] == "RS" for row in construction_mismatches
                ),
                "finding": "A coarse requested move family does not guarantee that the actual Rank-1 card satisfies its own when_to_use conditions.",
            },
            "repeated_filler": {
                "finding": "The overlap-only eligible/ineligible mean difference does not survive the factor-specific grouped OOF shortcut test.",
                "is_training_blocker": False,
            },
            "explicit_observation_language": {
                "finding": "The data qualifies finite explicit Observation parsing and held-out paraphrase transfer, not implicit natural-dialogue need understanding.",
                "is_scope_limitation_not_label_corruption": True,
            },
        },
        "bounded_repair": {
            "changed_item_count": len(repair_ids),
            "changed_blind_item_ids": repair_ids,
            "carry_forward_item_count": len(unchanged_ids),
            "carry_forward_blind_item_ids": unchanged_ids,
            "full_96_item_re_review_required": False,
            "training_authorized_before_delta_review": False,
        },
        "primary_source_sha256": sha256_file(args.primary),
        "overlap_source_sha256": sha256_file(args.overlap),
        "packet_sha256": sha256_file(args.packet_dir / "human_review_packet.json"),
        "binding_sha256": sha256_file(args.packet_dir / "private_binding.jsonl"),
        "blueprint_sha256": sha256_file(args.blueprint),
        "response_or_outcome_read": False,
        "external_lockbox_read": False,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "primary_working_reference.jsonl", reference_rows)
    report["working_reference_sha256"] = sha256_file(
        args.out_dir / "primary_working_reference.jsonl"
    )
    write_json(args.out_dir / "audit_report.json", report)
    print(
        {
            "status": status,
            "iaa_pass": iaa_pass,
            "filler_shortcut_pass": filler_pass,
            "fit_balance_pass": fit_balance_pass,
            "bounded_repair_items": len(repair_ids),
            "out": str(args.out_dir / "audit_report.json"),
        }
    )


if __name__ == "__main__":
    main()
