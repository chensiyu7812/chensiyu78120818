#!/usr/bin/env python3
"""Finalize the V3 H-Eligibility reference from exact-surface review lineage."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from metacom_pm.io import canonical_json, iter_jsonl, sha256_file, sha256_text, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v3-h-eligibility-reference-v3"
FIELDS = (
    "adjudicated_subtype",
    "owner_time_valid",
    "goal_function_fit",
    "boundary_burden_fit",
    "specific_nonredundant_increment",
    "derived_eligibility",
)
CURRENT_REQUEST_FAMILY = "RS_CURRENT_REQUEST_ALREADY_SPECIFIES_MOVE"
WRONG_OWNER_FAMILY = "MS_WRONG_OWNER_OR_GOAL"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _kappa(left: Sequence[str], right: Sequence[str]) -> float | None:
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    lc, rc = Counter(left), Counter(right)
    labels = set(lc) | set(rc)
    expected = sum((lc[x] / len(left)) * (rc[x] / len(right)) for x in labels)
    return None if expected == 1.0 else (observed - expected) / (1.0 - expected)


def _surface(row: Mapping[str, Any]) -> dict[str, Any]:
    candidate = row["exact_rank1_candidate"]
    return {
        "visible_dialogue": row["visible_dialogue"],
        "current_user_text": row["current_user_text"],
        "component": row["target_component_private_not_model_input"],
        "candidate_text": candidate["candidate_text"],
        "candidate_age_sessions": candidate["candidate_age_sessions"],
    }


def _surface_hash(row: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(_surface(row)))


def _derived(row: Mapping[str, Any]) -> str:
    gates = (
        row["owner_time_valid"],
        row["goal_function_fit"],
        row["boundary_burden_fit"],
        row["specific_nonredundant_increment"],
    )
    if "no" in gates:
        return "ineligible"
    if all(value == "yes" for value in gates):
        return "eligible"
    return "uncertain"


def _agreement(primary: Mapping[str, dict], independent: Mapping[str, dict]) -> dict[str, Any]:
    ids = sorted(primary)
    result: dict[str, Any] = {}
    for field in FIELDS:
        left = [str(primary[item][field]) for item in ids]
        right = [str(independent[item][field]) for item in ids]
        result[field] = {
            "n": len(ids),
            "agreement_count": sum(a == b for a, b in zip(left, right, strict=True)),
            "raw_agreement": sum(a == b for a, b in zip(left, right, strict=True)) / len(ids),
            "cohen_kappa": _kappa(left, right),
            "confusion": {
                f"{a} -> {b}": count
                for (a, b), count in Counter(zip(left, right)).items()
            },
        }
    return result


def _require_formal_python() -> None:
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V3 P2 requires {FORMAL_PYTHON}; got {sys.executable}")


def main() -> None:
    _require_formal_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument(
        "--v1-reference",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_h_eligibility_dual_review_audit_v1/adjudicated_reference_not_gold.jsonl",
    )
    parser.add_argument(
        "--v1-binding",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_h_eligibility_review_candidate/private_binding.jsonl",
    )
    parser.add_argument(
        "--delta-binding",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_h_eligibility_repair_delta_v2_candidate/private_binding.jsonl",
    )
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT / "data/pm_v1_5_v3_effect_blueprint_v1/private/construction_blueprint.jsonl",
    )
    parser.add_argument(
        "--v1-candidates",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_p2_exact_rank1_v1/candidate_rows_private.jsonl",
    )
    parser.add_argument(
        "--v2-candidates",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_p2_exact_rank1_v2/candidate_rows_private.jsonl",
    )
    parser.add_argument(
        "--v3-candidates",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_p2_exact_rank1_v3/candidate_rows_private.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_h_eligibility_reference_v3",
    )
    args = parser.parse_args()

    primary_rows, independent_rows = _rows(args.primary), _rows(args.independent)
    primary = {str(row["blind_item_id"]): row for row in primary_rows}
    independent = {str(row["blind_item_id"]): row for row in independent_rows}
    delta_binding = {str(row["blind_item_id"]): row for row in _rows(args.delta_binding)}
    if (
        len(primary_rows) != 24
        or len(independent_rows) != 24
        or len(primary) != 24
        or len(independent) != 24
        or set(primary) != set(independent)
        or set(primary) != set(delta_binding)
    ):
        raise RuntimeError("delta reviews must be two aligned 24-item files")
    for row in primary_rows + independent_rows:
        if row["derived_eligibility"] != _derived(row):
            raise RuntimeError(f"derived eligibility is inconsistent: {row['blind_item_id']}")
        if any(row[field] == "uncertain" for field in FIELDS[1:]):
            raise RuntimeError("repair delta cannot freeze uncertain gates")

    agreement = _agreement(primary, independent)
    if agreement["derived_eligibility"]["raw_agreement"] != 1.0:
        raise RuntimeError("repair delta final eligibility did not reproduce")

    blueprint = {str(row["blueprint_row_id"]): row for row in _rows(args.blueprint)}
    v1 = {str(row["state_id"]): row for row in _rows(args.v1_candidates)}
    v2 = {str(row["state_id"]): row for row in _rows(args.v2_candidates)}
    v3 = {str(row["state_id"]): row for row in _rows(args.v3_candidates)}
    v1_binding = {str(row["blind_item_id"]): row for row in _rows(args.v1_binding)}
    v1_reference = {
        str(v1_binding[str(row["blind_item_id"])]["state_id"]): row
        for row in _rows(args.v1_reference)
    }
    delta_by_state = {
        str(binding["state_id"]): blind_id for blind_id, binding in delta_binding.items()
    }
    if not (len(v1) == len(v2) == len(v3) == 640):
        raise RuntimeError("candidate versions differ in size")

    final_rows: list[dict[str, Any]] = []
    carry_counts = Counter()
    explicit_adjudications: list[dict[str, Any]] = []
    for state_id, construction in blueprint.items():
        if construction["track"] != "ELIGIBILITY_AUDIT":
            continue
        component = str(construction["target_component"])
        family = str(construction["logic_family"])
        current = v3[state_id]
        if state_id not in delta_by_state:
            if _surface_hash(v1[state_id]) != _surface_hash(current):
                raise RuntimeError(f"unchanged carry-forward surface changed: {state_id}")
            source = dict(v1_reference[state_id])
            basis = "V1_EXACT_DECISION_SURFACE_CARRY_FORWARD"
            reviewer_ids = [str(source["annotator_id"])]
            carry_counts["v1_exact_surface"] += 1
        else:
            blind_id = delta_by_state[state_id]
            source = dict(primary[blind_id])
            reviewer_ids = [
                str(primary[blind_id]["annotator_id"]),
                str(independent[blind_id]["annotator_id"]),
            ]
            if family == WRONG_OWNER_FAMILY:
                # Four gates are independent.  The friend-owned record is
                # concrete/new even though owner and current goal are wrong.
                source["specific_nonredundant_increment"] = "yes"
                source["derived_eligibility"] = _derived(source)
                basis = "V2_DUAL_REVIEW_WITH_FIELD_INDEPENDENCE_ADJUDICATION"
                explicit_adjudications.append(
                    {
                        "state_id": state_id,
                        "field": "specific_nonredundant_increment",
                        "final": "yes",
                        "reason": "specific/new information is distinct from owner and goal validity",
                    }
                )
                carry_counts["v2_dual_review_adjudicated"] += 1
            elif family == CURRENT_REQUEST_FAMILY:
                if _surface_hash(v2[state_id]) == _surface_hash(current):
                    raise RuntimeError("current-request profile repair did not change Rank-1")
                old_lines = str(v2[state_id]["exact_rank1_candidate"]["candidate_text"]).splitlines()
                new_lines = str(current["exact_rank1_candidate"]["candidate_text"]).splitlines()
                if not (
                    old_lines[0] == new_lines[0]
                    and "prefer this profile when the user requests low burden" in new_lines[1].lower()
                    and "do not use this profile to add any second support action" in new_lines[2].lower()
                    and "ask exactly one focused question to clarify the feeling i already indicated"
                    in str(current["current_user_text"]).lower()
                ):
                    raise RuntimeError("current-request monotone profile repair is not the frozen change")
                source.update(
                    {
                        "owner_time_valid": "yes",
                        "goal_function_fit": "yes",
                        "boundary_burden_fit": "yes",
                        "specific_nonredundant_increment": "no",
                        "derived_eligibility": "ineligible",
                    }
                )
                basis = "V3_LOW_BURDEN_PROFILE_REPAIR_PLUS_REDUNDANCY_ADJUDICATION"
                explicit_adjudications.append(
                    {
                        "state_id": state_id,
                        "field": "specific_nonredundant_increment",
                        "final": "no",
                        "reason": "the current request already specifies the card's complete atomic move; the resource adds no instruction over R0",
                    }
                )
                carry_counts["v3_monotone_profile_repair_adjudicated"] += 1
            else:
                if _surface_hash(v2[state_id]) != _surface_hash(current):
                    raise RuntimeError(f"reviewed V2 surface changed unexpectedly: {state_id}")
                if any(primary[blind_id][field] != independent[blind_id][field] for field in FIELDS):
                    raise RuntimeError(f"unadjudicated dual-review field difference: {blind_id}")
                basis = "V2_DUAL_REVIEW_EXACT_AGREEMENT"
                carry_counts["v2_exact_dual_review"] += 1

        candidate = current["exact_rank1_candidate"]
        final_rows.append(
            {
                "protocol": PROTOCOL,
                "state_id": state_id,
                "component": component,
                "adjudicated_subtype": source["adjudicated_subtype"],
                "owner_time_valid": source["owner_time_valid"],
                "goal_function_fit": source["goal_function_fit"],
                "boundary_burden_fit": source["boundary_burden_fit"],
                "specific_nonredundant_increment": source["specific_nonredundant_increment"],
                "derived_eligibility": _derived(source),
                "candidate_id": candidate["candidate_id"],
                "candidate_text_sha256": candidate["candidate_text_sha256"],
                "decision_surface_sha256": _surface_hash(current),
                "annotation_basis": basis,
                "reviewer_ids": reviewer_ids,
                "identified_human_gold": False,
                "is_step1_worth_opening_gold": False,
            }
        )

    final_rows.sort(key=lambda row: row["state_id"])
    shape = {
        component: dict(
            Counter(row["derived_eligibility"] for row in final_rows if row["component"] == component)
        )
        for component in ("MP", "MS", "ME", "RS")
    }
    expected_shape = {
        component: {"eligible": 16, "ineligible": 16}
        for component in ("MP", "MS", "ME", "RS")
    }
    intent_match = sum(
        row["derived_eligibility"]
        == str(blueprint[row["state_id"]]["private_eligibility_intent"]).lower()
        for row in final_rows
    )
    failures: list[str] = []
    if len(final_rows) != 128 or len({row["state_id"] for row in final_rows}) != 128:
        failures.append("final_reference_not_128_unique_states")
    if shape != expected_shape:
        failures.append("per_component_label_shape_not_16_16")
    if any(row["derived_eligibility"] == "uncertain" for row in final_rows):
        failures.append("uncertain_final_label")
    if intent_match != 128:
        failures.append("posthoc_construction_qa_mismatch")

    report = {
        "protocol": PROTOCOL,
        "status": "PASS_ELIGIBILITY_REFERENCE_FROZEN_OBSERVATION_QUALIFICATION_NEXT" if not failures else "FAIL",
        "rows": len(final_rows),
        "per_component_shape": shape,
        "delta_pre_adjudication_agreement": agreement,
        "delta_final_decision_agreement": agreement["derived_eligibility"],
        "explicit_adjudications": explicit_adjudications,
        "lineage_counts": dict(carry_counts),
        "posthoc_construction_qa_match": f"{intent_match}/128",
        "construction_intent_used_as_gold": False,
        "eligible_for_observation_development": not failures,
        "is_step1_worth_opening_gold": False,
        "identified_human_gold": False,
        "reviewer_identity_caveat": "Two independent review-agent identities are recorded; identified human status is not established.",
        "v1_reference_sha256": sha256_file(args.v1_reference),
        "primary_delta_sha256": sha256_file(args.primary),
        "independent_delta_sha256": sha256_file(args.independent),
        "v3_candidates_sha256": sha256_file(args.v3_candidates),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "responses_generated": 0,
        "external_lockbox_read": False,
        "failures": failures,
    }
    if failures:
        raise RuntimeError(report)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reference_path = args.out_dir / "eligibility_reference.jsonl"
    write_jsonl(reference_path, final_rows)
    report["eligibility_reference_sha256"] = sha256_file(reference_path)
    write_json(args.out_dir / "freeze_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
