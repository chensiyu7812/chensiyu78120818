#!/usr/bin/env python3
"""Finalize V5.2 FIT adjudication, restore full grain, and train Step-1 heads."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5.2-fit-final-adjudication-and-training-v1"
PANEL = ROOT / "outputs/pm_v1_5_v5_2_full_fit_outcome_review_v1_candidate"
ADJ_PACKET = ROOT / "outputs/pm_v1_5_v5_2_fit_disagreement_adjudication_v1_candidate"
FEATURE_DIR = ROOT / "outputs/pm_v1_5_v5_fit_training_surface_v1"
EXECUTION_DIR = ROOT / "outputs/pm_v1_5_v5_2_locked_fit_execution_v2"
CONTRACT = ROOT / "data/pm_v1_5_contracts/v5_fit_training_surface_freeze_v1.json"
OUT = ROOT / "outputs/pm_v1_5_v5_2_fit_final_training_v1"
DEFAULT_ADJUDICATIONS = {
    "quality": Path("/home/tokkio/.codex/attachments/3501fbe5-e7e2-44bd-94dd-a5bc4b431046/pasted-text.txt"),
    "risk": Path("/home/tokkio/.codex/attachments/43b3d33e-e263-49f4-9233-a61de84998b2/pasted-text.txt"),
    "function": Path("/home/tokkio/.codex/attachments/ddead6c7-5fc1-4a83-a8c2-c0a945c628e5/pasted-text.txt"),
}
COMPONENTS = ("MP", "MS", "ME", "RS")


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def index(data: Iterable[Mapping[str, Any]], field: str, name: str) -> dict[str, dict[str, Any]]:
    materialized = [dict(row) for row in data]
    result = {str(row[field]): row for row in materialized}
    if len(result) != len(materialized):
        raise RuntimeError(f"duplicate {field} in {name}")
    return result


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_adjudication(
    *,
    path: Path,
    packet_path: Path,
    field: str,
    label_field: str,
    allowed: set[str],
    protocol: str,
) -> dict[str, dict[str, Any]]:
    expected = index(rows(packet_path), field, str(packet_path))
    observed = index(rows(path), field, str(path))
    if set(observed) != set(expected):
        raise RuntimeError(
            f"adjudication coverage mismatch for {field}: "
            f"missing={sorted(set(expected)-set(observed))}, "
            f"extra={sorted(set(observed)-set(expected))}"
        )
    for item_id, row in observed.items():
        if row.get("protocol") != protocol:
            raise RuntimeError(f"protocol mismatch: {item_id}")
        if str(row.get(label_field)) not in allowed:
            raise RuntimeError(f"invalid {label_field}: {item_id}")
        if not str(row.get("annotator_id") or "").strip():
            raise RuntimeError(f"missing adjudicator identity: {item_id}")
    return observed


def flip_quality(label: str) -> str:
    return {"A": "B", "B": "A"}.get(label, label)


def quality_winner(label: str, key: Mapping[str, Any]) -> str:
    if label == "A":
        return str(key["a_arm"])
    if label == "B":
        return str(key["b_arm"])
    return label


def distribution(data: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in data).items()))


def main() -> None:
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V5.2 requires {FORMAL_PYTHON}; got {sys.executable}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-adjudication", type=Path, default=DEFAULT_ADJUDICATIONS["quality"])
    parser.add_argument("--risk-adjudication", type=Path, default=DEFAULT_ADJUDICATIONS["risk"])
    parser.add_argument("--function-adjudication", type=Path, default=DEFAULT_ADJUDICATIONS["function"])
    parser.add_argument("--panel-dir", type=Path, default=PANEL)
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--execution-dir", type=Path, default=EXECUTION_DIR)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    imported = args.panel_dir / "imported_annotations"
    agreement_report = read_json(imported / "data_quality_and_agreement_report.json")
    adjudications = {
        "quality": validate_adjudication(
            path=args.quality_adjudication,
            packet_path=ADJ_PACKET / "quality_disagreements.jsonl",
            field="blind_item_id",
            label_field="quality_preference",
            allowed={"A", "B", "tie", "uncertain"},
            protocol="pm-v1.5-v5.2-fit-quality-adjudication-v1",
        ),
        "risk": validate_adjudication(
            path=args.risk_adjudication,
            packet_path=ADJ_PACKET / "risk_disagreements.jsonl",
            field="risk_item_id",
            label_field="any_material_risk",
            allowed={"yes", "no", "uncertain"},
            protocol="pm-v1.5-v5.2-fit-risk-adjudication-v1",
        ),
        "function": validate_adjudication(
            path=args.function_adjudication,
            packet_path=ADJ_PACKET / "function_disagreements.jsonl",
            field="function_item_id",
            label_field="resource_functionally_contributed",
            allowed={"yes", "no", "uncertain"},
            protocol="pm-v1.5-v5.2-fit-function-adjudication-v1",
        ),
    }
    expected_shapes = {"quality": 35, "risk": 30, "function": 13}
    if {name: len(value) for name, value in adjudications.items()} != expected_shapes:
        raise RuntimeError("frozen adjudication shape changed")

    primary = {
        "quality": index(rows(imported / "primary_quality.jsonl"), "blind_item_id", "primary quality"),
        "risk": index(rows(imported / "primary_risk.jsonl"), "risk_item_id", "primary risk"),
        "function": index(rows(imported / "primary_function.jsonl"), "function_item_id", "primary function"),
    }
    final_rep: dict[str, dict[str, dict[str, Any]]] = {}
    for construct in ("quality", "risk", "function"):
        final_rep[construct] = {item_id: dict(row) for item_id, row in primary[construct].items()}
        for item_id, row in adjudications[construct].items():
            final_rep[construct][item_id] = dict(row)

    quality_keys = rows(args.panel_dir / "private_quality_key.jsonl")
    risk_keys = rows(args.panel_dir / "private_risk_key.jsonl")
    function_keys = rows(args.panel_dir / "private_function_key.jsonl")
    if (len(quality_keys), len(risk_keys), len(function_keys)) != (512, 1024, 512):
        raise RuntimeError("full private-key grain changed")

    full_quality: list[dict[str, Any]] = []
    for key in quality_keys:
        representative_id = str(key["manual_representative_id"])
        source = final_rep["quality"][representative_id]
        label = str(source["quality_preference"])
        if str(key["a_arm"]) != str(key["manual_representative_a_arm"]):
            label = flip_quality(label)
        full_quality.append({
            **key,
            "quality_preference": label,
            "quality_winner_arm": quality_winner(label, key),
            "final_label_source": "third_adjudication" if representative_id in adjudications["quality"] else "primary",
            "representative_annotator_id": source["annotator_id"],
        })

    full_risk: list[dict[str, Any]] = []
    for key in risk_keys:
        representative_id = str(key["manual_representative_id"])
        source = final_rep["risk"][representative_id]
        full_risk.append({
            **key,
            "any_material_risk": source["any_material_risk"],
            "selected_categories": source.get("selected_categories", []),
            "final_label_source": "third_adjudication" if representative_id in adjudications["risk"] else "primary",
            "representative_annotator_id": source["annotator_id"],
        })

    full_function: list[dict[str, Any]] = []
    for key in function_keys:
        representative_id = str(key["manual_representative_id"])
        source = final_rep["function"][representative_id]
        full_function.append({
            **key,
            "resource_functionally_contributed": source["resource_functionally_contributed"],
            "final_label_source": "third_adjudication" if representative_id in adjudications["function"] else "primary",
            "representative_annotator_id": source["annotator_id"],
        })

    risk_by_call = index(full_risk, "call_id", "full risk")
    function_by_call = index(full_function, "call_id", "full function")
    outcomes = index(rows(args.execution_dir / "outcomes_ordered_private.jsonl"), "call_id", "execution outcomes")
    if set(risk_by_call) != set(outcomes):
        raise RuntimeError("risk and execution call coverage differ")

    pair_rows: list[dict[str, Any]] = []
    for row in full_quality:
        on_call, off_call = str(row["on_call_id"]), str(row["off_call_id"])
        on_outcome, off_outcome = outcomes[on_call], outcomes[off_call]
        pair_rows.append({
            "protocol": PROTOCOL,
            "blind_item_id": row["blind_item_id"],
            "state_id": row["state_id"],
            "component": row["component"],
            "semantic_group_id": row["semantic_group_id"],
            "seed": row["seed"],
            "quality_preference": row["quality_preference"],
            "quality_winner_arm": row["quality_winner_arm"],
            "on_material_risk": risk_by_call[on_call]["any_material_risk"],
            "off_material_risk": risk_by_call[off_call]["any_material_risk"],
            "on_resource_functionally_contributed": function_by_call[on_call]["resource_functionally_contributed"],
            "on_total_tokens": int(on_outcome["usage"]["total_tokens"]),
            "off_total_tokens": int(off_outcome["usage"]["total_tokens"]),
            "incremental_total_tokens": int(on_outcome["usage"]["total_tokens"]) - int(off_outcome["usage"]["total_tokens"]),
            "on_fallback_used": bool(on_outcome["fallback_used"]),
            "itt_pair_valid": bool(on_outcome["itt_row_valid"] and off_outcome["itt_row_valid"]),
        })

    pairs_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        pairs_by_state[str(row["state_id"])].append(row)
    if len(pairs_by_state) != 256 or any(len(value) != 2 for value in pairs_by_state.values()):
        raise RuntimeError("expected 256 states with two frozen seeds each")

    state_labels: list[dict[str, Any]] = []
    for state_id, state_pairs in sorted(pairs_by_state.items()):
        state_pairs = sorted(state_pairs, key=lambda row: int(row["seed"]))
        winners = [str(row["quality_winner_arm"]) for row in state_pairs]
        on_risks = [str(row["on_material_risk"]) for row in state_pairs]
        off_risks = [str(row["off_material_risk"]) for row in state_pairs]
        functions = [str(row["on_resource_functionally_contributed"]) for row in state_pairs]
        quality_positive = "ON" in winners and "OFF" not in winners and "uncertain" not in winners
        risk_veto = "yes" in on_risks
        uncertainty_veto = "uncertain" in winners or "uncertain" in on_risks
        hard_label = int(quality_positive and not risk_veto and not uncertainty_veto)
        state_labels.append({
            "protocol": PROTOCOL,
            "state_id": state_id,
            "component": state_pairs[0]["component"],
            "semantic_group_id_v5_2": state_pairs[0]["semantic_group_id"],
            "seed_quality_winners": winners,
            "seed_on_material_risk": on_risks,
            "seed_off_material_risk": off_risks,
            "seed_on_functional_use_secondary": functions,
            "seed_incremental_total_tokens": [row["incremental_total_tokens"] for row in state_pairs],
            "hard_worth_opening": hard_label,
            "quality_positive_before_risk": quality_positive,
            "material_on_risk_veto": risk_veto,
            "uncertainty_veto": uncertainty_veto,
            "functional_use_is_secondary_only": True,
        })

    feature_rows = index(rows(args.feature_dir / "fit_feature_rows_private.jsonl"), "state_id", "frozen features")
    if set(feature_rows) != set(pairs_by_state):
        raise RuntimeError("frozen feature state coverage differs from V5.2 outcomes")
    joined: list[dict[str, Any]] = []
    for label in state_labels:
        feature = dict(feature_rows[str(label["state_id"])])
        joined.append({
            **feature,
            **label,
            # V5.2 replaced the older state-derived grouping with a canonical
            # visible-surface group before generation (160 groups).  This is
            # identical for MP/MS/ME and separates the two distinct RS
            # surfaces that the V5 field had conservatively co-grouped.
            "semantic_group_id": label["semantic_group_id_v5_2"],
            "semantic_family": feature["semantic_family_private_cv_only"],
        })

    old_training = load_module(ROOT / "scripts/v1_5/26a_aggregate_and_train_v5_fit_v1_5.py", "v52_fit_helper")
    contract = read_json(args.contract)
    heads = {
        component: old_training._fit_component(
            rows=[row for row in joined if row["component"] == component],
            contract=contract,
        )
        for component in COMPONENTS
    }

    for construct, data in final_rep.items():
        field = {"quality": "blind_item_id", "risk": "risk_item_id", "function": "function_item_id"}[construct]
        write_jsonl(args.out_dir / f"final_representative_{construct}.jsonl", [data[item] for item in sorted(data)])
    write_jsonl(args.out_dir / "final_full_quality_private.jsonl", full_quality)
    write_jsonl(args.out_dir / "final_full_risk_private.jsonl", full_risk)
    write_jsonl(args.out_dir / "final_full_function_private.jsonl", full_function)
    write_jsonl(args.out_dir / "final_seed_pair_outcomes_private.jsonl", pair_rows)
    write_jsonl(args.out_dir / "state_effect_labels_private.jsonl", state_labels)

    component_summary: dict[str, Any] = {}
    for component in COMPONENTS:
        component_pairs = [row for row in pair_rows if row["component"] == component]
        component_states = [row for row in state_labels if row["component"] == component]
        component_summary[component] = {
            "seed_pairs": len(component_pairs),
            "quality_winner_arm": distribution(component_pairs, "quality_winner_arm"),
            "on_material_risk": distribution(component_pairs, "on_material_risk"),
            "off_material_risk": distribution(component_pairs, "off_material_risk"),
            "on_resource_functionally_contributed": distribution(component_pairs, "on_resource_functionally_contributed"),
            "mean_incremental_total_tokens": round(sum(row["incremental_total_tokens"] for row in component_pairs) / len(component_pairs), 3),
            "positive_states": sum(int(row["hard_worth_opening"]) for row in component_states),
            "nonpositive_states": sum(1 - int(row["hard_worth_opening"]) for row in component_states),
        }

    report = {
        "protocol": PROTOCOL,
        "status": (
            "ALL_FOUR_HEADS_READY_FOR_FRESH_CONFIRMATION"
            if all(head.get("status") == "FIT_PROMOTED_TO_FRESH_CONFIRMATION" for head in heads.values())
            else "PARTIAL_OR_NOT_LEARNED_ON_FROZEN_V5_2_FIT"
        ),
        "data_quality": {
            "adjudication_coverage": {name: len(data) for name, data in adjudications.items()},
            "representative_coverage": {name: len(data) for name, data in final_rep.items()},
            "restored_full_grain": {"quality_pairs": len(full_quality), "risk_responses": len(full_risk), "function_on_responses": len(full_function)},
            "state_units": len(state_labels),
            "frozen_seeds_per_state": 2,
            "uncertain_state_labels": sum(bool(row["uncertainty_veto"]) for row in state_labels),
            "all_itt_pairs_valid": all(bool(row["itt_pair_valid"]) for row in pair_rows),
            "all_fallback_free": not any(bool(row["on_fallback_used"]) for row in pair_rows),
        },
        "agreement_before_adjudication": agreement_report["agreement"],
        "adjudication_distributions": {
            "quality": distribution(adjudications["quality"].values(), "quality_preference"),
            "risk": distribution(adjudications["risk"].values(), "any_material_risk"),
            "function": distribution(adjudications["function"].values(), "resource_functionally_contributed"),
        },
        "full_outcome_summary": component_summary,
        "hard_label_contract": contract["hard_label"],
        "heads": heads,
        "post_label_executor_or_method_change_allowed": False,
        "external_lockbox_read": False,
        "input_sha256": {
            "quality_adjudication": sha256_file(args.quality_adjudication),
            "risk_adjudication": sha256_file(args.risk_adjudication),
            "function_adjudication": sha256_file(args.function_adjudication),
            "quality_key": sha256_file(args.panel_dir / "private_quality_key.jsonl"),
            "risk_key": sha256_file(args.panel_dir / "private_risk_key.jsonl"),
            "function_key": sha256_file(args.panel_dir / "private_function_key.jsonl"),
            "feature_rows": sha256_file(args.feature_dir / "fit_feature_rows_private.jsonl"),
            "training_contract": sha256_file(args.contract),
            "execution_outcomes": sha256_file(args.execution_dir / "outcomes_ordered_private.jsonl"),
        },
    }
    write_json(args.out_dir / "final_fit_report.json", report)
    print(json.dumps({
        "protocol": PROTOCOL,
        "status": report["status"],
        "positive_states": {component: component_summary[component]["positive_states"] for component in COMPONENTS},
        "head_status": {component: heads[component]["status"] for component in COMPONENTS},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
