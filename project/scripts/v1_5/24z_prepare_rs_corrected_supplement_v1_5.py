#!/usr/bin/env python3
"""Prepare a fresh 16-pair RS effect supplement under the corrected target."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-rs-corrected-16-pair-supplement-plan-v1"
SELECTION_SEED = "pm-v1.5-rs-corrected-16-pair-selection-v1"
TARGET_BY_MOVE = {
    "AM01_invite_open_expression": 4,
    "AM04_tentative_paraphrase_check": 4,
    "AM05_grounded_validation": 4,
    "AM10_offer_one_optional_micro_step": 4,
}
PRIMARY_FEATURES = [
    "current_user_token_estimate",
    "history_turn_count",
    "move__AM01_invite_open_expression",
    "move__AM04_tentative_paraphrase_check",
    "move__AM05_grounded_validation",
    "move__AM10_offer_one_optional_micro_step",
]


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def build(
    root: Path = ROOT,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    source_dir = (
        root
        / "outputs/pm_v1_5_rs_six_card_clean_pair_v2_confirmation"
    )
    preflight_path = (
        root
        / "outputs/pm_v1_5_step1_factorized_preflight_v1"
        / "preflight_report.json"
    )
    correction_path = (
        root
        / "data/pm_v1_5_contracts"
        / "pm_step1_factorized_training_correction_v1.json"
    )
    review_path = (
        root
        / "data/pm_v1_5_contracts"
        / "rs_corrected_supplement_review_v1.json"
    )
    wave1_path = (
        root / "outputs/pm_v1_5_rs_six_card_clean_pair_v1/selected_states.jsonl"
    )
    old_outcomes = (
        root
        / "outputs/pm_v1_5_rs_six_card_clean_pair_v2_confirmation_execution"
        / "generation_outcomes.jsonl"
    )
    if old_outcomes.exists():
        raise RuntimeError("old 32-pair wave2 outcomes already exist")
    preflight = read_json(preflight_path)
    if (
        preflight["data_collection_gate"]["status"]
        != "READY_TO_PREPARE_CORRECTED_16_PAIR_SUPPLEMENT"
    ):
        raise RuntimeError("corrected data-collection gate has not passed")

    source_selected = _rows(source_dir / "selected_states.jsonl")
    source_states = {
        str(row["state_id"]): row
        for row in _rows(source_dir / "runtime_states.jsonl")
    }
    source_calls_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _rows(source_dir / "call_plan.jsonl"):
        source_calls_by_pair[str(row["pair_id"])].append(row)

    by_move: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_selected:
        move = str(row["selected_strategy_family"])
        if move in TARGET_BY_MOVE:
            by_move[move].append(row)
    chosen: list[dict[str, Any]] = []
    for move, target in TARGET_BY_MOVE.items():
        ranked = sorted(
            by_move[move],
            key=lambda row: stable_hex(
                SELECTION_SEED,
                move,
                row["user_id"],
                row["state_id"],
                n=32,
            ),
        )
        if len(ranked) < target:
            raise RuntimeError(f"insufficient fresh rows for {move}")
        chosen.extend(ranked[:target])
    chosen.sort(
        key=lambda row: (
            row["selected_strategy_family"],
            row["user_id"],
            row["state_id"],
        )
    )

    correction_sha = sha256_file(correction_path)
    review_sha = sha256_file(review_path)
    old_to_new: dict[str, str] = {}
    selected: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for row in chosen:
        old_pair = str(row["pair_id"])
        new_pair = "rs_corr_pair_" + stable_hex(
            PROTOCOL, old_pair, correction_sha, review_sha, n=24
        )
        old_to_new[old_pair] = new_pair
        selected.append(
            {
                **row,
                "protocol": PROTOCOL,
                "pair_id": new_pair,
                "source_prepared_pair_id": old_pair,
                "primary_feature_values": {
                    name: row["transparent_pm_features"][name]
                    for name in PRIMARY_FEATURES
                },
            }
        )
        state = dict(source_states[str(row["state_id"])])
        provenance = dict(state.get("provenance") or {})
        provenance.update(
            {
                "protocol": PROTOCOL,
                "source_prepared_pair_id": old_pair,
                "correction_contract_sha256": correction_sha,
            }
        )
        state["provenance"] = provenance
        states.append(state)
        for source_call in source_calls_by_pair[old_pair]:
            calls.append(
                {
                    **source_call,
                    "protocol": PROTOCOL,
                    "pair_id": new_pair,
                    "source_prepared_pair_id": old_pair,
                    "correction_contract_sha256": correction_sha,
                    "review_contract_sha256": review_sha,
                }
            )

    wave1_dialogues = {
        str(row["user_id"]) for row in _rows(wave1_path)
    }
    supplement_dialogues = {str(row["user_id"]) for row in selected}
    move_counts = Counter(
        str(row["selected_strategy_family"]) for row in selected
    )
    calls_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in calls:
        calls_by_pair[str(row["pair_id"])].append(row)
    treatment_parity = all(
        {str(row["arm"]) for row in pair_calls} == {"R0", "RS"}
        and len({int(row["generation"]["seed"]) for row in pair_calls}) == 1
        and len(
            {
                canonical_json(row["generator_identity"])
                for row in pair_calls
            }
        )
        == 1
        for pair_calls in calls_by_pair.values()
    )
    ready = (
        len(selected) == 16
        and len(supplement_dialogues) == 16
        and not (supplement_dialogues & wave1_dialogues)
        and move_counts == Counter(TARGET_BY_MOVE)
        and len(calls) == 32
        and treatment_parity
        and len(PRIMARY_FEATURES) <= (32 + 16) // 5
    )
    source_report = read_json(source_dir / "plan_report.json")
    report = {
        "protocol": PROTOCOL,
        "status": (
            "READY_FOR_CORRECTED_SUPPLEMENT_EXECUTION_PREFLIGHT"
            if ready
            else "BLOCKED_BY_CORRECTED_SUPPLEMENT_PLAN"
        ),
        "scope": (
            "Sixteen fresh train-only matched pairs to bring the shared RS "
            "quality-effect pilot from 32 to 48 independent groups; not a "
            "confirmation of the old 22-feature composite candidate."
        ),
        "planned_pairs": len(selected),
        "planned_generation_calls": len(calls),
        "independent_dialogue_groups": len(supplement_dialogues),
        "target_by_move": TARGET_BY_MOVE,
        "selected_by_move": dict(sorted(move_counts.items())),
        "future_pooled_groups_if_complete": 48,
        "primary_model_freeze_before_supplement_outcomes": {
            "model": "group-weighted L2 logistic regression",
            "target": "material RS quality benefit only",
            "feature_names": PRIMARY_FEATURES,
            "feature_count": len(PRIMARY_FEATURES),
            "maximum_at_48_groups": (32 + 16) // 5,
            "baai_features": False,
            "atomic_risk_heads": (
                "train only if each named risk has >=8 event and >=8 "
                "non-event independent groups; otherwise deterministic guard"
            ),
            "cost": "deterministic selector input, not a learned target",
        },
        "selection": {
            "seed_protocol": SELECTION_SEED,
            "uses_response_outcomes": False,
            "uses_wave1_target_labels": False,
            "uses_old_pm_predictions": False,
            "one_dialogue_per_pair": True,
            "zero_wave1_dialogue_overlap": not (
                supplement_dialogues & wave1_dialogues
            ),
        },
        "source_pool": {
            "path": str(source_dir.relative_to(root)),
            "source_plan_status": source_report["status"],
            "source_plan_sha256": sha256_file(
                source_dir / "plan_report.json"
            ),
            "old_32_pair_execution_outcomes_present": old_outcomes.exists(),
        },
        "generator_identity": source_report["generator_identity"],
        "supporter_generation_treatment_sha256": source_report[
            "supporter_generation_treatment_sha256"
        ],
        "checks": {
            "exactly_16_pairs": len(selected) == 16,
            "exactly_16_independent_dialogues": (
                len(supplement_dialogues) == 16
            ),
            "four_per_available_move": move_counts == Counter(TARGET_BY_MOVE),
            "zero_wave1_dialogue_overlap": not (
                supplement_dialogues & wave1_dialogues
            ),
            "same_seed_and_stack_within_pair": treatment_parity,
            "primary_features_within_48_group_capacity": (
                len(PRIMARY_FEATURES) <= (32 + 16) // 5
            ),
            "old_wave2_outcomes_absent": not old_outcomes.exists(),
        },
        "execution_policy": {
            "old_64_call_wave2_authorized": False,
            "this_32_call_supplement_authorized_by_plan": False,
            "next": (
                "Run the dedicated zero-API execution preflight; paid "
                "execution still requires the operator to invoke --run."
            ),
        },
        "lineage": {
            "correction_contract_sha256": correction_sha,
            "review_contract_sha256": review_sha,
            "scientific_preflight_sha256": sha256_file(preflight_path),
            "source_selected_states_sha256": sha256_file(
                source_dir / "selected_states.jsonl"
            ),
            "source_call_plan_sha256": sha256_file(
                source_dir / "call_plan.jsonl"
            ),
            "selection_mapping_sha256": sha256_text(
                canonical_json(old_to_new)
            ),
        },
    }
    return report, states, selected, calls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v1_5_rs_corrected_16_pair_supplement_v1"
        ),
    )
    args = parser.parse_args()
    report, states, selected, calls = build()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "plan_report.json", report)
    write_jsonl(args.out_dir / "runtime_states.jsonl", states)
    write_jsonl(args.out_dir / "selected_states.jsonl", selected)
    write_jsonl(args.out_dir / "call_plan.jsonl", calls)
    print(report)


if __name__ == "__main__":
    main()
