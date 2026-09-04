#!/usr/bin/env python3
"""Outcome-blind data-quality gate before the single final H1R review."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl
from metacom_pm.v1_5_final_dataset_blueprint import (
    H1R_BLUEPRINT_PROTOCOL,
    H1R_RS_FAMILIES,
    TOPIC_FAMILIES,
)
from metacom_pm.v1_5_observation_contract import (
    FACTOR_NAMES,
    transparent_semantic_observation,
)
from metacom_pm.v1_5_strategy_rag_repair import PRE_PM_CANDIDATE_RANK_PROTOCOL


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-final-h1r-pre-human-data-quality-gate-v1"
COMPONENTS = ("MP", "MS", "ME", "RS")
SPLITS = ("FIT", "FRESH_CONFIRMATION", "SEALED_INTERNAL_TEST")


def _surface_signature(row: dict[str, Any]) -> str:
    return json.dumps(
        [
            row["visible_dialogue"],
            row["current_user_text"],
            {
                component: row["candidate_surfaces"][component]["candidate_text"]
                for component in COMPONENTS
            },
        ],
        ensure_ascii=False,
        sort_keys=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT / "data/pm_v1_5_final_h1r_v1/private/construction_blueprint.jsonl",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_final_h1r_exact_rank1_v1",
    )
    parser.add_argument(
        "--strategy-cards",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_final_h1r_pre_human_gate_v1",
    )
    args = parser.parse_args()

    blueprints = [dict(row) for row in iter_jsonl(args.blueprint)]
    blueprint_by_state = {str(row["state_id"]): row for row in blueprints}
    candidate_path = args.candidate_dir / "candidate_rows_private.jsonl"
    feature_path = args.candidate_dir / "model_feature_rows.jsonl"
    candidates = [dict(row) for row in iter_jsonl(candidate_path)]
    features = [dict(row) for row in iter_jsonl(feature_path)]
    cards = {
        str(row["card_id"]): dict(row) for row in iter_jsonl(args.strategy_cards)
    }
    materialization = read_json(args.candidate_dir / "materialization_report.json")
    realization = read_json(args.candidate_dir / "construction_realization_private.json")
    failures: list[str] = []

    if len(blueprints) != 96 or len(blueprint_by_state) != 96:
        failures.append("blueprint_not_96_unique_states")
    if any(row.get("protocol") != H1R_BLUEPRINT_PROTOCOL for row in blueprints):
        failures.append("wrong_blueprint_protocol")
    if len(candidates) != 96 or len({row["state_id"] for row in candidates}) != 96:
        failures.append("candidate_rows_not_96_unique_states")
    if len(features) != 384:
        failures.append("feature_rows_not_384")
    if materialization.get("status") != "PASS_COMPLETE":
        failures.append("materialization_not_complete")
    if realization.get("status") != "PASS":
        failures.append("construction_candidate_type_realization_failed")
    if any(row.get("construction_intent_present") for row in candidates):
        failures.append("construction_intent_leaked_to_candidate_rows")
    if any(row.get("response_or_outcome_read") for row in candidates):
        failures.append("response_or_outcome_read")

    signatures: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        signatures[_surface_signature(row)].append(
            {"state_id": str(row["state_id"]), "split": str(row["split"])}
        )
    duplicates = [values for values in signatures.values() if len(values) > 1]
    cross_split_duplicates = [
        values for values in duplicates if len({value["split"] for value in values}) > 1
    ]
    if duplicates:
        failures.append("duplicate_decision_surface")

    topic_bit_grid: dict[str, Any] = {}
    for component in COMPONENTS:
        per_topic = {}
        for topic in TOPIC_FAMILIES:
            values = Counter(
                bool(row["private_construction_intent"]["intended_bits"][component])
                for row in blueprints
                if row["topic_family"] == topic
            )
            per_topic[topic] = {"off": values[False], "on": values[True]}
            if values != Counter({False: 3, True: 3}):
                failures.append(f"topic_shortcut_{component}_{topic}")
        topic_bit_grid[component] = per_topic

    scale_bit_grid: dict[str, Any] = {}
    for component in COMPONENTS:
        per_split = {}
        for split in SPLITS:
            cells = Counter(
                (
                    str(row["history_shape"]),
                    bool(row["private_construction_intent"]["intended_bits"][component]),
                )
                for row in blueprints
                if row["split"] == split
            )
            per_split[split] = {
                str(key): value for key, value in sorted(cells.items())
            }
            if any(
                cells[(scale, bit)] != 8
                for scale in ("SMALL", "EVO_LIKE_LARGE")
                for bit in (False, True)
            ):
                failures.append(f"history_size_shortcut_{component}_{split}")
        scale_bit_grid[component] = per_split

    rs_family_mismatches = []
    candidate_contract: dict[str, Any] = {}
    for component in COMPONENTS:
        malformed = 0
        subtype_mismatch = 0
        for row in candidates:
            surface = row["candidate_surfaces"][component]
            if not surface["candidate_present"] or not surface["candidate_text"]:
                malformed += 1
                continue
            target = blueprint_by_state[str(row["state_id"])][
                "private_construction_intent"
            ]["component_plans"][component]
            if component != "RS" and str(surface["compiler_subtype_hint"]) != str(
                target["candidate_subtype_target"]
            ):
                subtype_mismatch += 1
            if component == "RS":
                ids = row["private_rankings_not_model_input"]["RS"][
                    "ranked_card_ids"
                ]
                actual_family = (
                    str(cards[str(ids[0])]["strategy_family"])
                    if ids
                    else "CANDIDATE_ABSENT"
                )
                if actual_family != str(target["strategy_family_target"]):
                    rs_family_mismatches.append(
                        {
                            "state_id": row["state_id"],
                            "target": target["strategy_family_target"],
                            "actual": actual_family,
                        }
                    )
        candidate_contract[component] = {
            "present": 96 - malformed,
            "absent_or_malformed": malformed,
            "compiler_subtype_mismatches": subtype_mismatch,
        }
        if malformed:
            failures.append(f"{component}_candidate_absent_or_malformed")
        if subtype_mismatch:
            failures.append(f"{component}_candidate_subtype_mismatch")
    if rs_family_mismatches:
        failures.append("rs_rank1_family_mismatch")
    if any(
        row["candidate_surfaces"]["RS"]["compiler_protocol"]
        != PRE_PM_CANDIDATE_RANK_PROTOCOL
        for row in candidates
    ):
        failures.append("rs_not_using_pre_pm_candidate_ranker")

    observation_rows = []
    observation_distribution: dict[str, Any] = {}
    for component in COMPONENTS:
        per_split: dict[str, Any] = {}
        for row in candidates:
            surface = row["candidate_surfaces"][component]
            observation = transparent_semantic_observation(
                state_id=str(row["state_id"]),
                component=component,
                current_user_text=str(row["current_user_text"]),
                visible_dialogue=list(row["visible_dialogue"]),
                candidate_present=bool(surface["candidate_present"]),
                candidate_text=surface["candidate_text"],
                candidate_subtype=str(surface["compiler_subtype_hint"]),
            )
            observation_rows.append(
                {
                    **asdict(observation),
                    "split": row["split"],
                    "group_id": row["group_id"],
                }
            )
        for split in SPLITS:
            split_rows = [
                row
                for row in observation_rows
                if row["component"] == component and row["split"] == split
            ]
            factors = {
                factor: dict(
                    Counter(float(row["factor_scores"][factor]) for row in split_rows)
                )
                for factor in FACTOR_NAMES
            }
            specific_values = Counter(
                float(row["factor_scores"]["specific_increment"])
                for row in split_rows
            )
            if not specific_values[0.0] or not specific_values[1.0]:
                failures.append(f"{component}_{split}_specific_increment_constant")
            per_split[split] = factors
        observation_distribution[component] = per_split

    forbidden_feature_names = re.compile(
        r"(?:gold|label|construction|intended|response_winner|"
        r"future_response|judge_score|external_dataset|dataset_identity)",
        re.I,
    )
    leaked_feature_names = sorted(
        {
            name
            for row in features
            for name in row["model_features"]
            if forbidden_feature_names.search(name)
        }
    )
    if leaked_feature_names:
        failures.append("forbidden_model_feature_name")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    observation_path = args.out_dir / "semantic_observation_rows.jsonl"
    write_jsonl(observation_path, observation_rows)
    report = {
        "protocol": PROTOCOL,
        "status": "PASS" if not failures else "FAIL",
        "failures": sorted(set(failures)),
        "dataset_grain": "one unique current state/user with four exact Rank-1 component candidates",
        "states": len(candidates),
        "candidate_items": len(candidates) * 4,
        "model_feature_rows": len(features),
        "semantic_observation_rows": len(observation_rows),
        "candidate_contract": candidate_contract,
        "rs_family_targets": list(H1R_RS_FAMILIES),
        "rs_family_mismatches": rs_family_mismatches,
        "duplicate_decision_surfaces": len(duplicates),
        "cross_split_duplicate_decision_surfaces": len(cross_split_duplicates),
        "topic_bit_grid_construction_only_not_gold": topic_bit_grid,
        "history_scale_bit_grid_construction_only_not_gold": scale_bit_grid,
        "observation_factor_distribution_without_gold": observation_distribution,
        "leaked_feature_names": leaked_feature_names,
        "construction_intent_used_as_gold": False,
        "human_labels_read": 0,
        "response_or_outcome_read": False,
        "external_lockbox_read": False,
        "blueprint_sha256": sha256_file(args.blueprint),
        "candidate_rows_sha256": sha256_file(candidate_path),
        "model_feature_rows_sha256": sha256_file(feature_path),
        "semantic_observation_rows_sha256": sha256_file(observation_path),
        "next_step": "BUILD_SINGLE_H1R_REVIEW_PACKET" if not failures else "REPAIR_CONSTRUCTION_ONLY",
    }
    write_json(args.out_dir / "static_gate_report.json", report)
    if failures:
        raise RuntimeError(report)
    print(report)


if __name__ == "__main__":
    main()
