#!/usr/bin/env python3
"""Audit H1-v2 construction quality before any new human annotation."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json
from metacom_pm.v1_5_final_dataset_blueprint import H1_V2_BLUEPRINT_PROTOCOL
from metacom_pm.v1_5_final_raw_generation import TOPIC_BRIDGE_WORDS
from metacom_pm.v1_5_strategy_rag_repair import PRE_PM_CANDIDATE_RANK_PROTOCOL


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-final-h1-v2-pre-human-static-gate-v1"
COMPONENTS = ("MP", "MS", "ME", "RS")
SPLITS = ("FIT", "FRESH_CONFIRMATION", "SEALED_INTERNAL_TEST")


def _public_payload(row: dict[str, Any], component: str) -> str:
    surface = row["candidate_surfaces"][component]
    return json.dumps(
        [
            component,
            row["visible_dialogue"],
            row["current_user_text"],
            surface["candidate_text"],
            surface["compiler_subtype_hint"],
        ],
        ensure_ascii=False,
        sort_keys=True,
    )


def _decision_surface_payload(row: dict[str, Any]) -> str:
    """Represent the content that can change the four routing decisions.

    H1-v2 includes a generated two-turn history prefix.  That prefix can make the
    full JSON payload unique while current text and every exact candidate remain
    identical.  Such rows are not independent routing examples and must not cross
    splits.
    """
    return json.dumps(
        [
            row["current_user_text"],
            {
                component: row["candidate_surfaces"][component]["candidate_text"]
                for component in COMPONENTS
            },
        ],
        ensure_ascii=False,
        sort_keys=True,
    )


_TOPIC_WORDS = sorted(
    {word for pair in TOPIC_BRIDGE_WORDS.values() for word in pair},
    key=len,
    reverse=True,
)
_TOPIC_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in _TOPIC_WORDS) + r")\b",
    flags=re.IGNORECASE,
)


def _topic_normalized(text: str) -> str:
    return re.sub(r"\s+", " ", _TOPIC_RE.sub("<topic>", text).lower()).strip()


def _duplicate_report(
    rows: list[dict[str, Any]], component: str, *, normalized: bool
) -> dict[str, Any]:
    groups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if not row["candidate_surfaces"][component]["candidate_present"]:
            continue
        payload = _public_payload(row, component)
        key = _topic_normalized(payload) if normalized else payload
        groups[key].append({"state_id": row["state_id"], "split": row["split"]})
    duplicates = [values for values in groups.values() if len(values) > 1]
    cross_split = [
        values for values in duplicates if len({value["split"] for value in values}) > 1
    ]
    return {
        "candidate_items": sum(len(values) for values in groups.values()),
        "unique_payloads": len(groups),
        "duplicate_groups": len(duplicates),
        "duplicate_states": sum(len(values) for values in duplicates),
        "cross_split_duplicate_groups": len(cross_split),
        "cross_split_duplicate_states": sum(len(values) for values in cross_split),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_final_candidate_first_v9_h1_v2/private/construction_blueprint.jsonl",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_p2_h1_v2_exact_rank1_v10",
    )
    parser.add_argument(
        "--strategy-cards",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_final_h1_v2_pre_human_static_gate_v1",
    )
    args = parser.parse_args()

    blueprints = [dict(row) for row in iter_jsonl(args.blueprint)]
    blueprint_by_state = {row["state_id"]: row for row in blueprints}
    candidates_path = args.candidate_dir / "candidate_rows_private.jsonl"
    feature_path = args.candidate_dir / "model_feature_rows.jsonl"
    candidates = [dict(row) for row in iter_jsonl(candidates_path)]
    features = [dict(row) for row in iter_jsonl(feature_path)]
    cards = {str(row["card_id"]): dict(row) for row in iter_jsonl(args.strategy_cards)}
    materialization = read_json(args.candidate_dir / "materialization_report.json")

    failures: list[str] = []
    if len(blueprints) != 256 or len(blueprint_by_state) != 256:
        failures.append("blueprint_not_256_unique_states")
    if any(row.get("protocol") != H1_V2_BLUEPRINT_PROTOCOL for row in blueprints):
        failures.append("wrong_blueprint_protocol")
    if len(candidates) != 256 or len({row["state_id"] for row in candidates}) != 256:
        failures.append("candidate_rows_not_256_unique_states")
    if len(features) != 1024:
        failures.append("feature_rows_not_1024")
    if materialization.get("exact_rank1_binding_rate") != 1.0:
        failures.append("exact_rank1_binding_not_complete")
    if materialization.get("construction_realization_status") != "PASS":
        failures.append("construction_realization_failed")
    if any(row.get("construction_intent_present") for row in candidates):
        failures.append("private_construction_intent_leaked")
    if any(row.get("response_or_outcome_read") for row in candidates):
        failures.append("response_or_outcome_was_read")

    label_proxy_coverage: dict[str, Any] = {}
    candidate_contract: dict[str, Any] = {}
    duplicate_audit: dict[str, Any] = {}
    for component in COMPONENTS:
        per_split: dict[str, Any] = {}
        for split in SPLITS:
            counts: Counter[bool] = Counter()
            for row in candidates:
                if row["split"] != split:
                    continue
                surface = row["candidate_surfaces"][component]
                if surface["candidate_present"]:
                    bit = bool(
                        blueprint_by_state[row["state_id"]][
                            "private_construction_intent"
                        ]["intended_bits"][component]
                    )
                    counts[bit] += 1
            total = sum(counts.values())
            majority_share = max(counts.values(), default=0) / total if total else 1.0
            per_split[split] = {
                "intended_on": counts[True],
                "intended_off": counts[False],
                "candidate_present": total,
                "majority_share": majority_share,
            }
            if not counts[True] or not counts[False] or majority_share > 0.70:
                failures.append(f"{component}_{split}_construction_proxy_degenerate")
        label_proxy_coverage[component] = per_split

        present = 0
        absent = 0
        malformed = 0
        missed_positive = 0
        for row in candidates:
            surface = row["candidate_surfaces"][component]
            if surface["candidate_present"]:
                present += 1
                if not surface["candidate_id"] or not surface["candidate_text"]:
                    malformed += 1
            else:
                absent += 1
                if surface["candidate_id"] is not None or surface["candidate_text"] is not None:
                    malformed += 1
            intended_on = bool(
                blueprint_by_state[row["state_id"]]["private_construction_intent"][
                    "intended_bits"
                ][component]
            )
            if intended_on and not surface["candidate_present"]:
                missed_positive += 1
        candidate_contract[component] = {
            "present": present,
            "absent": absent,
            "malformed_surfaces": malformed,
            "intended_positive_without_candidate": missed_positive,
        }
        if malformed:
            failures.append(f"{component}_malformed_candidate_surface")
        if missed_positive:
            failures.append(f"{component}_intended_positive_not_retrieved")

        exact = _duplicate_report(candidates, component, normalized=False)
        topic_normalized = _duplicate_report(candidates, component, normalized=True)
        duplicate_audit[component] = {
            "exact_public_payload": exact,
            "topic_normalized_public_payload": topic_normalized,
        }
        if exact["duplicate_groups"]:
            failures.append(f"{component}_exact_public_duplicate")
        if topic_normalized["cross_split_duplicate_groups"]:
            failures.append(f"{component}_topic_normalized_cross_split_duplicate")

    decision_surface_groups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        decision_surface_groups[_decision_surface_payload(row)].append(
            {"state_id": row["state_id"], "split": row["split"]}
        )
    decision_surface_duplicates = [
        values for values in decision_surface_groups.values() if len(values) > 1
    ]
    decision_surface_cross_split = [
        values
        for values in decision_surface_duplicates
        if len({value["split"] for value in values}) > 1
    ]
    duplicate_audit["decision_surface"] = {
        "signature": "current_user_text + all four exact candidate texts",
        "duplicate_groups": len(decision_surface_duplicates),
        "duplicate_states": sum(len(values) for values in decision_surface_duplicates),
        "cross_split_duplicate_groups": len(decision_surface_cross_split),
        "cross_split_duplicate_states": sum(
            len(values) for values in decision_surface_cross_split
        ),
    }
    if decision_surface_cross_split:
        failures.append("decision_surface_cross_split_duplicate")

    rs_protocol_errors = sum(
        1
        for row in candidates
        if row["candidate_surfaces"]["RS"]["compiler_protocol"]
        != PRE_PM_CANDIDATE_RANK_PROTOCOL
    )
    vague_mp_profiles = sum(
        1
        for row in candidates
        if "recurring personal constraint involving"
        in str(row["candidate_surfaces"]["MP"]["candidate_text"] or "").lower()
    )
    if rs_protocol_errors:
        failures.append("rs_not_materialized_by_pre_pm_ranker")
    if vague_mp_profiles:
        failures.append("vague_mp_profile_surface")

    expected_rs_family = {
        "strategy_move_already_present": "Question",
        "clarification_move_fit": "Question",
        "listen_move_fit": "Restatement or Paraphrasing",
        "one_option_move_fit": "Providing Suggestions",
        "reflection_move_fit": "Reflection of feelings",
    }
    rs_family_mismatches = []
    for row in candidates:
        mode = str(
            blueprint_by_state[row["state_id"]]["private_construction_intent"][
                "component_plans"
            ]["RS"]["construction_mode"]
        )
        expected = expected_rs_family.get(mode)
        if expected is None:
            continue
        ranked_ids = row["private_rankings_not_model_input"]["RS"]["ranked_card_ids"]
        actual = (
            str(cards[str(ranked_ids[0])]["strategy_family"])
            if ranked_ids
            else "CANDIDATE_ABSENT"
        )
        if actual != expected:
            rs_family_mismatches.append(
                {
                    "state_id": row["state_id"],
                    "construction_mode": mode,
                    "expected_retrieval_family": expected,
                    "actual_retrieval_family": actual,
                }
            )
    if rs_family_mismatches:
        failures.append("rs_candidate_family_does_not_realize_construction_mode")

    report = {
        "protocol": PROTOCOL,
        "status": "PASS" if not failures else "FAIL",
        "failures": sorted(set(failures)),
        "states": len(candidates),
        "feature_rows": len(features),
        "candidate_contract": candidate_contract,
        "construction_proxy_coverage_not_gold": label_proxy_coverage,
        "duplicate_audit": duplicate_audit,
        "rs_pre_pm_ranker_protocol_errors": rs_protocol_errors,
        "vague_mp_profile_surfaces": vague_mp_profiles,
        "rs_candidate_family_realization": {
            "checked": sum(
                1
                for row in candidates
                if blueprint_by_state[row["state_id"]]["private_construction_intent"][
                    "component_plans"
                ]["RS"]["construction_mode"]
                in expected_rs_family
            ),
            "mismatches": rs_family_mismatches,
            "construction_family_is_gold": False,
        },
        "maximum_allowed_construction_proxy_majority_share": 0.70,
        "construction_proxy_used_as_h1_gold": False,
        "human_labels_read": 0,
        "response_or_outcome_read": False,
        "external_lockbox_read": False,
        "blueprint_sha256": sha256_file(args.blueprint),
        "candidate_rows_sha256": sha256_file(candidates_path),
        "model_feature_rows_sha256": sha256_file(feature_path),
        "next_step": "BUILD_H1_V2_REVIEW_PACKET" if not failures else "REPAIR_CONSTRUCTION_ONLY",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "static_gate_report.json", report)
    if failures:
        raise RuntimeError(report)
    print(report)


if __name__ == "__main__":
    main()
