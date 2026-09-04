#!/usr/bin/env python3
"""Compile raw P2 users and materialize actual MP/MS/ME/RS rank-1 candidates."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from metacom_pm.config import load_config
from metacom_pm.contracts import MemorySource
from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl
from metacom_pm.retrieval import MemoryRetriever, source_specific_memory_queries
from metacom_pm.text import normalize_space
from metacom_pm.v1_5_candidate_discovery import (
    FINAL_TYPED_CANDIDATE_DISCOVERY_PROTOCOL,
    discover_final_typed_memory_candidates,
)
from metacom_pm.v1_5_final_candidate_contract import (
    FINAL_FEATURE_CONTRACT_PROTOCOL,
    IndependentFeatureRecord,
    exact_rank1_memory_surface,
    exact_rank1_strategy_surface,
    final_model_feature_projection,
    final_rs_model_features,
)
from metacom_pm.v1_5_final_construction_realization import (
    audit_construction_realization,
)
from metacom_pm.v1_5_memory_opportunity_features import (
    SOURCE_SPECIFIC_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL,
    build_source_specific_memory_opportunity_observation,
)
from metacom_pm.v1_5_memory_transport import (
    BOUNDED_MEMORY_COMPILER_PROTOCOL,
    compile_bounded_memory_with_metadata,
)
from metacom_pm.v1_5_strategy_rag_repair import (
    PRE_PM_CANDIDATE_RANK_PROTOCOL,
    REPAIR_V2_PROTOCOL,
    effect_study_observable_flags,
    rank_pre_pm_strategy_candidates,
    repair_v2_rank_applicable_cards,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-p2-exact-rank1-candidate-materialization-v9"
RS_FEATURE_PROTOCOL = "pm-v1.5-final-rs-transparent-features-v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)] if path.is_file() else []


def _rs_query(dialogue: list[dict[str, str]], flags: dict[str, Any]) -> str:
    seekers = [row["content"] for row in dialogue if row["speaker"] == "seeker"]
    cues = [
        key.replace("_", " ")
        for key, value in flags.items()
        if isinstance(value, bool)
        and value
        and key
        not in {
            "substantive",
            "pure_phatic",
            "routine_closing",
            "active_high_stakes",
            "explicit_stop",
            "ordinary_rag_hard_off",
        }
    ]
    return (
        "Represent this sentence for searching relevant passages: Choose one safe, "
        "topic-agnostic emotional-support technique. Observable cues: "
        + (", ".join(cues) or "none")
        + ". Recent seeker context: "
        + " ".join(seekers[-3:])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-states",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_p2_raw_generation_execution_v9/raw_states.jsonl",
    )
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_final_candidate_first_v8/private/construction_blueprint.jsonl",
    )
    parser.add_argument(
        "--strategy-cards",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl",
    )
    parser.add_argument(
        "--pm-config", type=Path, default=ROOT / "configs/pm_v1_5.yaml"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_p2_exact_rank1_candidates_v9",
    )
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument(
        "--rs-ranker",
        choices=("post_h2_applicability", "pre_pm_content"),
        default="post_h2_applicability",
    )
    parser.add_argument("--expected-states", type=int, default=256)
    args = parser.parse_args()
    raw_rows = _rows(args.raw_states)
    if not raw_rows:
        raise RuntimeError("no accepted raw P2 states")
    if len({row["state_id"] for row in raw_rows}) != len(raw_rows):
        raise RuntimeError("duplicate raw state IDs")
    blueprint_rows = _rows(args.blueprint)
    blueprints_by_state = {str(row["state_id"]): row for row in blueprint_rows}
    if len(blueprints_by_state) != args.expected_states:
        raise RuntimeError("private construction blueprint is absent or invalid")
    cards = _rows(args.strategy_cards)
    cards_by_id = {str(row["card_id"]): row for row in cards}
    if len(cards_by_id) != len(cards):
        raise RuntimeError("duplicate Strategy Bank card IDs")
    config = load_config(args.pm_config)
    minimum_score = float(config["retrieval"]["memory_min_score"])
    retriever = MemoryRetriever(
        minimum_score_by_source={source: minimum_score for source in MemorySource}
    )

    candidate_rows = []
    feature_rows = []
    for raw in raw_rows:
        items, _docs, metadata = compile_bounded_memory_with_metadata(raw["user"])
        current_session_index = int(raw["current_session_index"])
        if any(item.created_session >= current_session_index for item in items):
            raise RuntimeError("current/future memory entered P2 catalog")
        visible_runtime = list(raw["visible_dialogue"])
        queries = source_specific_memory_queries(
            str(raw["current_user_text"]), visible_runtime, ""
        )
        discoveries = discover_final_typed_memory_candidates(
            queries=queries,
            items=items,
            source_metadata=metadata,
            session_index=current_session_index,
        )
        surfaces = {}
        private_rankings = {}
        for source in MemorySource:
            candidate = discoveries[source]
            surface = exact_rank1_memory_surface(
                state_id=str(raw["state_id"]),
                candidate=candidate,
                source_metadata=metadata,
                current_session_index=current_session_index,
                compiler_protocol=BOUNDED_MEMORY_COMPILER_PROTOCOL,
            )
            surfaces[source.value] = asdict(surface)
            private_rankings[source.value] = {
                "selected_memory_ids": [
                    item.memory_id for item in candidate.selected_items
                ],
                "descriptor": candidate.descriptor,
            }
            observation = build_source_specific_memory_opportunity_observation(
                candidate=candidate,
                catalog_items=items,
                catalog_user_id=str(raw["user_id"]),
                current_user_id=str(raw["user_id"]),
                current_session_index=current_session_index,
                current_user_text=str(raw["current_user_text"]),
                visible_dialogue=visible_runtime,
                source_metadata=metadata,
                background_action="M0+R0",
            )
            projected = final_model_feature_projection(
                component=source.value,
                source_features=observation["model_features"],
            )
            feature = IndependentFeatureRecord(
                protocol=FINAL_FEATURE_CONTRACT_PROTOCOL,
                state_id=str(raw["state_id"]),
                component=source.value,
                feature_builder_protocol=SOURCE_SPECIFIC_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL,
                model_features=projected,
            )
            feature_rows.append(
                {
                    **asdict(feature),
                    "split": raw["split"],
                    "group_id": raw["group_id"],
                    "semantic_family": raw["logic_family"],
                }
            )

        dialogue = [
            {
                "speaker": "seeker" if row["role"] == "user" else "supporter",
                "content": row["content"],
            }
            for row in visible_runtime
        ]
        dialogue.append({"speaker": "seeker", "content": raw["current_user_text"]})
        seekers = [row["content"] for row in dialogue if row["speaker"] == "seeker"]
        recent = normalize_space(" ".join(seekers[-3:]))
        flags = effect_study_observable_flags(
            current_user_text=str(raw["current_user_text"]),
            recent_user_text=recent,
            visible_dialogue=dialogue,
        )
        rs_query = _rs_query(dialogue, flags)
        ranker = (
            rank_pre_pm_strategy_candidates
            if args.rs_ranker == "pre_pm_content"
            else repair_v2_rank_applicable_cards
        )
        rs_ranker_protocol = (
            PRE_PM_CANDIDATE_RANK_PROTOCOL
            if args.rs_ranker == "pre_pm_content"
            else REPAIR_V2_PROTOCOL
        )
        ranked = ranker(
            query=rs_query,
            current_user_text=str(raw["current_user_text"]),
            recent_user_text=recent,
            visible_dialogue=dialogue,
            cards=cards,
        )
        top_card = cards_by_id[str(ranked[0]["card_id"])] if ranked else None
        rs_surface = exact_rank1_strategy_surface(
            state_id=str(raw["state_id"]),
            ranked_cards=ranked,
            cards_by_id=cards_by_id,
            compiler_protocol=rs_ranker_protocol,
        )
        surfaces["RS"] = asdict(rs_surface)
        private_rankings["RS"] = {
            "ranked_card_ids": [row["card_id"] for row in ranked],
            "observable_flags": flags,
        }
        rs_feature = IndependentFeatureRecord(
            protocol=FINAL_FEATURE_CONTRACT_PROTOCOL,
            state_id=str(raw["state_id"]),
            component="RS",
            feature_builder_protocol=RS_FEATURE_PROTOCOL,
            model_features=final_rs_model_features(
                current_user_text=str(raw["current_user_text"]),
                visible_dialogue=dialogue,
                observable_flags=flags,
                ranked_card=ranked[0] if ranked else None,
                bank_card=top_card,
            ),
        )
        feature_rows.append(
            {
                **asdict(rs_feature),
                "split": raw["split"],
                "group_id": raw["group_id"],
                "semantic_family": raw["logic_family"],
            }
        )
        candidate_rows.append(
            {
                "protocol": args.protocol,
                "state_id": raw["state_id"],
                "user_id": raw["user_id"],
                "group_id": raw["group_id"],
                "split": raw["split"],
                "logic_family_private_not_model_input": raw["logic_family"],
                "topic_family_private_not_model_input": raw["topic_family"],
                "visible_dialogue": visible_runtime,
                "current_user_text": raw["current_user_text"],
                "candidate_surfaces": surfaces,
                "private_rankings_not_model_input": private_rankings,
                "h1_gold": None,
                "construction_intent_present": False,
                "response_or_outcome_read": False,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.out_dir / "candidate_rows_private.jsonl"
    feature_path = args.out_dir / "model_feature_rows.jsonl"
    write_jsonl(candidate_path, candidate_rows)
    write_jsonl(feature_path, feature_rows)
    presence = Counter(
        (component, bool(row["candidate_surfaces"][component]["candidate_present"]))
        for row in candidate_rows
        for component in ("MP", "MS", "ME", "RS")
    )
    realization = audit_construction_realization(
        candidate_rows=candidate_rows,
        blueprints_by_state=blueprints_by_state,
    )
    write_json(args.out_dir / "construction_realization_private.json", realization)
    report = {
        "protocol": args.protocol,
        "status": (
            "PASS_COMPLETE"
            if len(candidate_rows) == args.expected_states
            else "PASS_PARTIAL_SMOKE"
        ),
        "expected_states": args.expected_states,
        "states": len(candidate_rows),
        "feature_rows": len(feature_rows),
        "candidate_presence": {
            component: {
                "present": presence[(component, True)],
                "absent": presence[(component, False)],
            }
            for component in ("MP", "MS", "ME", "RS")
        },
        "memory_compiler_protocol": BOUNDED_MEMORY_COMPILER_PROTOCOL,
        "memory_candidate_protocol": FINAL_TYPED_CANDIDATE_DISCOVERY_PROTOCOL,
        "strategy_ranker_protocol": rs_ranker_protocol,
        "exact_rank1_binding_rate": 1.0,
        "feature_rows_per_state": 4,
        "construction_intent_present": False,
        "response_or_outcome_read": False,
        "candidate_rows_sha256": sha256_file(candidate_path),
        "model_feature_rows_sha256": sha256_file(feature_path),
        "construction_realization_status": realization["status"],
        "construction_realization_rate": realization[
            "mechanically_checkable_rate"
        ],
        "construction_realization_is_gold": False,
    }
    write_json(args.out_dir / "materialization_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
