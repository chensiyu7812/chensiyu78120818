#!/usr/bin/env python3
"""Materialize the frozen, outcome-blind V5 FIT features and training contract."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

from metacom_pm.contracts import MemorySource
from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl
from metacom_pm.retrieval import source_specific_memory_queries
from metacom_pm.v1_5_candidate_discovery import (
    discover_final_typed_memory_candidates,
)
from metacom_pm.v1_5_final_candidate_contract import (
    FINAL_CANDIDATE_CONTRACT_PROTOCOL,
    FINAL_FEATURE_CONTRACT_PROTOCOL,
    FINAL_FEATURE_NAMES,
    IndependentFeatureRecord,
    final_model_feature_projection,
    final_rs_model_features,
)
from metacom_pm.v1_5_memory_opportunity_features import (
    SOURCE_SPECIFIC_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL,
    build_source_specific_memory_opportunity_observation,
)
from metacom_pm.v1_5_memory_transport import compile_bounded_memory_with_metadata
from metacom_pm.v1_5b_policy_runtime import COMPONENTS


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5-fit-training-surface-materialization-v1"
RS_FEATURE_PROTOCOL = "pm-v1.5-final-rs-transparent-features-v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def materialize(
    *,
    states_path: Path,
    blueprint_path: Path,
    candidates_path: Path,
    strategy_cards_path: Path,
    freeze_contract_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    states = {str(row["state_id"]): row for row in _rows(states_path)}
    blueprint = {
        str(row["blueprint_row_id"]): row for row in _rows(blueprint_path)
    }
    candidates = [
        row
        for row in _rows(candidates_path)
        if row["track_private_not_model_input"] == "COMPONENT_EFFECT"
        and row["split_private_not_model_input"] == "EFFECT_FIT"
    ]
    cards = {str(row["card_id"]): row for row in _rows(strategy_cards_path)}
    freeze = json.loads(freeze_contract_path.read_text(encoding="utf-8"))
    if len(candidates) != 256 or len(cards) != 80:
        raise RuntimeError("V5 FIT feature surface requires 256 units and 80 cards")

    feature_rows: list[dict[str, Any]] = []
    group_members: dict[str, list[str]] = {}
    for row in sorted(candidates, key=lambda value: str(value["state_id"])):
        state_id = str(row["state_id"])
        component = str(row["target_component_private_not_model_input"])
        if component not in COMPONENTS:
            raise RuntimeError(f"unknown component: {component}")
        state = states[state_id]
        source = blueprint[state_id]
        if (
            list(state["visible_dialogue"]) != list(row["visible_dialogue"])
            or str(state["current_user_text"]) != str(row["current_user_text"])
        ):
            raise RuntimeError(f"visible state drift: {state_id}")
        exact = dict(row["exact_rank1_candidate"])
        if exact.get("protocol") != FINAL_CANDIDATE_CONTRACT_PROTOCOL:
            raise RuntimeError(f"exact candidate contract drift: {state_id}")

        if component in {"MP", "MS", "ME"}:
            items, _session_docs, metadata = compile_bounded_memory_with_metadata(
                state["user"]
            )
            queries = source_specific_memory_queries(
                str(state["current_user_text"]),
                list(state["visible_dialogue"]),
                "",
            )
            discovered = discover_final_typed_memory_candidates(
                queries=queries,
                items=items,
                source_metadata=metadata,
                session_index=int(state["current_session_index"]),
            )
            memory_source = MemorySource(component)
            candidate = discovered[memory_source]
            if not candidate.selected_items:
                raise RuntimeError(f"frozen FIT candidate disappeared: {state_id}")
            if candidate.selected_items[0].memory_id != exact["candidate_id"]:
                raise RuntimeError(f"rank-1 candidate drift: {state_id}")
            observation = build_source_specific_memory_opportunity_observation(
                candidate=candidate,
                catalog_items=items,
                catalog_user_id=str(state["user_id"]),
                current_user_id=str(state["user_id"]),
                current_session_index=int(state["current_session_index"]),
                current_user_text=str(state["current_user_text"]),
                visible_dialogue=list(state["visible_dialogue"]),
                source_metadata=metadata,
                background_action="M0+R0",
            )
            values = final_model_feature_projection(
                component=component,
                source_features=observation["model_features"],
            )
            builder_protocol = SOURCE_SPECIFIC_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL
        else:
            candidate_id = str(exact["candidate_id"])
            card = cards.get(candidate_id)
            if card is None:
                raise RuntimeError(f"frozen RS candidate missing from bank: {state_id}")
            audit = dict(row["private_retrieval_audit_not_model_input"])
            values = final_rs_model_features(
                current_user_text=str(state["current_user_text"]),
                visible_dialogue=list(state["visible_dialogue"]),
                observable_flags=dict(audit["observable_flags"]),
                ranked_card={"card_id": candidate_id},
                bank_card=card,
            )
            builder_protocol = RS_FEATURE_PROTOCOL

        feature = IndependentFeatureRecord(
            protocol=FINAL_FEATURE_CONTRACT_PROTOCOL,
            state_id=state_id,
            component=component,  # type: ignore[arg-type]
            feature_builder_protocol=builder_protocol,
            model_features=values,
        )
        semantic_group = str(source["counterfactual_group_id"])
        group_members.setdefault(semantic_group, []).append(state_id)
        feature_rows.append(
            {
                **asdict(feature),
                "semantic_group_id_private_cv_only": semantic_group,
                "semantic_family_private_cv_only": str(source["logic_family"]),
                "state_candidate_unit_weight_before_class_balance": 0.5,
                "source_candidate_text_sha256": str(exact["candidate_text_sha256"]),
                "source_candidate_rows_sha256": sha256_file(candidates_path),
                "human_fit_outcome_labels_read": 0,
                "generated_response_or_execution_outcome_read": False,
                "external_lockbox_read": False,
            }
        )

    if len(feature_rows) != 256:
        raise RuntimeError("V5 feature row count drift")
    if len(group_members) != 128 or any(
        len(members) != 2 for members in group_members.values()
    ):
        raise RuntimeError("V5 requires 128 two-unit counterfactual groups")
    for component in COMPONENTS:
        rows = [row for row in feature_rows if row["component"] == component]
        if len(rows) != 64:
            raise RuntimeError(f"{component} feature row count drift")
        if any(
            set(row["model_features"]) != set(FINAL_FEATURE_NAMES[component])
            for row in rows
        ):
            raise RuntimeError(f"{component} feature schema drift")

    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = out_dir / "fit_feature_rows_private.jsonl"
    write_jsonl(feature_path, feature_rows)
    report = {
        "protocol": PROTOCOL,
        "status": "FROZEN_READY_FOR_SINGLE_HUMAN_OUTCOME_PANEL",
        "state_candidate_units": 256,
        "counterfactual_groups": 128,
        "units_per_component": {component: 64 for component in COMPONENTS},
        "groups_per_component": {component: 32 for component in COMPONENTS},
        "feature_names": {
            component: list(FINAL_FEATURE_NAMES[component])
            for component in COMPONENTS
        },
        "model": dict(freeze["primary_model"]),
        "promotion_gates": dict(freeze["promotion_gates"]),
        "inputs_sha256": {
            str(states_path.relative_to(ROOT)): sha256_file(states_path),
            str(blueprint_path.relative_to(ROOT)): sha256_file(blueprint_path),
            str(candidates_path.relative_to(ROOT)): sha256_file(candidates_path),
            str(strategy_cards_path.relative_to(ROOT)): sha256_file(
                strategy_cards_path
            ),
            str(freeze_contract_path.relative_to(ROOT)): sha256_file(
                freeze_contract_path
            ),
        },
        "fit_feature_rows_sha256": sha256_file(feature_path),
        "human_fit_outcome_labels_read": 0,
        "generated_response_or_execution_outcome_read": False,
        "external_lockbox_read": False,
        "post_label_feature_or_hyperparameter_change_allowed": False,
    }
    write_json(out_dir / "freeze_report.json", report)
    return report


def main() -> None:
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V5 requires {FORMAL_PYTHON}; got {sys.executable}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--states",
        type=Path,
        default=ROOT / "data/pm_v1_5_v3_visible_states_v2/private/raw_states.jsonl",
    )
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_v3_effect_blueprint_v1/private/construction_blueprint.jsonl",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_v3_p2_exact_rank1_v4/candidate_rows_private.jsonl",
    )
    parser.add_argument(
        "--strategy-cards",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl",
    )
    parser.add_argument(
        "--freeze-contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/v5_fit_training_surface_freeze_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_fit_training_surface_v1",
    )
    args = parser.parse_args()
    report = materialize(
        states_path=args.states,
        blueprint_path=args.blueprint,
        candidates_path=args.candidates,
        strategy_cards_path=args.strategy_cards,
        freeze_contract_path=args.freeze_contract,
        out_dir=args.out_dir,
    )
    print(report)


if __name__ == "__main__":
    main()
