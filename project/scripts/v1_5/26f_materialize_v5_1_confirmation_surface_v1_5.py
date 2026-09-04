#!/usr/bin/env python3
"""Materialize the outcome-blind V5.1 confirmation feature/data surface.

This is the last pre-API examination gate.  It rediscoveres the frozen actual
Rank-1 candidate, builds the exact V5 features, and checks that the exam does
not duplicate FIT decision surfaces or expose split/construction metadata.
It never reads generated replies, human outcomes, sealed data, or external
evaluation results.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys
from typing import Any
import unicodedata

from metacom_pm.contracts import MemorySource
from metacom_pm.io import canonical_json, iter_jsonl, sha256_file, sha256_text, write_json, write_jsonl
from metacom_pm.retrieval import source_specific_memory_queries
from metacom_pm.v1_5_candidate_discovery import discover_final_typed_memory_candidates
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
PROTOCOL = "pm-v1.5-v5.1-confirmation-surface-materialization-v1"
RS_FEATURE_PROTOCOL = "pm-v1.5-final-rs-transparent-features-v1"


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def decision_surface_signature(row: dict[str, Any]) -> str:
    dialogue = [
        {"role": str(turn["role"]), "content": normalize_text(str(turn["content"]))}
        for turn in row["visible_dialogue"]
    ]
    payload = {
        "component": str(row["target_component_private_not_model_input"]),
        "visible_dialogue": dialogue,
        "current_user_text": normalize_text(str(row["current_user_text"])),
        "actual_rank1_candidate": normalize_text(
            str(row["exact_rank1_candidate"]["candidate_text"])
        ),
    }
    return sha256_text(canonical_json(payload))


def materialize(
    *,
    states_path: Path,
    blueprint_path: Path,
    candidates_path: Path,
    strategy_cards_path: Path,
    contract_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    states = {str(row["state_id"]): row for row in rows(states_path)}
    blueprint = {str(row["blueprint_row_id"]): row for row in rows(blueprint_path)}
    all_effect = [
        row
        for row in rows(candidates_path)
        if row["track_private_not_model_input"] == "COMPONENT_EFFECT"
    ]
    fit_rows = [
        row for row in all_effect if row["split_private_not_model_input"] == "EFFECT_FIT"
    ]
    confirmation = [
        row
        for row in all_effect
        if row["split_private_not_model_input"] == "FRESH_CONFIRMATION"
    ]
    cards = {str(row["card_id"]): row for row in rows(strategy_cards_path)}
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    if contract.get("status") != "FROZEN_BEFORE_CONFIRMATION_OUTCOMES":
        raise RuntimeError("V5.1 system contract is not frozen")
    if len(fit_rows) != 256 or len(confirmation) != 128 or len(cards) != 80:
        raise RuntimeError("unexpected FIT/confirmation/card shape")
    observed = Counter(
        str(row["target_component_private_not_model_input"]) for row in confirmation
    )
    if observed != Counter({component: 32 for component in COMPONENTS}):
        raise RuntimeError(f"confirmation component shape mismatch: {observed}")

    fit_ids = {str(row["state_id"]) for row in fit_rows}
    confirmation_ids = {str(row["state_id"]) for row in confirmation}
    if fit_ids & confirmation_ids:
        raise RuntimeError("FIT and confirmation state IDs overlap")
    fit_groups = {str(blueprint[state_id]["counterfactual_group_id"]) for state_id in fit_ids}
    confirmation_groups = {
        str(blueprint[state_id]["counterfactual_group_id"])
        for state_id in confirmation_ids
    }
    if fit_groups & confirmation_groups:
        raise RuntimeError("FIT and confirmation counterfactual groups overlap")
    group_members: dict[str, list[str]] = defaultdict(list)
    for state_id in confirmation_ids:
        group_members[str(blueprint[state_id]["counterfactual_group_id"])].append(state_id)
    if len(group_members) != 64 or any(len(value) != 2 for value in group_members.values()):
        raise RuntimeError("confirmation requires 64 two-unit counterfactual groups")

    fit_signatures = {decision_surface_signature(row) for row in fit_rows}
    confirmation_signatures = [decision_surface_signature(row) for row in confirmation]
    if len(set(confirmation_signatures)) != 128:
        raise RuntimeError("confirmation contains duplicate decision surfaces")
    cross_split_duplicate_count = sum(
        signature in fit_signatures for signature in confirmation_signatures
    )
    if cross_split_duplicate_count:
        raise RuntimeError("confirmation decision surface duplicates EFFECT_FIT")

    forbidden_visible_tokens = re.compile(
        r"\b(?:effect[_ -]?fit|fresh[_ -]?confirmation|sealed[_ -]?internal|"
        r"private[_ -]?benefit[_ -]?enrichment|construction[_ -]?intent)\b",
        re.IGNORECASE,
    )
    for row in confirmation:
        if row.get("construction_intent_present_in_model_input") is not False:
            raise RuntimeError("construction intent entered the model input")
        if row.get("response_or_outcome_read") is not False:
            raise RuntimeError("confirmation materialization read a response/outcome")
        exact = dict(row["exact_rank1_candidate"])
        metadata = dict(row["structured_candidate_metadata"])
        if not exact.get("candidate_present") or int(exact.get("selected_rank", 0)) != 1:
            raise RuntimeError(f"missing actual Rank-1 candidate: {row['state_id']}")
        if row["target_component_private_not_model_input"] == "RS":
            if metadata.get("candidate_owner_id") != "SHARED_STRATEGY_BANK":
                raise RuntimeError(f"RS bank owner mismatch: {row['state_id']}")
        elif metadata.get("candidate_owner_id") != metadata.get("state_owner_id"):
            raise RuntimeError(f"candidate owner mismatch: {row['state_id']}")
        visible = " ".join(
            [str(turn["content"]) for turn in row["visible_dialogue"]]
            + [str(row["current_user_text"])]
        )
        if forbidden_visible_tokens.search(visible):
            raise RuntimeError(f"split/construction token leaked: {row['state_id']}")

    feature_rows: list[dict[str, Any]] = []
    for row in sorted(confirmation, key=lambda value: str(value["state_id"])):
        state_id = str(row["state_id"])
        component = str(row["target_component_private_not_model_input"])
        state = states[state_id]
        source = blueprint[state_id]
        if (
            list(state["visible_dialogue"]) != list(row["visible_dialogue"])
            or str(state["current_user_text"]) != str(row["current_user_text"])
        ):
            raise RuntimeError(f"visible state drift: {state_id}")
        exact = dict(row["exact_rank1_candidate"])
        if exact.get("protocol") != FINAL_CANDIDATE_CONTRACT_PROTOCOL:
            raise RuntimeError(f"candidate contract drift: {state_id}")

        if component in {"MP", "MS", "ME"}:
            items, _session_docs, metadata = compile_bounded_memory_with_metadata(state["user"])
            queries = source_specific_memory_queries(
                str(state["current_user_text"]), list(state["visible_dialogue"]), ""
            )
            discovered = discover_final_typed_memory_candidates(
                queries=queries,
                items=items,
                source_metadata=metadata,
                session_index=int(state["current_session_index"]),
            )
            candidate = discovered[MemorySource(component)]
            if not candidate.selected_items:
                raise RuntimeError(f"confirmation candidate disappeared: {state_id}")
            if candidate.selected_items[0].memory_id != exact["candidate_id"]:
                raise RuntimeError(f"actual Rank-1 drift: {state_id}")
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
                component=component, source_features=observation["model_features"]
            )
            builder_protocol = SOURCE_SPECIFIC_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL
        else:
            candidate_id = str(exact["candidate_id"])
            card = cards.get(candidate_id)
            if card is None:
                raise RuntimeError(f"confirmation RS card missing: {state_id}")
            audit = dict(row["private_retrieval_audit_not_model_input"])
            values = final_rs_model_features(
                current_user_text=str(state["current_user_text"]),
                visible_dialogue=list(state["visible_dialogue"]),
                observable_flags=dict(audit["observable_flags"]),
                ranked_card={"card_id": candidate_id},
                bank_card=card,
            )
            builder_protocol = RS_FEATURE_PROTOCOL

        if set(values) != set(FINAL_FEATURE_NAMES[component]):
            raise RuntimeError(f"feature schema drift: {state_id}")
        feature = IndependentFeatureRecord(
            protocol=FINAL_FEATURE_CONTRACT_PROTOCOL,
            state_id=state_id,
            component=component,  # type: ignore[arg-type]
            feature_builder_protocol=builder_protocol,
            model_features=values,
        )
        feature_rows.append(
            {
                **asdict(feature),
                "semantic_group_id_private_analysis_only": str(
                    source["counterfactual_group_id"]
                ),
                "semantic_family_private_analysis_only": str(source["logic_family"]),
                "source_candidate_text_sha256": str(exact["candidate_text_sha256"]),
                "decision_surface_sha256": decision_surface_signature(row),
                "construction_intent_or_enrichment_in_model_input": False,
                "confirmation_response_or_label_read": False,
                "sealed_or_external_read": False,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = out_dir / "confirmation_feature_rows_private.jsonl"
    write_jsonl(feature_path, feature_rows)
    subtype_enrichment = Counter(
        (
            str(row["target_component_private_not_model_input"]),
            str(row["exact_rank1_candidate"]["compiler_subtype_hint"]),
            str(blueprint[str(row["state_id"])]["private_benefit_enrichment"]),
        )
        for row in confirmation
    )
    candidate_only_duplicates = len(
        {
            (
                str(row["target_component_private_not_model_input"]),
                normalize_text(str(row["exact_rank1_candidate"]["candidate_text"])),
            )
            for row in confirmation
            if any(
                str(fit["target_component_private_not_model_input"])
                == str(row["target_component_private_not_model_input"])
                and normalize_text(str(fit["exact_rank1_candidate"]["candidate_text"]))
                == normalize_text(str(row["exact_rank1_candidate"]["candidate_text"]))
                for fit in fit_rows
            )
        }
    )
    report = {
        "protocol": PROTOCOL,
        "status": "PASS_READY_FOR_FROZEN_CONFIRMATION_CALL_PLAN",
        "state_candidate_units": len(feature_rows),
        "counterfactual_groups": len(group_members),
        "units_per_component": dict(sorted(observed.items())),
        "checks": {
            "same_frozen_retriever_actual_rank1_rediscovered": True,
            "owner_and_time_binding_valid": True,
            "fit_confirmation_state_id_overlap_zero": True,
            "fit_confirmation_counterfactual_group_overlap_zero": True,
            "fit_confirmation_decision_surface_duplicate_zero": True,
            "confirmation_internal_decision_surface_duplicate_zero": True,
            "pair_members_share_one_statistical_group": True,
            "construction_split_or_enrichment_absent_from_model_input": True,
            "response_label_sealed_and_external_read_zero": True,
        },
        "audited_not_prohibited": {
            "candidate_text_templates_shared_across_splits": candidate_only_duplicates,
            "interpretation": (
                "The same resource technique/template may recur across new users/topics; "
                "the complete current-state-plus-actual-candidate decision surface may not."
            ),
            "subtype_by_private_enrichment_counts": {
                "|".join(key): value for key, value in sorted(subtype_enrichment.items())
            },
        },
        "input_sha256": {
            str(states_path.relative_to(ROOT)): sha256_file(states_path),
            str(blueprint_path.relative_to(ROOT)): sha256_file(blueprint_path),
            str(candidates_path.relative_to(ROOT)): sha256_file(candidates_path),
            str(strategy_cards_path.relative_to(ROOT)): sha256_file(strategy_cards_path),
            str(contract_path.relative_to(ROOT)): sha256_file(contract_path),
        },
        "confirmation_feature_rows_sha256": sha256_file(feature_path),
        "api_calls": 0,
        "human_or_generated_outcomes_read": 0,
        "sealed_or_external_read": False,
    }
    write_json(out_dir / "data_integrity_report.json", report)
    return report


def main() -> None:
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V5.1 requires {FORMAL_PYTHON}; got {sys.executable}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--states",
        type=Path,
        default=ROOT / "data/pm_v1_5_v3_visible_states_v2/private/raw_states.jsonl",
    )
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT / "data/pm_v1_5_v3_effect_blueprint_v1/private/construction_blueprint.jsonl",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_p2_exact_rank1_v4/candidate_rows_private.jsonl",
    )
    parser.add_argument(
        "--strategy-cards",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "data/pm_v1_5_contracts/v5_1_system_pareto_confirmation_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_1_confirmation_surface_v1",
    )
    args = parser.parse_args()
    print(
        materialize(
            states_path=args.states,
            blueprint_path=args.blueprint,
            candidates_path=args.candidates,
            strategy_cards_path=args.strategy_cards,
            contract_path=args.contract,
            out_dir=args.out_dir,
        )
    )


if __name__ == "__main__":
    main()
