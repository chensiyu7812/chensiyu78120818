"""Formal same-stack exact Rank-1 materialization for the V3 P2 states."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from .contracts import MemorySource
from .io import canonical_json, sha256_text
from .retrieval import source_specific_memory_queries
from .text import normalize_for_hash, normalize_space
from .v1_5_candidate_discovery import (
    FINAL_TYPED_CANDIDATE_DISCOVERY_PROTOCOL,
    discover_final_typed_memory_candidates,
)
from .v1_5_final_candidate_contract import (
    ExactRank1CandidateSurface,
    exact_rank1_memory_surface,
    exact_rank1_strategy_surface,
)
from .v1_5_memory_transport import (
    BOUNDED_MEMORY_COMPILER_PROTOCOL,
    compile_bounded_memory_with_metadata,
)
from .v1_5_strategy_rag_repair import (
    PRE_PM_CANDIDATE_RANK_PROTOCOL,
    effect_study_observable_flags,
    rank_pre_pm_strategy_candidates,
)
from .v1_5_v3_state_realization import TOPIC_WORDS


PROTOCOL = "pm-v1.5-v3-p2-exact-rank1-materialization-v4"

_RS_FAMILY_BY_FUNCTION = {
    "focused_question": "Question",
    "tentative_paraphrase_check": "Restatement or Paraphrasing",
    "evidence_grounded_reflection": "Reflection of feelings",
    "one_reversible_suggestion": "Providing Suggestions",
}


def _rs_dialogue(state: Mapping[str, Any]) -> list[dict[str, str]]:
    dialogue = [
        {
            "speaker": "seeker" if row["role"] == "user" else "supporter",
            "content": str(row["content"]),
        }
        for row in state["visible_dialogue"]
    ]
    dialogue.append(
        {"speaker": "seeker", "content": str(state["current_user_text"])}
    )
    return dialogue


def _rs_query(dialogue: Sequence[Mapping[str, str]], flags: Mapping[str, Any]) -> str:
    seekers = [str(row["content"]) for row in dialogue if row["speaker"] == "seeker"]
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
    return normalize_space(
        "Represent this sentence for searching relevant passages: Choose one safe, "
        "topic-agnostic emotional-support technique. Observable cues: "
        + (", ".join(cues) or "none")
        + ". Recent seeker context: "
        + " ".join(seekers[-3:])
    )


def _target_memory_expected_id(
    *,
    component: str,
    state: Mapping[str, Any],
    items: Sequence[Any],
    metadata: Mapping[str, Mapping[str, Any]],
) -> str:
    if component == "MP":
        # Canonical JSON serialization sorts object keys.  Target binding must
        # therefore use the explicit field namespace, never dictionary order.
        target_fields = [
            str(field)
            for field in state["user"]["basic_info"]
            if str(field).startswith("stable_")
        ]
        if len(target_fields) != 1:
            raise RuntimeError(
                f"MP target field is not unique for {state['state_id']}: {target_fields}"
            )
        target_field = target_fields[0]
        matches = [
            item.memory_id
            for item in items
            if item.source is MemorySource.MP
            and str(metadata[item.memory_id].get("profile_field")) == target_field
        ]
    else:
        source = MemorySource(component)
        target_session = int(state["current_session_index"]) - 1
        matches = [
            item.memory_id
            for item in items
            if item.source is source and int(item.created_session) == target_session
        ]
    if len(matches) != 1:
        raise RuntimeError(
            f"target construction item is not unique for {state['state_id']}/{component}: {matches}"
        )
    return str(matches[0])


def materialize_v3_candidates(
    *,
    states: Sequence[Mapping[str, Any]],
    blueprint_rows: Sequence[Mapping[str, Any]],
    strategy_cards: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    blueprint_by_id = {
        str(row["blueprint_row_id"]): row for row in blueprint_rows
    }
    cards = [dict(row) for row in strategy_cards]
    cards_by_id = {str(row["card_id"]): row for row in cards}
    if len(cards) != 80 or len(cards_by_id) != 80:
        raise RuntimeError("V3 P2 requires the frozen 80-card Strategy Bank")
    failures: list[str] = []
    candidate_rows: list[dict[str, Any]] = []
    presence = Counter()
    subtype_match = Counter()
    expected_item_match = Counter()
    catalog_size_match = Counter()
    rs_family_match = Counter()
    for state in states:
        state_id = str(state["state_id"])
        blueprint = blueprint_by_id[state_id]
        component = str(blueprint["target_component"])
        items, _docs, metadata = compile_bounded_memory_with_metadata(state["user"])
        current_session_index = int(state["current_session_index"])
        if any(int(item.created_session) >= current_session_index for item in items):
            raise RuntimeError("current or future memory entered V3 P2")
        query_by_source = source_specific_memory_queries(
            str(state["current_user_text"]),
            list(state["visible_dialogue"]),
            "",
        )
        discoveries = discover_final_typed_memory_candidates(
            queries=query_by_source,
            items=items,
            source_metadata=metadata,
            session_index=current_session_index,
        )
        private_ranking: dict[str, Any]
        if component in {"MP", "MS", "ME"}:
            source = MemorySource(component)
            discovery = discoveries[source]
            surface = exact_rank1_memory_surface(
                state_id=state_id,
                candidate=discovery,
                source_metadata=metadata,
                current_session_index=current_session_index,
                compiler_protocol=BOUNDED_MEMORY_COMPILER_PROTOCOL,
            )
            expected_id = _target_memory_expected_id(
                component=component,
                state=state,
                items=items,
                metadata=metadata,
            )
            source_count = sum(item.source is source for item in items)
            expected_item_match[component] += int(surface.candidate_id == expected_id)
            catalog_size_match[component] += int(
                source_count == int(blueprint["source_catalog_size_target"])
            )
            private_ranking = {
                "selected_memory_ids": [
                    item.memory_id for item in discovery.selected_items
                ],
                "descriptor": discovery.descriptor,
                "target_construction_item_id": expected_id,
                "source_catalog_count": source_count,
            }
            selected_metadata = (
                dict(metadata[str(surface.candidate_id)])
                if surface.candidate_present and surface.candidate_id
                else {}
            )
            structured_metadata = {
                "candidate_present": bool(surface.candidate_present),
                "state_owner_id": str(state["user_id"]),
                "candidate_owner_id": selected_metadata.get("owner_id"),
                "candidate_active": selected_metadata.get("record_active"),
                "candidate_superseded": selected_metadata.get(
                    "record_superseded"
                ),
                "candidate_content_sha256": surface.candidate_text_sha256,
                "metadata_source": "shared_bounded_memory_compiler",
            }
        else:
            dialogue = _rs_dialogue(state)
            seekers = [
                row["content"] for row in dialogue if row["speaker"] == "seeker"
            ]
            recent = normalize_space(" ".join(seekers[-3:]))
            flags = effect_study_observable_flags(
                current_user_text=str(state["current_user_text"]),
                recent_user_text=recent,
                visible_dialogue=dialogue,
            )
            ranked = rank_pre_pm_strategy_candidates(
                query=_rs_query(dialogue, flags),
                current_user_text=str(state["current_user_text"]),
                recent_user_text=recent,
                visible_dialogue=dialogue,
                cards=cards,
            )
            surface = exact_rank1_strategy_surface(
                state_id=state_id,
                ranked_cards=ranked,
                cards_by_id=cards_by_id,
                compiler_protocol=PRE_PM_CANDIDATE_RANK_PROTOCOL,
            )
            expected_family = _RS_FAMILY_BY_FUNCTION[
                str(blueprint["required_candidate_function"])
            ]
            actual_family = (
                str(cards_by_id[str(surface.candidate_id)]["strategy_family"])
                if surface.candidate_present
                else None
            )
            rs_family_match[component] += int(actual_family == expected_family)
            expected_item_match[component] += int(actual_family == expected_family)
            catalog_size_match[component] += int(
                len(cards) == int(blueprint["source_catalog_size_target"])
            )
            private_ranking = {
                "ranked_card_ids": [str(row["card_id"]) for row in ranked],
                "observable_flags": flags,
                "expected_strategy_family": expected_family,
                "actual_strategy_family": actual_family,
                "source_catalog_count": len(cards),
            }
            structured_metadata = {
                "candidate_present": bool(surface.candidate_present),
                "state_owner_id": str(state["user_id"]),
                "candidate_owner_id": "SHARED_STRATEGY_BANK",
                "candidate_active": bool(surface.candidate_present),
                "candidate_superseded": False,
                "candidate_content_sha256": surface.candidate_text_sha256,
                "metadata_source": "shared_strategy_bank",
            }
        surface_row = asdict(surface)
        presence[component] += int(bool(surface.candidate_present))
        subtype_match[component] += int(
            surface.compiler_subtype_hint.value
            == str(blueprint["candidate_subtype_target"])
        )
        candidate_rows.append(
            {
                "protocol": PROTOCOL,
                "state_id": state_id,
                "user_id_private_not_model_input": str(state["user_id"]),
                "group_id_private_not_model_input": str(state["group_id"]),
                "split_private_not_model_input": str(state["split"]),
                "track_private_not_model_input": str(state["track"]),
                "target_component_private_not_model_input": component,
                "visible_dialogue": list(state["visible_dialogue"]),
                "current_user_text": str(state["current_user_text"]),
                "exact_rank1_candidate": surface_row,
                "structured_candidate_metadata": structured_metadata,
                "private_retrieval_audit_not_model_input": private_ranking,
                "construction_intent_present_in_model_input": False,
                "human_gold": None,
                "response_or_outcome_read": False,
                "external_text_read": False,
            }
        )

    decision_hashes = [
        sha256_text(
            canonical_json(
                {
                    "visible_dialogue": row["visible_dialogue"],
                    "current_user_text": row["current_user_text"],
                    "component": row["target_component_private_not_model_input"],
                    "candidate_text": row["exact_rank1_candidate"]["candidate_text"],
                }
            )
        )
        for row in candidate_rows
    ]
    duplicate_decision_surfaces = len(decision_hashes) - len(set(decision_hashes))
    expected_by_component = Counter(
        str(row["target_component"]) for row in blueprint_rows
    )
    for component, expected in expected_by_component.items():
        if presence[component] != expected:
            failures.append(f"candidate_absent_{component}")
        if subtype_match[component] != expected:
            failures.append(f"compiler_subtype_mismatch_{component}")
        if expected_item_match[component] != expected:
            failures.append(f"formal_rank1_missed_constructed_target_{component}")
        if catalog_size_match[component] != expected:
            failures.append(f"source_catalog_size_mismatch_{component}")
    if duplicate_decision_surfaces:
        failures.append("duplicate_complete_step1_decision_surface")
    if any(
        row["construction_intent_present_in_model_input"]
        or row["response_or_outcome_read"]
        or row["external_text_read"]
        or row["human_gold"] is not None
        for row in candidate_rows
    ):
        failures.append("forbidden_input_or_label_leakage")
    report = {
        "protocol": PROTOCOL,
        "status": "PASS" if not failures else "FAIL",
        "states": len(candidate_rows),
        "exact_rank1_present": dict(presence),
        "compiler_subtype_matches": dict(subtype_match),
        "formal_rank1_constructed_target_matches": dict(expected_item_match),
        "source_catalog_size_matches": dict(catalog_size_match),
        "rs_family_matches": dict(rs_family_match),
        "duplicate_complete_step1_decision_surfaces": duplicate_decision_surfaces,
        "memory_compiler_protocol": BOUNDED_MEMORY_COMPILER_PROTOCOL,
        "memory_candidate_protocol": FINAL_TYPED_CANDIDATE_DISCOVERY_PROTOCOL,
        "strategy_candidate_protocol": PRE_PM_CANDIDATE_RANK_PROTOCOL,
        "api_calls": 0,
        "human_labels_read": 0,
        "external_lockbox_read": False,
        "responses_generated": 0,
        "construction_intent_is_gold": False,
        "scientific_interpretation": (
            "P2 proves only that the actual frozen compiler/retriever produces the "
            "intended exact Rank-1 decision surfaces. It does not prove eligibility, "
            "resource benefit, generator execution, or PM learning."
        ),
        "failures": failures,
    }
    return candidate_rows, report
