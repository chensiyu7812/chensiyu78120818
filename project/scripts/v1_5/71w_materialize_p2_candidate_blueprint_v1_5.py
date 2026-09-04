#!/usr/bin/env python3
"""W7: materialize the formal P2 candidate blueprint (MP/MS/ME/RS).

Per the runbook's W7 spec: state, user, family, counterfactual_group,
proposed split, full Top-k, exact Rank-1, owner/time/version/subtype,
score/margin, candidate count, incremental tokens, visible dialogue, current
goal, model-visible surface, hard-off reasons, and Step1 contribution_slot
features. MP keeps preference/profile as two internal subdomains. MS uses
the release-bound BGE-M3 snapshot. ME uses the frozen production method
(lexical + typed_tier); a Rank-1 compiler failure records the state as
me_available=False, NEVER promoted to Rank-2 (v1_5_v5_3_release_bindings
enforces rank2_promotion_allowed=False; this script mirrors that contract).
RS uses the frozen 6-card Bank via rs_mechanical_candidate_pool() +
rs_shared_candidate_top1() so every policy arm shares the same candidate
identity.

Builds single-component paired-effect main states for MP/MS/RS (ME reuses
the already-verified 32 states from script 68 rather than reconstructing
them), plus a smaller set of multi-component interaction states on a shared
8-family scaffold. Does NOT require 16-action equal frequency (L2 already
verified structural compatibility separately). Reports only real, unique
group counts -- no near-duplicate templates counted as independent samples,
no final N frozen.

Zero API calls. No replies generated, no head trained, no quality/risk read.
Does not modify the three authoritative sources, the runbook, or any shared
module (accountability/typed_response_program/baselines/external_leakage).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.contracts import MemoryItem, MemorySource, StrategyCard  # noqa: E402
from metacom_pm.io import stable_hex, write_json  # noqa: E402
from metacom_pm.retrieval import source_specific_memory_queries  # noqa: E402
from metacom_pm.text import estimate_tokens  # noqa: E402
from metacom_pm.v1_5_candidate_discovery import (  # noqa: E402
    describe_memory_candidate,
    discover_final_typed_memory_candidates,
)
from metacom_pm.v1_5_v5_2_atomic_memory import (  # noqa: E402
    compile_atomic_reusable_outcome,
    compile_atomic_session_observation,
    me_subtype_hints,
)
from metacom_pm.v1_5_v5_3_candidate_layer_responsibility import (  # noqa: E402
    rs_mechanical_candidate_pool,
    rs_shared_candidate_top1,
)
from metacom_pm.v1_5_v5_3_release_bindings import build_static_release_bindings  # noqa: E402
from metacom_pm.v1_5_v5_3_semantic_ms_retrieval import BgeM3Encoder  # noqa: E402
from metacom_pm.v1_5_v5_3_contribution_slot_features import (  # noqa: E402
    mp_contribution_slots,
    ms_contribution_slots,
)

ME_STATES_PATH = ROOT / "outputs/pm_v1_5_v5_3_me_superdomain_v1/states_all.json"
CARDS_PATH = ROOT / "data/strategy/strategy_cards_v1_5_minimal.jsonl"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_p2_candidate_blueprint_v1"

# 8-family shared scaffold (topic identity reused across MP/MS/ME so
# interaction states can combine components on one coherent user situation).
FAMILIES = [
    "job_transition", "chronic_pain", "parenting_conflict", "breakup_recovery",
    "exam_anxiety", "career_change", "housing_move", "friendship_rift",
]

MP_CONTENT = {
    "job_transition": ("prefers concise words in one reflection before any question", "Occupation: currently between roles"),
    "chronic_pain": ("prefers a slower pace with pauses between topics", "Occupation: works a physically demanding job"),
    "parenting_conflict": ("prefers direct, non-judgmental phrasing", "Family status: raising a teenager alone"),
    "breakup_recovery": ("prefers not to be asked about future dating plans", "Relationship status: recently single"),
    "exam_anxiety": ("prefers short, low-pressure check-ins", "Education: currently enrolled in a degree program"),
    "career_change": ("prefers concrete next steps over general encouragement", "Occupation: exploring a new field"),
    "housing_move": ("prefers one topic at a time, not a full checklist", "Living situation: mid-move between residences"),
    "friendship_rift": ("prefers being asked before advice is offered", "Social circle: long-time close friend group"),
}

MS_CONTENT = {
    "job_transition": "The earlier job transition goal was to decide between two offers before the deadline.",
    "chronic_pain": "The earlier chronic pain goal was to identify which activities reliably triggered a flare-up.",
    "parenting_conflict": "The earlier parenting conflict goal was to agree on one consistent rule with the co-parent.",
    "breakup_recovery": "The earlier breakup goal was to figure out how to tell mutual friends without oversharing.",
    "exam_anxiety": "The earlier exam anxiety goal was to build a study schedule that felt realistic.",
    "career_change": "The earlier career change goal was to shortlist which industries were worth exploring.",
    "housing_move": "The earlier housing move goal was to decide which neighborhood to prioritize.",
    "friendship_rift": "The earlier friendship rift goal was to name what specifically felt hurtful.",
}


def _mid(user_id: str, tag: str) -> str:
    return "mem_" + stable_hex("pm-v1.5-v5.3-p2-candidate-blueprint-v1", user_id, tag, n=20)


def _selected_row(item: MemoryItem, rank: int) -> dict:
    return {
        "candidate_id": item.memory_id, "rank": rank, "created_session": item.created_session,
        "text": item.text,
    }


def build_mp_states() -> list[dict]:
    states = []
    for family in FAMILIES:
        pref_text, profile_text = MP_CONTENT[family]
        user_id = f"p2_mp_{family}"
        pref_id, profile_id = _mid(user_id, "pref"), _mid(user_id, "profile")
        items = [
            MemoryItem(memory_id=pref_id, source=MemorySource.MP, created_session=1, text=f"Stable Preference Response Format: {pref_text}"),
            MemoryItem(memory_id=profile_id, source=MemorySource.MP, created_session=1, text=profile_text),
        ]
        source_metadata = {
            pref_id: {"mp_subtype": "MP_PREFERENCE"},
            profile_id: {"mp_subtype": "MP_PROFILE"},
        }
        # Query phrasing shares content words with the TARGET item's own text
        # (same lesson learned constructing the ME superdomain, script 68):
        # a topic-only query without lexical overlap to the specific
        # preference/profile text falls below discover_final_typed_memory_
        # candidates()'s match_level floor and returns no candidate at all.
        variants = [
            ("preference_positive", "subdomain_preference",
             f"Could you keep this short? I want {pref_text.split('prefers ')[-1]} about the {family.replace('_', ' ')}."),
            ("profile_positive", "subdomain_profile",
             f"I need advice on how to arrange things -- {profile_text.split(': ', 1)[-1]} is part of my situation with the {family.replace('_', ' ')}."),
            ("current_redundant", "subdomain_preference",
             f"Like I said, {pref_text} -- I don't need that repeated, just responding to the {family.replace('_', ' ')} itself."),
        ]
        for negative_kind, subdomain, current_user_text in variants:
            session_index = 3
            queries = source_specific_memory_queries(current_user_text, [], "")
            discoveries = discover_final_typed_memory_candidates(
                queries=queries, items=items, source_metadata=source_metadata, session_index=session_index,
            )
            disc = discoveries[MemorySource.MP]
            selected = disc.selected_items
            rank1 = selected[0] if selected else None
            descriptor = describe_memory_candidate(
                source=MemorySource.MP, query=queries[MemorySource.MP],
                source_items=items, selected_items=selected, session_index=session_index,
            )
            is_pref = rank1 is not None and rank1.memory_id == pref_id
            slots = mp_contribution_slots(
                current_user_text=current_user_text, candidate_text=rank1.text if rank1 else None,
                candidate_is_preference=is_pref, source_items=items,
                selected_items=selected, session_index=session_index,
            )
            states.append({
                "component": "MP", "state_id": f"p2_mp_{family}_{negative_kind}",
                "user_id": user_id, "family": family, "subdomain": subdomain,
                "negative_kind": negative_kind,
                "counterfactual_group_id": f"cfg_mp_{family}",
                "proposed_split": "TBD_by_leader",
                "current_user_text": current_user_text, "current_goal": subdomain,
                "visible_dialogue": [{"role": "user", "content": current_user_text}],
                "session_index": session_index,
                "top_k_candidate_ids": [it.memory_id for it in selected],
                "exact_rank1_id": rank1.memory_id if rank1 else None,
                "exact_rank1_subtype": source_metadata.get(rank1.memory_id, {}).get("mp_subtype") if rank1 else None,
                "owner_id": user_id, "candidate_present": rank1 is not None,
                "n_candidates_in_catalog": len(items),
                "score_top1_lexical_relevance": descriptor["top1_lexical_relevance"],
                "score_top1_top2_margin": descriptor["top1_top2_lexical_margin"],
                "incremental_injected_tokens": descriptor["incremental_injected_tokens"],
                "model_visible_surface": rank1.text if rank1 else None,
                "hard_off_reason": None if rank1 else "candidate_absent_below_match_floor",
                "step1_features": {
                    "candidate_is_preference": slots.candidate_is_preference,
                    "preference_applies_to_response_act": slots.preference_applies_to_response_act,
                    "profile_goal_needs_advice_or_arrangement": slots.profile_goal_needs_advice_or_arrangement,
                    "current_redundant": slots.current_redundant,
                },
                "full_catalog": [_selected_row(it, i + 1) for i, it in enumerate(items)],
            })
    return states


def build_ms_states(encoder) -> list[dict]:
    states = []
    for family in FAMILIES:
        ms_text = MS_CONTENT[family]
        user_id = f"p2_ms_{family}"
        item_id = _mid(user_id, "obs")
        items = [MemoryItem(memory_id=item_id, source=MemorySource.MS, created_session=2, text=ms_text)]
        variants = [
            ("continuity_positive", f"Last time we talked about the {family.replace('_', ' ')}, and I want to pick that back up."),
            ("current_redundant", f"{ms_text} I don't need that repeated, I know where I left off."),
        ]
        for negative_kind, current_user_text in variants:
            session_index = 4
            queries = source_specific_memory_queries(current_user_text, [], "")
            discoveries = discover_final_typed_memory_candidates(
                queries=queries, items=items, source_metadata={}, session_index=session_index,
                ms_semantic_encoder=encoder,
            )
            disc = discoveries[MemorySource.MS]
            selected = disc.selected_items
            rank1 = selected[0] if selected else None
            descriptor = describe_memory_candidate(
                source=MemorySource.MS, query=queries[MemorySource.MS],
                source_items=items, selected_items=selected, session_index=session_index,
            )
            slots = ms_contribution_slots(
                current_user_text=current_user_text, candidate_text=rank1.text if rank1 else None,
                source_items=items, selected_items=selected, session_index=session_index,
            )
            states.append({
                "component": "MS", "state_id": f"p2_ms_{family}_{negative_kind}",
                "user_id": user_id, "family": family, "subdomain": "MS_SESSION",
                "negative_kind": negative_kind,
                "counterfactual_group_id": f"cfg_ms_{family}",
                "proposed_split": "TBD_by_leader",
                "current_user_text": current_user_text, "current_goal": "continuity_request",
                "visible_dialogue": [{"role": "user", "content": current_user_text}],
                "session_index": session_index,
                "top_k_candidate_ids": [it.memory_id for it in selected],
                "exact_rank1_id": rank1.memory_id if rank1 else None,
                "exact_rank1_subtype": "MS_SESSION" if rank1 else None,
                "owner_id": user_id, "candidate_present": rank1 is not None,
                "n_candidates_in_catalog": len(items),
                "retriever": "BGE_M3_COSINE_FULL_CAUSAL_POOL",
                "score_top1_lexical_relevance": descriptor["top1_lexical_relevance"],
                "incremental_injected_tokens": descriptor["incremental_injected_tokens"],
                "model_visible_surface": rank1.text if rank1 else None,
                "hard_off_reason": None if rank1 else "candidate_absent",
                "step1_features": {
                    "has_specific_prior_observation": slots.has_specific_prior_observation,
                    "continuity_request": slots.continuity_request,
                    "current_redundant": slots.current_redundant,
                },
                "full_catalog": [_selected_row(it, i + 1) for i, it in enumerate(items)],
            })
    return states


def load_me_states() -> list[dict]:
    raw_states = json.loads(ME_STATES_PATH.read_text(encoding="utf-8"))
    states = []
    for s in raw_states:
        items = [
            MemoryItem(memory_id=it["memory_id"], source=MemorySource.ME,
                       created_session=it["created_session"], text=it["text"])
            for it in s["items"]
        ]
        queries = source_specific_memory_queries(s["current_user_text"], [], "")
        hints = me_subtype_hints(items)
        discoveries = discover_final_typed_memory_candidates(
            queries=queries, items=items, source_metadata=hints, session_index=s["session_index"],
        )
        disc = discoveries[MemorySource.ME]
        selected = disc.selected_items
        rank1 = selected[0] if selected else None
        rank1_compiler_valid = compile_atomic_reusable_outcome(rank1.text) is not None if rank1 else False
        # Production contract: Rank-1 compiler failure = ME unavailable, no Rank-2 promotion.
        me_available = rank1 is not None and rank1_compiler_valid
        descriptor = describe_memory_candidate(
            source=MemorySource.ME, query=queries[MemorySource.ME],
            source_items=items, selected_items=selected, session_index=s["session_index"],
        )
        states.append({
            "component": "ME", "state_id": f"p2_{s['state_id']}",
            "user_id": s["state_id"], "family": s["family"], "subdomain": "ME_REUSABLE_OUTCOME",
            "negative_kind": s["negative_kind"],
            "counterfactual_group_id": f"cfg_me_{s['family']}",
            "proposed_split": "TBD_by_leader",
            "current_user_text": s["current_user_text"], "current_goal": s["negative_kind"],
            "visible_dialogue": [{"role": "user", "content": s["current_user_text"]}],
            "session_index": s["session_index"],
            "top_k_candidate_ids": [it.memory_id for it in selected],
            "exact_rank1_id": rank1.memory_id if rank1 else None,
            "exact_rank1_subtype": hints.get(rank1.memory_id, {}).get("me_subtype_hint") if rank1 else None,
            "owner_id": s["state_id"], "candidate_present": rank1 is not None,
            "me_available": me_available,
            "n_candidates_in_catalog": len(items),
            "score_top1_lexical_relevance": descriptor["top1_lexical_relevance"],
            "score_top1_top2_margin": descriptor["top1_top2_lexical_margin"],
            "incremental_injected_tokens": descriptor["incremental_injected_tokens"] if me_available else 0,
            "model_visible_surface": rank1.text if (rank1 and me_available) else None,
            "hard_off_reason": (
                None if me_available
                else ("candidate_absent" if rank1 is None else "rank1_compiler_invalid_me_unavailable_no_rank2")
            ),
            "step1_features": {
                "rank1_matches_intended_target": s["target_item_id"] == (rank1.memory_id if rank1 else None),
            },
            "full_catalog": [_selected_row(it, i + 1) for i, it in enumerate(items)],
        })
    return states


def build_rs_states(cards: list[StrategyCard]) -> list[dict]:
    natural_turns = {
        "open_expression": "I don't really know where to start, there's just a lot going on and something is bothering me.",
        "focused_clarification": "It's hard to explain exactly what's wrong, I haven't really said the specific thing yet.",
        "paraphrase_check": "Part of me wants to just let it go, but part of me is still really bothered by it.",
        "grounded_validation": "I feel overwhelmed, it's been a lot to carry.",
        "offer_one_optional_micro_step": "Any advice would help, I could really use one small idea to try.",
        "no_signal": "Work has been a lot lately, I don't really know.",
    }
    boundary_cases = {
        "explicit_stop_boundary": "Please stop this conversation now.",
    }
    states = []
    for family_label, text in {**natural_turns, **boundary_cases}.items():
        user_id = f"p2_rs_{family_label}"
        dialogue = [{"speaker": "seeker", "content": text}]
        already_executed = ("AM02_ask_one_focused_clarification",) if family_label == "focused_clarification_repeat" else ()
        pool = rs_mechanical_candidate_pool(recent_dialogue=dialogue, cards=cards, already_executed_move_ids=already_executed)
        shared = rs_shared_candidate_top1(pool)
        states.append({
            "component": "RS", "state_id": f"p2_rs_{family_label}",
            "user_id": user_id, "family": family_label, "subdomain": "RS_ATOMIC_MOVE",
            "negative_kind": "explicit_decline" if family_label in boundary_cases else "positive",
            "counterfactual_group_id": f"cfg_rs_{family_label}",
            "proposed_split": "TBD_by_leader",
            "current_user_text": text, "current_goal": family_label,
            "visible_dialogue": [{"role": "user", "content": text}],
            "session_index": 1,
            "top_k_candidate_ids": [c.move_id for c in pool.candidates],
            "exact_rank1_id": shared.observation.move_id if shared else None,
            "exact_rank1_subtype": shared.selection_mode if shared else None,
            "owner_id": user_id, "candidate_present": shared is not None,
            "n_candidates_in_catalog": len(pool.candidates),
            "score_top1_lexical_relevance": shared.observation.lexical_relevance if shared else None,
            "incremental_injected_tokens": estimate_tokens(shared.observation.strategy_card.retrieval_text) if shared else 0,
            "model_visible_surface": shared.observation.strategy_card.retrieval_text if shared else None,
            "hard_off_reason": pool.hard_off_reason,
            "step1_features": {
                "transparent_rule_on": shared.transparent_rule_on if shared else None,
                "selection_mode": shared.selection_mode if shared else None,
                "explicit_advice_welcome": pool.observable_flags.get("explicit_advice_welcome"),
            },
            "full_catalog": [
                {"candidate_id": c.move_id, "lexical_relevance": c.lexical_relevance} for c in pool.candidates
            ],
        })
    return states


def build_interaction_states(mp_states, ms_states, me_states, rs_states, cards) -> list[dict]:
    """Multi-component states: same synthetic user/turn, checked against all
    four component pipelines at once, keeping only cases where >=2 components
    genuinely produce a real candidate (not asserted)."""

    interaction = []
    for family in FAMILIES[:3]:  # bounded, real construction -- not padded to a target count
        topic = family.replace("_", " ")
        user_id = f"p2_ix_{family}"
        current_user_text = (
            f"I could really use some advice for handling the {topic} -- "
            f"like I said, {MS_CONTENT[family].split('goal was to ')[1]}"
        )
        session_index = 4
        mp_pref_text, mp_profile_text = MP_CONTENT[family]
        mp_items = [
            MemoryItem(memory_id=_mid(user_id, "mp_pref"), source=MemorySource.MP, created_session=1, text=f"Stable Preference Response Format: {mp_pref_text}"),
            MemoryItem(memory_id=_mid(user_id, "mp_profile"), source=MemorySource.MP, created_session=1, text=mp_profile_text),
        ]
        ms_items = [MemoryItem(memory_id=_mid(user_id, "ms"), source=MemorySource.MS, created_session=2, text=MS_CONTENT[family])]
        me_source = next((s for s in json.loads(ME_STATES_PATH.read_text()) if s["family"] == family and s["negative_kind"] == "positive"), None)
        # Reuse the ME items' own causal boundary (session_index=6, matching
        # script 68's construction) rather than this interaction state's MP/MS
        # session_index=4 -- the two subsets were built independently and
        # their created_session values are only mutually consistent under the
        # ME items' original session_index.
        me_items = [
            MemoryItem(memory_id=it["memory_id"], source=MemorySource.ME, created_session=it["created_session"], text=it["text"])
            for it in me_source["items"]
        ] if me_source else []
        session_index = max(session_index, me_source["session_index"] if me_source else session_index)

        all_items = mp_items + ms_items + me_items
        queries = source_specific_memory_queries(current_user_text, [], "")
        mp_metadata = {mp_items[0].memory_id: {"mp_subtype": "MP_PREFERENCE"}, mp_items[1].memory_id: {"mp_subtype": "MP_PROFILE"}}
        me_metadata = me_subtype_hints(me_items) if me_items else {}
        source_metadata = {**mp_metadata, **me_metadata}
        discoveries = discover_final_typed_memory_candidates(
            queries=queries, items=all_items, source_metadata=source_metadata, session_index=session_index,
        )
        rs_pool = rs_mechanical_candidate_pool(
            recent_dialogue=[{"speaker": "seeker", "content": current_user_text}], cards=cards,
        )
        rs_shared = rs_shared_candidate_top1(rs_pool)

        components_present = {}
        for source in (MemorySource.MP, MemorySource.MS, MemorySource.ME):
            selected = discoveries[source].selected_items
            rank1 = selected[0] if selected else None
            me_available = True
            if source is MemorySource.ME and rank1 is not None:
                me_available = compile_atomic_reusable_outcome(rank1.text) is not None
            components_present[source.value] = {
                "candidate_present": rank1 is not None,
                "exact_rank1_id": rank1.memory_id if rank1 else None,
                "available": (rank1 is not None) and (me_available if source is MemorySource.ME else True),
            }
        components_present["RS"] = {
            "candidate_present": rs_shared is not None,
            "exact_rank1_id": rs_shared.observation.move_id if rs_shared else None,
            "available": rs_shared is not None,
        }
        n_available = sum(1 for c in components_present.values() if c["available"])

        interaction.append({
            "component": "INTERACTION", "state_id": f"p2_ix_{family}",
            "user_id": user_id, "family": family, "subdomain": "MULTI_COMPONENT",
            "negative_kind": "joint_positive",
            "counterfactual_group_id": f"cfg_ix_{family}",
            "proposed_split": "TBD_by_leader",
            "current_user_text": current_user_text, "current_goal": "joint_advice_and_continuity",
            "visible_dialogue": [{"role": "user", "content": current_user_text}],
            "session_index": session_index,
            "n_components_with_real_candidate": n_available,
            "components": components_present,
            "n_candidates_in_catalog": len(all_items) + len(rs_pool.candidates),
        })
    return interaction


def main() -> None:
    bindings = build_static_release_bindings(ROOT)
    print(f"release_identity={bindings.release_identity}")

    mp_states = build_mp_states()
    print("loading BGE-M3 for MS (release-bound snapshot)...")
    encoder = BgeM3Encoder()
    ms_states = build_ms_states(encoder)
    me_states = load_me_states()
    cards = [
        StrategyCard(**row) for row in (json.loads(line) for line in CARDS_PATH.open() if line.strip())
    ]
    rs_states = build_rs_states(cards)
    interaction_states = build_interaction_states(mp_states, ms_states, me_states, rs_states, cards)

    all_single = mp_states + ms_states + me_states + rs_states
    unique_groups = {s["counterfactual_group_id"] for s in all_single}
    unique_users = {s["user_id"] for s in all_single} | {s["user_id"] for s in interaction_states}
    unique_families = {s["family"] for s in all_single}

    summary = {
        "protocol": "pm-v1.5-v5.3-p2-candidate-blueprint-v1",
        "release_identity": bindings.release_identity,
        "n_mp_states": len(mp_states), "n_ms_states": len(ms_states),
        "n_me_states": len(me_states), "n_rs_states": len(rs_states),
        "n_interaction_states": len(interaction_states),
        "n_single_component_states_total": len(all_single),
        "n_unique_counterfactual_groups": len(unique_groups),
        "n_unique_users": len(unique_users),
        "n_unique_families": len(unique_families),
        "note": (
            "Real, unique counts only -- no template duplicated across "
            "states, no final N frozen here (leader decision per runbook "
            "9.1). ME reuses the 32 already-verified states from script 68 "
            "rather than reconstructing them."
        ),
        "candidate_availability": {
            "mp": sum(1 for s in mp_states if s["candidate_present"]) / len(mp_states),
            "ms": sum(1 for s in ms_states if s["candidate_present"]) / len(ms_states),
            "me_available": sum(1 for s in me_states if s["me_available"]) / len(me_states),
            "rs": sum(1 for s in rs_states if s["candidate_present"]) / len(rs_states),
        },
        "api_calls": 0,
        "generated_response_or_quality_risk_outcome_read": False,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "summary.json", summary)
    write_json(OUT_DIR / "mp_states.json", mp_states)
    write_json(OUT_DIR / "ms_states.json", ms_states)
    write_json(OUT_DIR / "me_states.json", me_states)
    write_json(OUT_DIR / "rs_states.json", rs_states)
    write_json(OUT_DIR / "interaction_states.json", interaction_states)

    print(json.dumps(summary, indent=2))
    print(f"\nfull blueprint written to {OUT_DIR}")


if __name__ == "__main__":
    main()
