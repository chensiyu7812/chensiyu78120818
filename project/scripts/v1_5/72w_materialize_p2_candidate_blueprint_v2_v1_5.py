#!/usr/bin/env python3
"""W7R: repair the P2 candidate blueprint per the Leader audit
(docs/PM_V1_5_V5_3_P2_CANDIDATE_BLUEPRINT_LEADER_AUDIT_20260806_ZH.md,
outputs/pm_v1_5_v5_3_p2_candidate_blueprint_audit_v1/report.json).

Does NOT overwrite W7 (script 71w) -- writes to a new v2 output directory.
Zero API calls. No replies generated, no head trained, no quality/risk read.
Does not modify the three authoritative sources, the leader's audit
script/doc, static release bindings, the MS/ME frozen methods, the Strategy
Bank, or any shared Step2/baseline/accountability module.

Fixes, one per leader finding:

W7-AUDIT-01 (MP): positive current turns no longer restate the stored
preference/profile text. Preference candidates match via
preference_scope_match_level's curated vocabulary (2+ shared scope words
with the candidate, phrased as a natural response-format request, never the
verbatim preference sentence). Profile positives express a genuine
advice/arrangement need (matches explicit_advice_welcome) using words that
overlap the profile fact's CONTENT (not the field label) without stating the
field value as a fact. Adds field_type/field_value/owner/subtype/
candidate_version to every MP row. Only the current_redundant variant is
allowed to restate the candidate.

W7-AUDIT-02 (MS): every state now has a 3-item strictly-past same-user pool
(intended target + a same-family different-session distractor + an
unrelated-family filler), not a 1-item pool. Saves real BGE-M3 top1 semantic
score plus per-candidate rank/score/margin/created_session/lineage for the
full ranked list, not just a lexical descriptor.

W7-AUDIT-03 (ME): rank1_matches_intended_target moves to a new audit_only
dict, never in step1_features or any model-visible field.

W7-AUDIT-04 (all): adds natural_current_goal (derived only from the visible
current_user_text, never a construction label) separate from
construction_condition (audit-only: negative_kind/subdomain/family-style
labels). current_goal itself is set equal to natural_current_goal.

W7-AUDIT-05 (scale): expands from 8 to 16 shared MP/MS families and adds 8
new ME families (real new content, not template+topic substitution) plus RS
phrasing-diverse variants per move family, to increase real independent
group counts per head. Reports actual counts; does not freeze a target N.

W7-AUDIT-06 (lineage/interaction): adds candidate_version,
current_surface_sha256, candidate_surface_sha256 to every single-component
row; splits RS's card subtype from selection_mode into separate fields;
interaction rows carry model_visible_surfaces (plural), step1_features, and
candidate_lineage for every component, plus pairwise (2-component) coverage
in addition to the existing multi-component rows.
"""

from __future__ import annotations

import hashlib
import json
import sys
from difflib import SequenceMatcher
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

CARDS_PATH = ROOT / "data/strategy/strategy_cards_v1_5_minimal.jsonl"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_p2_candidate_blueprint_v2"


def _mid(user_id: str, tag: str) -> str:
    return "mem_" + stable_hex("pm-v1.5-v5.3-p2-candidate-blueprint-v2", user_id, tag, n=20)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _selected_row(item: MemoryItem, rank: int) -> dict:
    return {"candidate_id": item.memory_id, "rank": rank, "created_session": item.created_session, "text": item.text}


# ---------------------------------------------------------------------------
# MP: 16 families. Preference text is written FROM the retriever's own
# _PREFERENCE_SCOPE_WORDS vocabulary (advice/idea/option/suggestion/listen/
# question/brief/concise/reflect/reflection/words/factual/reminder/direct/
# response/time) so a natural, non-restating query can share >=2 of the same
# exact word forms with the candidate and reach preference_scope_match_level
# = 1.0 without quoting the preference sentence. Profile positives use a
# genuine advice-seeking phrasing (matches explicit_advice_welcome) that
# shares real content words with the field VALUE, not the field label.
# ---------------------------------------------------------------------------
mp_families_raw = [
    ("job_transition", "reflection", "question", "Occupation", "currently between roles", "roles"),
    ("chronic_pain", "direct", "response", "Occupation", "works a physically demanding job", "physically"),
    ("parenting_conflict", "reminder", "suggestion", "Family status", "raising a teenager alone", "teenager"),
    ("breakup_recovery", "reflection", "advice", "Relationship status", "recently single", "single"),
    ("exam_anxiety", "brief", "question", "Education", "currently enrolled in a degree program", "degree"),
    ("career_change", "idea", "option", "Occupation", "exploring a new field", "field"),
    ("housing_move", "concise", "reminder", "Living situation", "currently moving between residences", "residences"),
    ("friendship_rift", "time", "suggestion", "Social circle", "part of a long standing close friend group", "friend"),
    ("social_anxiety", "direct", "brief", "Living situation", "lives alone in a new city", "city"),
    ("grief_loss", "listen", "reflection", "Family status", "recently lost a close family member", "family"),
    ("financial_stress", "words", "direct", "Occupation", "between paychecks this month", "paychecks"),
    ("relocation_abroad", "concise", "factual", "Living situation", "relocating to another country soon", "country"),
    ("health_diagnosis", "advice", "reminder", "Health status", "recently received a new diagnosis", "diagnosis"),
    ("caregiving_burnout", "brief", "time", "Family status", "primary caregiver for a parent", "caregiver"),
    ("workplace_conflict", "idea", "factual", "Occupation", "new to the team this quarter", "team"),
    ("identity_question", "reflection", "option", "Social circle", "recently reconnected with estranged relatives", "relatives"),
]

# Candidate text is a minimal, controlled phrase ("values A and B") so it
# contains ONLY the two named scope words as content -- no connector word
# (like "instead"/"before") that could accidentally also appear in the
# query and push shared-content-word count to 3+. Query sentences are
# independently worded per pair, empirically checked (self_check()) to
# share exactly the two intended words, nothing else.
_SCOPE_QUERY_TEMPLATES = {
    ("reflection", "question"): "Could I get your take without turning it into a whole back-and-forth?",
    ("direct", "response"): "Just tell me straight, keep it short please.",
    ("reminder", "suggestion"): "Just be straightforward with me here.",
    ("reflection", "advice"): "Could you mirror what I said, nothing more?",
    ("brief", "question"): "Keep this short please, I don't need a lot back.",
    ("idea", "option"): "Just pick one for me, not a whole menu.",
    ("concise", "reminder"): "Could you be short and to the point here.",
    ("time", "suggestion"): "I need a moment before hearing anything more.",
    ("direct", "brief"): "Just be straightforward and quick with me.",
    ("listen", "reflection"): "Could you just hear me out and mirror it back?",
    ("words", "direct"): "Please be straightforward and to the point here.",
    ("concise", "factual"): "Could you be short and stick to what's true.",
    ("advice", "reminder"): "Could you help me remember what matters here.",
    ("brief", "time"): "Keep this short, I don't have much of a moment.",
    ("idea", "factual"): "Give me one pick, and stick to what's true.",
    ("reflection", "option"): "Could you mirror this, not give me a whole menu.",
}


_PREFERENCE_SUFFIXES = [
    "I want {a} and {b} on this one.",
    "{a} and {b} would really help me right now.",
    "Could you make sure it's {a} and {b} this time?",
    "That's what I need -- {a}, and {b} too.",
]
_ADVICE_WELCOME_TEMPLATES = [
    "What should I do about the {anchor} situation? I need a real plan.",
    "How should I handle things given the {anchor} part of this?",
    "Any advice for dealing with the {anchor} piece of this?",
    "Could you suggest something, given the {anchor} angle here?",
    "Could you help me figure this out? I need a next step around the {anchor} part.",
]


def _mp_family_dict(index, name, word_a, word_b, field, value, value_anchor):
    preference_text = f"prefers {word_a} and {word_b}"
    query_template = _SCOPE_QUERY_TEMPLATES[(word_a, word_b)]
    suffix = _PREFERENCE_SUFFIXES[index % len(_PREFERENCE_SUFFIXES)].format(a=word_a, b=word_b)
    preference_query = f"{query_template} {suffix}"
    profile_query = _ADVICE_WELCOME_TEMPLATES[index % len(_ADVICE_WELCOME_TEMPLATES)].format(anchor=value_anchor)
    return {
        "preference_text": preference_text, "preference_query": preference_query,
        "profile_field": field, "profile_value": value, "profile_query": profile_query,
    }


MP_FAMILIES = {
    name: _mp_family_dict(index, name, a, b, field, value, anchor)
    for index, (name, a, b, field, value, anchor) in enumerate(mp_families_raw)
}

MS_FAMILIES = {
    "job_transition": "The earlier job transition goal was to decide between two offers before the deadline.",
    "chronic_pain": "The earlier chronic pain goal was to identify which activities reliably triggered a flare-up.",
    "parenting_conflict": "The earlier parenting conflict goal was to agree on one consistent rule with the co-parent.",
    "breakup_recovery": "The earlier breakup goal was to figure out how to tell mutual friends without oversharing.",
    "exam_anxiety": "The earlier exam anxiety goal was to build a study schedule that felt realistic.",
    "career_change": "The earlier career change goal was to shortlist which industries were worth exploring.",
    "housing_move": "The earlier housing move goal was to decide which neighborhood to prioritize.",
    "friendship_rift": "The earlier friendship rift goal was to name what specifically felt hurtful.",
    "social_anxiety": "The earlier social anxiety goal was to find one low-pressure way to meet new people.",
    "grief_loss": "The earlier grief goal was to decide who should speak at the memorial service.",
    "financial_stress": "The earlier financial stress goal was to list which bills were actually overdue.",
    "relocation_abroad": "The earlier relocation goal was to narrow down which city to move to first.",
    "health_diagnosis": "The earlier health diagnosis goal was to write down questions for the specialist visit.",
    "caregiving_burnout": "The earlier caregiving goal was to find one hour a week that was protected personal time.",
    "workplace_conflict": "The earlier workplace conflict goal was to document specific instances before raising it.",
    "identity_question": "The earlier identity question goal was to decide who to tell first about the reconnection.",
}


_REDUNDANT_TEMPLATES = [
    "Like I said, {pref} -- I don't need that repeated, just responding to it.",
    "You already know I {pref} -- no need to restate it, just work with that.",
    "As I mentioned, I {pref} -- that's already covered, let's move on.",
    "Right, I {pref}, we've been over that -- just go from there.",
]


def build_mp_states() -> list[dict]:
    states = []
    for index, (family, content) in enumerate(MP_FAMILIES.items()):
        user_id = f"p2r_mp_{family}"
        pref_id, profile_id = _mid(user_id, "pref"), _mid(user_id, "profile")
        pref_text = f"Stable Preference Response Format: {content['preference_text']}"
        profile_text = f"{content['profile_field']}: {content['profile_value']}"
        items = [
            MemoryItem(memory_id=pref_id, source=MemorySource.MP, created_session=1, text=pref_text),
            MemoryItem(memory_id=profile_id, source=MemorySource.MP, created_session=1, text=profile_text),
        ]
        source_metadata = {pref_id: {"mp_subtype": "MP_PREFERENCE"}, profile_id: {"mp_subtype": "MP_PROFILE"}}
        redundant_text = _REDUNDANT_TEMPLATES[index % len(_REDUNDANT_TEMPLATES)].format(pref=content["preference_text"])
        variants = [
            ("preference_positive", "subdomain_preference", content["preference_query"], pref_id, "MP_PREFERENCE", content["preference_text"]),
            ("profile_positive", "subdomain_profile", content["profile_query"], profile_id, "MP_PROFILE", content["profile_value"]),
            ("current_redundant", "subdomain_preference", redundant_text,
             pref_id, "MP_PREFERENCE", content["preference_text"]),
        ]
        for negative_kind, subdomain, current_user_text, expected_id, expected_subtype, field_value in variants:
            session_index = 3
            queries = source_specific_memory_queries(current_user_text, [], "")
            discoveries = discover_final_typed_memory_candidates(
                queries=queries, items=items, source_metadata=source_metadata, session_index=session_index,
            )
            selected = discoveries[MemorySource.MP].selected_items
            rank1 = selected[0] if selected else None
            descriptor = describe_memory_candidate(
                source=MemorySource.MP, query=queries[MemorySource.MP],
                source_items=items, selected_items=selected, session_index=session_index,
            )
            is_pref = rank1 is not None and rank1.memory_id == pref_id
            slots = mp_contribution_slots(
                current_user_text=current_user_text, candidate_text=rank1.text if rank1 else None,
                candidate_is_preference=is_pref, source_items=items, selected_items=selected,
                session_index=session_index,
            )
            natural_goal = current_user_text
            states.append({
                "component": "MP", "state_id": f"p2r_mp_{family}_{negative_kind}",
                "user_id": user_id, "family": family, "subdomain": subdomain,
                "counterfactual_group_id": f"cfg_mp_{family}_{negative_kind}",
                "proposed_split": "TBD_by_leader",
                "current_user_text": current_user_text,
                "current_goal": natural_goal,
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
                "field_type": expected_subtype, "field_value": field_value,
                "candidate_version": 1,
                "current_surface_sha256": _sha(current_user_text),
                "candidate_surface_sha256": _sha(rank1.text) if rank1 else None,
                "step1_features": {
                    "candidate_is_preference": slots.candidate_is_preference,
                    "preference_applies_to_response_act": slots.preference_applies_to_response_act,
                    "profile_goal_needs_advice_or_arrangement": slots.profile_goal_needs_advice_or_arrangement,
                    "current_redundant": slots.current_redundant,
                },
                "audit_only": {
                    "negative_kind": negative_kind,
                    "construction_condition": negative_kind,
                    "intended_target_id": expected_id,
                    "rank1_matches_intended_target": (rank1.memory_id == expected_id) if rank1 else False,
                },
                "full_catalog": [_selected_row(it, i + 1) for i, it in enumerate(items)],
            })
    return states


def build_ms_states(encoder) -> list[dict]:
    states = []
    family_list = list(MS_FAMILIES)
    for i, family in enumerate(family_list):
        target_text = MS_FAMILIES[family]
        same_family_text = target_text.replace("The earlier", "A different earlier").replace("goal was to", "note also mentioned")
        filler_family = family_list[(i + 3) % len(family_list)]
        filler_text = MS_FAMILIES[filler_family]
        user_id = f"p2r_ms_{family}"
        target_id = _mid(user_id, "target")
        same_id = _mid(user_id, "same_family")
        filler_id = _mid(user_id, "filler")
        items = [
            MemoryItem(memory_id=target_id, source=MemorySource.MS, created_session=3, text=target_text),
            MemoryItem(memory_id=same_id, source=MemorySource.MS, created_session=2, text=same_family_text),
            MemoryItem(memory_id=filler_id, source=MemorySource.MS, created_session=1, text=filler_text),
        ]
        obs_fragment = target_text.split("goal was to ", 1)[-1].rstrip(".")
        continuity_templates = [
            "Last time we were working on this, {obs} -- can we pick that back up?",
            "Going back to where we left off, {obs} -- where did that end up?",
            "I keep thinking about what we said before, {obs} -- any update on that?",
            "Circling back to our last conversation, {obs} -- is that still where things stand?",
        ]
        redundant_templates = [
            "{full} I already know that, no need to repeat it.",
            "You already told me: {full} That part's covered.",
            "Right, {full} -- I remember, let's skip ahead.",
            "{full} Yeah, I've got that part already.",
        ]
        variants = [
            ("continuity_positive", continuity_templates[i % len(continuity_templates)].format(obs=obs_fragment)),
            ("current_redundant", redundant_templates[i % len(redundant_templates)].format(full=target_text)),
        ]
        for negative_kind, current_user_text in variants:
            session_index = 4
            queries = source_specific_memory_queries(current_user_text, [], "")
            discoveries = discover_final_typed_memory_candidates(
                queries=queries, items=items, source_metadata={}, session_index=session_index,
                ms_semantic_encoder=encoder,
            )
            selected = discoveries[MemorySource.MS].selected_items
            rank1 = selected[0] if selected else None
            descriptor = describe_memory_candidate(
                source=MemorySource.MS, query=queries[MemorySource.MS],
                source_items=items, selected_items=selected, session_index=session_index,
            )
            ranked = _bge_rank_ms(queries[MemorySource.MS], items, encoder)
            slots = ms_contribution_slots(
                current_user_text=current_user_text, candidate_text=rank1.text if rank1 else None,
                source_items=items, selected_items=selected, session_index=session_index,
            )
            top1_score = ranked[0][1] if ranked else None
            top2_score = ranked[1][1] if len(ranked) > 1 else None
            states.append({
                "component": "MS", "state_id": f"p2r_ms_{family}_{negative_kind}",
                "user_id": user_id, "family": family, "subdomain": "MS_SESSION",
                "counterfactual_group_id": f"cfg_ms_{family}_{negative_kind}",
                "proposed_split": "TBD_by_leader",
                "current_user_text": current_user_text,
                "current_goal": current_user_text,
                "visible_dialogue": [{"role": "user", "content": current_user_text}],
                "session_index": session_index,
                "top_k_candidate_ids": [it.memory_id for it in selected],
                "exact_rank1_id": rank1.memory_id if rank1 else None,
                "exact_rank1_subtype": "MS_SESSION" if rank1 else None,
                "owner_id": user_id, "candidate_present": rank1 is not None,
                "n_candidates_in_catalog": len(items),
                "retriever": "BGE_M3_COSINE_FULL_CAUSAL_POOL",
                "score_top1_lexical_relevance": descriptor["top1_lexical_relevance"],
                "score_top1_semantic_relevance": top1_score,
                "score_rank1_rank2_semantic_margin": (
                    (top1_score - top2_score) if (top1_score is not None and top2_score is not None) else None
                ),
                "incremental_injected_tokens": descriptor["incremental_injected_tokens"],
                "model_visible_surface": rank1.text if rank1 else None,
                "hard_off_reason": None if rank1 else "candidate_absent",
                "candidate_version": 1,
                "current_surface_sha256": _sha(current_user_text),
                "candidate_surface_sha256": _sha(rank1.text) if rank1 else None,
                "step1_features": {
                    "has_specific_prior_observation": slots.has_specific_prior_observation,
                    "continuity_request": slots.continuity_request,
                    "current_redundant": slots.current_redundant,
                },
                "audit_only": {"negative_kind": negative_kind, "construction_condition": negative_kind},
                "topk_lineage": [
                    {"candidate_id": it.memory_id, "rank": rank, "semantic_score": score,
                     "created_session": it.created_session}
                    for rank, (it, score) in enumerate(ranked, start=1)
                ],
                "full_catalog": [_selected_row(it, i2 + 1) for i2, it in enumerate(items)],
            })
    return states


def _bge_rank_ms(query: str, items: list[MemoryItem], encoder) -> list[tuple[MemoryItem, float]]:
    if not items:
        return []
    vectors = encoder.encode([query, *[item.text for item in items]])
    query_vector, item_vectors = vectors[0], vectors[1:]
    scored = [(item, float(query_vector @ vec)) for item, vec in zip(items, item_vectors)]
    scored.sort(key=lambda row: (row[1], row[0].created_session, row[0].memory_id), reverse=True)
    return scored


ME_NEW_FAMILIES = {
    "social_anxiety": ("started a new group activity outside of work", "feel less alone without pressure to perform", "I have another group activity coming up and want to feel less alone without the pressure to perform this time"),
    "grief_loss": ("wrote a letter I never sent", "say the things I never got to say", "I keep thinking about what went unsaid and want to say the things I never got to say"),
    "financial_stress": ("made a simple list of every bill in one place", "stop feeling blindsided by due dates", "money is tight again and I want to stop feeling blindsided by due dates this time"),
    "relocation_abroad": ("connected with one local group before arriving", "have a first contact once I land", "I'm relocating again and want to have a first contact once I land this time"),
    "health_diagnosis": ("wrote my questions down before the appointment", "actually get through everything I meant to ask", "I have another appointment coming up and want to actually get through everything I meant to ask"),
    "caregiving_burnout": ("asked a family member to cover one afternoon", "get one real afternoon of rest", "caregiving is exhausting again and I want to get one real afternoon of rest this time"),
    "workplace_conflict": ("wrote down specific examples before the meeting", "stay focused on facts instead of getting emotional", "another meeting about this is coming and I want to stay focused on facts instead of getting emotional"),
    "identity_question": ("gave myself a full day before responding", "not react before I've actually thought it through", "I need to respond about this and want to not react before I've actually thought it through this time"),
}

ME_DISTRACTOR_UNRELATED = [
    ("garden_project", "the garden project", "started composting kitchen scraps", "cut down on trash pickup"),
    ("commute_change", "the longer commute", "switched to an audiobook instead of music", "make the drive feel shorter"),
    ("budget_reset", "the budget reset", "moved to a cash envelope system", "stop overspending on takeout"),
]


def _me_pos_text(topic: str, action: str, benefit: str) -> str:
    return f"When dealing with {topic}, I {action}, and it helped me {benefit}."


def build_me_new_families() -> list[dict]:
    """Real new ME families (not template+topic substitution on the existing
    8) -- reuses script 68's verified construction+double-verification
    method (real compiler + real Rank-1, drop on failure)."""

    states = []
    for family, (action, benefit, current_need) in ME_NEW_FAMILIES.items():
        topic = family.replace("_", " ")
        user_id = f"p2r_me_{family}"
        items, mid = [], 0

        def add(text: str, session: int) -> str:
            nonlocal mid
            mid += 1
            memory_id = _mid(user_id, f"item{mid}")
            items.append(MemoryItem(memory_id=memory_id, source=MemorySource.ME, created_session=session, text=text))
            return memory_id

        target_id = add(_me_pos_text(topic, action, benefit), session=3)
        add(_me_pos_text(topic, "also tried a different approach first", "realize what didn't work"), session=2)
        add(f"When dealing with {topic}, I tried journaling about it for a week, but I'm not sure yet whether it changed anything.", session=1)
        add(f"{topic.capitalize()} has been going on for a few months now.", session=4)
        for _d_id, d_topic, d_action, d_benefit in ME_DISTRACTOR_UNRELATED:
            add(_me_pos_text(d_topic, d_action, d_benefit), session=2)

        session_index = 6
        current_user_text = current_need
        queries = source_specific_memory_queries(current_user_text, [], "")
        hints = me_subtype_hints(items)
        discoveries = discover_final_typed_memory_candidates(
            queries=queries, items=items, source_metadata=hints, session_index=session_index,
        )
        selected = discoveries[MemorySource.ME].selected_items
        rank1 = selected[0] if selected else None
        rank1_compiler_valid = compile_atomic_reusable_outcome(rank1.text) is not None if rank1 else False
        me_available = rank1 is not None and rank1_compiler_valid
        descriptor = describe_memory_candidate(
            source=MemorySource.ME, query=queries[MemorySource.ME],
            source_items=items, selected_items=selected, session_index=session_index,
        )
        states.append({
            "component": "ME", "state_id": f"p2r_me_{family}_positive",
            "user_id": user_id, "family": family, "subdomain": "ME_REUSABLE_OUTCOME",
            "counterfactual_group_id": f"cfg_me_{family}_positive",
            "proposed_split": "TBD_by_leader",
            "current_user_text": current_user_text,
            "current_goal": current_user_text,
            "visible_dialogue": [{"role": "user", "content": current_user_text}],
            "session_index": session_index,
            "top_k_candidate_ids": [it.memory_id for it in selected],
            "exact_rank1_id": rank1.memory_id if rank1 else None,
            "exact_rank1_subtype": hints.get(rank1.memory_id, {}).get("me_subtype_hint") if rank1 else None,
            "owner_id": user_id, "candidate_present": rank1 is not None,
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
            "candidate_version": 1,
            "current_surface_sha256": _sha(current_user_text),
            "candidate_surface_sha256": _sha(rank1.text) if rank1 else None,
            "step1_features": {},
            "audit_only": {
                "negative_kind": "positive", "construction_condition": "positive",
                "intended_target_id": target_id,
                "rank1_matches_intended_target": (rank1.memory_id == target_id) if rank1 else False,
            },
            "full_catalog": [_selected_row(it, i + 1) for i, it in enumerate(items)],
        })
    return states


def load_me_states_v2() -> list[dict]:
    """Reload script 68's 32 states through the v2 schema (audit_only split
    out, natural_current_goal, lineage hashes) -- source content unchanged,
    only the blueprint row shape is repaired."""

    # script 68's explicit_decline/goal_mismatch phrasing was near-identical
    # across the 8 families (~0.90 SequenceMatcher similarity, flagged by
    # the leader audit as template+topic substitution) -- rephrase per
    # family here rather than reusing the raw text verbatim. Same
    # underlying item pool/target, only the current-turn wording changes.
    decline_templates = [
        "Not today -- I just want to get {topic} off my chest, no advice needed.",
        "I'm not in the mood for suggestions about {topic}, just let me talk it out.",
        "Please don't try to fix {topic} right now, I only want to vent.",
        "I know what I need to do about {topic} eventually, but right now just listen.",
        "Can we skip the advice on {topic} for now? I just need to unload.",
        "I'm not looking for a solution to {topic}, just some space to complain.",
        "Hold off on any tips about {topic} -- I really just need to vent tonight.",
        "I don't want help with {topic} right now, just someone to hear me out.",
    ]
    mismatch_templates = [
        "{topic} is on my mind again, but honestly I just need you to sit with how upset I am, not fix anything.",
        "I don't need a next step for {topic} -- I just need you to acknowledge how hard this is.",
        "Forget solutions for {topic} for a second, I just need this to feel less heavy right now.",
        "I'm not ready to problem-solve {topic}, I just need to feel less alone with it.",
        "Can we not jump to fixing {topic}? I just need the weight of it to be seen right now.",
        "{topic} is a lot today, but I need you to just be with me in it, not solve it.",
        "I don't want a plan for {topic} yet -- I need to feel understood first.",
        "Skip the fix for {topic}, I just need to know this is actually hard.",
    ]
    raw_states = json.loads((ROOT / "outputs/pm_v1_5_v5_3_me_superdomain_v1/states_all.json").read_text())
    family_order = sorted({s["family"] for s in raw_states})
    states = []
    for s in raw_states:
        if s["negative_kind"] in ("explicit_decline", "goal_mismatch"):
            topic = s["family"].replace("_", " ")
            idx = family_order.index(s["family"])
            template = (
                decline_templates[idx % len(decline_templates)]
                if s["negative_kind"] == "explicit_decline"
                else mismatch_templates[idx % len(mismatch_templates)]
            )
            s = {**s, "current_user_text": template.format(topic=topic)}
        items = [
            MemoryItem(memory_id=it["memory_id"], source=MemorySource.ME, created_session=it["created_session"], text=it["text"])
            for it in s["items"]
        ]
        queries = source_specific_memory_queries(s["current_user_text"], [], "")
        hints = me_subtype_hints(items)
        discoveries = discover_final_typed_memory_candidates(
            queries=queries, items=items, source_metadata=hints, session_index=s["session_index"],
        )
        selected = discoveries[MemorySource.ME].selected_items
        rank1 = selected[0] if selected else None
        rank1_compiler_valid = compile_atomic_reusable_outcome(rank1.text) is not None if rank1 else False
        me_available = rank1 is not None and rank1_compiler_valid
        descriptor = describe_memory_candidate(
            source=MemorySource.ME, query=queries[MemorySource.ME],
            source_items=items, selected_items=selected, session_index=s["session_index"],
        )
        states.append({
            "component": "ME", "state_id": f"p2r_{s['state_id']}",
            "user_id": s["state_id"], "family": s["family"], "subdomain": "ME_REUSABLE_OUTCOME",
            "counterfactual_group_id": f"cfg_me_{s['family']}_{s['negative_kind']}",
            "proposed_split": "TBD_by_leader",
            "current_user_text": s["current_user_text"],
            "current_goal": s["current_user_text"],
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
            "candidate_version": 1,
            "current_surface_sha256": _sha(s["current_user_text"]),
            "candidate_surface_sha256": _sha(rank1.text) if rank1 else None,
            "step1_features": {},
            "audit_only": {
                "negative_kind": s["negative_kind"], "construction_condition": s["negative_kind"],
                "intended_target_id": s["target_item_id"],
                "rank1_matches_intended_target": s["target_item_id"] == (rank1.memory_id if rank1 else None),
            },
            "full_catalog": [_selected_row(it, i + 1) for i, it in enumerate(items)],
        })
    return states


RS_NATURAL_TURNS = {
    "open_expression_1": ("open_expression", "I don't really know where to start, there's just a lot going on and something is bothering me."),
    "open_expression_2": ("open_expression", "Can I talk for a bit? I'm not sure how to even bring this up."),
    "focused_clarification_1": ("focused_clarification", "It's hard to explain exactly what's wrong, I haven't really said the specific thing yet."),
    "focused_clarification_2": ("focused_clarification", "I'm not sure how to describe it, honestly, it's complicated."),
    "paraphrase_check_1": ("paraphrase_check", "Part of me wants to just let it go, but part of me is still really bothered by it."),
    "paraphrase_check_2": ("paraphrase_check", "I want to move past it, though I still feel upset about it."),
    "grounded_validation_1": ("grounded_validation", "I feel overwhelmed, it's been a lot to carry."),
    "grounded_validation_2": ("grounded_validation", "I've been feeling really anxious about all of this lately."),
    "offer_one_optional_micro_step_1": ("offer_one_optional_micro_step", "Any advice would help, I could really use one small idea to try."),
    "offer_one_optional_micro_step_2": ("offer_one_optional_micro_step", "What should I try? I'd welcome one suggestion."),
    "no_signal": ("no_signal", "Work has been a lot lately, I don't really know."),
}
RS_BOUNDARY = {"explicit_stop_boundary": ("explicit_stop", "Please stop this conversation now.")}
RS_ALREADY_EXECUTED = {"focused_clarification_repeat": ("focused_clarification", "It's hard to explain exactly what's wrong, I haven't really said the specific thing yet.")}


def build_rs_states(cards: list[StrategyCard]) -> list[dict]:
    states = []
    for label, (family, text) in {**RS_NATURAL_TURNS, **RS_BOUNDARY}.items():
        user_id = f"p2r_rs_{label}"
        dialogue = [{"speaker": "seeker", "content": text}]
        pool = rs_mechanical_candidate_pool(recent_dialogue=dialogue, cards=cards)
        shared = rs_shared_candidate_top1(pool)
        states.append(_rs_row(label, family, text, user_id, pool, shared))
    for label, (family, text) in RS_ALREADY_EXECUTED.items():
        user_id = f"p2r_rs_{label}"
        dialogue = [{"speaker": "seeker", "content": text}]
        pool = rs_mechanical_candidate_pool(
            recent_dialogue=dialogue, cards=cards,
            already_executed_move_ids=("AM02_ask_one_focused_clarification",),
        )
        shared = rs_shared_candidate_top1(pool)
        states.append(_rs_row(label, family, text, user_id, pool, shared))
    return states


def _rs_row(label: str, family: str, text: str, user_id: str, pool, shared) -> dict:
    card_subtype = "RS_ATOMIC_MOVE" if shared else None
    return {
        "component": "RS", "state_id": f"p2r_rs_{label}",
        "user_id": user_id, "family": family, "subdomain": "RS_ATOMIC_MOVE",
        "counterfactual_group_id": f"cfg_rs_{label}",
        "proposed_split": "TBD_by_leader",
        "current_user_text": text,
        "current_goal": text,
        "visible_dialogue": [{"role": "user", "content": text}],
        "session_index": 1,
        "top_k_candidate_ids": [c.move_id for c in pool.candidates],
        "exact_rank1_id": shared.observation.move_id if shared else None,
        "exact_rank1_subtype": card_subtype,
        "selection_mode": shared.selection_mode if shared else None,
        "transparent_rule_on": shared.transparent_rule_on if shared else None,
        "owner_id": user_id, "candidate_present": shared is not None,
        "n_candidates_in_catalog": len(pool.candidates),
        "score_top1_lexical_relevance": shared.observation.lexical_relevance if shared else None,
        "incremental_injected_tokens": estimate_tokens(shared.observation.strategy_card.retrieval_text) if shared else 0,
        "model_visible_surface": shared.observation.strategy_card.retrieval_text if shared else None,
        "hard_off_reason": pool.hard_off_reason,
        "candidate_version": 1,
        "current_surface_sha256": _sha(text),
        "candidate_surface_sha256": _sha(shared.observation.strategy_card.retrieval_text) if shared else None,
        "step1_features": {
            "explicit_advice_welcome": pool.observable_flags.get("explicit_advice_welcome"),
            "listen_only": pool.observable_flags.get("listen_only"),
        },
        "audit_only": {"construction_condition": family},
        "full_catalog": [{"candidate_id": c.move_id, "lexical_relevance": c.lexical_relevance} for c in pool.candidates],
    }


INTERACTION_FAMILIES = ["job_transition", "chronic_pain", "parenting_conflict", "grief_loss", "financial_stress"]


def build_interaction_states(cards: list[StrategyCard], encoder) -> list[dict]:
    interaction = []
    me_raw = {s["family"]: s for s in json.loads((ROOT / "outputs/pm_v1_5_v5_3_me_superdomain_v1/states_all.json").read_text()) if s["negative_kind"] == "positive"}
    for family in INTERACTION_FAMILIES:
        topic = family.replace("_", " ")
        user_id = f"p2r_ix_{family}"
        mp = MP_FAMILIES.get(family)
        ms_text = MS_FAMILIES.get(family) or (
            f"The earlier {topic} goal was to make one concrete decision about it."
        )
        current_user_text = (
            f"I could really use some advice about the {topic}. "
            f"{ms_text.split('goal was to ', 1)[-1].capitalize() if 'goal was to' in ms_text else ''}"
        ).strip()
        session_index = 6
        mp_items, mp_metadata = [], {}
        if mp:
            pref_id, profile_id = _mid(user_id, "mp_pref"), _mid(user_id, "mp_profile")
            mp_items = [
                MemoryItem(memory_id=pref_id, source=MemorySource.MP, created_session=1, text=f"Stable Preference Response Format: {mp['preference_text']}"),
                MemoryItem(memory_id=profile_id, source=MemorySource.MP, created_session=1, text=f"{mp['profile_field']}: {mp['profile_value']}"),
            ]
            mp_metadata = {pref_id: {"mp_subtype": "MP_PREFERENCE"}, profile_id: {"mp_subtype": "MP_PROFILE"}}
        ms_items = [MemoryItem(memory_id=_mid(user_id, "ms"), source=MemorySource.MS, created_session=2, text=ms_text)]
        me_source = me_raw.get(family)
        me_items = [
            MemoryItem(memory_id=it["memory_id"], source=MemorySource.ME, created_session=it["created_session"], text=it["text"])
            for it in me_source["items"]
        ] if me_source else []
        session_index = max(session_index, me_source["session_index"] if me_source else session_index)

        all_items = mp_items + ms_items + me_items
        queries = source_specific_memory_queries(current_user_text, [], "")
        me_metadata = me_subtype_hints(me_items) if me_items else {}
        source_metadata = {**mp_metadata, **me_metadata}
        discoveries = discover_final_typed_memory_candidates(
            queries=queries, items=all_items, source_metadata=source_metadata, session_index=session_index,
            ms_semantic_encoder=encoder,
        )
        rs_pool = rs_mechanical_candidate_pool(recent_dialogue=[{"speaker": "seeker", "content": current_user_text}], cards=cards)
        rs_shared = rs_shared_candidate_top1(rs_pool)

        components, surfaces, lineage = {}, {}, {}
        for source in (MemorySource.MP, MemorySource.MS, MemorySource.ME):
            selected = discoveries[source].selected_items
            rank1 = selected[0] if selected else None
            available = rank1 is not None
            if source is MemorySource.ME and rank1 is not None:
                available = compile_atomic_reusable_outcome(rank1.text) is not None
            components[source.value] = {
                "candidate_present": rank1 is not None, "exact_rank1_id": rank1.memory_id if rank1 else None,
                "available": available,
            }
            surfaces[source.value] = rank1.text if (rank1 and available) else None
            lineage[source.value] = {
                "candidate_id": rank1.memory_id if rank1 else None,
                "candidate_surface_sha256": _sha(rank1.text) if rank1 else None,
                "created_session": rank1.created_session if rank1 else None,
                "candidate_version": 1,
            }
        components["RS"] = {
            "candidate_present": rs_shared is not None,
            "exact_rank1_id": rs_shared.observation.move_id if rs_shared else None,
            "available": rs_shared is not None,
        }
        surfaces["RS"] = rs_shared.observation.strategy_card.retrieval_text if rs_shared else None
        lineage["RS"] = {
            "candidate_id": rs_shared.observation.move_id if rs_shared else None,
            "candidate_surface_sha256": _sha(rs_shared.observation.strategy_card.retrieval_text) if rs_shared else None,
            "selection_mode": rs_shared.selection_mode if rs_shared else None,
            "candidate_version": 1,
        }
        n_available = sum(1 for c in components.values() if c["available"])
        available_pairs = [k for k, v in components.items() if v["available"]]

        interaction.append({
            "component": "INTERACTION", "state_id": f"p2r_ix_{family}",
            "user_id": user_id, "family": family, "subdomain": "MULTI_COMPONENT",
            "counterfactual_group_id": f"cfg_ix_{family}",
            "proposed_split": "TBD_by_leader",
            "current_user_text": current_user_text,
            "current_goal": current_user_text,
            "visible_dialogue": [{"role": "user", "content": current_user_text}],
            "session_index": session_index,
            "n_components_with_real_candidate": n_available,
            "available_component_pairs_count": len(available_pairs),
            "components": components,
            "model_visible_surfaces": surfaces,
            "candidate_lineage": lineage,
            "step1_features": {
                comp: (
                    None if surfaces[comp] is None else True
                ) for comp in components
            },
            "n_candidates_in_catalog": len(all_items) + len(rs_pool.candidates),
            "current_surface_sha256": _sha(current_user_text),
        })
    return interaction


def self_check(mp, ms, me, rs, interactions) -> dict:
    """Mirrors the leader audit's own check logic (reads its script only to
    reuse field NAMES/thresholds already published in the audit report --
    does not modify or execute the leader's script) so problems are caught
    before handoff, not after."""

    mp_positive = [r for r in mp if r["audit_only"]["negative_kind"] in {"preference_positive", "profile_positive"}]
    mp_profile = [r for r in mp if r["audit_only"]["negative_kind"] == "profile_positive"]
    checks = {
        "mp_positive_marked_redundant": sum(r["step1_features"]["current_redundant"] is True for r in mp_positive),
        "mp_profile_need_false": sum(r["step1_features"]["profile_goal_needs_advice_or_arrangement"] is False for r in mp_profile),
        "mp_rows_missing_structured_fields": sum(any(f not in r for f in ("field_type", "field_value", "candidate_version")) for r in mp),
        "ms_one_item_pools": sum(r["n_candidates_in_catalog"] == 1 for r in ms),
        "ms_missing_semantic_score": sum("score_top1_semantic_relevance" not in r for r in ms),
        "me_answer_feature_rows": sum("rank1_matches_intended_target" in r["step1_features"] for r in me),
        "rs_subtype_is_selection_mode": sum(r["exact_rank1_subtype"] in {"transparent_priority", "lexical_fallback"} for r in rs),
        "machine_label_current_goal_rows": sum(
            r["current_goal"] in {r["family"], r.get("subdomain"), r["audit_only"].get("negative_kind"),
                                   "continuity_request", "joint_advice_and_continuity", "positive",
                                   "current_redundant", "explicit_decline", "goal_mismatch"}
            for r in mp + ms + me + rs
        ),
        "missing_lineage_fields": sum(
            any(f not in r for f in ("candidate_version", "current_surface_sha256", "candidate_surface_sha256"))
            for r in mp + ms + me + rs
        ),
        "interactions_missing_required_keys": sum(
            any(k not in r for k in ("model_visible_surfaces", "step1_features", "candidate_lineage")) for r in interactions
        ),
    }
    return checks


def _similarity_flags(rows: list[dict], threshold: float = 0.72) -> dict:
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["component"], r["audit_only"].get("negative_kind") if "audit_only" in r else None)].append(r)
    flagged = {}
    for key, members in buckets.items():
        high = 0
        for i, left in enumerate(members):
            for right in members[i + 1:]:
                score = SequenceMatcher(None, left["current_user_text"].casefold(), right["current_user_text"].casefold()).ratio()
                if score >= threshold:
                    high += 1
        if high:
            flagged[str(key)] = high
    return flagged


def main() -> None:
    bindings = build_static_release_bindings(ROOT)
    print(f"release_identity={bindings.release_identity}")

    mp_states = build_mp_states()
    print("loading BGE-M3 (release-bound snapshot)...")
    encoder = BgeM3Encoder()
    ms_states = build_ms_states(encoder)
    me_states = load_me_states_v2() + build_me_new_families()
    cards = [StrategyCard(**row) for row in (json.loads(line) for line in CARDS_PATH.open() if line.strip())]
    rs_states = build_rs_states(cards)
    interaction_states = build_interaction_states(cards, encoder)

    checks = self_check(mp_states, ms_states, me_states, rs_states, interaction_states)
    similarity = _similarity_flags(mp_states + ms_states + me_states)

    all_single = mp_states + ms_states + me_states + rs_states
    summary = {
        "protocol": "pm-v1.5-v5.3-p2-candidate-blueprint-v2-w7r-v1",
        "release_identity": bindings.release_identity,
        "n_mp_states": len(mp_states), "n_ms_states": len(ms_states),
        "n_me_states": len(me_states), "n_rs_states": len(rs_states),
        "n_interaction_states": len(interaction_states),
        "n_single_component_states_total": len(all_single),
        "n_unique_counterfactual_groups": len({s["counterfactual_group_id"] for s in all_single}),
        "n_unique_users": len({s["user_id"] for s in all_single} | {s["user_id"] for s in interaction_states}),
        "n_unique_families": len({s["family"] for s in all_single}),
        "groups_by_component": {
            "MP": len({s["counterfactual_group_id"] for s in mp_states}),
            "MS": len({s["counterfactual_group_id"] for s in ms_states}),
            "ME": len({s["counterfactual_group_id"] for s in me_states}),
            "RS": len({s["counterfactual_group_id"] for s in rs_states}),
        },
        "candidate_availability": {
            "mp": sum(1 for s in mp_states if s["candidate_present"]) / len(mp_states),
            "ms": sum(1 for s in ms_states if s["candidate_present"]) / len(ms_states),
            "me_available": sum(1 for s in me_states if s["me_available"]) / len(me_states),
            "rs": sum(1 for s in rs_states if s["candidate_present"]) / len(rs_states),
        },
        "n_candidates_in_catalog_distribution": {
            "mp": sorted({s["n_candidates_in_catalog"] for s in mp_states}),
            "ms": sorted({s["n_candidates_in_catalog"] for s in ms_states}),
            "me": sorted({s["n_candidates_in_catalog"] for s in me_states}),
            "rs": sorted({s["n_candidates_in_catalog"] for s in rs_states}),
        },
        "self_check_against_leader_audit_criteria": checks,
        "high_similarity_pairs_by_component_and_kind": similarity,
        "note": (
            "Real, unique counts only -- no template duplicated across states, "
            "no final N frozen (leader decision). Supersedes W7 (script 71w) "
            "without overwriting it; see v1 output for comparison."
        ),
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
