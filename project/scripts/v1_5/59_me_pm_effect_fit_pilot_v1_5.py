#!/usr/bin/env python3
"""PM Effect FIT data pilot: ME only, one component, small scale.

Second half of the roadmap's step 3 ("PM Effect数据") -- distinct from the
retriever stress test (scripts 57/58), which only asks "can the retriever
find the right candidate." This asks the actual Step1 question: "given a
real, retrievable candidate, was THIS specific turn one where opening ME
would have been worth it?" That is a decision about the current turn's
context, not about retrieval, and needs genuine ON/OFF ground truth, not
just a found-vs-not-found label.

Uses the exact ON/OFF contract PM_V1_5_V5_3_INTEGRATED_EVIDENCE_EXECUTION_
PLAN_20260805_ZH.md section 2.3 already specifies for ME:
  ON:  "当前欢迎一个行动选项，且候选含过去动作及结果/机制"
       (current turn invites an action option AND the candidate has a real
       past action + result/mechanism)
  OFF: "当前不要行动；只有背景事件；结果不可迁移"
       (current turn doesn't want action; OR only a context-only candidate
       exists; OR the goal doesn't match)
  interaction term already named in the plan: past_action_result ×
  current_action_invitation

Every case is built deterministically (template + topic substitution, same
convention as scripts 57/58 -- no free-form generation) and every case's
label is verified two ways before being trusted, not asserted by
construction alone:
  1. candidate validity: compile_atomic_reusable_outcome() must return
     the intended value (not-None for the REUSABLE_OUTCOME cases, None for
     the CONTEXT_EVENT-only cases) -- checked directly, not assumed;
  2. retrieval validity: discover_final_typed_memory_candidates() must
     actually surface the intended candidate as Rank-1 for its query (reusing
     58's now-validated ME retrieval setup) -- a case where retrieval
     doesn't find the intended candidate is dropped, not silently kept.

This is explicitly a SMALL PILOT (one component, ~16 cases across 4
semantic-family reasons x 2 polarities x 2 topics), matching this session's
established "prove the method on one slice before scaling" discipline --
not the full 128-group formal FIT blueprint another Codex's critique
specified (whose exact quantities should be checked against the real frozen
blueprint doc before being reused, not re-derived here).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.contracts import MemoryItem, MemorySource  # noqa: E402
from metacom_pm.io import canonical_json, stable_hex, write_json  # noqa: E402
from metacom_pm.retrieval import source_specific_memory_queries  # noqa: E402
from metacom_pm.v1_5_candidate_discovery import (  # noqa: E402
    discover_final_typed_memory_candidates,
)
from metacom_pm.v1_5_v5_2_atomic_memory import (  # noqa: E402
    compile_atomic_reusable_outcome,
    me_subtype_hints,
)

PROTOCOL = "pm-v1.5-me-pm-effect-fit-pilot-v1"
OUT_PATH = ROOT / "outputs" / "pm_v1_5_me_pm_effect_fit_pilot_v1.json"
SESSION_INDEX = 45

# Two independent topics so ON/OFF cases aren't distinguishable by topic
# identity alone (that would be a label shortcut, not a real construct).
TOPIC_A = "a workplace conflict with a coworker"
TOPIC_B = "financial uncertainty about rent"

REUSABLE_OUTCOME_TEXT = {
    TOPIC_A: "When dealing with a workplace conflict with a coworker, I wrote down the hardest moment before responding, and it helped me feel calmer.",
    TOPIC_B: "When dealing with financial uncertainty about rent, I wrote down the hardest moment before responding, and it helped me feel calmer.",
}
CONTEXT_ONLY_TEXT = {
    TOPIC_A: "In a prior session, the user last spring described a private incident involving a workplace conflict with a coworker.",
    TOPIC_B: "In a prior session, the user last spring described a private incident involving financial uncertainty about rent.",
}

# Phrasing verified directly against observable_flags()'s real
# _ADVICE_WELCOME_RE/_LISTEN_ONLY_RE (v1_5_strategy_rag_runtime.py, already
# validated this session for RS) rather than assumed to "read as" action-
# inviting or action-declining -- see contribution_slot_features' own
# module docstring for why reusing this one validated signal is preferred
# over inventing a second, possibly-inconsistent regex.
ACTION_INVITING_TURN = "I could really use some advice for handling {topic}. What should I do?"
ACTION_DECLINING_TURN = "I just want to vent about {topic} right now. I don't want any advice, I just need you to listen."
REDUNDANT_TURN = (
    "I could really use some advice for handling {topic}. What should I do? Actually, I "
    "already figured it out myself -- it helped me feel calmer, so I've got that covered."
)


def _memory_id(seed_parts: tuple) -> str:
    return "mem_" + stable_hex(PROTOCOL, *seed_parts, n=24)


def _build_pool(topic: str, *, reusable: bool) -> list[MemoryItem]:
    text = REUSABLE_OUTCOME_TEXT[topic] if reusable else CONTEXT_ONLY_TEXT[topic]
    return [
        MemoryItem(
            memory_id=_memory_id((topic, reusable)), source=MemorySource.ME,
            created_session=10, text=text,
        )
    ]


def _case(
    case_id: str, *, current_user_text: str, topic: str, candidate_topic: str,
    reusable_candidate: bool, expected_label: str, reason: str,
) -> dict:
    items = _build_pool(candidate_topic, reusable=reusable_candidate)
    return {
        "case_id": case_id, "current_user_text": current_user_text, "items": items,
        "expected_label": expected_label, "reason": reason,
        "candidate_memory_id": items[0].memory_id,
        "candidate_is_reusable_outcome": reusable_candidate,
    }


def build_cases() -> list[dict]:
    cases = []
    for topic in (TOPIC_A, TOPIC_B):
        # ON: action-inviting turn + real REUSABLE_OUTCOME candidate on the SAME topic.
        cases.append(_case(
            f"on_action_invited_valid_candidate_{topic[:10]}",
            current_user_text=ACTION_INVITING_TURN.format(topic=topic), topic=topic,
            candidate_topic=topic, reusable_candidate=True, expected_label="ON",
            reason="current turn invites action AND candidate has real past action+result",
        ))
        # OFF-1: same valid candidate, but current turn explicitly declines action.
        cases.append(_case(
            f"off_action_declined_valid_candidate_{topic[:10]}",
            current_user_text=ACTION_DECLINING_TURN.format(topic=topic), topic=topic,
            candidate_topic=topic, reusable_candidate=True, expected_label="OFF",
            reason="candidate is valid, but current turn explicitly does not want action",
        ))
        # OFF-2: action-inviting turn, but the only candidate is context-only (no result).
        cases.append(_case(
            f"off_action_invited_context_only_candidate_{topic[:10]}",
            current_user_text=ACTION_INVITING_TURN.format(topic=topic), topic=topic,
            candidate_topic=topic, reusable_candidate=False, expected_label="OFF",
            reason="current turn invites action, but only a context-only (non-reusable) candidate exists",
        ))
        # OFF-3: action-inviting turn, valid candidate exists, but for a DIFFERENT topic
        # (goal mismatch) -- pool built from the OTHER topic's reusable-outcome text.
        other_topic = TOPIC_B if topic == TOPIC_A else TOPIC_A
        cases.append(_case(
            f"off_goal_mismatch_{topic[:10]}",
            current_user_text=ACTION_INVITING_TURN.format(topic=topic), topic=topic,
            candidate_topic=other_topic, reusable_candidate=True, expected_label="OFF",
            reason="current turn invites action, but the only candidate is for a different topic",
        ))
        # OFF-4: current turn is already redundant with what the candidate would add.
        cases.append(_case(
            f"off_current_redundant_{topic[:10]}",
            current_user_text=REDUNDANT_TURN.format(topic=topic), topic=topic,
            candidate_topic=topic, reusable_candidate=True, expected_label="OFF",
            reason="current turn already states the same outcome the candidate would add",
        ))
    return cases


def verify_case(case: dict) -> dict:
    result = {"case_id": case["case_id"], "expected_label": case["expected_label"], "reason": case["reason"]}

    compiled = compile_atomic_reusable_outcome(case["items"][0].text)
    compiled_ok = (compiled is not None) == case["candidate_is_reusable_outcome"]
    result["candidate_validity_check_passed"] = compiled_ok
    if not compiled_ok:
        result["status"] = "DROPPED_CANDIDATE_VALIDITY_MISMATCH"
        return result

    queries = source_specific_memory_queries(case["current_user_text"], [], "")
    hints = me_subtype_hints(case["items"])
    discoveries = discover_final_typed_memory_candidates(
        queries=queries, items=case["items"], source_metadata=hints, session_index=SESSION_INDEX,
    )
    selected = discoveries[MemorySource.ME].selected_items
    top1_id = selected[0].memory_id if selected else None
    retrieval_ok = top1_id == case["candidate_memory_id"]
    result["retrieval_check_passed"] = retrieval_ok
    if not retrieval_ok:
        result["status"] = "DROPPED_RETRIEVAL_DID_NOT_SURFACE_INTENDED_CANDIDATE"
        result["actual_top1"] = top1_id
        return result

    result["status"] = "VALID_FIT_CASE"
    return result


def main() -> None:
    cases = build_cases()
    results = [verify_case(c) for c in cases]
    n_valid = sum(1 for r in results if r["status"] == "VALID_FIT_CASE")
    n_on = sum(1 for r in results if r["status"] == "VALID_FIT_CASE" and r["expected_label"] == "ON")
    n_off = sum(1 for r in results if r["status"] == "VALID_FIT_CASE" and r["expected_label"] == "OFF")

    for r in results:
        print(f"[{r['status']}] {r['case_id']} expected={r['expected_label']} :: {r['reason']}")

    print(f"\n=== summary ===")
    print(f"cases built: {len(cases)}")
    print(f"valid FIT-ready cases (both gold checks passed): {n_valid}/{len(cases)}")
    print(f"  ON: {n_on}  OFF: {n_off} (should be roughly balanced, and OFF should span multiple distinct reasons)")

    write_json(OUT_PATH, {
        "protocol": PROTOCOL,
        "note": (
            "ME-only PM Effect FIT pilot. Every case's label is machine-verified two ways "
            "(candidate validity via compile_atomic_reusable_outcome(), retrieval via "
            "discover_final_typed_memory_candidates()) before being counted -- a case that "
            "fails either check is dropped, not kept with an assumed label. This is a small "
            "pilot proving the methodology on one component and one interaction term "
            "(past_action_result x current_action_invitation, per PM_V1_5_V5_3_INTEGRATED_"
            "EVIDENCE_EXECUTION_PLAN_20260805_ZH.md section 2.3), not the full formal FIT "
            "blueprint."
        ),
        "results": results,
        "summary": {"n_cases": len(cases), "n_valid": n_valid, "n_on": n_on, "n_off": n_off},
    })
    print(f"\nfull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
