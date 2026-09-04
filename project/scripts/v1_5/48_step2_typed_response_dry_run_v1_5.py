"""Step2 typed_response_program runner: real candidates, dry-run by default.

Status: supports a real small-batch --live run (explicitly authorized by the
user on 2026-08-06, "step2可以继续", after being told this needs "source a
key + agree to a small spend"). Wires v1_5_v5_3_typed_response_program.py to
real production candidates (via the same discover_final_typed_memory_
candidates()/build_evo_memory() functions used throughout the ME/MS/MP
verification work) and to the real `generator` endpoint config in
project/configs/experiment.yaml (Llama-3.1-8B-Instruct via NVIDIA), closing
the gap the module's own docstring flags: "implemented, unit-tested, NOT yet
wired into any generation pipeline."

This script makes NO network call unless invoked with --live. Without
--live, it builds real TypedResponseProgram(s) and real generation messages
from real qualifying-panel states, prints them, and validates a synthetic
(hand-written, clearly fake) response against parse_generator_response_dict
+ typed_response_guard_errors, to prove the downstream parsing/validation
path is correct before any money is spent.

--live requires NVIDIA_API_KEY (`source ~/.metacom_v1_5_secrets.env`) and
makes --n real, billable API calls (default 10, one per distinct user where
possible, to avoid all calls landing on one user's writing style). Real
responses are checked with the real typed_response_guard_errors (not the
synthetic ones dry-run uses) and saved to
outputs/pm_v1_5_v5_3_step2_live_test_v1/ (gitignored per this project's
artifact policy -- may contain model-generated text derived from private
EvoEmo user content).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.contracts import MemoryItem, MemorySource  # noqa: E402
from metacom_pm.evoemo import build_evo_memory  # noqa: E402
from metacom_pm.retrieval import source_specific_memory_queries  # noqa: E402
from metacom_pm.v1_5_candidate_discovery import (  # noqa: E402
    discover_final_typed_memory_candidates,
)
from metacom_pm.v1_5_typed_resource_adapter import TypedResourceCandidate  # noqa: E402
from metacom_pm.v1_5_v5_3_typed_response_program import (  # noqa: E402
    build_typed_response_program,
    call_with_guard_and_rewrite,
    content_words,
    evidence_aware_generation_messages,
    evidence_usage_plausibility_errors,
    parse_generator_response_dict,
    speaker_attribution_guard_errors,
    typed_response_guard_errors,
)

EVOEMO = ROOT / "data/external/evo_emo.json"
PANEL_DIR = ROOT / "outputs/pm_v1_5b_corrected_external_split_v1"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_step2_live_test_v1"

# 2026-08-06: EvoEmo's frozen runtime_state has no current_goal-like field
# (checked: allowed_actions, card_id, current_session_history,
# current_session_summary, current_user_text, inventory, provenance,
# semantic_family, session_index, split, state_id, user_id -- nothing else).
# The original version of this script wrote a plausible-sounding goal by
# hand, which is exactly the kind of hidden/invented-intent leak this
# project's own rule forbids (evaluator-only signals must never enter the
# policy/generator view). Use an honest, observable task description
# instead of pretending a real intent-recognition step already ran.
OBSERVABLE_RESPONSE_TASK_GOAL = (
    "Respond to the user's latest message based on the visible conversation, "
    "respecting any explicit request or boundary stated in it."
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_states() -> list[dict]:
    states = []
    for fname in ("evoemo_qualification_panel_private.jsonl", "evoemo_lockbox_panel_private.jsonl"):
        for row in load_jsonl(PANEL_DIR / fname):
            rt = row["runtime_state"]
            states.append(
                {
                    "state_id": row["state_id"],
                    "user_id": row["user_id_private_analysis_only"],
                    "current_user_text": rt.get("current_user_text", ""),
                    "current_session_history": rt.get("current_session_history", []),
                    "current_session_summary": rt.get("current_session_summary", ""),
                }
            )
    return states


def _ms_candidate(item: MemoryItem, session_index: int, user_id: str) -> TypedResourceCandidate:
    return TypedResourceCandidate(
        component="MS",
        subtype="MS_SESSION_OBSERVATION",
        resource_id=item.memory_id,
        candidate_version="v1",
        source_kind="session",
        owner_id=user_id,
        strictly_prior=True,
        age_sessions=session_index - item.created_session,
        prior_observation=item.text,
    )


def _mp_candidate(item: MemoryItem, user_id: str) -> TypedResourceCandidate:
    return TypedResourceCandidate(
        component="MP",
        subtype="MP_PROFILE",
        resource_id=item.memory_id,
        candidate_version="v1",
        source_kind="profile",
        owner_id=user_id,
        profile_fact=item.text,
    )


def find_demo_states(
    states: list[dict], users: dict, n: int, exclude_state_ids: frozenset[str] = frozenset(),
    ms_semantic_encoder=None,
) -> list[tuple[dict, dict, int]]:
    """Find up to n real states with a usable MS candidate (MP optional).

    Prefers spreading across distinct users first (so a small live batch
    isn't accidentally all one person's writing style), then fills any
    remaining slots from additional states of users already picked.

    exclude_state_ids lets a re-test draw a genuinely independent sample
    instead of resending the exact same states a previous run already used
    (Codex's correct point: the original 10 became a regression set once
    they were used to find/verify the fix, not fresh qualifying evidence).

    ME is deliberately not attempted here: the ME verification work already
    established a 0.40% strict-compiler pass rate, so requiring a real ME
    candidate for the demo would very likely fail on almost any state, for
    reasons already documented, not a bug in this script.

    2026-08-06: scanning always used the cheap lexical-tiebreak path
    (ms_semantic_encoder=None) even after BGE-M3 became the decided
    default -- whether a state has >=1 MS candidate at all does not depend
    on which scorer ranks it, so this does not change which states qualify
    or their selection order. ms_semantic_encoder (if given) is only
    applied once, to the final n-sized batch, not to every state examined
    while scanning -- CPU BGE inference on every candidate pool of every
    scanned state (not just the ones actually used) made this function slow
    enough to matter in practice once it started running by default.
    """
    qualifying: list[tuple[dict, dict, int]] = []
    for state in states:
        if state["state_id"] in exclude_state_ids:
            continue
        uid = state["user_id"]
        user = users[uid]
        session_index = len(user.get("dialog_history") or []) + 1
        items, _extra = build_evo_memory(user)
        queries = source_specific_memory_queries(
            state["current_user_text"], state["current_session_history"], state["current_session_summary"]
        )
        discoveries = discover_final_typed_memory_candidates(
            queries=queries, items=items, source_metadata={}, session_index=session_index,
        )
        if discoveries[MemorySource.MS].selected_items:
            qualifying.append((state, items, queries, session_index))
    if not qualifying:
        raise RuntimeError("no state in the panel had any MS candidate -- unexpected, investigate")

    seen_users: set[str] = set()
    first_pass: list[tuple] = []
    rest: list[tuple] = []
    for entry in qualifying:
        uid = entry[0]["user_id"]
        if uid not in seen_users:
            seen_users.add(uid)
            first_pass.append(entry)
        else:
            rest.append(entry)
    ordered = (first_pass + rest)[:n]

    # Real discovery (with the real scorer, if any) only for the states
    # actually selected.
    final: list[tuple[dict, dict, int]] = []
    for state, items, queries, session_index in ordered:
        discoveries = discover_final_typed_memory_candidates(
            queries=queries, items=items, source_metadata={}, session_index=session_index,
            ms_semantic_encoder=ms_semantic_encoder,
        )
        final.append((state, discoveries, session_index))
    return final


def known_aliases_for(user: dict) -> tuple[str, ...]:
    """EvoEmo's own basic_info.name is known, structured data -- e.g. "Anna
    Li" -- used by ~20-79% of a given user's MS session summaries to refer
    to them in the third person (checked directly against the real data,
    not assumed). Return both the full name and first name: MS text uses
    either ("Anna" alone, or occasionally the full "Anna Li")."""

    name = str((user.get("basic_info") or {}).get("name") or "").strip()
    if not name:
        return ()
    first = name.split()[0]
    return (first, name) if first != name else (name,)


def build_program_and_messages(
    state: dict, discoveries: dict, session_index: int, uid: str, user: dict
) -> tuple:
    candidates: dict[str, TypedResourceCandidate] = {}
    ms_item = discoveries[MemorySource.MS].selected_items[0]
    candidates["MS"] = _ms_candidate(ms_item, session_index, uid)
    requested_action_id = "MS+R0"
    mp_selected = discoveries[MemorySource.MP].selected_items
    if mp_selected:
        candidates["MP"] = _mp_candidate(mp_selected[0], uid)
        requested_action_id = "MPMS+R0" if "MS" in candidates else "MP+R0"

    program = build_typed_response_program(
        current_user_known_aliases=known_aliases_for(user),
        requested_action_id=requested_action_id,
        current_goal=OBSERVABLE_RESPONSE_TASK_GOAL,
        current_user_id=uid,
        candidates=candidates,
    )
    messages = evidence_aware_generation_messages(
        current_context=state["current_user_text"], program=program
    )
    return program, messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true",
        help="Make real, billable API calls. Requires NVIDIA_API_KEY.",
    )
    parser.add_argument(
        "--n", type=int, default=10,
        help="Number of real states to process (default 10, spread across distinct users where possible).",
    )
    parser.add_argument(
        "--no-bge-ms", action="store_true",
        help="Use the legacy lexical-tiebreak MS ranking instead of the decided BGE-M3 "
        "default (for A/B comparison only -- see PM_V1_5_V5_3_MS_QUALIFYING_TRIAL_"
        "FINDINGS_20260806_ZH.md 's 'decision' section for why BGE-M3 is now the default).",
    )
    args = parser.parse_args()

    # 2026-08-06: BGE-M3 is the decided default for MS ranking (see the
    # qualifying-trial doc's decision section) -- every real batch from this
    # script now uses it unless --no-bge-ms explicitly asks for the legacy
    # lexical-tiebreak baseline. Local model, no API key/network needed.
    ms_semantic_encoder = None
    if not args.no_bge_ms:
        from metacom_pm.v1_5_v5_3_semantic_ms_retrieval import BgeM3Encoder  # noqa: PLC0415
        ms_semantic_encoder = BgeM3Encoder()

    # A previous run's output becomes the exclusion set for this run (and is
    # archived, not overwritten) -- reusing the same states after changing
    # the prompt/guard would test "does it work on states we already looked
    # at", not "does it generalize" (Codex's correct point: the original 10
    # are a regression set now, not fresh qualifying evidence).
    exclude_state_ids: frozenset[str] = frozenset()
    out_path = OUT_DIR / "step2_live_responses.jsonl"
    if args.live and out_path.exists():
        previous = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
        exclude_state_ids = frozenset(r["state_id"] for r in previous if "state_id" in r)
        archive_path = OUT_DIR / "step2_live_responses_before_prompt_fix_20260806.jsonl"
        if not archive_path.exists():
            out_path.rename(archive_path)
            print(f"archived previous run ({len(exclude_state_ids)} states) to {archive_path}")

    users = {str(u["id"]): u for u in json.loads(EVOEMO.read_text(encoding="utf-8"))}
    states = load_states()
    batch = find_demo_states(
        states, users, args.n, exclude_state_ids=exclude_state_ids,
        ms_semantic_encoder=ms_semantic_encoder,
    )
    print(f"selected {len(batch)} states across {len({s['user_id'] for s, _, _ in batch})} distinct users"
          f" (excluded {len(exclude_state_ids)} previously-tested states)")

    client = None
    response_schema = None
    if args.live:
        from metacom_pm.api import OpenAICompatibleClient  # noqa: PLC0415
        from metacom_pm.config import endpoint_from_config, load_config  # noqa: PLC0415
        from metacom_pm.contracts import StrictModel  # noqa: PLC0415

        class _GeneratorResponseSchema(StrictModel):
            reply: str
            used_evidence_ids: list[str]
            realized_response_act: str

        response_schema = _GeneratorResponseSchema
        config = load_config(ROOT / "configs/experiment.yaml")
        endpoint = endpoint_from_config(config, "generator")
        client = OpenAICompatibleClient(endpoint)

    results: list[dict] = []
    try:
        for i, (state, discoveries, session_index) in enumerate(batch, 1):
            uid = state["user_id"]
            program, messages = build_program_and_messages(
                state, discoveries, session_index, uid, users[uid]
            )
            print(f"\n=== [{i}/{len(batch)}] state={state['state_id']} user={uid} "
                  f"action={program.requested_action_id} evidence={len(program.evidence)} ===")

            if args.live:
                response, status, first_pass_errors = call_with_guard_and_rewrite(
                    client, response_schema, messages, program
                )
                if response is None:
                    print(f"  no structured output: {first_pass_errors}")
                    results.append({"state_id": state["state_id"], "user_id": uid, "error": str(first_pass_errors)})
                    continue
                print(f"  status: {status}  (first-pass guard errors: {first_pass_errors})")
                print(f"  reply: {response.reply[:200]}")
                print(f"  used_evidence_ids: {response.used_evidence_ids}")
                if response.reported_used_evidence_ids != response.used_evidence_ids:
                    print(
                        "  generator_reported_used_evidence_ids: "
                        f"{response.reported_used_evidence_ids}"
                    )
                results.append(
                    {
                        "state_id": state["state_id"],
                        "user_id": uid,
                        "evidence_ids": [e.evidence_id for e in program.evidence],
                        "reply": response.reply,
                        "generator_reported_used_evidence_ids": list(
                            response.reported_used_evidence_ids
                        ),
                        "used_evidence_ids": list(response.used_evidence_ids),
                        "realized_response_act": response.realized_response_act,
                        "status": status,
                        "first_pass_guard_errors": list(first_pass_errors),
                    }
                )
            else:
                ms_item = discoveries[MemorySource.MS].selected_items[0]
                # Exercise the lexical-use guard without copying a raw
                # record fragment into the synthetic visible reply.  Some
                # source summaries legitimately contain record-like words;
                # quoting their first 60 characters made this dry-run test
                # the leak guard rather than the intended parse/binding path.
                safe_words = sorted(
                    content_words(ms_item.text)
                    - {"earlier", "record", "recorded", "seeker", "session"}
                )
                content_hint = safe_words[0] if safe_words else "situation"
                fake_good = {
                    "reply": (
                        "That sounds like a lot to carry. "
                        f"Does the {content_hint} part still feel relevant today?"
                    ),
                    "used_evidence_ids": [item.evidence_id for item in program.evidence],
                    "realized_response_act": "reflection",
                }
                response = parse_generator_response_dict(fake_good)
                errors = (
                    typed_response_guard_errors(response=response, program=program)
                    + speaker_attribution_guard_errors(response=response)
                    + evidence_usage_plausibility_errors(response=response, program=program)
                )
                print(f"  dry-run synthetic response parses cleanly, guard errors: {errors}")
    finally:
        if client is not None:
            client.close()

    if args.live:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        from collections import Counter  # noqa: PLC0415
        status_counts = Counter(r.get("status", "error") for r in results)
        print(f"\nwrote {out_path}: {len(results)} results, status breakdown: {dict(status_counts)}")
        print(
            "NOTE: 'clean' = first pass had no guard violation. 'fixed_by_rewrite'/"
            "'fell_back_to_m0' mean a violation WAS caught on the first pass -- still "
            "read the full reply text manually, do not trust guard-pass counts alone "
            "(the speaker-attribution guard is a calibrated heuristic, not a complete "
            "semantic check; see its docstring)."
        )


if __name__ == "__main__":
    main()
