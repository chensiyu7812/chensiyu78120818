"""MP effect test: does including a correctly-matched MP candidate actually
change the real generated reply, versus generating from MS alone?

Status: dry-run by default, real --live calls only on explicit request.

The MP false-positive bugs (field-name-prefix, parenthetical filler words)
are fixed and verified (see PM_V1_5_V5_3_MP_VERIFICATION_FINDINGS_
20260806_ZH.md): MP's remaining nonzero selections are now 100% real
value-word matches, not noise. What has never been tested is the next
question up the chain -- an accurately-matched MP fact might still be
inert (the response reads the same with or without it). This script
generates two real replies for the same real state: one with MP+MS, one
with MS alone, holding everything else constant, so the two can be read
side by side.
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
    evidence_aware_generation_messages,
)

EVOEMO = ROOT / "data/external/evo_emo.json"
PANEL_DIR = ROOT / "outputs/pm_v1_5b_corrected_external_split_v1"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_mp_effect_test_v1"

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


def known_aliases_for(user: dict) -> tuple[str, ...]:
    name = str((user.get("basic_info") or {}).get("name") or "").strip()
    if not name:
        return ()
    first = name.split()[0]
    return (first, name) if first != name else (name,)


def _ms_candidate(item: MemoryItem, session_index: int, user_id: str) -> TypedResourceCandidate:
    return TypedResourceCandidate(
        component="MS", subtype="MS_SESSION_OBSERVATION", resource_id=item.memory_id,
        candidate_version="v1", source_kind="session", owner_id=user_id, strictly_prior=True,
        age_sessions=session_index - item.created_session, prior_observation=item.text,
    )


def _mp_candidate(item: MemoryItem, user_id: str) -> TypedResourceCandidate:
    return TypedResourceCandidate(
        component="MP", subtype="MP_PROFILE", resource_id=item.memory_id,
        candidate_version="v1", source_kind="profile", owner_id=user_id, profile_fact=item.text,
    )


def find_mp_states(states: list[dict], users: dict, n: int) -> list[dict]:
    """Real states where MP selects >=1 candidate AND MS also has a
    candidate (so the "without MP" arm is still MS+R0, not M0+R0 -- keeps
    the comparison to exactly one variable, presence/absence of MP)."""

    picked: list[dict] = []
    seen_users: set[str] = set()
    for state in states:
        if len(picked) >= n:
            break
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
        mp_selected = discoveries[MemorySource.MP].selected_items
        ms_selected = discoveries[MemorySource.MS].selected_items
        if not mp_selected or not ms_selected:
            continue
        if uid in seen_users:
            continue
        seen_users.add(uid)
        picked.append(
            {"state": state, "user": user, "session_index": session_index, "discoveries": discoveries}
        )
    return picked


def build_pair(entry: dict) -> tuple:
    state, user, session_index = entry["state"], entry["user"], entry["session_index"]
    discoveries = entry["discoveries"]
    uid = state["user_id"]
    aliases = known_aliases_for(user)
    ms_item = discoveries[MemorySource.MS].selected_items[0]
    mp_item = discoveries[MemorySource.MP].selected_items[0]

    def make(with_mp: bool):
        candidates: dict[str, TypedResourceCandidate] = {
            "MS": _ms_candidate(ms_item, session_index, uid)
        }
        action = "MS+R0"
        if with_mp:
            candidates["MP"] = _mp_candidate(mp_item, uid)
            action = "MPMS+R0"
        program = build_typed_response_program(
            requested_action_id=action, current_goal=OBSERVABLE_RESPONSE_TASK_GOAL,
            current_user_id=uid, candidates=candidates, current_user_known_aliases=aliases,
        )
        messages = evidence_aware_generation_messages(
            current_context=state["current_user_text"], program=program
        )
        return program, messages

    with_program, with_messages = make(True)
    without_program, without_messages = make(False)
    return with_program, with_messages, without_program, without_messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Make real, billable API calls.")
    parser.add_argument("--n", type=int, default=6, help="Number of MP-firing states to test.")
    args = parser.parse_args()

    users = {str(u["id"]): u for u in json.loads(EVOEMO.read_text(encoding="utf-8"))}
    states = load_states()
    batch = find_mp_states(states, users, args.n)
    print(f"selected {len(batch)} states where MP fires, across "
          f"{len({e['state']['user_id'] for e in batch})} distinct users")

    client = None
    response_schema = None
    if args.live:
        from metacom_pm.api import OpenAICompatibleClient  # noqa: PLC0415
        from metacom_pm.config import endpoint_from_config, load_config  # noqa: PLC0415
        from metacom_pm.contracts import StrictModel  # noqa: PLC0415

        class _Schema(StrictModel):
            reply: str
            used_evidence_ids: list[str]
            realized_response_act: str

        response_schema = _Schema
        config = load_config(ROOT / "configs/experiment.yaml")
        endpoint = endpoint_from_config(config, "generator")
        client = OpenAICompatibleClient(endpoint)

    results = []
    try:
        for i, entry in enumerate(batch, 1):
            state = entry["state"]
            with_program, with_messages, without_program, without_messages = build_pair(entry)
            print(f"\n=== [{i}/{len(batch)}] state={state['state_id']} user={state['user_id']} ===")
            mp_item = entry["discoveries"][MemorySource.MP].selected_items[0]
            print(f"  MP fact: {mp_item.text}")

            if args.live:
                with_resp, with_status, with_errs = call_with_guard_and_rewrite(
                    client, response_schema, with_messages, with_program
                )
                without_resp, without_status, without_errs = call_with_guard_and_rewrite(
                    client, response_schema, without_messages, without_program
                )
                print(f"  [WITH MP, status={with_status}, first_pass_errors={with_errs}]: "
                      f"{with_resp.reply if with_resp else None}")
                print(f"  [WITHOUT MP, status={without_status}, first_pass_errors={without_errs}]: "
                      f"{without_resp.reply if without_resp else None}")
                results.append(
                    {
                        "state_id": state["state_id"], "user_id": state["user_id"],
                        "mp_fact": mp_item.text,
                        "with_mp_reply": with_resp.reply if with_resp else None,
                        "with_mp_status": with_status,
                        "with_mp_first_pass_guard_errors": list(with_errs),
                        "without_mp_reply": without_resp.reply if without_resp else None,
                        "without_mp_status": without_status,
                        "without_mp_first_pass_guard_errors": list(without_errs),
                    }
                )
    finally:
        if client is not None:
            client.close()

    if args.live:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / "mp_paired_generation_results.jsonl"
        with out_path.open("w") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nwrote {out_path}: {len(results)} paired results")


if __name__ == "__main__":
    main()
