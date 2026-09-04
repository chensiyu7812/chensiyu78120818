"""MS effect test: does switching the MS candidate (production vs BGE) for
the same real state actually change the real generated reply?

Status: dry-run by default, real --live calls only on explicit request.

Everything up to this point (script 45, PM_V1_5_V5_3_MS_QUALIFYING_TRIAL_
FINDINGS_20260806_ZH.md) established that BGE and production disagree on
which MS candidate to pick 82.6% of the time, and that BGE's candidate is
topically better by several measures. What has never been tested is whether
that actually changes the *generated response* -- this script generates two
real replies for the same state (same MP if any, same current_context,
same everything else), one built from the production MS candidate, one from
the BGE MS candidate, so the two outputs can be read side by side.

Uses the same real functions as scripts 45-48: discover_final_typed_memory_
candidates() (production, default) and again with ms_semantic_encoder=...
(BGE), build_typed_response_program/evidence_aware_generation_messages/
call_with_guard_and_rewrite from v1_5_v5_3_typed_response_program.py.
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
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_ms_effect_test_v1"

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


def find_disagreement_states(states: list[dict], users: dict, encoder, n: int) -> list[dict]:
    """States where production and BGE pick a genuinely different MS
    candidate (same causal-boundary-checked pool, both real functions)."""

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
        prod = discover_final_typed_memory_candidates(
            queries=queries, items=items, source_metadata={}, session_index=session_index,
        )
        bge = discover_final_typed_memory_candidates(
            queries=queries, items=items, source_metadata={}, session_index=session_index,
            ms_semantic_encoder=encoder,
        )
        prod_ms = prod[MemorySource.MS].selected_items
        bge_ms = bge[MemorySource.MS].selected_items
        if not prod_ms or not bge_ms:
            continue
        if prod_ms[0].memory_id == bge_ms[0].memory_id:
            continue  # not a real disagreement, skip
        if uid in seen_users:
            continue  # spread across distinct users first
        seen_users.add(uid)
        picked.append(
            {
                "state": state, "user": user, "session_index": session_index,
                "prod_discoveries": prod, "bge_discoveries": bge,
            }
        )
    return picked


def build_pair(entry: dict) -> tuple:
    state, user, session_index = entry["state"], entry["user"], entry["session_index"]
    uid = state["user_id"]
    aliases = known_aliases_for(user)

    def one(discoveries):
        candidates: dict[str, TypedResourceCandidate] = {}
        ms_item = discoveries[MemorySource.MS].selected_items[0]
        candidates["MS"] = _ms_candidate(ms_item, session_index, uid)
        action = "MS+R0"
        mp_selected = discoveries[MemorySource.MP].selected_items
        if mp_selected:
            candidates["MP"] = _mp_candidate(mp_selected[0], uid)
            action = "MPMS+R0"
        program = build_typed_response_program(
            requested_action_id=action, current_goal=OBSERVABLE_RESPONSE_TASK_GOAL,
            current_user_id=uid, candidates=candidates, current_user_known_aliases=aliases,
        )
        messages = evidence_aware_generation_messages(
            current_context=state["current_user_text"], program=program
        )
        return program, messages

    prod_program, prod_messages = one(entry["prod_discoveries"])
    bge_program, bge_messages = one(entry["bge_discoveries"])
    return prod_program, prod_messages, bge_program, bge_messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Make real, billable API calls.")
    parser.add_argument("--n", type=int, default=8, help="Number of disagreement states to test.")
    args = parser.parse_args()

    from metacom_pm.v1_5_v5_3_semantic_ms_retrieval import BgeM3Encoder  # noqa: PLC0415
    encoder = BgeM3Encoder()

    users = {str(u["id"]): u for u in json.loads(EVOEMO.read_text(encoding="utf-8"))}
    states = load_states()
    batch = find_disagreement_states(states, users, encoder, args.n)
    print(f"selected {len(batch)} states with real production/BGE MS disagreement, "
          f"across {len({e['state']['user_id'] for e in batch})} distinct users")

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
            prod_program, prod_messages, bge_program, bge_messages = build_pair(entry)
            print(f"\n=== [{i}/{len(batch)}] state={state['state_id']} user={state['user_id']} ===")
            print(f"  production MS: {prod_program.evidence[-1].literal_evidence[:100]}")
            print(f"  BGE MS:        {bge_program.evidence[-1].literal_evidence[:100]}")

            if args.live:
                prod_resp, prod_status, prod_errs = call_with_guard_and_rewrite(
                    client, response_schema, prod_messages, prod_program
                )
                bge_resp, bge_status, bge_errs = call_with_guard_and_rewrite(
                    client, response_schema, bge_messages, bge_program
                )
                print(f"  [production reply, status={prod_status}]: {prod_resp.reply if prod_resp else None}")
                print(f"  [BGE reply, status={bge_status}]: {bge_resp.reply if bge_resp else None}")
                results.append(
                    {
                        "state_id": state["state_id"], "user_id": state["user_id"],
                        "production_ms_evidence": prod_program.evidence[-1].literal_evidence,
                        "bge_ms_evidence": bge_program.evidence[-1].literal_evidence,
                        "production_reply": prod_resp.reply if prod_resp else None,
                        "production_status": prod_status,
                        "production_first_pass_guard_errors": list(prod_errs),
                        "bge_reply": bge_resp.reply if bge_resp else None,
                        "bge_status": bge_status,
                        "bge_first_pass_guard_errors": list(bge_errs),
                    }
                )
    finally:
        if client is not None:
            client.close()

    if args.live:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / "ms_paired_generation_results.jsonl"
        with out_path.open("w") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nwrote {out_path}: {len(results)} paired results")


if __name__ == "__main__":
    main()
