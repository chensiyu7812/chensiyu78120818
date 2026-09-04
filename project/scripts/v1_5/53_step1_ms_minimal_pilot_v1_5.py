"""Step1 minimal pilot: MS only, real generation + real judge scoring.

Status: dry-run by default, real --live calls only on explicit request.

Same pattern and same caveats as 52_step1_mp_minimal_pilot_v1_5.py (see that
module's docstring): a cheap plumbing + directional-signal pilot, not the
formally qualified decision-grade judge. This one asks the MS-specific
question the qualifying trial (45/script 49, BGE vs lexical top-1) never
asked: given the DECIDED production MS method (BGE-M3, see "决定
（2026-08-06）：MS默认使用BGE-M3" in PM_V1_5_V5_3_MS_QUALIFYING_TRIAL_
FINDINGS_20260806_ZH.md), does including an MS candidate at all beat having
none, in real judged pairwise comparison? MS fires on 138/138 real panel
states (confirmed separately), so state availability is not a concern here
the way it is for ME.

Per state: 2 real generator calls (with-MS, without-MS/R0) + 2 real
quality-judge calls (forward, reverse order) + 2 real risk-judge calls (one
per arm) = 6 real calls. Default n=15.
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
from metacom_pm.io import stable_hex  # noqa: E402
from metacom_pm.retrieval import source_specific_memory_queries  # noqa: E402
from metacom_pm.v1_5_candidate_discovery import (  # noqa: E402
    discover_final_typed_memory_candidates,
)
from metacom_pm.v1_5_mvp_judge import (  # noqa: E402
    ImmediateSupportJudgment,
    PointwiseRiskJudgment,
    build_immediate_support_messages,
    build_pointwise_risk_messages,
    resolve_ab_ba_quality,
    reverse_quality_pair,
    validate_quality_excerpts,
    validate_risk_excerpts,
)
from metacom_pm.v1_5_typed_resource_adapter import TypedResourceCandidate  # noqa: E402
from metacom_pm.v1_5_v5_3_typed_response_program import (  # noqa: E402
    build_typed_response_program,
    call_with_guard_and_rewrite,
    evidence_aware_generation_messages,
)

EVOEMO = ROOT / "data/external/evo_emo.json"
PANEL_DIR = ROOT / "outputs/pm_v1_5b_corrected_external_split_v1"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_step1_ms_minimal_pilot_v1"
PROTOCOL = "pm-v1.5-step1-ms-minimal-pilot-v1"

QUALITY_MAX_OUTPUT_TOKENS = 900
RISK_MAX_OUTPUT_TOKENS = 1300

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


def find_ms_states(states: list[dict], users: dict, n: int, encoder) -> list[dict]:
    """One state per user where MS (ranked by the decided production
    method, BGE-M3) selects >=1 candidate. MS fires on 138/138 real states
    (verified separately), so this is a selection among plenty, not a
    scarcity search like ME's."""

    picked: list[dict] = []
    seen_users: set[str] = set()
    for state in states:
        if len(picked) >= n:
            break
        uid = state["user_id"]
        if uid in seen_users:
            continue
        user = users[uid]
        session_index = len(user.get("dialog_history") or []) + 1
        items, _extra = build_evo_memory(user)
        queries = source_specific_memory_queries(
            state["current_user_text"], state["current_session_history"], state["current_session_summary"]
        )
        discoveries = discover_final_typed_memory_candidates(
            queries=queries, items=items, source_metadata={}, session_index=session_index,
            ms_semantic_encoder=encoder,
        )
        ms_selected = discoveries[MemorySource.MS].selected_items
        if not ms_selected:
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

    def make(with_ms: bool):
        candidates: dict[str, TypedResourceCandidate] = {}
        action = "M0+R0"
        if with_ms:
            candidates["MS"] = _ms_candidate(ms_item, session_index, uid)
            action = "MS+R0"
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
    return with_program, with_messages, without_program, without_messages, ms_item


def visible_dialogue_for(state: dict) -> dict:
    return {
        "current_session_summary": state["current_session_summary"],
        "current_session_history": state["current_session_history"],
        "current_user_text": state["current_user_text"],
    }


def _judge_call(client, schema, messages, *, seed_parts: tuple, max_tokens: int, validate=None):
    """Same as 52_step1_mp_minimal_pilot_v1_5.py's _judge_call -- see that
    module's docstring for the two real bugs this shape was built to avoid
    (uncaught transient provider errors crashing the batch; excerpt
    validation happening outside the retry loop so repair never fires)."""

    seed = int(stable_hex(PROTOCOL, *seed_parts, n=8), 16) & 0x7FFFFFFF
    result = None
    for attempt in range(2):
        attempt_messages = list(messages)
        if attempt:
            attempt_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Format repair only: return the same required JSON schema, "
                        "but every non-sentinel excerpt must be copied verbatim as an "
                        "exact substring from the supplied candidate response or "
                        "visible evidence. Do not paraphrase excerpts."
                    ),
                }
            )
        try:
            result, parsed = client.chat(
                attempt_messages, temperature=0.0, max_tokens=max_tokens,
                seed=seed + attempt, response_schema=schema, retries=2,
            )
        except Exception as exc:  # noqa: BLE001 -- provider failure degrades this
            # one call, not the whole batch.
            print(f"    (judge call failed, skipping this verdict: {exc})")
            return None, None
        if parsed is None:
            continue
        if validate is not None:
            try:
                validate(parsed)
            except ValueError as exc:
                if attempt == 0:
                    print(f"    (judge excerpt validation failed on attempt 1, retrying: {exc})")
                    continue
                return None, None
        return parsed, result
    return None, result


def run_pair_judgment(judge_client, entry: dict, state: dict, with_reply: str, without_reply: str,
                       ms_item: MemoryItem, aliases: tuple[str, ...]) -> dict:
    visible = visible_dialogue_for(state)

    fwd_messages = build_immediate_support_messages(
        visible_dialogue=visible, response_a=with_reply, response_b=without_reply,
        current_user_known_aliases=aliases,
    )
    rev_a, rev_b = reverse_quality_pair(response_a=with_reply, response_b=without_reply)
    rev_messages = build_immediate_support_messages(
        visible_dialogue=visible, response_a=rev_a, response_b=rev_b,
        current_user_known_aliases=aliases,
    )
    fwd_parsed, _ = _judge_call(
        judge_client, ImmediateSupportJudgment, fwd_messages,
        seed_parts=(state["state_id"], "quality", "fwd"), max_tokens=QUALITY_MAX_OUTPUT_TOKENS,
        validate=lambda p: validate_quality_excerpts(p, response_a=with_reply, response_b=without_reply),
    )
    rev_parsed, _ = _judge_call(
        judge_client, ImmediateSupportJudgment, rev_messages,
        seed_parts=(state["state_id"], "quality", "rev"), max_tokens=QUALITY_MAX_OUTPUT_TOKENS,
        validate=lambda p: validate_quality_excerpts(p, response_a=rev_a, response_b=rev_b),
    )
    quality: dict = {"status": "no_structured_output"}
    if fwd_parsed is not None and rev_parsed is not None:
        quality = resolve_ab_ba_quality(fwd_parsed, rev_parsed)
        quality["status"] = "ok"

    risk_results = {}
    for arm, reply, selected_evidence in (
        ("with_ms", with_reply, {"MS": ms_item.text}),
        ("without_ms", without_reply, {}),
    ):
        risk_messages = build_pointwise_risk_messages(
            visible_dialogue=visible, selected_evidence=selected_evidence, response=reply,
            current_user_known_aliases=aliases,
        )
        risk_parsed, _ = _judge_call(
            judge_client, PointwiseRiskJudgment, risk_messages,
            seed_parts=(state["state_id"], "risk", arm), max_tokens=RISK_MAX_OUTPUT_TOKENS,
            validate=lambda p: validate_risk_excerpts(
                p, response=reply, visible_dialogue=visible, selected_evidence=selected_evidence
            ),
        )
        if risk_parsed is None:
            risk_results[arm] = {"status": "no_structured_output_or_excerpt_validation_failed"}
            continue
        risk_results[arm] = {
            "status": "ok",
            "findings": [f.model_dump() for f in risk_parsed.findings],
        }

    return {"quality": quality, "risk": risk_results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Make real, billable API calls.")
    parser.add_argument("--n", type=int, default=15, help="Number of MS-firing states to test.")
    parser.add_argument("--judge-endpoint", default="training_judge",
                         help="Endpoint name in configs/experiment.yaml (default: Gemini 2.5 Flash "
                              "Lite -- NOT the formally qualified decision-grade judge).")
    args = parser.parse_args()

    from metacom_pm.v1_5_v5_3_semantic_ms_retrieval import BgeM3Encoder  # noqa: PLC0415
    encoder = BgeM3Encoder()

    users = {str(u["id"]): u for u in json.loads(EVOEMO.read_text(encoding="utf-8"))}
    states = load_states()
    batch = find_ms_states(states, users, args.n, encoder)
    print(f"selected {len(batch)} states where MS fires (BGE-M3), across "
          f"{len({e['state']['user_id'] for e in batch})} distinct users")

    if not args.live:
        for entry in batch:
            print(f"  {entry['state']['state_id']} user={entry['state']['user_id']}")
        print("\n(dry-run: no generation or judge calls made)")
        return

    from metacom_pm.api import make_client  # noqa: PLC0415
    from metacom_pm.config import endpoint_from_config, load_config  # noqa: PLC0415
    from metacom_pm.contracts import StrictModel  # noqa: PLC0415

    class _GenSchema(StrictModel):
        reply: str
        used_evidence_ids: list[str]
        realized_response_act: str

    config = load_config(ROOT / "configs/experiment.yaml")
    gen_endpoint = endpoint_from_config(config, "generator")
    gen_client = make_client(gen_endpoint)
    judge_endpoint = endpoint_from_config(config, args.judge_endpoint)
    judge_client = make_client(judge_endpoint)

    results = []
    try:
        for i, entry in enumerate(batch, 1):
            state = entry["state"]
            with_program, with_messages, without_program, without_messages, ms_item = build_pair(entry)
            print(f"\n=== [{i}/{len(batch)}] state={state['state_id']} user={state['user_id']} ===")
            print(f"  MS candidate: {ms_item.text[:100]!r}")

            with_resp, with_status, with_errs = call_with_guard_and_rewrite(
                gen_client, _GenSchema, with_messages, with_program
            )
            without_resp, without_status, without_errs = call_with_guard_and_rewrite(
                gen_client, _GenSchema, without_messages, without_program
            )
            print(f"  [WITH MS, status={with_status}]: {with_resp.reply if with_resp else None}")
            print(f"  [WITHOUT MS, status={without_status}]: {without_resp.reply if without_resp else None}")

            row = {
                "state_id": state["state_id"], "user_id": state["user_id"], "ms_candidate": ms_item.text,
                "with_ms_reply": with_resp.reply if with_resp else None,
                "with_ms_status": with_status, "with_ms_first_pass_guard_errors": list(with_errs),
                "without_ms_reply": without_resp.reply if without_resp else None,
                "without_ms_status": without_status, "without_ms_first_pass_guard_errors": list(without_errs),
            }
            if with_resp is not None and without_resp is not None:
                judgment = run_pair_judgment(
                    judge_client, entry, state, with_resp.reply, without_resp.reply, ms_item,
                    known_aliases_for(entry["user"]),
                )
                row["judgment"] = judgment
                print(f"  [JUDGE quality={judgment['quality'].get('status')} "
                      f"resolved={judgment['quality'].get('resolved_preference')}]")
                print(f"  [JUDGE risk with_ms={judgment['risk']['with_ms'].get('status')} "
                      f"without_ms={judgment['risk']['without_ms'].get('status')}]")
            else:
                row["judgment"] = {"status": "skipped_generation_incomplete"}
                print("  [JUDGE skipped: one or both generations produced no structured output]")
            results.append(row)
    finally:
        gen_client.close()
        judge_client.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "step1_ms_minimal_pilot_results.jsonl"
    with out_path.open("w") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    resolved = [
        r["judgment"]["quality"].get("resolved_preference")
        for r in results
        if r.get("judgment", {}).get("quality", {}).get("status") == "ok"
    ]
    print(f"\n=== summary ===")
    print(f"pairs generated: {len(results)}")
    print(f"pairs with a valid resolved quality judgment: {len(resolved)}")
    for label in ("A", "B", "tie", "abstain"):
        print(f"  resolved={label} (A=with_ms): {resolved.count(label)}")
    print("NOTE: training_judge is not the formally qualified decision-grade judge -- "
          "this is a pipeline-plumbing + directional-signal pilot, not a publication-grade conclusion.")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
