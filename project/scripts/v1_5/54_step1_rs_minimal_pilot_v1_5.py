"""Step1 minimal pilot: RS only, real generation + real judge scoring.

Status: dry-run by default, real --live calls only on explicit request.

Same pattern and caveats as 52/53 (see those modules' docstrings): a cheap
plumbing + directional-signal pilot, not the formally qualified
decision-grade judge.

RS is structurally different from MP/MS/ME: it is not a per-user memory
retrieved via build_evo_memory()/discover_final_typed_memory_candidates().
It is the frozen, ACTIVE_FOR_NEW_PM_V1_5_G3_WORK six-card strategy bank
(data/strategy/strategy_cards_v1_5_minimal.jsonl) retrieved by
QualifiedStrategyRAG (src/metacom_pm/v1_5_strategy_rag_runtime.py):
regex-derived observable eligibility flags -> lexical top-1 ranking ->
abstain below a frozen score floor. This script's only new code is a small
adapter that (a) builds recent_dialogue in the shape that runtime expects
and (b) parses a selected StrategyCard's retrieval_text ("Support move: ...
Use conditions: ... Observable goals: ... Do not use: ...") into the three
typed fields TypedResourceCandidate's RS_ATOMIC_MOVE subtype requires
(support_move/when_to_use/when_not_to_use) -- verified against all 6 real
cards before use, not invented.

Per state: 2 real generator calls (with-RS, without-RS/R0) + 2 real
quality-judge calls (forward, reverse order) + 2 real risk-judge calls (one
per arm) = 6 real calls. Default n=15.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.contracts import StrategyCard  # noqa: E402
from metacom_pm.io import stable_hex  # noqa: E402
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
from metacom_pm.v1_5_strategy_rag_runtime import QualifiedStrategyRAG  # noqa: E402
from metacom_pm.v1_5_typed_resource_adapter import TypedResourceCandidate  # noqa: E402
from metacom_pm.v1_5_v5_3_typed_response_program import (  # noqa: E402
    build_typed_response_program,
    call_with_guard_and_rewrite,
    evidence_aware_generation_messages,
)

EVOEMO = ROOT / "data/external/evo_emo.json"
PANEL_DIR = ROOT / "outputs/pm_v1_5b_corrected_external_split_v1"
STRATEGY_BANK = ROOT / "data/strategy/strategy_cards_v1_5_minimal.jsonl"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_step1_rs_minimal_pilot_v1"
PROTOCOL = "pm-v1.5-step1-rs-minimal-pilot-v1"

QUALITY_MAX_OUTPUT_TOKENS = 900
RISK_MAX_OUTPUT_TOKENS = 1300

OBSERVABLE_RESPONSE_TASK_GOAL = (
    "Respond to the user's latest message based on the visible conversation, "
    "respecting any explicit request or boundary stated in it."
)

# Verified to match all 6 real cards in strategy_cards_v1_5_minimal.jsonl
# (checked directly before use, not assumed): "Support move: X. Use
# conditions: Y. Observable goals: Z. Do not use: W."
_RETRIEVAL_TEXT_RE = re.compile(
    r"Support move:\s*(?P<move>.*?)\s*Use conditions:\s*(?P<use>.*?)\s*"
    r"Observable goals:\s*(?P<goals>.*?)\s*Do not use:\s*(?P<not_use>.*)$",
    re.DOTALL,
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


def recent_dialogue_for(state: dict) -> list[dict]:
    history = [
        {"role": row.get("role", ""), "content": row.get("content", "")}
        for row in state["current_session_history"]
    ]
    return [*history, {"role": "user", "content": state["current_user_text"]}]


def rs_candidate(card: StrategyCard) -> TypedResourceCandidate:
    match = _RETRIEVAL_TEXT_RE.match(card.retrieval_text)
    if match is None:
        raise RuntimeError(f"strategy card retrieval_text does not match the expected shape: {card.strategy_id}")
    return TypedResourceCandidate(
        component="RS", subtype="RS_ATOMIC_MOVE", resource_id=card.strategy_id,
        candidate_version="v1", source_kind="strategy",
        support_move=match.group("move").strip(),
        when_to_use=match.group("use").strip(),
        when_not_to_use=match.group("not_use").strip(),
    )


def find_rs_states(states: list[dict], users: dict, n: int, rag: QualifiedStrategyRAG) -> list[dict]:
    """One state per user where the real, ACTIVE_FOR_NEW_PM_V1_5_G3_WORK
    QualifiedStrategyRAG retrieves a top-1 card above its frozen score
    floor (status == "retrieved_top1")."""

    picked: list[dict] = []
    seen_users: set[str] = set()
    for state in states:
        if len(picked) >= n:
            break
        uid = state["user_id"]
        if uid in seen_users:
            continue
        decision = rag.retrieve(recent_dialogue_for(state))
        if decision.status != "retrieved_top1":
            continue
        seen_users.add(uid)
        picked.append({"state": state, "user": users[uid], "decision": decision})
    return picked


def build_pair(entry: dict) -> tuple:
    state, user = entry["state"], entry["user"]
    uid = state["user_id"]
    aliases = known_aliases_for(user)
    card = entry["decision"].selected_cards[0]
    candidate = rs_candidate(card)

    def make(with_rs: bool):
        candidates: dict[str, TypedResourceCandidate] = {}
        action = "M0+R0"
        if with_rs:
            candidates["RS"] = candidate
            action = "M0+RS"
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
    return with_program, with_messages, without_program, without_messages, card, candidate


def visible_dialogue_for(state: dict) -> dict:
    return {
        "current_session_summary": state["current_session_summary"],
        "current_session_history": state["current_session_history"],
        "current_user_text": state["current_user_text"],
    }


def _judge_call(client, schema, messages, *, seed_parts: tuple, max_tokens: int, validate=None):
    """Same as 52/53's _judge_call -- see those modules' docstrings for the
    two real bugs this shape was built to avoid."""

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
        except Exception as exc:  # noqa: BLE001
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


def run_pair_judgment(judge_client, state: dict, with_reply: str, without_reply: str,
                       card: StrategyCard, aliases: tuple[str, ...]) -> dict:
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
        ("with_rs", with_reply, {"RS": card.guidance_text}),
        ("without_rs", without_reply, {}),
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
    parser.add_argument("--n", type=int, default=15, help="Number of RS-firing states to test.")
    parser.add_argument("--judge-endpoint", default="training_judge",
                         help="Endpoint name in configs/experiment.yaml (default: Gemini 2.5 Flash "
                              "Lite -- NOT the formally qualified decision-grade judge).")
    args = parser.parse_args()

    cards = [StrategyCard(**row) for row in load_jsonl(STRATEGY_BANK)]
    rag = QualifiedStrategyRAG(cards)

    users = {str(u["id"]): u for u in json.loads(EVOEMO.read_text(encoding="utf-8"))}
    states = load_states()
    batch = find_rs_states(states, users, args.n, rag)
    print(f"selected {len(batch)} states where RS retrieves a real top-1 card above the score floor, "
          f"across {len({e['state']['user_id'] for e in batch})} distinct users")

    if not args.live:
        for entry in batch:
            card = entry["decision"].selected_cards[0]
            print(f"  {entry['state']['state_id']} user={entry['state']['user_id']} "
                  f"card={card.strategy_label} score={entry['decision'].selected_score:.3f}")
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
            with_program, with_messages, without_program, without_messages, card, candidate = build_pair(entry)
            print(f"\n=== [{i}/{len(batch)}] state={state['state_id']} user={state['user_id']} "
                  f"card={card.strategy_label} ===")
            print(f"  support_move: {candidate.support_move}")

            with_resp, with_status, with_errs = call_with_guard_and_rewrite(
                gen_client, _GenSchema, with_messages, with_program
            )
            without_resp, without_status, without_errs = call_with_guard_and_rewrite(
                gen_client, _GenSchema, without_messages, without_program
            )
            print(f"  [WITH RS, status={with_status}]: {with_resp.reply if with_resp else None}")
            print(f"  [WITHOUT RS, status={without_status}]: {without_resp.reply if without_resp else None}")

            row = {
                "state_id": state["state_id"], "user_id": state["user_id"],
                "strategy_label": card.strategy_label, "strategy_id": card.strategy_id,
                "support_move": candidate.support_move,
                "with_rs_reply": with_resp.reply if with_resp else None,
                "with_rs_status": with_status, "with_rs_first_pass_guard_errors": list(with_errs),
                "without_rs_reply": without_resp.reply if without_resp else None,
                "without_rs_status": without_status, "without_rs_first_pass_guard_errors": list(without_errs),
            }
            if with_resp is not None and without_resp is not None:
                judgment = run_pair_judgment(
                    judge_client, state, with_resp.reply, without_resp.reply, card,
                    known_aliases_for(entry["user"]),
                )
                row["judgment"] = judgment
                print(f"  [JUDGE quality={judgment['quality'].get('status')} "
                      f"resolved={judgment['quality'].get('resolved_preference')}]")
                print(f"  [JUDGE risk with_rs={judgment['risk']['with_rs'].get('status')} "
                      f"without_rs={judgment['risk']['without_rs'].get('status')}]")
            else:
                row["judgment"] = {"status": "skipped_generation_incomplete"}
                print("  [JUDGE skipped: one or both generations produced no structured output]")
            results.append(row)
    finally:
        gen_client.close()
        judge_client.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "step1_rs_minimal_pilot_results.jsonl"
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
        print(f"  resolved={label} (A=with_rs): {resolved.count(label)}")
    print("NOTE: training_judge is not the formally qualified decision-grade judge -- "
          "this is a pipeline-plumbing + directional-signal pilot, not a publication-grade conclusion.")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
