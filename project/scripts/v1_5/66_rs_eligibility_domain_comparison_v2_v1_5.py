#!/usr/bin/env python3
"""RS domain comparison, corrected (supersedes script 65's finding, does not
delete it -- see the correction doc for why).

Two real, distinct problems with script 65, both raised by an independent
review, both verified true before this rewrite:

1. Identity confusion: V3's effect-blueprint construction gates RS through
   `v1_5_strategy_rag_repair.effect_study_observable_flags()` (built on the
   v4 "50 core / 100 execution card" Strategy Bank, ESConv's own 5 safe
   strategy families) -- NOT `v1_5_strategy_rag_runtime.observable_flags()`/
   `eligible_moves()` (the newer, separate "6 atomic move" AM01-AM14 bank,
   `strategy_cards_v1_5_minimal.jsonl`). Script 65 audited the training
   construction with the WRONG one of these two real, both-still-live
   implementations. This script runs BOTH, clearly labeled, and does not
   pick a winner -- which bank is canonical is not this script's call
   (data/pm_v1_5_contracts/strategy_bank_v2_candidate_v2.json vs
   strategy_cards_v1_5_minimal.jsonl sizing is reported for the leader).

2. Sampling identity: script 65 used the first 300 of 1300 raw ESConv.json
   dialogues ("not the full ESConv") and the 138-state
   qualification+lockbox EvoEmo panel. This uses the real, already-frozen
   169-dialogue ESConv test split (data/esconv_test_v1_5/pm_v2_states.jsonl,
   2112 support-eligible turns) and the real 204-state EvoEmo panel
   (outputs/pm_v1_5b_final_external_panel_v2/evoemo_panel_private.jsonl =
   60 qualification + 78 lockbox + 66 diagnostic), not a 138-state subset.

These rates are domain description only -- NOT precision/recall or PM
correctness. Zero API calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.io import write_json  # noqa: E402
from metacom_pm.v1_5_strategy_rag_repair import (  # noqa: E402
    effect_study_observable_flags,
)
from metacom_pm.v1_5_strategy_rag_runtime import (  # noqa: E402
    eligible_moves,
    observable_flags as sixcard_observable_flags,
)

V3_CANDIDATES = ROOT / "outputs/pm_v1_5_v3_p2_exact_rank1_v4/candidate_rows_private.jsonl"
ESCONV_STATES = ROOT / "data/esconv_test_v1_5/pm_v2_states.jsonl"
EVOEMO_PANEL_V2 = ROOT / "outputs/pm_v1_5b_final_external_panel_v2/evoemo_panel_private.jsonl"
OUT_PATH = ROOT / "outputs/pm_v1_5_v5_3_rs_eligibility_domain_comparison_v2/report.json"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _visible_dialogue(history: list[dict], current_user_text: str) -> list[dict]:
    """Normalize to the {speaker, content} shape
    latest_visible_turn_is_seeker() actually reads (it only checks
    "speaker", not "role" -- unlike the 6-card module's _speaker() helper,
    which accepts both). Missing this normalization would make every row
    register as "latest turn not seeker" and hard-off everything."""

    out = []
    for turn in history:
        role = str(turn.get("role") or turn.get("speaker") or "")
        speaker = "seeker" if role in {"user", "seeker"} else "supporter"
        out.append({"speaker": speaker, "content": turn.get("content", "")})
    out.append({"speaker": "seeker", "content": current_user_text})
    return out


def _rows_for_domain(name: str) -> list[tuple[str, list[dict]]]:
    """Returns (current_user_text, visible_dialogue) pairs."""

    if name == "train_v3_rs_blueprint":
        rows = [
            r for r in _jsonl(V3_CANDIDATES)
            if r["target_component_private_not_model_input"] == "RS"
        ]
        return [
            (r["current_user_text"], [{"speaker": "seeker", "content": r["current_user_text"]}])
            for r in rows
        ]
    if name == "esconv_169_formal_test":
        rows = _jsonl(ESCONV_STATES)
        return [
            (r["current_user_text"], _visible_dialogue(r.get("current_session_history", []), r["current_user_text"]))
            for r in rows
        ]
    if name == "evoemo_204_formal_panel":
        rows = _jsonl(EVOEMO_PANEL_V2)
        out = []
        for r in rows:
            rt = r.get("runtime_state", {})
            hist = rt.get("current_session_history", [])
            text = r.get("current_user_text") or rt.get("current_user_text", "")
            out.append((text, _visible_dialogue(hist, text)))
        return out
    raise ValueError(name)


def _summarize(name: str) -> dict:
    pairs = _rows_for_domain(name)
    n = len(pairs)
    hard_off = 0
    hard_off_reasons: dict[str, int] = {}
    any_move_eligible = 0
    for text, dialogue in pairs:
        recent = " ".join(t["content"] for t in dialogue[-3:] if t["speaker"] == "seeker")
        es_flags = effect_study_observable_flags(
            current_user_text=text, recent_user_text=recent, visible_dialogue=dialogue,
        )
        if es_flags["ordinary_rag_hard_off"]:
            hard_off += 1
            for reason in es_flags["ordinary_rag_hard_off_reasons"]:
                hard_off_reasons[reason] = hard_off_reasons.get(reason, 0) + 1
        sc_flags = sixcard_observable_flags(dialogue)
        if eligible_moves(sc_flags):
            any_move_eligible += 1
    return {
        "n": n,
        "effect_study_ordinary_rag_hard_off_rate": hard_off / n if n else None,
        "effect_study_hard_off_reason_counts": hard_off_reasons,
        "effect_study_not_hard_off_rate": (n - hard_off) / n if n else None,
        "sixcard_am_any_move_eligible_rate": any_move_eligible / n if n else None,
    }


def main() -> None:
    report = {
        "protocol": "pm-v1.5-v5.3-rs-eligibility-domain-comparison-v2",
        "supersedes": "pm-v1.5-v5.3-rs-eligibility-domain-comparison-v1 (script 65) -- identity and sampling corrected",
        "note": (
            "effect_study_ordinary_rag_hard_off_rate uses the SAME eligibility "
            "implementation (v1_5_strategy_rag_repair.effect_study_observable_flags, "
            "built on v1_5_strategy_rag_v4) that gated the V3 blueprint's own "
            "RS construction -- 'not hard-off' does not mean a specific card "
            "ranked nonempty, only that the structural exclusion gate did not fire. "
            "sixcard_am_any_move_eligible_rate is the separate, newer 6-card "
            "AM01-AM14 runtime (v1_5_strategy_rag_runtime), used by prompts.py. "
            "These are two real, currently-coexisting implementations; this "
            "script does not decide which is canonical."
        ),
        "domains": {},
    }
    for name in ("train_v3_rs_blueprint", "esconv_169_formal_test", "evoemo_204_formal_panel"):
        d = _summarize(name)
        report["domains"][name] = d
        print(f"{name}: n={d['n']} effect_study_not_hard_off={d['effect_study_not_hard_off_rate']} "
              f"sixcard_any_eligible={d['sixcard_am_any_move_eligible_rate']}")
        print(f"  hard_off_reasons: {d['effect_study_hard_off_reason_counts']}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUT_PATH, report)
    print(f"\nfull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
