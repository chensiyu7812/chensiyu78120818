#!/usr/bin/env python3
"""RS domain comparison (roadmap item: "MP/MS/RS各自的域对比审计", RS part).

RS has no memory catalog and no compile_atomic_* contract -- its
"eligibility" is entirely observable_flags()/eligible_moves() over recent
dialogue text (v1_5_strategy_rag_runtime.py). So the meaningful domain
comparison for RS is not "does the candidate pass a compiler" (there is no
such gate) but "does the SAME deterministic eligibility runtime fire at a
comparable, sensible rate on real natural dialogue vs the synthetic V3
construction's intended distribution."

Three domains, same functions, zero API calls:
  1. training (V3 effect blueprint, RS rows only -- flags precomputed at
     construction time, read directly from the audit trail);
  2. real ESConv (data/external/ESConv.json, 1300 real single-session
     dialogues -- RS's native domain);
  3. real EvoEmo (data/external/evo_emo.json, current-session windows from
     the same 138-state panel used all session).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.io import write_json  # noqa: E402
from metacom_pm.v1_5_strategy_rag_runtime import (  # noqa: E402
    eligible_moves,
    observable_flags,
)

V3_CANDIDATES = ROOT / "outputs/pm_v1_5_v3_p2_exact_rank1_v4/candidate_rows_private.jsonl"
ESCONV_PATH = ROOT / "data/external/ESConv.json"
EVOEMO_PATH = ROOT / "data/external/evo_emo.json"
PANEL_DIR = ROOT / "outputs/pm_v1_5b_corrected_external_split_v1"
OUT_PATH = ROOT / "outputs/pm_v1_5_v5_3_rs_eligibility_domain_comparison_v1/report.json"

FLAG_KEYS = [
    "active_high_stakes", "explicit_stop", "listen_only", "question_repetition_block",
    "open_expression_opportunity", "focused_clarification_opportunity",
    "paraphrase_check_opportunity", "grounded_validation_opportunity",
    "explicit_advice_welcome", "one_low_risk_step_available", "transition_opportunity",
]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def training_domain_flags() -> list[dict]:
    """The V3 blueprint export's own precomputed observable_flags uses a
    different, richer schema (a separate hard-off gate module, not
    v1_5_strategy_rag_runtime.observable_flags) -- reusing it directly threw
    a KeyError on eligible_moves(), which expects THIS module's 11-key
    schema. Recomputing from current_user_text with the SAME function used
    for the other two domains keeps this an honest apples-to-apples
    comparison instead of silently mixing two different eligibility
    implementations."""

    rows = [
        r for r in _jsonl(V3_CANDIDATES)
        if r["target_component_private_not_model_input"] == "RS"
    ]
    out = []
    for r in rows:
        text = r.get("current_user_text") or ""
        if text:
            out.append(observable_flags([{"role": "seeker", "content": text}]))
    return out


def esconv_domain_flags(limit: int = 300) -> list[dict]:
    dialogues = json.loads(ESCONV_PATH.read_text(encoding="utf-8"))
    out = []
    for convo in dialogues[:limit]:
        dialog = convo.get("dialog") or []
        window: list[dict] = []
        for turn in dialog:
            window.append(turn)
            if turn.get("speaker") == "seeker":
                out.append(observable_flags(window[-8:]))
    return out


def evoemo_domain_flags() -> list[dict]:
    users = {str(u["id"]): u for u in json.loads(EVOEMO_PATH.read_text(encoding="utf-8"))}
    states = []
    for fname in ("evoemo_qualification_panel_private.jsonl", "evoemo_lockbox_panel_private.jsonl"):
        for row in _jsonl(PANEL_DIR / fname):
            rt = row["runtime_state"]
            states.append(rt.get("current_session_history", []) + [
                {"role": "seeker", "content": rt.get("current_user_text", "")}
            ])
    return [observable_flags(window[-8:]) for window in states if window]


def _summarize(flag_rows: list[dict]) -> dict:
    n = len(flag_rows)
    rates = {k: sum(1 for r in flag_rows if r.get(k)) / n for k in FLAG_KEYS} if n else {}
    move_counts: Counter[str] = Counter()
    any_eligible = 0
    for r in flag_rows:
        moves = eligible_moves(r)
        if moves:
            any_eligible += 1
        for m in moves:
            move_counts[m] += 1
    return {
        "n": n,
        "flag_rates": rates,
        "any_move_eligible_rate": (any_eligible / n) if n else None,
        "move_eligibility_rates": {k: v / n for k, v in move_counts.items()} if n else {},
    }


def main() -> None:
    train = training_domain_flags()
    esconv = esconv_domain_flags()
    evoemo = evoemo_domain_flags()
    print(f"n_train={len(train)} n_esconv={len(esconv)} n_evoemo={len(evoemo)}")

    report = {
        "protocol": "pm-v1.5-v5.3-rs-eligibility-domain-comparison-v1",
        "note": (
            "RS has no memory catalog/compiler; eligibility is entirely "
            "observable_flags()/eligible_moves() over recent dialogue text. "
            "This compares that SAME deterministic runtime's firing rate "
            "across training construction vs two real natural-dialogue domains."
        ),
        "train": _summarize(train),
        "esconv": _summarize(esconv),
        "evoemo": _summarize(evoemo),
        "api_calls": 0,
    }
    for domain in ("train", "esconv", "evoemo"):
        d = report[domain]
        print(f"\n{domain}: n={d['n']} any_move_eligible_rate={d['any_move_eligible_rate']}")
        for k, v in sorted(d["flag_rates"].items()):
            print(f"  {k}: {v:.3f}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUT_PATH, report)
    print(f"\nfull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
