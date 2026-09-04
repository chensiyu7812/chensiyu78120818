#!/usr/bin/env python3
"""New RS natural-language superdomain (roadmap item: "构造RS natural-
language superdomain"). Directly answers what V15-DATA-81/the RS correction
found: the V3 blueprint's RS construction used instructional phrasing
("Please ask exactly one focused question... do not add another support
move") that never reads as natural user speech. This constructs genuinely
natural turns for each of the 6 real AM01-AM14 move families, plus two
boundary negatives, and verifies them through the REAL mechanical candidate
pool (rs_mechanical_candidate_pool, this round's candidate-layer
responsibility split) rather than asserting the intended label.

Per this round's design: RS cards enter the candidate pool through normal
(always-nonempty-unless-mechanically-blocked) discovery; eligible_moves()-
style semantic matching is checked here only as an attached FEATURE /
transparent-rule preference (rs_transparent_rule_top1), never as the final
opportunity decision -- matching what Step1 is supposed to own.

Zero API calls. All text newly authored, not copied from ESConv/EvoEmo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.contracts import StrategyCard  # noqa: E402
from metacom_pm.io import write_json  # noqa: E402
from metacom_pm.v1_5_v5_3_candidate_layer_responsibility import (  # noqa: E402
    rs_mechanical_candidate_pool,
    rs_transparent_rule_top1,
)

CARDS = [
    StrategyCard(**row)
    for row in (
        json.loads(line)
        for line in (ROOT / "data/strategy/strategy_cards_v1_5_minimal.jsonl").open()
        if line.strip()
    )
]
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_rs_superdomain_v1"

# (family label, natural user turn, expected transparent-rule top-1 move,
# already_executed_move_ids)
FAMILY_CASES = [
    (
        "open_expression",
        "I don't really know where to start, there's just a lot going on and something is bothering me.",
        "AM01_invite_open_expression",
        (),
    ),
    (
        "focused_clarification",
        "It's hard to explain exactly what's wrong, I haven't really said the specific thing yet.",
        "AM02_ask_one_focused_clarification",
        (),
    ),
    (
        "paraphrase_check",
        "Part of me wants to just let it go, but part of me is still really bothered by it.",
        "AM04_tentative_paraphrase_check",
        (),
    ),
    (
        "grounded_validation",
        "I feel so overwhelmed by all of this, it's been a lot to carry.",
        "AM05_grounded_validation",
        (),
    ),
    (
        "offer_one_optional_micro_step",
        "Any advice would help, I could really use one small idea to try.",
        "AM10_offer_one_optional_micro_step",
        (),
    ),
]

BOUNDARY_CASES = [
    (
        "explicit_stop_boundary",
        "Please stop this conversation now.",
        None,
        (),
    ),
    (
        "already_executed_repeat",
        "It's hard to explain exactly what's wrong, I haven't really said the specific thing yet.",
        None,  # AM02 excluded via already_executed -- next best should not be AM02
        ("AM02_ask_one_focused_clarification",),
    ),
]


def run_case(label: str, text: str, expected_move: str | None, already_executed: tuple[str, ...]) -> dict:
    dialogue = [{"speaker": "seeker", "content": text}]
    pool = rs_mechanical_candidate_pool(
        recent_dialogue=dialogue, cards=CARDS, already_executed_move_ids=already_executed,
    )
    top = rs_transparent_rule_top1(pool)
    top_move = top.move_id if top else None
    return {
        "family_or_case": label,
        "text": text,
        "hard_off": pool.hard_off,
        "hard_off_reason": pool.hard_off_reason,
        "n_candidates_in_pool": len(pool.candidates),
        "candidate_move_ids": sorted(c.move_id for c in pool.candidates),
        "transparent_rule_top1": top_move,
        "expected_move": expected_move,
        "matches_expected": (
            top_move == expected_move if expected_move is not None
            else (pool.hard_off or top_move != "AM02_ask_one_focused_clarification")
        ),
    }


def main() -> None:
    results = [run_case(*case) for case in FAMILY_CASES]
    boundary_results = [run_case(*case) for case in BOUNDARY_CASES]

    n_family = len(results)
    n_family_ok = sum(r["matches_expected"] for r in results)
    n_boundary = len(boundary_results)
    n_boundary_ok = sum(r["matches_expected"] for r in boundary_results)

    report = {
        "protocol": "pm-v1.5-v5.3-rs-superdomain-construction-v1",
        "note": (
            "Cards always enter the pool through rs_mechanical_candidate_pool "
            "(mechanical-only gate); rs_transparent_rule_top1 is checked as an "
            "explicit transparent-rule baseline, not a stand-in for a learned "
            "Step1 decision."
        ),
        "family_cases": {"n": n_family, "n_matches_expected": n_family_ok, "results": results},
        "boundary_cases": {"n": n_boundary, "n_matches_expected": n_boundary_ok, "results": boundary_results},
        "api_calls": 0,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "report.json", report)
    print(f"family cases: {n_family_ok}/{n_family} matched expected transparent-rule move")
    for r in results:
        print(f"  [{r['family_or_case']}] top1={r['transparent_rule_top1']} expected={r['expected_move']} ok={r['matches_expected']}")
    print(f"\nboundary cases: {n_boundary_ok}/{n_boundary}")
    for r in boundary_results:
        print(f"  [{r['family_or_case']}] hard_off={r['hard_off']} reason={r['hard_off_reason']} "
              f"top1={r['transparent_rule_top1']} n_candidates={r['n_candidates_in_pool']}")
    print(f"\nfull report written to {OUT_DIR}")


if __name__ == "__main__":
    main()
