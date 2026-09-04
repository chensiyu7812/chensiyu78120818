from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import write_json
from metacom_pm.pm_v2_generation_review_v8 import (
    audit_generation_pilot_semantics,
    load_v8_cases,
    semantic_audit_markdown,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = (
    ROOT / "outputs" / "pm_v2_generation_pilot_semantic_review_v8"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run the PM V2 V8 semantic leakage and grounding audit."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_DIR / "generation_pilot_semantic_review_cases.json",
    )
    parser.add_argument(
        "--strategy-bank",
        type=Path,
        default=ROOT / "data" / "strategy" / "strategy_cards.jsonl",
    )
    parser.add_argument(
        "--v7-packet",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_generation_pilot_semantic_review_v7"
        / "generation_pilot_semantic_review_packet.csv",
    )
    parser.add_argument(
        "--json-out", type=Path, default=DEFAULT_DIR / "semantic_review_audit.json"
    )
    parser.add_argument(
        "--md-out", type=Path, default=DEFAULT_DIR / "semantic_review_audit.md"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Print status without replacing the frozen audit artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_v8_cases(args.cases)
    report = audit_generation_pilot_semantics(
        cases,
        strategy_bank_path=args.strategy_bank,
        v7_packet_path=args.v7_packet,
    )
    if not args.check_only:
        write_json(args.json_out, report)
        args.md_out.write_text(semantic_audit_markdown(report), encoding="utf-8")
    print(report)
    if report["technical_checks_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
