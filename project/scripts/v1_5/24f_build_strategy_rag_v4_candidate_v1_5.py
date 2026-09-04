#!/usr/bin/env python3
"""Build the 100-card Strategy RAG V4 development candidate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl
from metacom_pm.v1_5_strategy_rag_v4 import (
    V4_PROTOCOL,
    bank_semantic_digest,
    build_v4_candidate_cards,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--core-cards",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v1_5_strategy_bank_v3_core_candidate_v1"
            / "strategy_cards_v3_core_candidate.jsonl"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_strategy_rag_v4_candidate_v1",
    )
    args = parser.parse_args()

    core = [dict(row) for row in iter_jsonl(args.core_cards)]
    cards = build_v4_candidate_cards(core)
    family_counts = Counter(str(row["strategy_family"]) for row in cards)
    profile_counts = Counter(str(row["execution_profile"]) for row in cards)
    support_by_family: dict[str, dict[str, int]] = defaultdict(
        lambda: {"provisional_pass": 0, "provisional_fail": 0}
    )
    for row in cards:
        key = (
            "provisional_pass"
            if row["source_support"]["provisional_source_support_pass"]
            else "provisional_fail"
        )
        support_by_family[str(row["strategy_family"])][key] += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cards_path = args.out_dir / "strategy_cards_v4_candidate.jsonl"
    write_jsonl(cards_path, cards)
    report = {
        "protocol": V4_PROTOCOL,
        "status": "100_CARD_DEVELOPMENT_CANDIDATE_READY_FOR_RETRIEVAL_AUDIT",
        "card_count": len(cards),
        "core_submove_count": len(
            {str(row["core_submove_id"]) for row in cards}
        ),
        "family_counts": dict(sorted(family_counts.items())),
        "execution_profile_counts": dict(sorted(profile_counts.items())),
        "source_support_by_family": {
            family: dict(values)
            for family, values in sorted(support_by_family.items())
        },
        "hard_quota_claimed": False,
        "formal_rs_authorized": False,
        "raw_source_responses_exposed_to_generator": False,
        "candidate_design": (
            "fifty outcome-blind core submoves times minimal/dialogic "
            "execution profiles"
        ),
        "core_cards_path": str(args.core_cards),
        "core_cards_sha256": sha256_file(args.core_cards),
        "cards_path": str(cards_path),
        "cards_sha256": sha256_file(cards_path),
        "bank_semantic_digest": bank_semantic_digest(cards),
        "next_gate": (
            "family-first retrieval audit followed by no-PM matched R0/RS "
            "direct-effect pilot"
        ),
    }
    write_json(args.out_dir / "build_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
