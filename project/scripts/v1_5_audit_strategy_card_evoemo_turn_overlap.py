#!/usr/bin/env python3
"""Free, no-API turn/card-level dedup audit: Strategy Bank vs. EvoEmo turns.

`strategy_bank.py`'s existing overlap screen (`find_esconv_evoemo_overlaps`)
compares whole ESConv dialogues against whole EvoEmo *sessions* and excludes
ESConv dialogues that match closely. That is the right check for "did we
build cards from a source conversation that is actually an EvoEmo session,"
but it cannot catch a short, generic response that happens to duplicate one
specific EvoEmo *turn* without the surrounding dialogue also matching (a
short reply like "That sounds really hard" can be near-identical across
totally unrelated conversations).

This script checks the same thing at finer grain: every strategy card's
`retrieval_text` and `example_response` against every individual EvoEmo turn,
looking for exact (post-normalization) or high-shingle-Jaccard matches. No
API calls, no cost. Ad-hoc, not part of the PM-v2.2 gated pipeline.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from metacom_pm.contracts import StrategyCard
from metacom_pm.io import iter_jsonl, sha256_file, write_json
from metacom_pm.text import jaccard, normalize_for_hash, shingle_set

ROOT = Path(__file__).resolve().parents[1]


def load_strategy_cards(path: Path) -> list[StrategyCard]:
    return [StrategyCard.model_validate(row) for row in iter_jsonl(path)]


def load_evoemo_turns(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    turns = []
    for user in data:
        user_id = user.get("id") or user.get("user_id")
        for session in user.get("dialog_history") or []:
            for turn in session.get("dialogue") or []:
                content = str(turn.get("content") or "").strip()
                if content:
                    turns.append(
                        {
                            "user_id": user_id,
                            "session_id": session.get("id"),
                            "turn_idx": turn.get("idx"),
                            "role": turn.get("role"),
                            "content": content,
                        }
                    )
    return turns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy-bank", type=Path, default=ROOT / "data" / "strategy" / "strategy_cards.jsonl")
    parser.add_argument("--evoemo", type=Path, default=ROOT / "data" / "external" / "evo_emo.json")
    parser.add_argument("--jaccard-threshold", type=float, default=0.85)
    parser.add_argument("--min-turn-tokens", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "v1_5_strategy_card_evoemo_turn_overlap_audit.json",
    )
    return parser.parse_args()


def _deterministic_source_matches(
    card: StrategyCard, matched_turns: list[dict]
) -> list[str]:
    matches = []
    for turn in matched_turns:
        match = re.fullmatch(r"esc(\d+)", str(turn.get("session_id") or ""))
        if match and card.source_dialogue_id == f"esconv_{int(match.group(1)):04d}":
            matches.append(str(turn["session_id"]))
    return sorted(set(matches))


def main() -> None:
    args = parse_args()
    cards = load_strategy_cards(args.strategy_bank)
    turns = load_evoemo_turns(args.evoemo)

    # Only check turns with enough content to make near-duplication a
    # meaningful signal; very short turns ("okay", "thank you") are excluded
    # since generic short replies are expected to recur across any corpus and
    # are not evidence of copying a specific EvoEmo instance.
    turns = [t for t in turns if len(t["content"].split()) >= args.min_turn_tokens]
    exact_by_normalized: dict[str, list[dict]] = {}
    shingles_by_turn: list[tuple[dict, set]] = []
    turn_indices_by_shingle: dict[str, set[int]] = defaultdict(set)
    for turn in turns:
        normalized = normalize_for_hash(turn["content"])
        exact_by_normalized.setdefault(normalized, []).append(turn)
        turn_shingles = shingle_set(turn["content"])
        turn_index = len(shingles_by_turn)
        shingles_by_turn.append((turn, turn_shingles))
        for shingle in turn_shingles:
            turn_indices_by_shingle[shingle].add(turn_index)

    findings = []
    for card in cards:
        for field_name in ("retrieval_text", "example_response"):
            text = getattr(card, field_name)
            if len(text.split()) < args.min_turn_tokens:
                continue
            normalized = normalize_for_hash(text)
            exact_hits = exact_by_normalized.get(normalized)
            if exact_hits:
                matched_turns = exact_hits[:3]
                findings.append(
                    {
                        "strategy_id": card.strategy_id,
                        "source_dialogue_id": card.source_dialogue_id,
                        "field": field_name,
                        "match_type": "exact",
                        "similarity": 1.0,
                        "matched_turns": matched_turns,
                        "deterministic_source_matches": (
                            _deterministic_source_matches(card, matched_turns)
                        ),
                    }
                )
                continue
            card_shingles = shingle_set(text)
            if not card_shingles:
                continue
            best_score = 0.0
            best_turn = None
            candidate_indices: set[int] = set()
            for shingle in card_shingles:
                candidate_indices.update(turn_indices_by_shingle.get(shingle, ()))
            for turn_index in candidate_indices:
                turn, turn_shingles = shingles_by_turn[turn_index]
                # Jaccard(A, B) cannot reach the threshold when the set-size
                # ratio alone is already below it. This exact upper-bound
                # filter and the inverted shingle index preserve findings
                # while avoiding an 11,590 x 8,596 all-pairs scan.
                size_ratio = min(len(card_shingles), len(turn_shingles)) / max(
                    len(card_shingles), len(turn_shingles)
                )
                if size_ratio < args.jaccard_threshold:
                    continue
                score = jaccard(card_shingles, turn_shingles)
                if score > best_score:
                    best_score = score
                    best_turn = turn
            if best_score >= args.jaccard_threshold:
                matched_turns = [best_turn]
                findings.append(
                    {
                        "strategy_id": card.strategy_id,
                        "source_dialogue_id": card.source_dialogue_id,
                        "field": field_name,
                        "match_type": "shingle_jaccard",
                        "similarity": best_score,
                        "matched_turns": matched_turns,
                        "deterministic_source_matches": (
                            _deterministic_source_matches(card, matched_turns)
                        ),
                    }
                )

    deterministic_source_findings = sum(
        bool(row["deterministic_source_matches"]) for row in findings
    )
    report = {
        "protocol": "pm-v1.5-strategy-card-evoemo-turn-overlap-audit-v2",
        "status": (
            "DETERMINISTIC_SOURCE_LEAKAGE"
            if deterministic_source_findings
            else "PASS"
            if not findings
            else "GENERIC_FINDINGS_ONLY"
        ),
        "strategy_bank_path": str(args.strategy_bank.resolve()),
        "strategy_bank_sha256": sha256_file(args.strategy_bank),
        "evoemo_path": str(args.evoemo.resolve()),
        "evoemo_sha256": sha256_file(args.evoemo),
        "n_cards_checked": len(cards),
        "n_evoemo_turns_checked": len(turns),
        "jaccard_threshold": args.jaccard_threshold,
        "min_turn_tokens": args.min_turn_tokens,
        "n_findings": len(findings),
        "n_deterministic_source_findings": deterministic_source_findings,
        "findings": findings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print(
        {
            "status": report["status"],
            "n_cards_checked": report["n_cards_checked"],
            "n_evoemo_turns_checked": report["n_evoemo_turns_checked"],
            "n_findings": report["n_findings"],
            "n_deterministic_source_findings": report[
                "n_deterministic_source_findings"
            ],
        }
    )


if __name__ == "__main__":
    main()
