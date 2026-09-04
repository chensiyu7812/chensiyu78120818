#!/usr/bin/env python3
"""Freeze an outcome-blind H2 state slice before the final Bank is known."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, sha256_text, stable_hex
from metacom_pm.text import normalize_space
from metacom_pm.v1_5_strategy_rag_repair import (
    REPAIR_PROTOCOL,
    repaired_observable_opportunity_flags,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-h2-retrieval-state-freeze-v1"
TARGETS = {
    "effort_or_progress": 8,
    "advice_welcome": 8,
    "uncertainty": 8,
    "emotion": 8,
    "ordinary_hard_off": 8,
    "phatic_stop_closing": 8,
}


def _excluded_dialogues(
    *,
    h1_packet: Path,
    h1_private_retrieval: Path,
    relink_finalists: Path,
) -> tuple[set[str], dict[str, int]]:
    packet = json.loads(h1_packet.read_text(encoding="utf-8"))
    old_sources = {
        str(example["source_dialogue_id"])
        for item in packet["card_items"]
        for example in item["source_examples"]
    }
    h1_queries = {
        str(row["source_dialogue_id"])
        for row in iter_jsonl(h1_private_retrieval)
    }
    relink = {
        str(row["source_dialogue_id"])
        for row in iter_jsonl(relink_finalists)
    }
    union = old_sources | h1_queries | relink
    return union, {
        "old_h1_source_dialogues": len(old_sources),
        "h1_query_dialogues": len(h1_queries),
        "source_relink_finalist_dialogues": len(relink),
        "union_dialogues": len(union),
    }


def _strata(flags: dict[str, Any]) -> list[str]:
    phatic = bool(
        flags["pure_phatic"]
        or flags["explicit_stop"]
        or flags["expanded_routine_closing"]
    )
    if phatic:
        return ["phatic_stop_closing"]
    if flags["ordinary_rag_hard_off"]:
        return ["ordinary_hard_off"]
    labels: list[str] = []
    if flags["effort_or_progress_visible"]:
        labels.append("effort_or_progress")
    if flags["advice_welcome"]:
        labels.append("advice_welcome")
    if flags["uncertainty_or_multi_concern"]:
        labels.append("uncertainty")
    if flags["emotion_visible"]:
        labels.append("emotion")
    return labels


def _select(
    pools: dict[str, list[dict[str, Any]]],
    *,
    targets: dict[str, int],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_dialogues: set[str] = set()
    # Positives first, then controls. Within each stratum, the protocol hash
    # fixes order before the final Bank or either reranker's results exist.
    for stratum in targets:
        problem_counts: Counter[str] = Counter()
        taken = 0
        for row in sorted(
            pools[stratum],
            key=lambda value: stable_hex(
                PROTOCOL,
                stratum,
                value["source_dialogue_id"],
                value["visible_dialogue_sha256"],
                n=32,
            ),
        ):
            dialogue = str(row["source_dialogue_id"])
            problem = str(row.get("problem_type") or "unknown")
            if dialogue in used_dialogues or problem_counts[problem] >= 3:
                continue
            selected.append({**row, "selection_stratum": stratum})
            used_dialogues.add(dialogue)
            problem_counts[problem] += 1
            taken += 1
            if taken == targets[stratum]:
                break
        if taken != targets[stratum]:
            raise RuntimeError(f"H2 stratum {stratum}: {taken}/{targets[stratum]}")
    return selected


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_esconv_train_strategy_inductive_audit_v1/"
        "clean_train_strategy_universe.jsonl",
    )
    parser.add_argument(
        "--h1-packet",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h1_complete_rag_review_v1_candidate/"
        "h1_review_packet.json",
    )
    parser.add_argument(
        "--h1-private-retrieval",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h1_complete_rag_review_v1_candidate/"
        "private_retrieval_audit.jsonl",
    )
    parser.add_argument(
        "--relink-finalists",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h1_source_relink_review_v1_candidate/"
        "private_candidate_scores.jsonl",
    )
    parser.add_argument(
        "--formal-test",
        type=Path,
        default=ROOT / "data/splits/formal_test.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_h2_retrieval_state_freeze_v1",
    )
    args = parser.parse_args()

    excluded, exclusion_counts = _excluded_dialogues(
        h1_packet=args.h1_packet,
        h1_private_retrieval=args.h1_private_retrieval,
        relink_finalists=args.relink_finalists,
    )
    if args.formal_test.is_file():
        formal = {
            str(row.get("user_id") or row.get("source_dialogue_id"))
            for row in iter_jsonl(args.formal_test)
        }
        formal.discard("None")
        excluded |= formal
        exclusion_counts["formal_test_dialogues"] = len(formal)
        exclusion_counts["union_with_formal_test"] = len(excluded)

    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_states: set[tuple[str, str]] = set()
    for source in iter_jsonl(args.universe):
        dialogue_id = str(source["source_dialogue_id"])
        dialogue = [
            {
                "speaker": str(turn["speaker"]),
                "content": normalize_space(turn["content"]),
            }
            for turn in source["recent_dialogue"]
            if str(turn.get("speaker")) in {"seeker", "supporter"}
            and normalize_space(turn.get("content", ""))
        ]
        if (
            dialogue_id in excluded
            or not dialogue
            or dialogue[-1]["speaker"] != "seeker"
        ):
            continue
        visible_text = "\n".join(
            f"{turn['speaker']}: {turn['content']}" for turn in dialogue
        )
        digest = sha256_text(visible_text)
        state_key = (dialogue_id, digest)
        if state_key in seen_states:
            continue
        seen_states.add(state_key)
        seekers = [
            turn["content"] for turn in dialogue if turn["speaker"] == "seeker"
        ]
        flags = repaired_observable_opportunity_flags(
            current_user_text=seekers[-1],
            recent_user_text=" ".join(seekers[-3:]),
            visible_dialogue=dialogue,
        )
        row = {
            "h2_state_id": "h2_state_"
            + stable_hex(PROTOCOL, dialogue_id, digest, n=20),
            "source_dialogue_id": dialogue_id,
            "source_turn_index": int(source["source_turn_index"]),
            "problem_type": str(source.get("problem_type") or "unknown"),
            "visible_dialogue": dialogue,
            "visible_dialogue_sha256": digest,
            "observable_flags_at_freeze": flags,
        }
        for stratum in _strata(flags):
            pools[stratum].append(row)

    selected = _select(pools, targets=TARGETS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    states_path = args.out_dir / "h2_states_private.jsonl"
    _write_jsonl(states_path, selected)
    manifest = {
        "protocol": PROTOCOL,
        "repair_protocol": REPAIR_PROTOCOL,
        "status": "H2_STATES_FROZEN_BEFORE_FINAL_BANK_AND_RERANKER",
        "purpose": (
            "Fresh internal qualification slice for the final Bank and the "
            "transparent-vs-BGE reranker comparison."
        ),
        "states": len(selected),
        "targets": TARGETS,
        "observed_strata": dict(
            sorted(Counter(row["selection_stratum"] for row in selected).items())
        ),
        "unique_dialogues": len(
            {str(row["source_dialogue_id"]) for row in selected}
        ),
        "selection_constraints": {
            "outcome_blind": True,
            "selected_before_final_bank": True,
            "selected_before_h2_retrieval_results": True,
            "latest_visible_turn_must_be_seeker": True,
            "one_state_per_dialogue": True,
            "maximum_three_same_problem_per_stratum": True,
            "ordering": "stable protocol hash; no seed search",
        },
        "excluded": exclusion_counts,
        "next_stage": (
            "After the 36-core source relink review, freeze the qualified "
            "Bank, render a blinded union of transparent/BGE candidates for "
            "these exact states, and choose the reranker once."
        ),
        "inputs": {
            "universe": str(args.universe.relative_to(ROOT)),
            "h1_packet": str(args.h1_packet.relative_to(ROOT)),
            "h1_private_retrieval": str(
                args.h1_private_retrieval.relative_to(ROOT)
            ),
            "relink_finalists": str(args.relink_finalists.relative_to(ROOT)),
            "formal_test": (
                str(args.formal_test.relative_to(ROOT))
                if args.formal_test.is_file()
                else None
            ),
        },
        "outputs": {states_path.name: sha256_file(states_path)},
    }
    _write_json(args.out_dir / "manifest.json", manifest)
    print(
        {
            "protocol": PROTOCOL,
            "status": manifest["status"],
            "states": len(selected),
            "unique_dialogues": manifest["unique_dialogues"],
            "strata": manifest["observed_strata"],
            "out_dir": str(args.out_dir),
        }
    )


if __name__ == "__main__":
    main()
