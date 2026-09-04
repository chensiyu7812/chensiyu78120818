#!/usr/bin/env python3
"""Bind H1 source labels and freeze the 40-core/80-card V1.5 Bank."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-strategy-bank-v4-final-freeze-v1"
SOURCE_PROTOCOL = "pm-v1.5-h1-source-relink-human-review-v1"


def _load_source_annotations(
    *,
    annotations_path: Path,
    packet_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    annotations = [dict(row) for row in iter_jsonl(annotations_path)]
    packet = [dict(row) for row in iter_jsonl(packet_path)]
    by_review = {str(row["review_item_id"]): row for row in packet}
    if len(annotations) != 36 or len(by_review) != 36:
        raise RuntimeError("source relink must contain exactly 36 review units")
    if len({str(row["review_item_id"]) for row in annotations}) != 36:
        raise RuntimeError("duplicate source-relink review ids")
    if {str(row["review_item_id"]) for row in annotations} != set(by_review):
        raise RuntimeError("source annotations do not exactly match packet")
    for row in annotations:
        review_id = str(row["review_item_id"])
        packet_row = by_review[review_id]
        if row.get("protocol") != SOURCE_PROTOCOL:
            raise RuntimeError(f"wrong source protocol: {review_id}")
        if row.get("reviewed") is not True:
            raise RuntimeError(f"unreviewed source core: {review_id}")
        if str(row["core_submove_id"]) != str(
            packet_row["core_submove_id"]
        ):
            raise RuntimeError(f"source core binding mismatch: {review_id}")
        numbers = list(row.get("accepted_candidate_numbers") or [])
        if (
            len(numbers) != len(set(numbers))
            or not set(numbers) <= set(range(1, 6))
        ):
            raise RuntimeError(f"invalid accepted candidate set: {review_id}")
    return annotations, by_review


def _validate_final_bank(cards: list[dict[str, Any]]) -> None:
    if len(cards) != 80:
        raise RuntimeError(f"expected 80 final cards, found {len(cards)}")
    if len({str(row["card_id"]) for row in cards}) != 80:
        raise RuntimeError("final card ids must be unique")
    core_profiles = [
        (str(row["core_submove_id"]), str(row["execution_profile"]))
        for row in cards
    ]
    if len(set(core_profiles)) != 80:
        raise RuntimeError("final Bank repeats a core/profile pair")
    core_counts = Counter(core for core, _ in core_profiles)
    if set(core_counts.values()) != {2} or len(core_counts) != 40:
        raise RuntimeError("every final core must have minimal/dialogic variants")
    if Counter(profile for _, profile in core_profiles) != {
        "minimal": 40,
        "dialogic": 40,
    }:
        raise RuntimeError("final Bank profile balance is invalid")
    for row in cards:
        if row.get("raw_source_response_exposed_to_generator") is not False:
            raise RuntimeError("final Bank may not expose raw source responses")
        if row.get("content_scope") != "technique_only":
            raise RuntimeError("final Bank must remain topic-agnostic techniques")
        source = row.get("source_qualification") or {}
        if source.get("status") != "HUMAN_SOURCE_EVIDENCE_QUALIFIED":
            raise RuntimeError("final card lacks human source qualification")
        if int(source.get("accepted_source_dialogues") or 0) < 2:
            raise RuntimeError("final card has fewer than two source dialogues")


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
        "--source-annotations",
        type=Path,
        default=Path(
            "/home/tokkio/.codex/attachments/"
            "ded4d596-7932-4212-a6f2-915b9a7145a5/pasted-text.txt"
        ),
    )
    parser.add_argument(
        "--source-packet",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h1_source_relink_review_v1_candidate/"
        "source_relink_packet.jsonl",
    )
    parser.add_argument(
        "--h1-packet",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h1_complete_rag_review_v1_candidate/"
        "h1_review_packet.json",
    )
    parser.add_argument(
        "--h1-annotations",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h1_complete_rag_review_v1_analysis/"
        "h1_annotations_frozen.jsonl",
    )
    parser.add_argument(
        "--candidate-cards",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_rag_v4_candidate_v1/"
        "strategy_cards_v4_candidate.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_strategy_bank_v4_final_v1",
    )
    args = parser.parse_args()

    source_annotations, source_packet = _load_source_annotations(
        annotations_path=args.source_annotations,
        packet_path=args.source_packet,
    )
    h1_packet = json.loads(args.h1_packet.read_text(encoding="utf-8"))
    h1_annotations = {
        str(row["review_item_id"]): dict(row)
        for row in iter_jsonl(args.h1_annotations)
        if row.get("item_type") == "card"
    }
    full_cards = {
        str(row["card_id"]): dict(row)
        for row in iter_jsonl(args.candidate_cards)
    }
    source_by_core = {
        str(row["core_submove_id"]): row for row in source_annotations
    }
    packet_by_core = {
        str(row["core_submove_id"]): row
        for row in source_packet.values()
    }

    final_cores: list[dict[str, Any]] = []
    excluded_cores: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    final_cards: list[dict[str, Any]] = []
    frozen_annotations = sorted(
        source_annotations, key=lambda row: str(row["core_submove_id"])
    )

    for item in sorted(
        h1_packet["card_items"],
        key=lambda row: str(row["core_submove_id"]),
    ):
        core_id = str(item["core_submove_id"])
        h1 = h1_annotations[str(item["review_item_id"])]
        qualification: dict[str, Any] | None = None
        accepted_sources: list[dict[str, Any]] = []
        exclusion_reason = ""
        if (
            h1["distinctness"] == "distinct"
            and h1["source_support_valid"] == "yes"
            and h1["minimal_variant"] == "approve"
            and h1["dialogic_variant"] == "approve"
        ):
            for example in item["source_examples"]:
                accepted_sources.append(
                    {
                        "source_dialogue_id": example["source_dialogue_id"],
                        "source_turn_index": example["source_turn_index"],
                        "strategy_id": example["strategy_id"],
                    }
                )
            qualification = {
                "status": "HUMAN_SOURCE_EVIDENCE_QUALIFIED",
                "qualification_stage": "original_h1_core_review",
                "review_protocol": h1["protocol"],
                "review_item_id": h1["review_item_id"],
                "judgment_granularity": "displayed_source_set_overall",
                "accepted_source_dialogues": len(
                    {row["source_dialogue_id"] for row in accepted_sources}
                ),
                "raw_examples_exposed_to_generator": False,
            }
        elif core_id in source_by_core:
            source_label = source_by_core[core_id]
            accepted_numbers = list(
                source_label["accepted_candidate_numbers"]
            )
            packet_row = packet_by_core[core_id]
            by_number = {
                int(row["candidate_number"]): row
                for row in packet_row["candidates"]
            }
            accepted_sources = [
                {
                    "candidate_id": by_number[number]["candidate_id"],
                    "source_dialogue_id": by_number[number][
                        "source_dialogue_id"
                    ],
                    "source_turn_index": by_number[number][
                        "source_turn_index"
                    ],
                }
                for number in accepted_numbers
            ]
            independent_dialogues = len(
                {row["source_dialogue_id"] for row in accepted_sources}
            )
            if independent_dialogues >= 2:
                qualification = {
                    "status": "HUMAN_SOURCE_EVIDENCE_QUALIFIED",
                    "qualification_stage": "bounded_h1_source_relink",
                    "review_protocol": source_label["protocol"],
                    "review_item_id": source_label["review_item_id"],
                    "judgment_granularity": "per_candidate_literal_move_fit",
                    "accepted_source_dialogues": independent_dialogues,
                    "raw_examples_exposed_to_generator": False,
                }
            else:
                exclusion_reason = (
                    "source_relink_fewer_than_two_independent_examples"
                )
        else:
            exclusion_reason = (
                "h1_duplicate_or_rejected_core:"
                + str(h1.get("duplicate_of") or "not_approved")
            )

        if qualification is None:
            excluded_cores.append(
                {
                    "core_submove_id": core_id,
                    "strategy_family": item["strategy_family"],
                    "reason": exclusion_reason,
                    "h1_disposition": {
                        "distinctness": h1["distinctness"],
                        "minimal_variant": h1["minimal_variant"],
                        "dialogic_variant": h1["dialogic_variant"],
                        "source_support_valid": h1["source_support_valid"],
                    },
                    "source_relink_accepted_count": (
                        len(
                            source_by_core[core_id][
                                "accepted_candidate_numbers"
                            ]
                        )
                        if core_id in source_by_core
                        else None
                    ),
                    "notes": (
                        source_by_core[core_id].get("notes", "")
                        if core_id in source_by_core
                        else h1.get("required_corrections", "")
                    ),
                }
            )
            continue

        provenance.extend(
            {
                "core_submove_id": core_id,
                "strategy_family": item["strategy_family"],
                **row,
                "raw_response_in_runtime_bank": False,
            }
            for row in accepted_sources
        )
        final_cores.append(
            {
                "protocol": PROTOCOL,
                "core_submove_id": core_id,
                "strategy_family": item["strategy_family"],
                "support_move": item["support_move"],
                "when_to_use": item["when_to_use"],
                "when_not_to_use": item["when_not_to_use"],
                "source_qualification": qualification,
            }
        )
        for visible_variant in item["variants"]:
            card = dict(full_cards[str(visible_variant["card_id"])])
            card["protocol"] = PROTOCOL
            card["source_qualification"] = qualification
            card["source_support"] = {
                "human_verified": True,
                "accepted_source_dialogues": qualification[
                    "accepted_source_dialogues"
                ],
                "raw_examples_exposed_to_generator": False,
            }
            card["quality_status"] = (
                "BANK_CONTENT_FROZEN_PENDING_H2_RETRIEVAL_QUALIFICATION"
            )
            card["eligible_for_formal_rs"] = False
            card["raw_source_response_exposed_to_generator"] = False
            final_cards.append(card)

    final_cards.sort(
        key=lambda row: (
            str(row["strategy_family"]),
            str(row["core_submove_id"]),
            str(row["execution_profile"]),
        )
    )
    final_cores.sort(key=lambda row: str(row["core_submove_id"]))
    excluded_cores.sort(key=lambda row: str(row["core_submove_id"]))
    provenance.sort(
        key=lambda row: (
            str(row["core_submove_id"]),
            str(row["source_dialogue_id"]),
        )
    )
    _validate_final_bank(final_cards)
    if len(excluded_cores) != 10:
        raise RuntimeError(
            f"expected 10 excluded cores, found {len(excluded_cores)}"
        )
    provenance_pairs = {
        (str(row["core_submove_id"]), str(row["source_dialogue_id"]))
        for row in provenance
    }
    if len(provenance_pairs) != len(provenance):
        raise RuntimeError("final provenance repeats a core/dialogue pair")
    provenance_counts = Counter(
        str(row["core_submove_id"]) for row in provenance
    )
    if set(provenance_counts) != {
        str(row["core_submove_id"]) for row in final_cores
    } or min(provenance_counts.values()) < 2:
        raise RuntimeError("every final core needs two provenance dialogues")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cards_path = args.out_dir / "strategy_cards_v4_final.jsonl"
    cores_path = args.out_dir / "core_catalog.jsonl"
    excluded_path = args.out_dir / "excluded_cores.jsonl"
    provenance_path = args.out_dir / "source_provenance_private.jsonl"
    annotations_path = args.out_dir / "source_relink_annotations_frozen.jsonl"
    _write_jsonl(cards_path, final_cards)
    _write_jsonl(cores_path, final_cores)
    _write_jsonl(excluded_path, excluded_cores)
    _write_jsonl(provenance_path, provenance)
    _write_jsonl(annotations_path, frozen_annotations)
    manifest = {
        "protocol": PROTOCOL,
        "status": "BANK_CONTENT_FROZEN_H2_RETRIEVAL_PENDING",
        "core_count": len(final_cores),
        "card_count": len(final_cards),
        "profile_counts": dict(
            sorted(Counter(row["execution_profile"] for row in final_cards).items())
        ),
        "family_core_counts": dict(
            sorted(Counter(row["strategy_family"] for row in final_cores).items())
        ),
        "family_card_counts": dict(
            sorted(Counter(row["strategy_family"] for row in final_cards).items())
        ),
        "excluded_core_count": len(excluded_cores),
        "source_provenance_rows": len(provenance),
        "source_relink_accepted_rows": sum(
            len(row["accepted_candidate_numbers"])
            for row in source_annotations
        ),
        "raw_source_responses_in_runtime_cards": 0,
        "formal_rs_runtime_enabled": False,
        "next_gate": (
            "One frozen H2 comparison of transparent and BGE rankers; Bank "
            "content may not change based on H2 outcomes."
        ),
        "inputs": {
            "source_annotations_original_path": str(args.source_annotations),
            "source_annotations_sha256": sha256_file(args.source_annotations),
            "source_packet": str(args.source_packet.relative_to(ROOT)),
            "h1_packet": str(args.h1_packet.relative_to(ROOT)),
            "h1_annotations": str(args.h1_annotations.relative_to(ROOT)),
            "candidate_cards": str(args.candidate_cards.relative_to(ROOT)),
        },
    }
    manifest["outputs"] = {
        path.name: sha256_file(path)
        for path in (
            cards_path,
            cores_path,
            excluded_path,
            provenance_path,
            annotations_path,
        )
    }
    _write_json(args.out_dir / "manifest.json", manifest)
    print(
        {
            "protocol": PROTOCOL,
            "status": manifest["status"],
            "cores": len(final_cores),
            "cards": len(final_cards),
            "families": manifest["family_core_counts"],
            "excluded_cores": len(excluded_cores),
            "source_rows": len(provenance),
            "out_dir": str(args.out_dir),
        }
    )


if __name__ == "__main__":
    main()
