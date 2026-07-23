#!/usr/bin/env python3
"""Select a small, real, outcome-free pilot subset of ESConv-auxiliary states.

Certifying the generation/judging pipeline (13b/13c) on a handful of real
states before committing to the full 719-state run needs a pilot sample.
This script selects it under a frozen, disclosed, purely observable rule:

- 12 *distinct* train dialogues (never repeating a dialogue), taken as the
  first 12 in the frozen ``pm_v1_5_selected_seed_sources.jsonl`` ordinal
  order (the same canonical order the 24/12/16 train/calibration/
  internal_test split assignment already uses) -- deterministic, not random,
  and not re-sorted by anything outcome-derived.
- Within each selected dialogue, one state is chosen by its position in the
  dialogue's own observable turn sequence (``provenance.turn_index``,
  written by the auxiliary builder itself): the earliest available turn,
  the middle available turn, or the latest available turn.
- Dialogues at ordinal position 0-3 (of the 12 selected) contribute an
  "early" state, 4-7 a "mid" state, 8-11 a "late" state -- 4 of each.

This reads only ``runtime_states.jsonl``, ``memory_backend.jsonl`` and
``pm_v2_states.jsonl`` (observable state) plus the frozen seed-source
manifest (dialogue identity only). It never opens ``audit_only.jsonl`` and
never references gold_response, gold_strategy, judge results, or generation
outcomes -- the selection is fully computable before a single API call is
made, let alone before any label exists.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, read_json, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
PILOT_SELECTION_PROTOCOL = "pm-v1.5-esconv-auxiliary-pilot-selection-stratified-v1"
DIALOGUES_PER_STRATUM = 4
STRATA = ("early", "mid", "late")


def select_pilot_states(
    *,
    selected_seed_sources_path: Path,
    train_split_dir: Path,
    dialogue_offset: int = 0,
) -> dict[str, Any]:
    seeds = list(iter_jsonl(selected_seed_sources_path))
    train_dialogue_ids_ordered = [
        str(row["dialogue_id"])
        for row in sorted(seeds, key=lambda r: int(r["selection_ordinal"]))
        if int(row["selection_ordinal"]) < 24
    ]
    if len(train_dialogue_ids_ordered) != 24:
        raise RuntimeError(
            "expected exactly 24 train dialogues in the frozen seed-source "
            f"manifest, found {len(train_dialogue_ids_ordered)}"
        )
    n_pilot_dialogues = DIALOGUES_PER_STRATUM * len(STRATA)
    if dialogue_offset < 0 or dialogue_offset + n_pilot_dialogues > len(
        train_dialogue_ids_ordered
    ):
        raise ValueError(
            f"dialogue_offset {dialogue_offset} leaves fewer than "
            f"{n_pilot_dialogues} train dialogues; must be in "
            f"[0, {len(train_dialogue_ids_ordered) - n_pilot_dialogues}]"
        )
    # Still deterministic, still computable before any real API call --
    # offset only shifts which contiguous window of the frozen canonical
    # ordinal order is used (e.g. a second, disjoint pilot after an earlier
    # one's real artifacts were lost, so the fresh run is not just
    # recomputing byte-identical content).
    pilot_dialogue_ids = train_dialogue_ids_ordered[
        dialogue_offset : dialogue_offset + n_pilot_dialogues
    ]
    stratum_by_dialogue_id = {
        dialogue_id: STRATA[index // DIALOGUES_PER_STRATUM]
        for index, dialogue_id in enumerate(pilot_dialogue_ids)
    }

    runtime_rows = list(iter_jsonl(train_split_dir / "runtime_states.jsonl"))
    backend_rows = list(iter_jsonl(train_split_dir / "memory_backend.jsonl"))
    pmv2_rows = list(iter_jsonl(train_split_dir / "pm_v2_states.jsonl"))
    backend_by_card = {row["card_id"]: row for row in backend_rows}
    pmv2_by_state = {row["state_id"]: row for row in pmv2_rows}

    by_dialogue: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in runtime_rows:
        dialogue_id = str(row["provenance"]["dialogue_id"])
        by_dialogue[dialogue_id].append(row)

    selected: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for dialogue_id in pilot_dialogue_ids:
        stratum = stratum_by_dialogue_id[dialogue_id]
        candidates = sorted(
            by_dialogue[dialogue_id],
            key=lambda row: int(row["provenance"]["turn_index"]),
        )
        if not candidates:
            raise RuntimeError(f"pilot dialogue {dialogue_id} has no eligible states")
        if stratum == "early":
            chosen = candidates[0]
        elif stratum == "late":
            chosen = candidates[-1]
        else:
            chosen = candidates[len(candidates) // 2]
        selected.append(chosen)
        manifest_rows.append(
            {
                "dialogue_id": dialogue_id,
                "state_id": chosen["state_id"],
                "card_id": chosen["card_id"],
                "stratum": stratum,
                "turn_index": int(chosen["provenance"]["turn_index"]),
                "dialogue_eligible_turn_count": len(candidates),
            }
        )

    unique_dialogue_ids = {row["dialogue_id"] for row in manifest_rows}
    if len(unique_dialogue_ids) != n_pilot_dialogues:
        raise RuntimeError(
            f"pilot selection must cover {n_pilot_dialogues} distinct "
            f"dialogues, got {len(unique_dialogue_ids)}"
        )
    if len(selected) != n_pilot_dialogues:
        raise RuntimeError("pilot selection must choose exactly one state per dialogue")
    stratum_counts = collections.Counter(row["stratum"] for row in manifest_rows)
    if any(stratum_counts[s] != DIALOGUES_PER_STRATUM for s in STRATA):
        raise RuntimeError(f"pilot stratum counts must be 4/4/4, got {dict(stratum_counts)}")

    selected_card_ids = [row["card_id"] for row in selected]
    selected_state_ids = [row["state_id"] for row in selected]
    missing_backend = [c for c in selected_card_ids if c not in backend_by_card]
    missing_pmv2 = [s for s in selected_state_ids if s not in pmv2_by_state]
    if missing_backend or missing_pmv2:
        raise RuntimeError(
            f"pilot selection references missing backend/pm_v2 rows: "
            f"backend={missing_backend[:5]}, pm_v2={missing_pmv2[:5]}"
        )

    return {
        "protocol": PILOT_SELECTION_PROTOCOL,
        "n_dialogues": n_pilot_dialogues,
        "n_states": len(selected),
        "strata": list(STRATA),
        "dialogues_per_stratum": DIALOGUES_PER_STRATUM,
        "manifest": manifest_rows,
        "runtime_rows": selected,
        "backend_rows": [backend_by_card[c] for c in selected_card_ids],
        "pmv2_rows": [pmv2_by_state[s] for s in selected_state_ids],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-seed-sources",
        type=Path,
        default=ROOT / "data" / "strategy" / "pm_v1_5_selected_seed_sources.jsonl",
    )
    parser.add_argument(
        "--train-split-dir",
        type=Path,
        default=ROOT / "data" / "esconv_auxiliary_v1_5" / "train",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "esconv_auxiliary_v1_5_pilot" / "train",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=ROOT / "outputs" / "esconv_auxiliary_pilot_selection_manifest.json",
    )
    parser.add_argument(
        "--dialogue-offset",
        type=int,
        default=0,
        help=(
            "Shift which contiguous window of the 24 frozen train dialogues "
            "is used (still deterministic, still selectable before any real "
            "API call). Use a nonzero offset to pick a disjoint pilot "
            "sample, e.g. after an earlier pilot's real artifacts were lost "
            "and a fresh, genuinely different sample is wanted rather than "
            "recomputing byte-identical content."
        ),
    )
    args = parser.parse_args()

    result = select_pilot_states(
        selected_seed_sources_path=args.selected_seed_sources,
        train_split_dir=args.train_split_dir,
        dialogue_offset=args.dialogue_offset,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "runtime_states.jsonl", result["runtime_rows"])
    write_jsonl(args.out_dir / "memory_backend.jsonl", result["backend_rows"])
    write_jsonl(args.out_dir / "pm_v2_states.jsonl", result["pmv2_rows"])
    manifest = {
        "protocol": result["protocol"],
        "dialogue_offset": int(args.dialogue_offset),
        "n_dialogues": result["n_dialogues"],
        "n_states": result["n_states"],
        "strata": result["strata"],
        "dialogues_per_stratum": result["dialogues_per_stratum"],
        "manifest": result["manifest"],
    }
    write_json(args.manifest_out, manifest)
    print(manifest)


if __name__ == "__main__":
    main()
