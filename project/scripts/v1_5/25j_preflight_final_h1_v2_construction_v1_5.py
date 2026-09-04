#!/usr/bin/env python3
"""Realize the H1-v2 construction set without API calls or human labels."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl
from metacom_pm.v1_5_final_dataset_blueprint import H1_V2_BLUEPRINT_PROTOCOL
from metacom_pm.v1_5_final_raw_generation import (
    H1_V2_RAW_GENERATION_PROTOCOL,
    FinalRawStateDraft,
    compile_raw_user_state,
    deterministically_realize_h1_v2_raw_draft,
    validate_h1_v2_raw_draft,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-p2-h1-v2-zero-api-construction-preflight-v1"


def _neutral_draft(prior_session_count: int) -> FinalRawStateDraft:
    return FinalRawStateDraft.model_validate(
        {
            "profile_entries": [],
            "critical_prior_sessions": [
                {
                    "summary": f"Neutral earlier summary {index}.",
                    "seeker_text": f"I discussed a neutral earlier topic number {index}.",
                    "supporter_text": f"You acknowledged neutral topic number {index}.",
                }
                for index in range(3)
            ],
            "distractor_session_summaries": [
                f"Neutral unrelated session summary {index}."
                for index in range(max(0, prior_session_count - 3))
            ],
            "recent_dialogue": [
                {
                    "seeker_text": "I want to talk about what is happening.",
                    "supporter_text": "I am listening to what you want to share.",
                }
            ],
            "current_user_text": "I want to explain what is happening now.",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blueprint-dir",
        type=Path,
        default=ROOT / "data/pm_v1_5_final_candidate_first_v9_h1_v2/private",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_p2_h1_v2_zero_api_raw_v10",
    )
    args = parser.parse_args()
    blueprint_path = args.blueprint_dir / "construction_blueprint.jsonl"
    blueprint_report = read_json(args.blueprint_dir / "blueprint_report.json")
    rows = [dict(row) for row in iter_jsonl(blueprint_path)]
    if (
        blueprint_report.get("status") != "PASS"
        or blueprint_report.get("protocol") != H1_V2_BLUEPRINT_PROTOCOL
        or blueprint_report.get("blueprint_sha256") != sha256_file(blueprint_path)
        or len(rows) != 256
        or any(row.get("protocol") != H1_V2_BLUEPRINT_PROTOCOL for row in rows)
    ):
        raise RuntimeError("H1-v2 private blueprint is stale or invalid")

    raw_states = []
    validation_errors: Counter[str] = Counter()
    public_surfaces = set()
    for row in rows:
        draft = deterministically_realize_h1_v2_raw_draft(
            draft=_neutral_draft(int(row["prior_session_count_target"])),
            row=row,
        )
        validation = validate_h1_v2_raw_draft(draft=draft, row=row)
        validation_errors.update(validation["errors"])
        if validation["status"] != "PASS":
            continue
        raw = compile_raw_user_state(
            draft=draft,
            row=row,
            validation_mode="h1_v2",
            protocol=H1_V2_RAW_GENERATION_PROTOCOL,
        )
        raw_states.append(raw)
        public_surfaces.add(
            (
                tuple(
                    (turn["role"], turn["content"])
                    for turn in raw["visible_dialogue"]
                ),
                raw["current_user_text"],
            )
        )

    status = "PASS" if len(raw_states) == 256 and not validation_errors else "FAIL"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "raw_states.jsonl"
    write_jsonl(raw_path, raw_states)
    report = {
        "protocol": PROTOCOL,
        "status": status,
        "states": len(raw_states),
        "split_counts": dict(Counter(row["split"] for row in raw_states)),
        "unique_public_dialogue_current_surfaces": len(public_surfaces),
        "validation_errors": dict(validation_errors),
        "raw_state_protocol": H1_V2_RAW_GENERATION_PROTOCOL,
        "raw_states_sha256": sha256_file(raw_path),
        "api_calls_made": 0,
        "human_labels_read": 0,
        "response_or_outcome_read": False,
        "external_lockbox_read": False,
    }
    write_json(args.out_dir / "preflight_report.json", report)
    if status != "PASS":
        raise RuntimeError(report)
    print(report)


if __name__ == "__main__":
    main()
