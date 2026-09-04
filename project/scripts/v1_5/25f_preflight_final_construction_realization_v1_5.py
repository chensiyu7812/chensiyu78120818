#!/usr/bin/env python3
"""Build a zero-API mechanical P2 corpus for full rank-1 preflight.

This is not training, confirmation, test, or human-review data.  It exercises
the frozen private matrix and deterministic realizer across all 256 rows so
the formal compiler/retriever can be checked before any additional paid run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl
from metacom_pm.v1_5_final_dataset_blueprint import BLUEPRINT_PROTOCOL
from metacom_pm.v1_5_final_raw_generation import (
    FinalCriticalPriorSessionDraft,
    FinalCurrentExchangeDraft,
    FinalRawStateDraft,
    RAW_GENERATION_PROTOCOL,
    RAW_REALIZATION_PROTOCOL,
    compile_raw_user_state,
    deterministically_realize_raw_draft,
    validate_raw_draft_for_blueprint,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-p2-zero-api-mechanical-realization-preflight-v1"


def _neutral_draft(distractor_count: int) -> FinalRawStateDraft:
    return FinalRawStateDraft(
        profile_entries=[],
        critical_prior_sessions=[
            FinalCriticalPriorSessionDraft(
                summary=f"Neutral prior summary number {index}.",
                seeker_text=f"I mentioned an ordinary neutral topic number {index}.",
                supporter_text=f"You acknowledged neutral topic number {index}.",
            )
            for index in range(1, 4)
        ],
        distractor_session_summaries=[
            f"Neutral provider distractor number {index}."
            for index in range(distractor_count)
        ],
        recent_dialogue=[
            FinalCurrentExchangeDraft(
                seeker_text="An ordinary concern was on my mind earlier.",
                supporter_text="You were beginning to describe what felt difficult.",
            )
        ],
        current_user_text="An ordinary current concern needs a response.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_final_candidate_first_v8/private/construction_blueprint.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_p2_zero_api_mechanical_raw_v9",
    )
    args = parser.parse_args()
    blueprints = [dict(row) for row in iter_jsonl(args.blueprint)]
    if len(blueprints) != 256 or any(
        row.get("protocol") != BLUEPRINT_PROTOCOL for row in blueprints
    ):
        raise RuntimeError("mechanical preflight requires the frozen 256-row blueprint")

    rows = []
    for blueprint in blueprints:
        draft = deterministically_realize_raw_draft(
            draft=_neutral_draft(
                int(blueprint["prior_session_count_target"]) - 3
            ),
            row=blueprint,
        )
        validation = validate_raw_draft_for_blueprint(
            draft=draft, row=blueprint
        )
        if validation["status"] != "PASS":
            raise RuntimeError(
                f"mechanical realization failed {blueprint['state_id']}: "
                f"{validation['errors']}"
            )
        rows.append(
            {
                **compile_raw_user_state(draft=draft, row=blueprint),
                "mechanical_preflight_only": True,
                "generator_identity": {
                    "model": "NONE_ZERO_API_MECHANICAL_PREFLIGHT"
                },
                "construction_realizer_protocol": RAW_REALIZATION_PROTOCOL,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "raw_states.jsonl"
    write_jsonl(raw_path, rows)
    report = {
        "protocol": PROTOCOL,
        "status": "PASS_STATIC_REALIZATION",
        "rows": len(rows),
        "raw_state_protocol": RAW_GENERATION_PROTOCOL,
        "construction_realizer_protocol": RAW_REALIZATION_PROTOCOL,
        "api_calls": 0,
        "human_labels_read": 0,
        "response_or_outcome_read": False,
        "eligible_for_training_or_evaluation": False,
        "blueprint_sha256": sha256_file(args.blueprint),
        "raw_states_sha256": sha256_file(raw_path),
        "next_step": "RUN_FORMAL_COMPILER_AND_RANK1_MATERIALIZER_AS_PREFLIGHT_ONLY",
    }
    write_json(args.out_dir / "preflight_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
