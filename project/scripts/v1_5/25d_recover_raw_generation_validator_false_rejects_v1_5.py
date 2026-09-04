#!/usr/bin/env python3
"""Recover paid drafts rejected only by the superseded ME/me regex bug."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import append_jsonl, canonical_json, iter_jsonl, sha256_text, utc_now, write_json
from metacom_pm.v1_5_final_raw_generation import (
    FinalRawStateDraft,
    compile_raw_user_state,
    validate_raw_draft_for_blueprint,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-p2-validator-false-reject-recovery-v1"


def _rows(path: Path):
    return [dict(row) for row in iter_jsonl(path)] if path.is_file() else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT / "data/pm_v1_5_final_candidate_first_v1/private/construction_blueprint.jsonl",
    )
    parser.add_argument(
        "--execution-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_p2_raw_generation_execution_v1",
    )
    args = parser.parse_args()
    blueprints = {str(row["state_id"]): row for row in _rows(args.blueprint)}
    rejected = _rows(args.execution_dir / "rejected_attempts.jsonl")
    accepted_path = args.execution_dir / "raw_states.jsonl"
    accepted_ids = {str(row["state_id"]) for row in _rows(accepted_path)}
    recovered = 0
    still_rejected = []
    for row in rejected:
        state_id = str(row["state_id"])
        if state_id in accepted_ids:
            continue
        old_errors = set(row.get("validation", {}).get("errors") or [])
        if old_errors != {"experimental_meta_language_leak"}:
            still_rejected.append({"state_id": state_id, "reason": "not_only_regex_bug"})
            continue
        draft = FinalRawStateDraft.model_validate(row["trace"]["validated"])
        validation = validate_raw_draft_for_blueprint(
            draft=draft, row=blueprints[state_id]
        )
        if validation["status"] != "PASS":
            still_rejected.append(
                {"state_id": state_id, "reason": validation["errors"]}
            )
            continue
        compiled = compile_raw_user_state(draft=draft, row=blueprints[state_id])
        trace = row["trace"]
        append_jsonl(
            accepted_path,
            {
                **compiled,
                "generator_identity": {
                    "base_url": trace["base_url"],
                    "model": trace["model"],
                    "family": trace["model_family"],
                    "transport": trace["transport"],
                },
                "provider_request_hash": trace["request_hash"],
                "provider_draft_sha256": sha256_text(
                    canonical_json(draft.model_dump(mode="json"))
                ),
                "usage": trace["usage"],
                "completed_at": utc_now(),
                "recovery_protocol": PROTOCOL,
                "recovered_from_paid_attempt_without_recall": True,
            },
        )
        accepted_ids.add(state_id)
        recovered += 1
    report = {
        "protocol": PROTOCOL,
        "status": "COMPLETE",
        "rejected_rows_examined": len(rejected),
        "recovered_without_api_recall": recovered,
        "still_rejected": still_rejected,
        "api_calls_made": 0,
        "old_bug": "case_insensitive_ME_acronym_matched_English_pronoun_me",
        "new_rule": "component_acronyms_are_case_sensitive",
    }
    write_json(args.execution_dir / "validator_recovery_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
