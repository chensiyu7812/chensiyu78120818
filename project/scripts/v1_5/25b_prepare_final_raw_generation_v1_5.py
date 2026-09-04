#!/usr/bin/env python3
"""Freeze the 256-call raw-user generation plan for P2."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)
from metacom_pm.text import estimate_tokens
from metacom_pm.v1_5_final_dataset_blueprint import BLUEPRINT_PROTOCOL
from metacom_pm.v1_5_final_raw_generation import (
    RAW_GENERATION_PROTOCOL,
    generation_messages_for_blueprint,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN_PROTOCOL = "pm-v1.5-p2-raw-user-generation-plan-v9"
PLAN_STATUS = "FROZEN_READY_FOR_256_RAW_USER_CALLS"
MAX_OUTPUT_BY_SHAPE = {"SMALL": 1800, "MEDIUM": 2800, "EVO_LIKE_LARGE": 5200}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blueprint-dir",
        type=Path,
        default=ROOT / "data/pm_v1_5_final_candidate_first_v8/private",
    )
    parser.add_argument(
        "--experiment-config", type=Path, default=ROOT / "configs/experiment.yaml"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_p2_raw_generation_plan_v9",
    )
    args = parser.parse_args()
    blueprint_path = args.blueprint_dir / "construction_blueprint.jsonl"
    blueprint_report = read_json(args.blueprint_dir / "blueprint_report.json")
    rows = [dict(row) for row in iter_jsonl(blueprint_path)]
    if (
        blueprint_report.get("status") != "PASS"
        or blueprint_report.get("blueprint_sha256") != sha256_file(blueprint_path)
        or len(rows) != 256
        or any(row.get("protocol") != BLUEPRINT_PROTOCOL for row in rows)
    ):
        raise RuntimeError("P2 private blueprint is stale or invalid")

    config = load_config(args.experiment_config)
    endpoint = endpoint_from_config(config, "p2_data_generator")
    identity = {
        "base_url": endpoint.base_url,
        "model": endpoint.model,
        "family": endpoint.family,
        "transport": endpoint.transport,
    }
    calls = []
    for row in rows:
        messages = generation_messages_for_blueprint(row)
        messages_sha = sha256_text(canonical_json(messages))
        calls.append(
            {
                "protocol": PLAN_PROTOCOL,
                "call_id": "raw_call_" + stable_hex(PLAN_PROTOCOL, row["state_id"], n=24),
                "state_id": row["state_id"],
                "blueprint_index": row["blueprint_index"],
                "split": row["split"],
                "history_shape": row["history_shape"],
                "messages": messages,
                "messages_sha256": messages_sha,
                "estimated_input_tokens": estimate_tokens(canonical_json(messages)),
                "generation": {
                    "temperature": 0.4,
                    "max_output_tokens": MAX_OUTPUT_BY_SHAPE[row["history_shape"]],
                    "seed": int(stable_hex(PLAN_PROTOCOL, row["state_id"], n=8), 16),
                    "response_schema": "FinalRawStateDraft",
                },
                "generator_identity": identity,
                "h1_gold_read": False,
                "external_lockbox_read": False,
            }
        )
    if len({row["call_id"] for row in calls}) != 256:
        raise RuntimeError("raw generation call IDs are not unique")
    if Counter(row["split"] for row in calls) != Counter(
        {"FIT": 128, "FRESH_CONFIRMATION": 64, "SEALED_INTERNAL_TEST": 64}
    ):
        raise RuntimeError("raw generation call split mismatch")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    call_path = args.out_dir / "call_plan.jsonl"
    write_jsonl(call_path, calls)
    report = {
        "protocol": PLAN_PROTOCOL,
        "status": PLAN_STATUS,
        "raw_output_protocol": RAW_GENERATION_PROTOCOL,
        "planned_calls": 256,
        "api_calls_made": 0,
        "split_counts": dict(Counter(row["split"] for row in calls)),
        "history_shape_counts": dict(Counter(row["history_shape"] for row in calls)),
        "estimated_input_tokens": sum(row["estimated_input_tokens"] for row in calls),
        "maximum_output_tokens": sum(
            row["generation"]["max_output_tokens"] for row in calls
        ),
        "generator_identity": identity,
        "blueprint_sha256": sha256_file(blueprint_path),
        "call_plan_sha256": sha256_file(call_path),
        "resumable": True,
        "private_construction_intent_in_model_features": False,
        "external_lockbox_read": False,
    }
    write_json(args.out_dir / "generation_plan_report.json", report)
    write_json(
        args.out_dir / "freeze_manifest.json",
        {
            "protocol": PLAN_PROTOCOL,
            "status": PLAN_STATUS,
            "blueprint_sha256": sha256_file(blueprint_path),
            "call_plan_sha256": sha256_file(call_path),
            "generation_plan_report_sha256": sha256_file(
                args.out_dir / "generation_plan_report.json"
            ),
            "generator_identity": identity,
        },
    )
    print(report)


if __name__ == "__main__":
    main()
