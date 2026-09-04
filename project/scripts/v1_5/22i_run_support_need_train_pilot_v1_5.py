#!/usr/bin/env python3
"""Run the zero-API, train-only SupportNeedObservation pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.config import load_config
from metacom_pm.io import (
    iter_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from metacom_pm.pm_v1_5_semantic import (
    FrozenTransformerSemanticEncoder,
    require_semantic_runtime_contract,
    semantic_encoder_spec_from_config,
)
from metacom_pm.pm_v2_data import load_states
from metacom_pm.v1_5_support_need import (
    SupportNeedPartialLabel,
    prepare_support_need_views,
    run_support_need_learning_canary,
    run_train_only_support_need_pilot,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pm_v1_5.yaml")
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--partial-labels", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    states = load_states(args.states)
    labels = [
        SupportNeedPartialLabel.model_validate(row)
        for row in iter_jsonl(args.partial_labels)
    ]
    canary = run_support_need_learning_canary()
    if canary["status"] != "PASS":
        raise RuntimeError("support-need implementation canary did not PASS")

    encoder = FrozenTransformerSemanticEncoder.load(
        semantic_encoder_spec_from_config(config)
    )
    runtime = require_semantic_runtime_contract(config, encoder)
    prepared = [prepare_support_need_views(encoder, state) for state in states]
    report = run_train_only_support_need_pilot(
        states=states,
        prepared_views=prepared,
        labels=labels,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "learning_canary.json", canary)
    write_jsonl(
        args.out_dir / "visible_view_audit.jsonl",
        [
            {
                "state_id": row.state_id,
                "user_id": row.user_id,
                "explicit_boundaries": row.explicit_boundaries.model_dump(
                    mode="json"
                ),
                "audit": dict(row.audit),
            }
            for row in prepared
        ],
    )
    write_json(args.out_dir / "pilot_report.json", report)
    write_json(
        args.out_dir / "summary.json",
        {
            "protocol": "pm-v1.5-support-need-train-only-pilot-run-v1",
            "status": report["status"],
            "source_lineage": {
                "config_sha256": sha256_file(args.config),
                "states_sha256": sha256_file(args.states),
                "partial_labels_sha256": sha256_file(args.partial_labels),
            },
            "semantic_runtime_contract_sha256": runtime["contract_sha256"],
            "learning_canary_report_sha256": canary["report_sha256"],
            "pilot_report_sha256": report["report_sha256"],
            "api_calls_made": 0,
            "api_clients_created": 0,
            "training_labels_created": False,
            "formal_fit_authorized": False,
            "internal_test_outcomes_opened": False,
            "external_outcomes_opened": False,
        },
    )


if __name__ == "__main__":
    main()
