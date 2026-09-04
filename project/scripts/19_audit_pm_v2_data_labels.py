#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.config import load_config
from metacom_pm.io import iter_jsonl, sha256_file, write_json
from metacom_pm.pm_v2_audit import audit_training_labels
from metacom_pm.pm_v2_contracts import ActionLabel, CompositeSpec, PMV2Split
from metacom_pm.pm_v2_data import load_evaluator_context_index, load_states

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pm-v2-config",
        type=Path,
        default=ROOT / "configs" / "pm_v2.yaml",
    )
    parser.add_argument("--states", type=Path, default=ROOT / "data" / "pm_v2" / "pm_v2_states.jsonl")
    parser.add_argument(
        "--evaluator-contexts",
        type=Path,
        default=ROOT / "data" / "pm_v2" / "evaluator_contexts.jsonl",
    )
    parser.add_argument("--labels", type=Path, default=ROOT / "outputs" / "pm_v2_judging" / "action_labels.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "pm_v2_judging" / "data_label_audit.json")
    args = parser.parse_args()

    pm_config = load_config(args.pm_v2_config)
    if pm_config.get("version") != "pm-v2.2":
        raise ValueError("unsupported PM-v2 config version")
    quality_cfg = pm_config["quality_composite"]
    composite_spec = CompositeSpec(
        version=str(quality_cfg["version"]),
        weights={
            str(key): float(value) for key, value in quality_cfg["weights"].items()
        },
    )
    selection_cfg = pm_config["selection"]
    gate_cfg = pm_config["data_label_gate"]
    selected_splits = {PMV2Split.TRAIN, PMV2Split.CALIBRATION}
    all_states = load_states(args.states)
    evaluator_contexts = load_evaluator_context_index(
        args.evaluator_contexts,
        states=all_states,
        require_exact=True,
    )
    states = [state for state in all_states if state.split in selected_splits]
    state_ids = {state.state_id for state in states}
    labels = [
        ActionLabel.model_validate(row)
        for row in iter_jsonl(args.labels)
        if str(row.get("state_id")) in state_ids
    ]
    report = audit_training_labels(
        states,
        labels,
        evaluator_contexts=evaluator_contexts,
        composite_spec=composite_spec,
        risk_weight=float(selection_cfg["risk_weight"]),
        cost_weight=float(selection_cfg["cost_weight"]),
        maximum_single_action_share=float(gate_cfg["maximum_single_action_share"]),
        minimum_m0_share=float(gate_cfg["minimum_m0_share"]),
        minimum_r0_share=float(gate_cfg["minimum_r0_share"]),
        minimum_rs_share=float(gate_cfg["minimum_rs_share"]),
        minimum_distinct_actions=int(gate_cfg["minimum_distinct_actions"]),
        minimum_regime_pass_rate=float(gate_cfg["minimum_regime_pass_rate"]),
        minimum_reliable_rate=float(gate_cfg["minimum_reliable_rate"]),
    )
    report.update(
        {
            "pm_v2_config": str(args.pm_v2_config),
            "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
            "audited_splits": sorted(split.value for split in selected_splits),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print({key: value for key, value in report.items() if key != "rows"})


if __name__ == "__main__":
    main()
