#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.freeze import require_study_freeze
from metacom_pm.io import canonical_json, read_json, sha256_text
from metacom_pm.pm_v2_external_eval import expected_external_units
from metacom_pm.pm_v2_forced_swap import run_forced_swap_evaluation


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--pm-v2-config", type=Path, default=ROOT / "configs" / "pm_v2.yaml")
    parser.add_argument("--freeze", type=Path, default=ROOT / "outputs" / "pm_v2_study_freeze.json")
    parser.add_argument("--evoemo", type=Path, default=ROOT / "data" / "external" / "evo_emo.json")
    parser.add_argument(
        "--turn-paths",
        type=Path,
        nargs=2,
        default=[
            ROOT / "outputs" / "evoemo_pm_v2" / "turns.jsonl",
            ROOT / "outputs" / "evoemo_pm_v2_cost_matched_fixed" / "turns.jsonl",
        ],
        metavar=("TREATMENT_TURNS", "BASELINE_TURNS"),
    )
    parser.add_argument(
        "--generation-attestations",
        type=Path,
        nargs=2,
        default=[
            ROOT / "outputs" / "evoemo_pm_v2" / "artifact_attestation.json",
            ROOT
            / "outputs"
            / "evoemo_pm_v2_cost_matched_fixed"
            / "artifact_attestation.json",
        ],
        metavar=("TREATMENT_ATTESTATION", "BASELINE_ATTESTATION"),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "outputs" / "pm_v2_forced_swap"
    )
    parser.add_argument("--max-api-calls", type=int, default=500)
    parser.add_argument("--max-estimated-usd", type=float, default=10.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=12000)
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.run and args.overwrite:
        raise RuntimeError(
            "paid API runs prohibit --overwrite; use a new output directory"
        )

    experiment = load_config(args.config)
    load_config(args.pm_v2_config)
    verification = require_study_freeze(
        args.freeze,
        release_root=ROOT,
        config_path=args.config,
        required_files=[args.pm_v2_config, args.evoemo],
    )
    freeze_sha256 = str(verification["freeze_sha256"])
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    notes = freeze.get("notes") or {}
    generation_contract = notes.get("generation_contract") or {}
    external_contract = notes.get("external_evaluation_contract") or {}
    forced_swap = external_contract.get("forced_swap") or {}
    if not generation_contract or not forced_swap:
        raise RuntimeError("study freeze lacks generation/forced-swap contracts")

    treatment = str(forced_swap["treatment"])
    baseline = str(forced_swap["baseline"])
    expected_conditions = [treatment, baseline]
    for path, expected_condition in zip(
        args.generation_attestations, expected_conditions
    ):
        attestation = read_json(path)
        parameters = attestation.get("parameters") or {}
        if parameters.get("condition") != expected_condition:
            raise RuntimeError(
                f"generation attestation condition/order mismatch: {path}"
            )
        for key in (
            "protocol",
            "simulator_id",
            "max_turns",
            "seeds",
            "strategy_action_tokens",
            "strategy_top_k",
            "memory_min_score",
            "strategy_min_score",
            "action_preflight_gates",
            "maximum_cost_matched_relative_deviation",
            "generator_retries",
            "input_token_safety_factor",
            "fail_on_reported_input_overrun",
            "evaluation_unit_contract",
            "paid_generation_scope",
        ):
            if parameters.get(key) != generation_contract.get(key):
                raise RuntimeError(
                    f"generation attestation violates frozen {key}: {path}"
                )

    endpoint_names = [
        str(value["name"]) for value in forced_swap.get("judge_endpoints") or []
    ]
    endpoints = [endpoint_from_config(experiment, name) for name in endpoint_names]
    if len(endpoints) != len(forced_swap.get("judge_endpoints") or []):
        raise RuntimeError("forced-swap endpoint contract is incomplete")
    for endpoint, frozen_endpoint in zip(
        endpoints, forced_swap.get("judge_endpoints") or []
    ):
        endpoint_sha256 = sha256_text(
            canonical_json(
                {
                    "model": endpoint.model,
                    "family": endpoint.family,
                    "base_url": endpoint.base_url,
                }
            )
        )
        if endpoint_sha256 != frozen_endpoint.get("sha256"):
            raise RuntimeError("forced-swap judge endpoint changed after freeze")

    expected_units = expected_external_units(
        args.evoemo,
        seeds=[int(value) for value in generation_contract["seeds"]],
        simulator_id=str(generation_contract["simulator_id"]),
        turn_indices=[int(value) for value in external_contract["turn_indices"]],
    )
    evaluation_unit_contract = generation_contract.get("evaluation_unit_contract") or {}
    if (
        evaluation_unit_contract
        != external_contract.get("evaluation_unit_contract")
        or evaluation_unit_contract.get("expected_unit_count") != len(expected_units)
        or evaluation_unit_contract.get("expected_units_sha256")
        != sha256_text(canonical_json(expected_units))
    ):
        raise RuntimeError("frozen forced-swap generation universe is stale")
    result = run_forced_swap_evaluation(
        evoemo_path=args.evoemo,
        study_freeze_path=args.freeze,
        study_freeze_sha256=freeze_sha256,
        turn_paths=args.turn_paths,
        generation_attestation_paths=args.generation_attestations,
        expected_units=expected_units,
        treatment=treatment,
        baseline=baseline,
        endpoints=endpoints,
        sample_units=int(forced_swap["sample_units"]),
        order_variants=[int(value) for value in forced_swap["order_variants"]],
        judge_seed=int(forced_swap["judge_seed"]),
        estimated_output_tokens_per_call=int(
            forced_swap["estimated_output_tokens_per_call"]
        ),
        maximum_order_disagreement_rate=float(
            forced_swap["maximum_order_disagreement_rate"]
        ),
        minimum_schema_success_rate=float(
            forced_swap["minimum_schema_success_rate"]
        ),
        minimum_cross_family_support_delta_correlation=float(
            forced_swap["minimum_cross_family_support_delta_correlation"]
        ),
        require_cross_family_direction_agreement=bool(
            forced_swap["require_cross_family_direction_agreement"]
        ),
        minimum_support_delta_ci_upper_for_continuation=float(
            forced_swap["minimum_support_delta_ci_upper_for_continuation"]
        ),
        minimum_support_delta_ci_lower_for_advantage=float(
            forced_swap["minimum_support_delta_ci_lower_for_advantage"]
        ),
        require_positive_support_delta_every_family=bool(
            forced_swap["require_positive_support_delta_every_family"]
        ),
        minimum_resolved_preference_margin=int(
            forced_swap["minimum_resolved_preference_margin"]
        ),
        maximum_cost_matched_relative_deviation=float(
            external_contract["maximum_cost_matched_relative_deviation"]
        ),
        pricing_usd_per_mtok=forced_swap["judge_pricing_usd_per_mtok"],
        api_cost_planning=forced_swap["api_cost_planning"],
        out_dir=args.out_dir,
        run=bool(args.run),
        max_api_calls=args.max_api_calls,
        max_estimated_usd=args.max_estimated_usd,
        max_input_tokens_per_call=args.max_input_tokens_per_call,
        accept_cost_estimate_sha256=args.accept_cost_estimate_sha256,
        overwrite=args.overwrite,
    )
    print(result)


if __name__ == "__main__":
    main()
