#!/usr/bin/env python3
"""Run PM-v1.5's real four-call external pointwise schema smoke.

The smoke unit is deterministically the first unit in the frozen 12-unit
forced-swap canary set. The whole canary set is excluded from the main scored
external sample, so neither smoke nor sensitivity-check judgments are reused
as efficacy observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metacom_pm.artifacts import require_artifact_attestation
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.paid_run_release import require_paid_run_release
from metacom_pm.freeze import require_study_freeze
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import canonical_json, read_json, sha256_text
from metacom_pm.pm_v2_external_eval import expected_external_units
from metacom_pm.pm_v2_external_schema_smoke import (
    POINTWISE_SCHEMA_SMOKE_PROTOCOL,
    run_external_pointwise_schema_smoke,
)
from metacom_pm.v1_5_forced_swap_canary import require_v1_5_forced_swap_canary

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--pm-v1-5-config", type=Path, default=ROOT / "configs" / "pm_v1_5.yaml")
    parser.add_argument("--freeze", type=Path, default=ROOT / "outputs" / "pm_v1_5_study_freeze.json")
    parser.add_argument("--evoemo", type=Path, default=ROOT / "data" / "external" / "evo_emo.json")
    parser.add_argument("--turns", type=Path, default=ROOT / "outputs" / "evoemo_pm_v1_5" / "turns.jsonl")
    parser.add_argument(
        "--generation-attestation",
        type=Path,
        default=ROOT / "outputs" / "evoemo_pm_v1_5" / "artifact_attestation.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_external_pointwise_schema_smoke",
    )
    parser.add_argument(
        "--forced-swap-summary",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_forced_swap_canary" / "summary.json",
    )
    parser.add_argument(
        "--forced-swap-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v1_5_forced_swap_canary"
        / "artifact_attestation.json",
    )
    parser.add_argument("--max-api-calls", type=int, default=4)
    parser.add_argument("--max-estimated-usd", type=float, default=1.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=12000)
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.run and args.overwrite:
        raise RuntimeError("paid schema-smoke runs prohibit --overwrite")

    experiment = load_config(args.config)
    pm_v1_5_config = load_config(args.pm_v1_5_config)
    require_paid_run_release(
        pm_v1_5_config,
        config_path=args.pm_v1_5_config,
        stage="external_pointwise_schema_smoke",
        run=bool(args.run),
        run_identity=args.accept_cost_estimate_sha256,
    )
    if pm_v1_5_config.get("version") != "pm-v1.5":
        raise RuntimeError("this schema-smoke driver requires a pm-v1.5 config")
    supporter_generation_contract = SupporterGenerationContract.from_config(pm_v1_5_config)

    verification = require_study_freeze(
        args.freeze,
        release_root=ROOT,
        config_path=args.config,
        required_files=[args.pm_v1_5_config, args.evoemo],
    )
    freeze_sha256 = str(verification["freeze_sha256"])
    notes = json.loads(args.freeze.read_text(encoding="utf-8")).get("notes") or {}
    generation_contract = notes.get("generation_contract") or {}
    external_contract = notes.get("external_evaluation_contract") or {}
    smoke_contract = external_contract.get("pointwise_schema_smoke") or {}
    if (
        not generation_contract
        or smoke_contract.get("protocol") != POINTWISE_SCHEMA_SMOKE_PROTOCOL
        or not (
            bool(smoke_contract.get("required_before_full_external_client_creation"))
            or bool(smoke_contract.get("legacy_pointwise_fallback_only"))
        )
    ):
        raise RuntimeError("study freeze lacks the legacy pointwise schema smoke")
    if (
        generation_contract.get("supporter_generation_treatment")
        != supporter_generation_contract.payload()
        or generation_contract.get("supporter_generation_treatment_sha256")
        != supporter_generation_contract.digest()
    ):
        raise RuntimeError("study freeze supporter-generation treatment is stale")

    evaluation_unit_contract = generation_contract.get("evaluation_unit_contract") or {}
    if evaluation_unit_contract != external_contract.get("evaluation_unit_contract"):
        raise RuntimeError("generation/external evaluation-unit contracts differ")
    full_expected_units = expected_external_units(
        args.evoemo,
        seeds=[int(v) for v in generation_contract["seeds"]],
        simulator_id=str(generation_contract["simulator_id"]),
        turn_indices=[int(v) for v in external_contract["turn_indices"]],
    )
    if (
        evaluation_unit_contract.get("expected_unit_count") != len(full_expected_units)
        or evaluation_unit_contract.get("expected_units_sha256")
        != sha256_text(canonical_json(full_expected_units))
    ):
        raise RuntimeError("frozen pointwise schema-smoke unit universe is stale")

    smoke_unit_raw = tuple(smoke_contract.get("unit") or ())
    if len(smoke_unit_raw) != 5:
        raise RuntimeError("frozen pointwise schema-smoke unit is malformed")
    smoke_unit = (
        str(smoke_unit_raw[0]),
        int(smoke_unit_raw[1]),
        int(smoke_unit_raw[2]),
        str(smoke_unit_raw[3]),
        int(smoke_unit_raw[4]),
    )
    forced_verification = require_v1_5_forced_swap_canary(
        args.forced_swap_summary,
        args.forced_swap_attestation,
        study_freeze_sha256=freeze_sha256,
        full_expected_units=full_expected_units,
        contract=external_contract.get("forced_swap") or {},
    )
    selected_units = sorted(forced_verification["excluded_units"])
    if smoke_unit != selected_units[0]:
        raise RuntimeError(
            "PM-v1.5 schema-smoke unit must be the first frozen canary unit"
        )
    if smoke_contract.get(
        "forced_swap_selected_units_sha256"
    ) != sha256_text(canonical_json(selected_units)):
        raise RuntimeError("frozen schema-smoke hold-out set is stale")

    condition = str(smoke_contract["condition"])
    generation_verification = require_artifact_attestation(
        args.generation_attestation,
        required_stage="evoemo_pm_v2_generation",
        required_output_paths={"turns": args.turns},
        expected_freeze_sha256=freeze_sha256,
    )
    generation_attestation = read_json(args.generation_attestation)
    parameters = generation_attestation.get("parameters") or {}
    expected_generation_parameters = {
        "condition": condition,
        "protocol": supporter_generation_contract.version,
        "simulator_id": generation_contract["simulator_id"],
        "max_turns": generation_contract["max_turns"],
        "seeds": generation_contract["seeds"],
        "evaluation_unit_contract": evaluation_unit_contract,
        "paid_generation_scope": "frozen_evaluation_turns_only",
    }
    for key, expected in expected_generation_parameters.items():
        if parameters.get(key) != expected:
            raise RuntimeError(f"schema-smoke generation attestation mismatch: {key}")
    if not generation_verification.get("ok"):
        raise RuntimeError("schema-smoke generation attestation failed")

    frozen_endpoints = smoke_contract.get("judge_endpoints") or []
    endpoints = [endpoint_from_config(experiment, str(row["name"])) for row in frozen_endpoints]
    if len(endpoints) != 2:
        raise RuntimeError("schema-smoke requires exactly two frozen endpoints")
    for endpoint, frozen in zip(endpoints, frozen_endpoints):
        endpoint_hash = sha256_text(
            canonical_json(
                {"model": endpoint.model, "family": endpoint.family, "base_url": endpoint.base_url}
            )
        )
        if endpoint_hash != frozen.get("sha256"):
            raise RuntimeError("schema-smoke endpoint changed after study freeze")

    result = run_external_pointwise_schema_smoke(
        evoemo_path=args.evoemo,
        study_freeze_path=args.freeze,
        study_freeze_sha256=freeze_sha256,
        turn_path=args.turns,
        generation_attestation_path=args.generation_attestation,
        forced_swap_summary_path=args.forced_swap_summary,
        forced_swap_attestation_path=args.forced_swap_attestation,
        full_expected_units=full_expected_units,
        smoke_unit=smoke_unit,
        condition=condition,
        endpoints=endpoints,
        pricing_usd_per_mtok=smoke_contract["judge_pricing_usd_per_mtok"],
        api_cost_planning=smoke_contract["api_cost_planning"],
        judge_seed=int(smoke_contract["judge_seed"]),
        estimated_response_output_tokens=int(smoke_contract["estimated_response_output_tokens"]),
        estimated_risk_output_tokens=int(smoke_contract["estimated_risk_output_tokens"]),
        out_dir=args.out_dir,
        run=bool(args.run),
        accept_cost_estimate_sha256=args.accept_cost_estimate_sha256,
        max_api_calls=args.max_api_calls,
        max_estimated_usd=args.max_estimated_usd,
        max_input_tokens_per_call=args.max_input_tokens_per_call,
        overwrite=args.overwrite,
    )
    print(result)


if __name__ == "__main__":
    main()
