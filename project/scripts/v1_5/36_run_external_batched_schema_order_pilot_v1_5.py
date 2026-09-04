#!/usr/bin/env python3
"""Run PM-v1.5's frozen eight-call batched schema/order transport pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from metacom_pm.artifacts import require_artifact_attestation
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.freeze import require_study_freeze
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import canonical_json, iter_jsonl, read_json, sha256_file, sha256_text
from metacom_pm.pm_v2_external_eval import expected_external_units
from metacom_pm.pm_v22_reference_baselines import PMV22_REFERENCE_BASELINE_STAGE
from metacom_pm.v1_5_external_batched import (
    PILOT_PROTOCOL,
    batched_prompt_contract_hash,
    run_v1_5_batched_schema_order_pilot,
)
from metacom_pm.v1_5_forced_swap_canary import require_v1_5_forced_swap_canary


ROOT = Path(__file__).resolve().parents[2]
GENERATION_STAGE = "evoemo_pm_v2_generation"
KNOWN_STAGES = {GENERATION_STAGE, PMV22_REFERENCE_BASELINE_STAGE}


def _require_treatment_bound_turns(
    path: Path,
    *,
    expected_conditions: Sequence[str],
    expected_units: Sequence[tuple[str, int, int, str, int]],
    treatment: Mapping[str, Any],
    treatment_sha256: str,
    fixed_seeker_treatment: Mapping[str, Any],
    fixed_seeker_treatment_sha256: str,
) -> None:
    rows = [dict(row) for row in iter_jsonl(path)]
    conditions = {str(value) for value in expected_conditions}
    if not rows or {str(row.get("condition") or "") for row in rows} != conditions:
        raise RuntimeError("batched pilot turn file does not cover attested conditions")
    expected = sorted(expected_units)
    for condition in sorted(conditions):
        values = [row for row in rows if str(row.get("condition")) == condition]
        units = sorted(
            (
                str(row.get("user_id") or ""),
                int(row.get("topic_index") or 0),
                int(row.get("seed") or 0),
                str(row.get("simulator_id") or ""),
                int(row.get("turn_index") or 0),
            )
            for row in values
        )
        if units != expected:
            raise RuntimeError(f"batched pilot turns for {condition} are incomplete")
        if any(
            row.get("supporter_generation_treatment") != dict(treatment)
            or row.get("supporter_generation_treatment_sha256") != treatment_sha256
            or row.get("fixed_seeker_generation_treatment")
            != dict(fixed_seeker_treatment)
            or row.get("fixed_seeker_generation_treatment_sha256")
            != fixed_seeker_treatment_sha256
            or row.get("normalized_finish_reason") != "complete"
            or not str(row.get("supporter_message") or "").strip()
            for row in values
        ):
            raise RuntimeError(f"batched pilot turns for {condition} mix treatments")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--pm-v1-5-config", type=Path, default=ROOT / "configs" / "pm_v1_5.yaml")
    parser.add_argument("--freeze", type=Path, default=ROOT / "outputs" / "pm_v1_5_study_freeze.json")
    parser.add_argument("--evoemo", type=Path, default=ROOT / "data" / "external" / "evo_emo.json")
    parser.add_argument(
        "--turn-paths",
        type=Path,
        nargs="+",
        default=[
            ROOT / "outputs" / "evoemo_pm_v1_5_reference_baselines" / "turns.jsonl",
            ROOT / "outputs" / "evoemo_pm_v1_5" / "turns.jsonl",
            ROOT / "outputs" / "evoemo_pm_v1_5_cost_matched_fixed" / "turns.jsonl",
            ROOT / "outputs" / "evoemo_pm_v1_5_me_r0_fixed" / "turns.jsonl",
        ],
    )
    parser.add_argument(
        "--generation-attestations",
        type=Path,
        nargs="+",
        default=[
            ROOT / "outputs" / "evoemo_pm_v1_5_reference_baselines" / "artifact_attestation.json",
            ROOT / "outputs" / "evoemo_pm_v1_5" / "artifact_attestation.json",
            ROOT / "outputs" / "evoemo_pm_v1_5_cost_matched_fixed" / "artifact_attestation.json",
            ROOT / "outputs" / "evoemo_pm_v1_5_me_r0_fixed" / "artifact_attestation.json",
        ],
    )
    parser.add_argument(
        "--forced-swap-summary",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_forced_swap_canary" / "summary.json",
    )
    parser.add_argument(
        "--forced-swap-attestation",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_forced_swap_canary" / "artifact_attestation.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_external_batched_schema_order_pilot",
    )
    parser.add_argument("--max-api-calls", type=int, default=8)
    parser.add_argument("--max-estimated-usd", type=float, default=2.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=24000)
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.run and args.overwrite:
        raise RuntimeError("paid batched pilot runs prohibit --overwrite")
    if len(args.turn_paths) != len(args.generation_attestations):
        raise ValueError("each batched pilot turn file requires one attestation")

    experiment = load_config(args.config)
    pm_config = load_config(args.pm_v1_5_config)
    if pm_config.get("version") != "pm-v1.5":
        raise RuntimeError("batched pilot requires the honest pm-v1.5 config")
    supporter = SupporterGenerationContract.from_config(pm_config)
    verification = require_study_freeze(
        args.freeze,
        release_root=ROOT,
        config_path=args.config,
        required_files=[args.pm_v1_5_config, args.evoemo],
    )
    freeze_sha = str(verification["freeze_sha256"])
    notes = json.loads(args.freeze.read_text(encoding="utf-8")).get("notes") or {}
    generation = notes.get("generation_contract") or {}
    external = notes.get("external_evaluation_contract") or {}
    pilot = external.get("batched_schema_order_pilot") or {}
    if (
        pilot.get("protocol") != PILOT_PROTOCOL
        or pilot.get("batched_prompt_contract_sha256")
        != batched_prompt_contract_hash()
        or not bool(pilot.get("required_before_full_external_client_creation"))
    ):
        raise RuntimeError("study freeze lacks the current batched schema pilot")
    if (
        generation.get("supporter_generation_treatment") != supporter.payload()
        or generation.get("supporter_generation_treatment_sha256") != supporter.digest()
    ):
        raise RuntimeError("frozen supporter-generation treatment is stale")
    unit_contract = generation.get("evaluation_unit_contract") or {}
    if unit_contract != external.get("evaluation_unit_contract"):
        raise RuntimeError("generation/external evaluation-unit contracts differ")
    full_units = expected_external_units(
        args.evoemo,
        seeds=[int(value) for value in generation["seeds"]],
        simulator_id=str(generation["simulator_id"]),
        turn_indices=[int(value) for value in external["turn_indices"]],
    )
    if (
        unit_contract.get("expected_unit_count") != len(full_units)
        or unit_contract.get("expected_units_sha256")
        != sha256_text(canonical_json(full_units))
    ):
        raise RuntimeError("frozen batched pilot unit universe is stale")
    smoke_unit = tuple(pilot.get("unit") or ())
    if len(smoke_unit) != 5:
        raise RuntimeError("frozen batched pilot unit is malformed")
    smoke_unit = (
        str(smoke_unit[0]),
        int(smoke_unit[1]),
        int(smoke_unit[2]),
        str(smoke_unit[3]),
        int(smoke_unit[4]),
    )
    forced = require_v1_5_forced_swap_canary(
        args.forced_swap_summary,
        args.forced_swap_attestation,
        study_freeze_sha256=freeze_sha,
        full_expected_units=full_units,
        contract=external.get("forced_swap") or {},
    )
    selected = sorted(forced["excluded_units"])
    if smoke_unit != selected[0] or pilot.get(
        "forced_swap_selected_units_sha256"
    ) != sha256_text(canonical_json(selected)):
        raise RuntimeError("batched pilot is not bound to the frozen canary set")

    attested_conditions: set[str] = set()
    for turns, attestation in zip(args.turn_paths, args.generation_attestations):
        raw = read_json(attestation)
        stage = str(raw.get("stage") or "")
        if stage not in KNOWN_STAGES:
            raise RuntimeError(f"unknown V1.5 generation stage: {stage!r}")
        require_artifact_attestation(
            attestation,
            required_stage=stage,
            required_output_paths={"turns": turns},
            expected_freeze_sha256=(freeze_sha if stage == GENERATION_STAGE else None),
        )
        parameters = raw.get("parameters") or {}
        values = [
            str(value)
            for value in (parameters.get("conditions") or [parameters.get("condition")])
            if value
        ]
        if not values:
            raise RuntimeError("batched pilot generation attestation has no conditions")
        attested_conditions.update(values)
        if stage == PMV22_REFERENCE_BASELINE_STAGE:
            freeze_input = (raw.get("inputs") or {}).get("strategy_bank_approval") or {}
            if (
                Path(str(freeze_input.get("path") or "")).resolve() != args.freeze.resolve()
                or freeze_input.get("sha256") != sha256_file(args.freeze)
            ):
                raise RuntimeError("reference-baseline pilot input is not freeze-bound")
        _require_treatment_bound_turns(
            turns,
            expected_conditions=values,
            expected_units=full_units,
            treatment=generation["supporter_generation_treatment"],
            treatment_sha256=str(generation["supporter_generation_treatment_sha256"]),
            fixed_seeker_treatment=generation["fixed_seeker_generation_treatment"],
            fixed_seeker_treatment_sha256=str(
                generation["fixed_seeker_generation_treatment_sha256"]
            ),
        )
    conditions = [str(value) for value in external["conditions"]]
    if attested_conditions != set(conditions):
        raise RuntimeError("batched pilot inputs do not cover all frozen conditions")

    frozen_endpoints = pilot.get("judge_endpoints") or []
    endpoints = [
        endpoint_from_config(experiment, str(row["name"])) for row in frozen_endpoints
    ]
    for endpoint, frozen in zip(endpoints, frozen_endpoints):
        current = sha256_text(
            canonical_json(
                {
                    "model": endpoint.model,
                    "family": endpoint.family,
                    "base_url": endpoint.base_url,
                }
            )
        )
        if current != frozen.get("sha256"):
            raise RuntimeError("batched pilot endpoint changed after the freeze")

    result = run_v1_5_batched_schema_order_pilot(
        evoemo_path=args.evoemo,
        study_freeze_path=args.freeze,
        study_freeze_sha256=freeze_sha,
        turn_paths=args.turn_paths,
        generation_attestation_paths=args.generation_attestations,
        full_expected_units=full_units,
        smoke_unit=smoke_unit,
        conditions=conditions,
        endpoints=endpoints,
        contract=pilot,
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
