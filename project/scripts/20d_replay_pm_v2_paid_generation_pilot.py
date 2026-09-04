#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
)
from metacom_pm.pm_v2_contracts import ResourceNeedRegime
from metacom_pm.pm_v2_data import (
    GeneratedBundleDraft,
    compile_generation_draft,
)
from metacom_pm.pm_v2_generation_pilot import (
    GENERATION_PILOT_REPLAY_PROTOCOL,
    GENERATION_PILOT_STAGE,
    build_generation_compatibility_contract,
    read_generation_seed_dialogues_from_contract,
    require_generation_compatibility_attestation,
    validate_generation_pilot_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = (
    ROOT / "outputs" / "pm_v2_generation_compatibility_pilot_orthogonal_v6"
)
DEFAULT_OUT = (
    ROOT
    / "outputs"
    / "pm_v2_generation_compatibility_pilot_deterministic_evidence_v7"
)


def _exact_source_attempt(
    *, source_dir: Path
) -> tuple[dict, dict, dict, dict]:
    ledger_path = source_dir / "physical_attempt_ledger.jsonl"
    failure_path = source_dir / "failed_attempt_01.json"
    call_plan_path = source_dir / "call_plan.jsonl"
    rows = list(iter_jsonl(ledger_path))
    if len(rows) != 2 or [row.get("event") for row in rows] != [
        "STARTED",
        "FAILED",
    ]:
        raise RuntimeError(
            "compiler replay requires exactly one immutable STARTED -> FAILED "
            "source attempt"
        )
    started, failed = rows
    if (
        started.get("call_key") != failed.get("call_key")
        or started.get("attempt_key") != failed.get("attempt_key")
        or failed.get("usage") is None
        or failed.get("request_hash") is None
    ):
        raise RuntimeError("source physical-attempt lineage is incomplete")
    failure = read_json(failure_path)
    if canonical_json(failed.get("result")) != canonical_json(failure):
        raise RuntimeError("saved failure differs from immutable ledger result")
    plan = list(iter_jsonl(call_plan_path))
    if len(plan) != 1 or plan[0].get("physical_call_key") != failed.get("call_key"):
        raise RuntimeError("source call plan differs from its physical ledger")
    return started, failed, failure, plan[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one paid schema-valid PM-v2 provider response through the "
            "current deterministic evidence compiler without another API call."
        )
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument(
        "--pm-v2-config", type=Path, default=ROOT / "configs/pm_v2.yaml"
    )
    parser.add_argument(
        "--seed-dialogues",
        type=Path,
        default=ROOT / "data/pm_v2/train_seed_dialogues.jsonl",
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.source_dir = args.source_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    experiment = load_config(args.config)
    pm_config = load_config(args.pm_v2_config)
    generation = dict(pm_config["data_generation"])
    endpoint = endpoint_from_config(
        experiment, str(generation["generator_endpoint"])
    )
    full_user_count = sum(
        int(generation[key])
        for key in ("train_users", "calibration_users", "internal_test_users")
    )
    pricing = dict(generation["pricing_usd_per_mtok"])
    api_cost = dict(pm_config["api_cost_planning"])
    contract = build_generation_compatibility_contract(
        project_root=ROOT,
        experiment_config_path=args.config,
        pm_v2_config_path=args.pm_v2_config,
        seed_dialogues_path=args.seed_dialogues,
        endpoint=endpoint,
        base_generation_seed=int(generation["base_seed"]),
        full_user_count=full_user_count,
        input_token_safety_factor=float(
            api_cost["input_token_safety_factor"]
        ),
        fail_on_reported_input_overrun=bool(
            api_cost["fail_on_reported_input_overrun"]
        ),
        input_usd_per_mtok=float(pricing["input"]),
        output_usd_per_mtok=float(pricing["output"]),
    )

    started, failed, failure, source_plan = _exact_source_attempt(
        source_dir=args.source_dir
    )
    parsed_payload = failure.get("parsed_payload")
    provider_response = failure.get("provider_response")
    if not isinstance(parsed_payload, dict) or not isinstance(provider_response, dict):
        raise RuntimeError("source attempt lacks preserved provider payload/response")
    draft = GeneratedBundleDraft.model_validate(parsed_payload)
    try:
        raw_payload = json.loads(
            provider_response["choices"][0]["message"]["content"]
        )
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("source raw response cannot reconstruct JSON") from exc
    if raw_payload != draft.model_dump(mode="json"):
        raise RuntimeError("source raw response differs from parsed provider draft")

    seed_dialogue = read_generation_seed_dialogues_from_contract(contract)[
        int(contract["held_out_seed_index"])
    ]
    bundle = compile_generation_draft(
        draft=draft,
        seed_dialogue=seed_dialogue,
        user_id=str(contract["pilot_user_id"]),
        semantic_families=[str(value) for value in contract["semantic_families"]],
        regimes=list(ResourceNeedRegime),
    )
    usage = {
        key: int(failed["usage"][key])
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    replay_binding = {
        "protocol": GENERATION_PILOT_REPLAY_PROTOCOL,
        "source_stage": str(failed["stage"]),
        "source_call_key": str(failed["call_key"]),
        "source_attempt_key": str(failed["attempt_key"]),
        "source_request_hash": str(failed["request_hash"]),
        "source_prompt_sha256": str(failed["prompt_sha256"]),
        "source_provider_response_sha256": sha256_text(
            canonical_json(provider_response)
        ),
        "source_provider_draft_sha256": sha256_text(
            canonical_json(draft.model_dump(mode="json"))
        ),
        "provider_schema_sha256": contract["schema_sha256"],
        "compatibility_contract_sha256": contract["contract_sha256"],
        "source_physical_attempts": 1,
        "new_physical_api_attempts": 0,
    }
    bundle.provenance.update(
        {
            "generator_model": str(
                provider_response.get("model") or endpoint.model
            ),
            "generator_family": endpoint.family,
            "request_hash": str(failed["request_hash"]),
            "reported_usage": usage,
            "provider_draft": draft.model_dump(mode="json"),
            "provider_response": provider_response,
            "provider_response_sha256": sha256_text(
                canonical_json(provider_response)
            ),
            "generation_compatibility_contract_sha256": contract[
                "contract_sha256"
            ],
            "compatibility_replay": replay_binding,
        }
    )
    bundle_report = validate_generation_pilot_bundle(bundle, contract)
    if bundle_report["status"] != "PASS":
        raise RuntimeError(f"offline compiler replay failed: {bundle_report}")

    replay = {
        **replay_binding,
        "status": "PASS",
        "source_payload_validates_current_provider_schema": True,
        "raw_response_reconstructs_provider_draft": True,
        "provider_oracle_evidence_used": False,
        "provider_oracle_rationale_used": False,
        "source_ledger_started_event": started["event"],
        "source_ledger_terminal_event": failed["event"],
        "source_usage": usage,
        "source_call_plan": source_plan,
        "surface_selection": bundle.provenance["surface_selection"],
        "evidence_blueprint_sha256": bundle.provenance[
            "evidence_blueprint_sha256"
        ],
    }
    summary = {
        "status": "PASS",
        "compatibility_mode": GENERATION_PILOT_REPLAY_PROTOCOL,
        "compatibility_contract_sha256": contract["contract_sha256"],
        "source_physical_attempts": 1,
        "new_physical_api_attempts": 0,
        "bundle_validation": bundle_report,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = args.out_dir / "pilot_bundle.json"
    summary_path = args.out_dir / "summary.json"
    replay_path = args.out_dir / "compiler_replay_provenance.json"
    attestation_path = args.out_dir / "artifact_attestation.json"
    for path in (bundle_path, summary_path, replay_path, attestation_path):
        if path.exists() and not args.overwrite:
            raise RuntimeError(f"refusing to overwrite replay artifact: {path}")
    write_json(bundle_path, bundle.model_dump(mode="json"))
    write_json(summary_path, summary)
    write_json(replay_path, replay)
    create_artifact_attestation(
        attestation_path,
        stage=GENERATION_PILOT_STAGE,
        inputs={
            "experiment_config": args.config,
            "pm_v2_config": args.pm_v2_config,
            "seed_dialogues": args.seed_dialogues,
            "source_run_manifest": args.source_dir / "run_manifest.json",
            "source_cost_estimate": args.source_dir / "cost_estimate.json",
            "source_call_plan": args.source_dir / "call_plan.jsonl",
            "source_physical_attempt_ledger": (
                args.source_dir / "physical_attempt_ledger.jsonl"
            ),
            "source_failed_attempt": args.source_dir / "failed_attempt_01.json",
        },
        outputs={
            "pilot_bundle": (bundle_path, False),
            "summary": (summary_path, False),
            "compiler_replay_provenance": (replay_path, False),
        },
        parameters={
            "compatibility_mode": GENERATION_PILOT_REPLAY_PROTOCOL,
            "compatibility_contract": contract,
            "compatibility_contract_sha256": contract["contract_sha256"],
        },
        expected={
            "source_physical_attempts": 1,
            "new_physical_api_attempts": 0,
            "regimes": len(ResourceNeedRegime),
            "provider_schema_sha256": contract["schema_sha256"],
        },
    )
    result = require_generation_compatibility_attestation(
        attestation_path, expected_contract=contract
    )
    print(result)


if __name__ == "__main__":
    main()
