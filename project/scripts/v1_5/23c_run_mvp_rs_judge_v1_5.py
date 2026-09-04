#!/usr/bin/env python3
"""Preflight or run the resumable minimum RS judge controls/outcomes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Any

from metacom_pm.api import make_client, require_reported_usage
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.io import (
    append_jsonl,
    canonical_json,
    dict_field_diff,
    iter_jsonl,
    read_json,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from metacom_pm.v1_5_mvp_judge import (
    ImmediateSupportJudgment,
    PointwiseRiskJudgment,
)
from metacom_pm.v1_5_mvp_judge_runner import (
    MVP_JUDGE_RUNNER_PROTOCOL,
    aggregate_outcome_measurement,
    build_control_call_plan,
    build_outcome_call_plan,
    qualify_controls,
    validate_parsed_judgment,
    validate_rs_judge_inputs,
)


ROOT = Path(__file__).resolve().parents[2]


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [dict(row) for row in iter_jsonl(path)]


def _endpoint_record(name: str, endpoint: Any) -> dict[str, Any]:
    return {
        "name": name,
        "base_url": endpoint.base_url,
        "model": endpoint.model,
        "family": endpoint.family or endpoint.model,
        "transport": endpoint.transport,
    }


def _persist_rows_exact(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.is_file():
        if _rows(path) != rows:
            raise RuntimeError(f"existing judge plan drifted: {path}")
        return
    write_jsonl(path, rows)


def _load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    plan_report = read_json(args.plan_dir / "plan_report.json")
    generation_summary = read_json(
        args.generation_dir / "generation_summary.json"
    )
    runtime_path = Path(str(plan_report["runtime_states_path"]))
    validated = validate_rs_judge_inputs(
        plan_report=plan_report,
        generation_summary=generation_summary,
        selected_states=_rows(args.plan_dir / "selected_states.jsonl"),
        generation_plan=_rows(args.plan_dir / "call_plan.jsonl"),
        generation_outcomes=_rows(
            args.generation_dir / "generation_outcomes.jsonl"
        ),
        runtime_states=_rows(runtime_path),
    )
    return {
        **validated,
        "lineage": {
            "plan_report": str(args.plan_dir / "plan_report.json"),
            "plan_report_sha256": sha256_file(
                args.plan_dir / "plan_report.json"
            ),
            "selected_states_sha256": sha256_file(
                args.plan_dir / "selected_states.jsonl"
            ),
            "generation_plan_sha256": sha256_file(
                args.plan_dir / "call_plan.jsonl"
            ),
            "generation_outcomes_sha256": sha256_file(
                args.generation_dir / "generation_outcomes.jsonl"
            ),
            "generation_summary_sha256": sha256_file(
                args.generation_dir / "generation_summary.json"
            ),
            "runtime_states_sha256": sha256_file(runtime_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("controls", "outcomes"), required=True
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=ROOT / "configs/experiment.yaml",
    )
    parser.add_argument("--judge-endpoint", default="training_judge")
    parser.add_argument(
        "--qualification-contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/minimum_rs_judge_qualification_v1.json",
    )
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_minimum_rs_clean_pair_pilot_v1",
    )
    parser.add_argument(
        "--generation-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_minimum_rs_clean_pair_pilot_v1_execution",
    )
    parser.add_argument(
        "--control-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_minimum_rs_judge_controls_v1",
    )
    parser.add_argument(
        "--qualification-report",
        type=Path,
        help=(
            "Material-rule qualification report for outcome judging. "
            "Defaults to CONTROL_DIR/qualification_report_material_v2.json."
        ),
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--max-new-calls", type=int)
    parser.add_argument(
        "--inter-call-delay-seconds",
        type=float,
        default=1.0,
        help="Polite delay after each completed provider call.",
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = (
            args.control_dir
            if args.phase == "controls"
            else ROOT / "outputs/pm_v1_5_minimum_rs_judging_v1"
        )
    if args.qualification_report is None:
        args.qualification_report = (
            args.control_dir / "qualification_report_material_v2.json"
        )
    experiment = load_config(args.experiment_config)
    endpoint = endpoint_from_config(experiment, args.judge_endpoint)
    endpoint_record = _endpoint_record(args.judge_endpoint, endpoint)
    qualification_contract = read_json(args.qualification_contract)
    inputs = _load_inputs(args)
    if args.phase == "controls":
        plan = build_control_call_plan(
            qualification_contract=qualification_contract,
            endpoint=endpoint_record,
        )
        qualification_report = None
    else:
        qualification_path = args.qualification_report
        if not qualification_path.is_file():
            raise RuntimeError(
                "outcome judging requires a completed control qualification"
            )
        qualification_report = read_json(qualification_path)
        if qualification_report.get("qualified") is not True:
            raise RuntimeError("judge controls did not qualify")
        control_plan = _rows(args.control_dir / "call_plan.jsonl")
        if any(
            dict(row.get("endpoint") or {}) != endpoint_record
            for row in control_plan
        ):
            raise RuntimeError(
                "outcome judge endpoint differs from the qualified endpoint"
            )
        plan = build_outcome_call_plan(
            pairs=inputs["pairs"],
            endpoint=endpoint_record,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.out_dir / "call_plan.jsonl"
    _persist_rows_exact(plan_path, plan)
    write_json(args.out_dir / "input_data_audit.json", inputs["audit"])
    write_json(args.out_dir / "input_lineage.json", inputs["lineage"])

    logical_calls = len(plan)
    max_new_calls = (
        logical_calls if args.max_new_calls is None else args.max_new_calls
    )
    if max_new_calls < 0:
        raise ValueError("--max-new-calls must be non-negative")
    if args.inter_call_delay_seconds < 0:
        raise ValueError("--inter-call-delay-seconds must be non-negative")
    estimated_input_tokens = sum(
        int(row["estimated_input_tokens"]) for row in plan
    )
    preflight = {
        "protocol": MVP_JUDGE_RUNNER_PROTOCOL,
        "status": (
            "READY_FOR_CONTROL_QUALIFICATION"
            if args.phase == "controls"
            else "READY_FOR_TRAIN_ONLY_OUTCOME_JUDGING"
        ),
        "phase": args.phase,
        "endpoint": endpoint_record,
        "logical_calls": logical_calls,
        "quality_calls": sum(row["role"] == "quality" for row in plan),
        "risk_calls": sum(row["role"] == "risk" for row in plan),
        "estimated_input_tokens": estimated_input_tokens,
        "maximum_output_tokens": sum(
            int(row["request_parameters"]["max_output_tokens"])
            for row in plan
        ),
        "input_data_audit_status": inputs["audit"]["status"],
        "control_qualification_required_before_outcomes": True,
        "control_qualification_passed": (
            qualification_report is not None
            and qualification_report.get("qualified") is True
        ),
        "resumable": True,
        "one_shot_execution_required": False,
        "api_key_environment_variable": endpoint.api_key_env,
        "api_key_present": bool(os.environ.get(endpoint.api_key_env, "")),
        "run_requested": args.run,
        "inter_call_delay_seconds": args.inter_call_delay_seconds,
        "api_calls_made": 0,
    }
    write_json(args.out_dir / "preflight.json", preflight)
    if not args.run:
        print(canonical_json(preflight))
        return
    if not os.environ.get(endpoint.api_key_env, ""):
        raise RuntimeError(
            f"{endpoint.api_key_env} is absent; preflight completed with zero API calls"
        )

    result_path = args.out_dir / "judge_results.jsonl"
    existing = {
        str(row["call_id"]): dict(row)
        for row in (_rows(result_path) if result_path.is_file() else [])
    }
    if len(existing) != (
        len(_rows(result_path)) if result_path.is_file() else 0
    ):
        raise RuntimeError("judge results contain duplicate call_id")
    pending: list[dict[str, Any]] = []
    for row in plan:
        prior = existing.get(str(row["call_id"]))
        if prior is not None:
            if prior.get("prompt_sha256") != row["prompt_sha256"]:
                raise RuntimeError("completed judge result prompt drifted")
            continue
        pending.append(row)
    if len(pending) > max_new_calls:
        raise RuntimeError(
            f"{len(pending)} calls remain but --max-new-calls={max_new_calls}"
        )

    client = make_client(endpoint)
    completed_now = 0
    run_error: Exception | None = None
    try:
        for row in pending:
            schema = (
                ImmediateSupportJudgment
                if row["role"] == "quality"
                else PointwiseRiskJudgment
            )
            request = dict(row["request_parameters"])
            result, parsed = client.chat(
                list(row["messages"]),
                temperature=float(request["temperature"]),
                max_tokens=int(request["max_output_tokens"]),
                seed=int(request["seed"]),
                response_schema=schema,
                retries=2,
            )
            if parsed is None:
                raise RuntimeError("judge returned no structured output")
            parsed_payload = parsed.model_dump(mode="json")
            try:
                normalized = validate_parsed_judgment(
                    call=row,
                    parsed=parsed_payload,
                )
            except ValueError as exc:
                append_jsonl(
                    args.out_dir / "judge_validation_failures.jsonl",
                    {
                        "protocol": MVP_JUDGE_RUNNER_PROTOCOL,
                        "phase": args.phase,
                        "call_id": row["call_id"],
                        "role": row["role"],
                        "record_ids": row["record_ids"],
                        "prompt_sha256": row["prompt_sha256"],
                        "endpoint": endpoint_record,
                        "request_hash": result.request_hash,
                        "parsed": parsed_payload,
                        "validation_error": str(exc),
                        "failed_at": utc_now(),
                    },
                )
                raise
            usage = require_reported_usage(
                result.usage,
                stage=f"minimum_rs_judge_{args.phase}",
            )
            append_jsonl(
                result_path,
                {
                    "protocol": MVP_JUDGE_RUNNER_PROTOCOL,
                    "phase": args.phase,
                    "call_id": row["call_id"],
                    "role": row["role"],
                    "record_ids": row["record_ids"],
                    "order_variant": row.get("order_variant"),
                    "prompt_sha256": row["prompt_sha256"],
                    "endpoint": endpoint_record,
                    "request_hash": result.request_hash,
                    "usage": usage,
                    "latency_ms": result.latency_ms,
                    "provider_finish_reason": result.provider_finish_reason,
                    "normalized_finish_reason": result.normalized_finish_reason,
                    "parsed": normalized,
                    "deterministic_output_normalization": (
                        parsed_payload != normalized
                    ),
                    "deterministic_output_normalized_fields": (
                        dict_field_diff(parsed_payload, normalized)
                    ),
                    "schema_validated": True,
                    "literal_excerpts_validated": True,
                    "completed_at": utc_now(),
                },
            )
            completed_now += 1
            if args.inter_call_delay_seconds:
                time.sleep(args.inter_call_delay_seconds)
    except Exception as exc:
        run_error = exc
    finally:
        client.close()

    results = _rows(result_path)
    summary = {
        "protocol": MVP_JUDGE_RUNNER_PROTOCOL,
        "status": (
            "COMPLETE" if len(results) == logical_calls else "PARTIAL_RESUMABLE"
        ),
        "phase": args.phase,
        "logical_calls": logical_calls,
        "completed_calls": len(results),
        "completed_now": completed_now,
        "remaining_calls": logical_calls - len(results),
        "resumable": True,
        "one_shot_execution_required": False,
        "last_run_error": (
            None
            if run_error is None
            else {
                "type": type(run_error).__name__,
                "message": str(run_error),
            }
        ),
    }
    write_json(args.out_dir / "execution_summary.json", summary)
    if run_error is not None:
        print(canonical_json(summary))
        raise run_error
    if len(results) == logical_calls:
        if args.phase == "controls":
            report = qualify_controls(
                qualification_contract=qualification_contract,
                call_plan=plan,
                results=results,
            )
            write_json(
                args.out_dir / "qualification_report_material_v2.json",
                report,
            )
            summary["qualification_status"] = report["status"]
        else:
            assert qualification_report is not None
            measurement = aggregate_outcome_measurement(
                qualification_report=qualification_report,
                call_plan=plan,
                results=results,
                pairs=inputs["pairs"],
            )
            quality_rows = list(measurement.pop("quality_pair_rows"))
            risk_rows = list(measurement.pop("risk_event_rows"))
            write_jsonl(args.out_dir / "resolved_quality_pairs.jsonl", quality_rows)
            write_jsonl(args.out_dir / "atomic_risk_events.jsonl", risk_rows)
            write_json(args.out_dir / "measurement_report.json", measurement)
            summary["measurement_status"] = measurement["status"]
        write_json(args.out_dir / "execution_summary.json", summary)
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
