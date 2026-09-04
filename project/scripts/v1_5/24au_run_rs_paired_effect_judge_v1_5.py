#!/usr/bin/env python3
"""Run the qualified train-only judge on corrected RS paired effects."""

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
    build_outcome_call_plan,
    deterministic_risk_applicability,
    validate_parsed_judgment,
)
from metacom_pm.v1_5_mvp_rs_pilot import explicit_boundary_cue


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-rs-paired-effect-qualified-judge-v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _endpoint_record(name: str, endpoint: Any) -> dict[str, Any]:
    return {
        "name": name,
        "base_url": endpoint.base_url,
        "model": endpoint.model,
        "family": endpoint.family or endpoint.model,
        "transport": endpoint.transport,
    }


def _ungrounded_violation_to_insufficient(
    *, call: dict[str, Any], parsed: dict[str, Any]
) -> dict[str, Any]:
    """Fail unsupported risk events to missing instead of inventing evidence."""

    if str(call["role"]) != "risk":
        return parsed
    response = str(call["response"])
    evidence_surface = canonical_json(
        {
            "visible_dialogue": dict(call["visible_dialogue"]),
            "selected_evidence": call.get("selected_evidence") or {},
        }
    )
    repaired = dict(parsed)
    findings: list[dict[str, Any]] = []
    for source in list(parsed.get("findings") or []):
        finding = dict(source)
        if finding.get("verdict") == "violation":
            response_excerpt = str(finding.get("response_excerpt") or "")
            evidence_excerpt = str(finding.get("evidence_excerpt") or "")
            response_exact = (
                response_excerpt in {"[none]", "[not applicable]"}
                or response_excerpt in response
            )
            evidence_exact = (
                evidence_excerpt in {"[none]", "[not applicable]"}
                or evidence_excerpt in evidence_surface
            )
            if not response_exact or not evidence_exact:
                finding.update(
                    {
                        "verdict": "insufficient_evidence",
                        "severity": None,
                        "response_excerpt": "[none]",
                        "evidence_excerpt": "[none]",
                        "reason": (
                            "Original violation lacked exact auditable "
                            "response/evidence excerpts; excluded rather than "
                            "counted as a material event."
                        ),
                    }
                )
        findings.append(finding)
    repaired["findings"] = findings
    return repaired


def _pairs(
    *,
    selected: list[dict[str, Any]],
    call_plan: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_by_pair = {str(row["pair_id"]): row for row in selected}
    plan_by_key = {
        (str(row["pair_id"]), str(row["arm"])): row for row in call_plan
    }
    outcome_by_key = {
        (str(row["pair_id"]), str(row["arm"])): row for row in outcomes
    }
    expected = {
        (pair_id, arm)
        for pair_id in selected_by_pair
        for arm in ("R0", "RS")
    }
    if (
        len(selected_by_pair) != 32
        or set(plan_by_key) != expected
        or set(outcome_by_key) != expected
    ):
        raise RuntimeError("RS pair/arm inputs are incomplete or duplicated")
    pairs: list[dict[str, Any]] = []
    prompt_tokens = 0
    completion_tokens = 0
    for pair_id, selected_row in sorted(selected_by_pair.items()):
        dialogue = list(selected_row["visible_dialogue"])
        if not dialogue or dialogue[-1]["speaker"] != "seeker":
            raise RuntimeError(f"{pair_id}: visible dialogue does not end in seeker")
        history = [
            {
                "role": (
                    "user" if row["speaker"] == "seeker" else "assistant"
                ),
                "content": str(row["content"]),
            }
            for row in dialogue[:-1]
        ]
        current = str(dialogue[-1]["content"])
        visible = {
            "current_session_summary": "",
            "current_session_history": history,
            "current_user_text": current,
        }
        arms: dict[str, dict[str, Any]] = {}
        for arm in ("R0", "RS"):
            plan = plan_by_key[(pair_id, arm)]
            outcome = outcome_by_key[(pair_id, arm)]
            if (
                outcome["state_id"] != selected_row["state_id"]
                or outcome["user_id"] != selected_row["user_id"]
                or outcome["prompt_sha256"] != plan["prompt_sha256"]
                or outcome["normalized_finish_reason"] != "complete"
            ):
                raise RuntimeError(f"{pair_id}/{arm}: generation lineage drift")
            usage = dict(outcome["usage"])
            prompt_tokens += int(usage["prompt_tokens"])
            completion_tokens += int(usage["completion_tokens"])
            arms[arm] = {
                "response": str(outcome["response"]),
                "prompt_tokens": int(usage["prompt_tokens"]),
                "completion_tokens": int(usage["completion_tokens"]),
                "selected_strategy_card_id": outcome.get(
                    "selected_strategy_card_id"
                ),
                "selected_strategy_family": outcome.get(
                    "selected_strategy_family"
                ),
            }
        if arms["R0"]["selected_strategy_card_id"] is not None:
            raise RuntimeError(f"{pair_id}: R0 contains a strategy card")
        if not arms["RS"]["selected_strategy_card_id"]:
            raise RuntimeError(f"{pair_id}: RS did not realize Top-1")
        boundary = explicit_boundary_cue(current) or "none"
        pairs.append(
            {
                "pair_id": pair_id,
                "state_id": selected_row["state_id"],
                "user_id": selected_row["user_id"],
                "boundary_cue": boundary,
                "visible_dialogue": visible,
                "selected_evidence": {},
                "risk_applicability": deterministic_risk_applicability(
                    visible_dialogue=visible,
                    selected_evidence={},
                ),
                "arms": arms,
            }
        )
    return pairs, {
        "protocol": PROTOCOL,
        "status": "PASS",
        "pair_count": len(pairs),
        "response_count": len(outcomes),
        "independent_user_groups": len({row["user_id"] for row in pairs}),
        "generator_prompt_tokens": prompt_tokens,
        "generator_completion_tokens": completion_tokens,
        "same_state_r0_rs_complete": True,
        "outcomes_not_used_for_state_selection": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_rs_paired_effect_v1",
    )
    parser.add_argument(
        "--generation-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_rs_paired_effect_v1_execution",
    )
    parser.add_argument(
        "--control-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_minimum_rs_judge_controls_v1_openai_mini",
    )
    parser.add_argument(
        "--qualification-report",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_minimum_rs_judge_controls_v1_openai_mini/"
        "qualification_report_material_v2.json",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=ROOT / "configs/experiment.yaml",
    )
    parser.add_argument(
        "--judge-endpoint", default="training_judge_openai_mini"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_rs_paired_effect_v1_judging",
    )
    parser.add_argument("--max-new-calls", type=int, default=128)
    parser.add_argument(
        "--inter-call-delay-seconds", type=float, default=0.0
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    qualification = read_json(args.qualification_report)
    if qualification.get("qualified") is not True:
        raise RuntimeError("the selected train-only judge is not qualified")
    experiment = load_config(args.experiment_config)
    endpoint = endpoint_from_config(experiment, args.judge_endpoint)
    endpoint_record = _endpoint_record(args.judge_endpoint, endpoint)
    control_plan = _rows(args.control_dir / "call_plan.jsonl")
    if any(
        dict(row.get("endpoint") or {}) != endpoint_record
        for row in control_plan
    ):
        raise RuntimeError("judge endpoint differs from qualified controls")

    pairs, audit = _pairs(
        selected=_rows(args.plan_dir / "selected_states.jsonl"),
        call_plan=_rows(args.plan_dir / "call_plan.jsonl"),
        outcomes=_rows(args.generation_dir / "generation_outcomes.jsonl"),
    )
    plan = build_outcome_call_plan(pairs=pairs, endpoint=endpoint_record)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.out_dir / "call_plan.jsonl"
    if plan_path.is_file() and _rows(plan_path) != plan:
        raise RuntimeError("existing paired-effect judge plan drifted")
    if not plan_path.is_file():
        write_jsonl(plan_path, plan)
    write_json(args.out_dir / "input_data_audit.json", audit)
    write_json(
        args.out_dir / "input_lineage.json",
        {
            "protocol": PROTOCOL,
            "selected_states_sha256": sha256_file(
                args.plan_dir / "selected_states.jsonl"
            ),
            "generation_plan_sha256": sha256_file(
                args.plan_dir / "call_plan.jsonl"
            ),
            "generation_outcomes_sha256": sha256_file(
                args.generation_dir / "generation_outcomes.jsonl"
            ),
            "qualification_report_sha256": sha256_file(
                args.qualification_report
            ),
        },
    )
    preflight = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_TRAIN_ONLY_OUTCOME_JUDGING",
        "logical_calls": len(plan),
        "quality_calls": sum(row["role"] == "quality" for row in plan),
        "risk_calls": sum(row["role"] == "risk" for row in plan),
        "qualified_control_report": qualification["status"],
        "endpoint": endpoint_record,
        "api_key_environment_variable": endpoint.api_key_env,
        "api_key_present": bool(os.environ.get(endpoint.api_key_env, "")),
        "run_requested": args.run,
        "api_calls_made": 0,
    }
    write_json(args.out_dir / "preflight.json", preflight)
    if not args.run:
        print(canonical_json(preflight))
        return
    if not os.environ.get(endpoint.api_key_env, ""):
        raise RuntimeError(f"{endpoint.api_key_env} is absent")

    result_path = args.out_dir / "judge_results.jsonl"
    existing = {
        str(row["call_id"]): row
        for row in (_rows(result_path) if result_path.is_file() else [])
    }
    pending = [
        row for row in plan if str(row["call_id"]) not in existing
    ]
    if len(pending) > args.max_new_calls:
        raise RuntimeError(
            f"{len(pending)} calls remain but max is {args.max_new_calls}"
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
            validation_error: Exception | None = None
            result = None
            raw: dict[str, Any] | None = None
            normalized: dict[str, Any] | None = None
            for content_attempt in range(3):
                messages = list(row["messages"])
                if content_attempt:
                    messages = [
                        *messages,
                        {
                            "role": "user",
                            "content": (
                                "Format repair only: return the same required "
                                "JSON schema, but every non-sentinel excerpt "
                                "must be copied verbatim as an exact substring "
                                "from the supplied candidate response or "
                                "visible evidence. Do not paraphrase excerpts."
                            ),
                        },
                    ]
                result, parsed = client.chat(
                    messages,
                    temperature=float(request["temperature"]),
                    max_tokens=int(request["max_output_tokens"]),
                    seed=int(request["seed"]) + content_attempt,
                    response_schema=schema,
                    retries=2,
                )
                if parsed is None:
                    validation_error = RuntimeError(
                        "judge returned no structured output"
                    )
                    continue
                raw = parsed.model_dump(mode="json")
                auditable_raw = _ungrounded_violation_to_insufficient(
                    call=row, parsed=raw
                )
                try:
                    normalized = validate_parsed_judgment(
                        call=row, parsed=auditable_raw
                    )
                    validation_error = None
                    break
                except ValueError as exc:
                    validation_error = exc
                    append_jsonl(
                        args.out_dir / "content_validation_failures.jsonl",
                        {
                            "protocol": PROTOCOL,
                            "call_id": row["call_id"],
                            "role": row["role"],
                            "content_attempt": content_attempt,
                            "validation_error": str(exc),
                            "parsed": raw,
                            "auditable_parsed": auditable_raw,
                            "failed_at": utc_now(),
                        },
                    )
            if validation_error is not None or result is None or raw is None:
                raise validation_error or RuntimeError(
                    "judge content repair exhausted"
                )
            assert normalized is not None
            usage = require_reported_usage(
                result.usage, stage="rs_paired_effect_judge"
            )
            append_jsonl(
                result_path,
                {
                    "protocol": MVP_JUDGE_RUNNER_PROTOCOL,
                    "phase": "outcomes",
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
                    "deterministic_output_normalization": raw != normalized,
                    "deterministic_output_normalized_fields": dict_field_diff(
                        raw, normalized
                    ),
                    "content_attempts_used": content_attempt + 1,
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

    results = _rows(result_path) if result_path.is_file() else []
    summary = {
        "protocol": PROTOCOL,
        "status": (
            "COMPLETE" if len(results) == len(plan) else "PARTIAL_RESUMABLE"
        ),
        "logical_calls": len(plan),
        "completed_calls": len(results),
        "completed_now": completed_now,
        "remaining_calls": len(plan) - len(results),
        "resumable": True,
        "last_run_error": (
            None
            if run_error is None
            else {"type": type(run_error).__name__, "message": str(run_error)}
        ),
    }
    if run_error is None and len(results) == len(plan):
        measurement = aggregate_outcome_measurement(
            qualification_report=qualification,
            call_plan=plan,
            results=results,
            pairs=pairs,
        )
        quality_rows = list(measurement.pop("quality_pair_rows"))
        risk_rows = list(measurement.pop("risk_event_rows"))
        write_jsonl(
            args.out_dir / "resolved_quality_pairs.jsonl", quality_rows
        )
        write_jsonl(args.out_dir / "atomic_risk_events.jsonl", risk_rows)
        write_json(args.out_dir / "measurement_report.json", measurement)
        summary["measurement_status"] = measurement["status"]
    write_json(args.out_dir / "execution_summary.json", summary)
    print(canonical_json(summary))
    if run_error is not None:
        raise run_error


if __name__ == "__main__":
    main()
