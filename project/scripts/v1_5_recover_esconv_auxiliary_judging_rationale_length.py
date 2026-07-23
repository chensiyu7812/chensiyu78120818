#!/usr/bin/env python3
"""Offline, zero-API-cost recovery for ESConv-auxiliary judging's rationale-
length failures (currently: esconv_auxiliary_judging_full_train).

Context: the real train-split judging run (output_directory outputs/
esconv_auxiliary_judging_v1_5_full_train) completed 2537/2544 logical calls
(629/636 judge pairs); the remaining 7 are all retry_class=structured_output_
validation_error, i.e. a complete, well-formed, non-truncated provider
response (normalize_provider_finish_reason(...) == "complete" in all 7, both
OpenAI-compatible choices[].finish_reason and Gemini-native candidates[].
finishReason surfaces) rejected only because ResponseJudgeOutput.rationale's
max_length=800 -- a prompt-shaping constant with no scientific meaning, since
the real, scientific output bound is the judge call's own max_tokens (600 for
response, 700 for risk) -- was exceeded. This script requires the *exact*
failure shape: each of the 7 must have precisely one validation error, on
`rationale`, of type `string_too_long`; any other shape, or any different
failure count, fails closed (raises, writes nothing).

max_length has since been removed from ResponseJudgeOutput/RiskJudgeOutput
(src/metacom_pm/pm_v2_judging.py) as a permanent fix, not a one-off patch:
this script re-validates each of the 7 already-received raw responses against
the *current* (uncapped) schema class, and fails closed if that re-validation
raises for any other reason.

This is NOT ordinary carry-forward: --carry-forward-from re-attempts a
previously-failed logical call over the network under the current contract.
This script never re-calls any provider, never re-attempts anything, and
never mutates the original output directory's run_manifest.json, cost_
estimate.json, call_plan.jsonl, or physical_attempt_ledger.jsonl -- it only
reads them (with a fail-closed sha256 cross-check against the frozen paid-
run-release manifest) and the already-generated action_outcomes.jsonl/pm_v2_
states.jsonl, then re-runs the *same* label-aggregation and quality-gate
functions the live runner uses, over 629 original-successful + 7 offline-
recovered pairs.

Writes two new files inside the existing output directory (never touching
any existing file): recovery_report.json (per-call recovery detail) and
recovered_summary.json (the full 636/636 aggregated summary, in the same
shape 13c_judge_esconv_auxiliary_v1_5.py itself would have produced, labeled
as a post-run engineering-contract amendment). The original run_manifest.
json/summary.json are left untouched and the original stage_consumptions
entry in outputs/pm_v1_5_paid_run_release.json stays CONSUMED_INCOMPLETE_
NONREPORTABLE permanently -- this recovery is recorded as its own, separate
addendum, not a silent overwrite.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from metacom_pm.api import normalize_provider_finish_reason
from metacom_pm.attempt_ledger import PersistentAttemptLedger
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import ActionOutcome
from metacom_pm.io import (
    append_jsonl,
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
)
from metacom_pm.pm_v2_data import load_states
from metacom_pm.pm_v2_judging import (
    JudgeResult,
    ResponseJudgeOutput,
    RiskJudgeOutput,
    build_action_label,
    composite_spec_from_config,
    composite_weights_hash,
    dimension_applicability_by_action,
    dimension_applicability_contract_sha256,
    dimensions_inapplicable_to_every_action,
    judge_family_directional_preference_report,
    labeling_settings_from_config,
    prompt_contract_hash,
    validate_action_applicable_risk_signal,
    validate_judge_table,
    validate_raw_judge_family_health,
    validate_raw_judge_family_subgroup_health,
)

ROOT = Path(__file__).resolve().parents[1]
RATIONALE_LENGTH_RECOVERY_PROTOCOL = (
    "pm-v1.5-esconv-auxiliary-judging-rationale-length-recovery-v1"
)
MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL = 10
MINIMUM_NONZERO_OBSERVATIONS_FOR_DUPLICATE_CHECK = 10


def outcome_key(outcome: ActionOutcome) -> tuple[str, str]:
    return (outcome.state_id, outcome.action_id)


def call_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["state_id"]),
        str(row["action_id"]),
        str(row["judge_family"]),
        str(row["judge_type"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "esconv_auxiliary_judging_v1_5_full_train",
    )
    parser.add_argument(
        "--stage-consumption-key",
        default="esconv_auxiliary_judging_full_train",
        help="Key under stage_consumptions in the paid-run-release manifest.",
    )
    parser.add_argument(
        "--paid-run-release",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_paid_run_release.json",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "experiment.yaml"
    )
    parser.add_argument(
        "--pm-v1-5-config", type=Path, default=ROOT / "configs" / "pm_v1_5.yaml"
    )
    parser.add_argument(
        "--auxiliary-dir", type=Path, default=ROOT / "data" / "esconv_auxiliary_v1_5"
    )
    parser.add_argument(
        "--generation-dir",
        type=Path,
        default=ROOT / "outputs" / "esconv_auxiliary_generation_v1_5_full_train",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--expected-succeeded-calls",
        type=int,
        required=True,
        help="Fail closed unless exactly this many logical calls already succeeded.",
    )
    parser.add_argument(
        "--expected-recoverable-failures",
        type=int,
        required=True,
        help=(
            "Fail closed unless exactly this many logical calls failed, and "
            "every one is a pure rationale/string_too_long, complete-response "
            "failure."
        ),
    )
    return parser.parse_args()


def _schema_for(judge_type: str):
    if judge_type == "response":
        return ResponseJudgeOutput
    if judge_type == "risk":
        return RiskJudgeOutput
    raise RuntimeError(f"unknown judge_type: {judge_type!r}")


def require_complete_provider_response(
    provider_response: dict[str, Any],
) -> str | None:
    """Reject truncated recovery inputs across native and compatible surfaces."""

    provider_finish_reason, normalized_finish_reason = (
        normalize_provider_finish_reason(provider_response)
    )
    if normalized_finish_reason != "complete":
        raise RuntimeError(
            "provider response does not have a complete (stop/STOP) finish "
            f"reason (got {normalized_finish_reason!r}); refusing to recover "
            "a possibly-truncated response"
        )
    return provider_finish_reason


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    manifest = read_json(args.paid_run_release)
    stage_consumption = manifest["stage_consumptions"][args.stage_consumption_key]
    if stage_consumption["status"] != "CONSUMED_INCOMPLETE_NONREPORTABLE":
        raise RuntimeError(
            "recovery requires the manifest to record this run as "
            "CONSUMED_INCOMPLETE_NONREPORTABLE; refusing to recover any "
            "other status"
        )
    if Path(stage_consumption["output_directory"]) != out_dir:
        raise RuntimeError(
            "recovery --out-dir does not match the manifest's recorded "
            "output_directory for this stage"
        )
    ledger_path = out_dir / "physical_attempt_ledger.jsonl"
    if sha256_file(ledger_path) != stage_consumption["physical_attempt_ledger_sha256"]:
        raise RuntimeError(
            "physical_attempt_ledger.jsonl does not match the manifest's "
            "recorded sha256; refusing to recover from an altered ledger"
        )

    run_manifest = read_json(out_dir / "run_manifest.json")
    if sha256_file(args.pm_v1_5_config) != run_manifest["pm_v1_5_config_sha256"]:
        raise RuntimeError(
            "current pm_v1_5.yaml does not match the original run's config"
        )
    if sha256_file(args.config) != run_manifest["experiment_config_sha256"]:
        raise RuntimeError(
            "current experiment.yaml does not match the original run's config"
        )

    call_plan = list(iter_jsonl(out_dir / "call_plan.jsonl"))
    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=run_manifest["stage"],
        expected_calls={
            str(row["physical_call_key"]): MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL
            for row in call_plan
        },
        maximum_total_attempts=ledger_started_attempts_bound(ledger_path),
    )

    succeeded_rows = [
        row for row in call_plan if ledger.succeeded(str(row["physical_call_key"]))
    ]
    failed_rows = [
        row for row in call_plan if not ledger.succeeded(str(row["physical_call_key"]))
    ]
    if len(succeeded_rows) != int(args.expected_succeeded_calls):
        raise RuntimeError(
            f"expected exactly {args.expected_succeeded_calls} succeeded logical "
            f"calls, found {len(succeeded_rows)}"
        )
    if len(failed_rows) != int(args.expected_recoverable_failures):
        raise RuntimeError(
            f"expected exactly {args.expected_recoverable_failures} failed "
            f"logical calls, found {len(failed_rows)}"
        )

    recovery_detail: list[dict[str, Any]] = []
    recovered_parsed_by_key: dict[str, dict[str, Any]] = {}
    for row in failed_rows:
        physical_key = str(row["physical_call_key"])
        terminal = ledger.terminal_row(physical_key)
        if terminal is None or terminal.get("event") != "FAILED":
            raise RuntimeError(f"expected a terminal FAILED row for {physical_key}")
        metadata = terminal.get("metadata") or {}
        retry_class = str(metadata.get("retry_class") or "")
        if retry_class != "structured_output_validation_error":
            raise RuntimeError(
                f"call {physical_key} failed with retry_class={retry_class!r}, "
                "not structured_output_validation_error -- cannot be recovered "
                "offline; refusing to proceed"
            )
        result = terminal.get("result") or {}
        validation_errors = result.get("validation_errors") or []
        if validation_errors != [
            {
                "type": "string_too_long",
                "loc": ["rationale"],
                "msg": "String should have at most 800 characters",
                "ctx": {"max_length": 800},
            }
        ]:
            raise RuntimeError(
                f"call {physical_key} has an unexpected validation_errors shape "
                f"(not exactly one rationale/string_too_long error): "
                f"{validation_errors!r}; refusing to recover"
            )
        provider_response = result.get("provider_response")
        if not isinstance(provider_response, dict):
            raise RuntimeError(
                f"call {physical_key} has no provider_response to verify "
                "completeness against"
            )
        provider_finish_reason = require_complete_provider_response(
            provider_response
        )
        parsed_payload = result.get("parsed_payload")
        if not isinstance(parsed_payload, dict):
            raise RuntimeError(f"call {physical_key} has no parsed_payload to recover")
        schema = _schema_for(str(row["judge_type"]))
        recovered = schema.model_validate(parsed_payload)
        # No modification: the recovered object's dump must byte-match the
        # original parsed_payload exactly (only the schema's own strictness
        # changed, never the content).
        if recovered.model_dump(mode="json") != parsed_payload:
            raise RuntimeError(
                f"call {physical_key} recovered value differs from the "
                "original parsed_payload; refusing to silently alter content"
            )
        recovered_parsed_by_key[physical_key] = parsed_payload
        recovery_detail.append(
            {
                "physical_call_key": physical_key,
                "state_id": row["state_id"],
                "action_id": row["action_id"],
                "judge_family": row["judge_family"],
                "judge_type": row["judge_type"],
                "rationale_length": len(str(parsed_payload.get("rationale", ""))),
                "provider_finish_reason": provider_finish_reason,
            }
        )

    # --- Reload the same inputs the live runner uses to build labels ---
    config = load_config(args.config)
    pm_v1_5_config = load_config(args.pm_v1_5_config)
    states_path = args.auxiliary_dir / args.split / "pm_v2_states.jsonl"
    states = load_states(states_path)
    state_by_id = {state.state_id: state for state in states}
    outcomes_path = args.generation_dir / "action_outcomes.jsonl"
    outcomes = [ActionOutcome.model_validate(row) for row in iter_jsonl(outcomes_path)]
    outcome_by_key = {outcome_key(o): o for o in outcomes}

    composite_spec = composite_spec_from_config(pm_v1_5_config)
    composite_weights_sha256 = composite_weights_hash(composite_spec)
    labeling = labeling_settings_from_config(pm_v1_5_config)
    judging_config = dict(pm_v1_5_config["development_judging"])
    endpoint_names = [str(value) for value in judging_config["judge_endpoints"]]
    endpoints = [endpoint_from_config(config, name) for name in endpoint_names]

    # --- Reconstruct raw_by_key exactly as the live runner would, using
    # succeeded ledger rows for the 629 pairs and the freshly-recovered
    # parsed payloads for the 7 previously-failed calls. ---
    successful_call_rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in succeeded_rows:
        physical_key = str(row["physical_call_key"])
        terminal = ledger.terminal_row(physical_key)
        result_payload = (terminal or {}).get("result") or {}
        schema = _schema_for(str(row["judge_type"]))
        successful_call_rows[call_key(row)] = {
            **row,
            "parsed": schema.model_validate(result_payload.get("parsed")).model_dump(
                mode="json"
            ),
            "request_hash": str((terminal or {}).get("request_hash")),
        }
    for row in failed_rows:
        physical_key = str(row["physical_call_key"])
        terminal = ledger.terminal_row(physical_key)
        successful_call_rows[call_key(row)] = {
            **row,
            "parsed": recovered_parsed_by_key[physical_key],
            "request_hash": str((terminal or {}).get("request_hash")),
        }

    expected_pairs = {outcome_key(o) for o in outcomes}
    raw_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for state_id, action_id in sorted(expected_pairs):
        for family in sorted(str(e.family) for e in endpoints):
            response_row = successful_call_rows.get(
                (state_id, action_id, family, "response")
            )
            risk_row = successful_call_rows.get((state_id, action_id, family, "risk"))
            if response_row is None or risk_row is None:
                raise RuntimeError(
                    f"missing complete family panel for {(state_id, action_id, family)} "
                    "after recovery -- this should be impossible given the exact "
                    "succeeded/failed counts already verified above"
                )
            raw_by_key[(state_id, action_id, family)] = {
                "state_id": state_id,
                "card_id": outcome_by_key[(state_id, action_id)].card_id,
                "action_id": action_id,
                "judge_family": family,
                "judge_model": response_row["judge_model"],
                "prompt_contract_hash": prompt_contract_hash(),
                "response": response_row["parsed"],
                "risk": risk_row["parsed"],
                "response_request_hash": response_row["request_hash"],
                "risk_request_hash": risk_row["request_hash"],
            }

    canonical_raw_rows = [
        {
            "judge_family": row["judge_family"],
            "action_id": row["action_id"],
            "response": row["response"],
            "risk": row["risk"],
        }
        for row in raw_by_key.values()
    ]
    # Unlike canonical_raw_rows (deliberately state_id-free, matching what
    # validate_raw_judge_family_health/validate_raw_judge_family_subgroup_
    # health/validate_action_applicable_risk_signal consume), the directional
    # preference diagnostic below needs state_id to pair each family's M0+R0
    # vs M0+RS judgment for the *same* state.
    preference_rows = [
        {
            "state_id": state_id,
            "action_id": action_id,
            "judge_family": family,
            "response": row["response"],
            "risk": row["risk"],
        }
        for (state_id, action_id, family), row in raw_by_key.items()
    ]
    # ESConv-auxiliary's legal actions are frozen to {M0+R0, M0+RS}: no
    # memory source is ever selected, so every memory-misuse risk dimension
    # (applicable_risk_fields, pm_v2_model.py) is structurally a zero here,
    # not a judge defect. Matches the same fix applied to
    # 13c_judge_esconv_auxiliary_v1_5.py itself.
    action_ids_present = sorted({o.action_id for o in outcomes})
    inapplicable_risk_dimensions = dimensions_inapplicable_to_every_action(
        action_ids_present
    )
    inapplicable_risk_dimensions_by_action = dimension_applicability_by_action(
        action_ids_present
    )
    raw_family_global_gate = validate_raw_judge_family_health(
        canonical_raw_rows,
        expected_families=[str(endpoint.family) for endpoint in endpoints],
        duplicate_exact_match_rate=labeling["duplicate_exact_match_rate"],
        maximum_absolute_dimension_correlation=labeling[
            "maximum_absolute_dimension_correlation"
        ],
        composite_support_exact_match_rate=labeling["composite_support_exact_match_rate"],
        maximum_absolute_composite_support_correlation=labeling[
            "maximum_absolute_composite_support_correlation"
        ],
        inapplicable_risk_dimensions=inapplicable_risk_dimensions,
        inapplicable_risk_dimensions_by_action=inapplicable_risk_dimensions_by_action,
        minimum_nonzero_observations=MINIMUM_NONZERO_OBSERVATIONS_FOR_DUPLICATE_CHECK,
        split_correlation_by_sign=True,
        reject_constant_response_dimensions=labeling["reject_constant_response_dimensions"],
        reject_constant_risk_dimensions=labeling["reject_constant_risk_dimensions"],
        composite_spec=composite_spec,
        raise_on_failure=False,
    )
    raw_family_action_gate = validate_raw_judge_family_subgroup_health(
        canonical_raw_rows,
        subgroup_key="action_id",
        expected_subgroups=sorted({o.action_id for o in outcomes}),
        expected_families=[str(endpoint.family) for endpoint in endpoints],
        duplicate_exact_match_rate=labeling["duplicate_exact_match_rate"],
        maximum_absolute_dimension_correlation=labeling[
            "maximum_absolute_dimension_correlation"
        ],
        composite_support_exact_match_rate=labeling["composite_support_exact_match_rate"],
        maximum_absolute_composite_support_correlation=labeling[
            "maximum_absolute_composite_support_correlation"
        ],
        reject_constant_response_dimensions=labeling["reject_constant_response_dimensions"],
        reject_constant_risk_dimensions=labeling["reject_constant_risk_dimensions"],
        composite_spec=composite_spec,
        raise_on_failure=False,
    )
    action_applicable_risk_gate = validate_action_applicable_risk_signal(
        canonical_raw_rows,
        expected_actions=sorted({o.action_id for o in outcomes}),
        expected_families=[str(endpoint.family) for endpoint in endpoints],
        minimum_signal_rate=labeling["minimum_action_applicable_risk_signal_rate"],
        minimum_distinct_values=labeling["minimum_action_applicable_risk_distinct_values"],
        raise_on_failure=False,
    )
    # ESConv-auxiliary's legal action set is frozen to exactly {M0+R0,
    # M0+RS}: whether independent judge families agree on which one is
    # *better* per state (not just per-dimension score agreement) is a
    # distinct, diagnostic-only reliability question -- never a hard gate.
    if len(action_ids_present) != 2:
        raise RuntimeError(
            "directional preference diagnostic requires exactly two legal "
            f"actions, got {action_ids_present}"
        )
    directional_preference = judge_family_directional_preference_report(
        preference_rows,
        action_a=action_ids_present[0],
        action_b=action_ids_present[1],
        expected_families=[str(endpoint.family) for endpoint in endpoints],
        dialogue_by_state={state.state_id: state.user_id for state in states},
        risk_weight=float(pm_v1_5_config["selection"]["risk_weight"]),
        bootstrap_seed=0,
    )

    labels = []
    for state_id, action_id in sorted(expected_pairs):
        state = state_by_id[state_id]
        outcome = outcome_by_key[(state_id, action_id)]
        results = [
            JudgeResult(
                family=str(row["judge_family"]),
                model=str(row["judge_model"]),
                response=ResponseJudgeOutput.model_validate(row["response"]),
                risk=RiskJudgeOutput.model_validate(row["risk"]),
                response_request_hash=str(row["response_request_hash"]),
                risk_request_hash=str(row["risk_request_hash"]),
            )
            for endpoint in endpoints
            for row in [raw_by_key[(state_id, action_id, str(endpoint.family))]]
        ]
        label = build_action_label(
            state=state,
            action_id=action_id,
            observed_input_tokens=outcome.cost.total_input_tokens,
            retrieval_calls=outcome.cost.retrieval_calls,
            results=results,
            composite_spec=composite_spec,
            minimum_families=labeling["minimum_families"],
            reliable_mad_threshold=labeling["reliable_mad_threshold"],
            provenance={"esconv_auxiliary_split": args.split},
        )
        labels.append(label)

    recovered_labels_path = out_dir / "recovered_action_labels.jsonl"
    if recovered_labels_path.exists():
        raise RuntimeError(
            f"{recovered_labels_path} already exists; refusing to overwrite a "
            "prior recovery's output"
        )
    for label in labels:
        append_jsonl(recovered_labels_path, label.model_dump(mode="json"))

    quality_gate = validate_judge_table(
        labels,
        minimum_families=labeling["minimum_families"],
        minimum_reliable_rate=labeling["minimum_reliable_rate"],
        reliable_mad_threshold=labeling["reliable_mad_threshold"],
        minimum_low_mad_coverage_per_dimension=labeling[
            "minimum_low_mad_coverage_per_dimension"
        ],
        minimum_low_mad_coverage_per_action_dimension=labeling[
            "minimum_low_mad_coverage_per_action_dimension"
        ],
        duplicate_exact_match_rate=labeling["duplicate_exact_match_rate"],
        maximum_absolute_dimension_correlation=labeling[
            "maximum_absolute_dimension_correlation"
        ],
        composite_support_exact_match_rate=labeling["composite_support_exact_match_rate"],
        maximum_absolute_composite_support_correlation=labeling[
            "maximum_absolute_composite_support_correlation"
        ],
        reject_constant_response_dimensions=labeling["reject_constant_response_dimensions"],
        reject_constant_risk_dimensions=labeling["reject_constant_risk_dimensions"],
        inapplicable_risk_dimensions=inapplicable_risk_dimensions,
        inapplicable_risk_dimensions_by_action=inapplicable_risk_dimensions_by_action,
        minimum_nonzero_observations=MINIMUM_NONZERO_OBSERVATIONS_FOR_DUPLICATE_CHECK,
        split_correlation_by_sign=True,
        composite_spec=composite_spec,
        raise_on_failure=False,
    )
    raw_family_quality_status = (
        "PASS"
        if raw_family_global_gate.get("status") == "PASS"
        and raw_family_action_gate.get("status") == "PASS"
        and action_applicable_risk_gate.get("status") == "PASS"
        else "FAIL"
    )
    recovery_report = {
        "protocol": RATIONALE_LENGTH_RECOVERY_PROTOCOL,
        "status": "RECOVERED",
        "note": (
            f"Post-run engineering-contract amendment: recovers all "
            f"{len(recovery_detail)} logical calls rejected only for "
            "exceeding ResponseJudgeOutput/RiskJudgeOutput.rationale's former "
            "max_length=800 (since removed as scientifically meaningless -- "
            "the real bound is the judge call's own max_tokens). Every "
            "recovered response has a real, normal, non-truncated provider "
            "finish reason (normalize_provider_finish_reason == 'complete') "
            "and exactly one validation error (rationale/string_too_long); "
            "no other schema constraint was bypassed and no field value was "
            "altered (recovered.model_dump() byte-matches the original "
            "parsed_payload for every one). Zero new API calls; zero new "
            "cost. This is NOT ordinary carry-forward: no provider was "
            "re-contacted for these 7 calls."
        ),
        "original_output_directory": str(out_dir),
        "original_physical_attempt_ledger_sha256": sha256_file(ledger_path),
        "n_recovered": len(recovery_detail),
        "recovered_calls": recovery_detail,
    }
    write_json(out_dir / "recovery_report.json", recovery_report)

    recovered_summary = {
        "protocol": RATIONALE_LENGTH_RECOVERY_PROTOCOL,
        "status": "COMPLETE",
        "pilot_mode": False,
        "reportability_status": (
            "REPORTABLE"
            if raw_family_quality_status == "PASS" and quality_gate.get("status") == "PASS"
            else "FORMAL_GATE_FAILED"
        ),
        "split": args.split,
        "expected_judge_pairs": len(expected_pairs),
        "completed_judge_pairs": len(labels),
        "recovery_report_sha256": sha256_text(canonical_json(recovery_report)),
        "composite_weights_sha256": composite_weights_sha256,
        "raw_family_quality_gate": {
            "status": raw_family_quality_status,
            "global": raw_family_global_gate,
            "family_by_action": raw_family_action_gate,
            "action_applicable_risk_signal": action_applicable_risk_gate,
        },
        "quality_gate": quality_gate,
        "judge_family_directional_preference": directional_preference,
        "dimension_applicability_contract_sha256": dimension_applicability_contract_sha256(
            action_ids_present
        ),
        "note": (
            "Supersedes the original summary.json's INCOMPLETE_NONREPORTABLE_"
            "MATRIX via the offline recovery documented in recovery_report."
            "json. The original summary.json/run_manifest.json/call_plan."
            "jsonl/physical_attempt_ledger.jsonl are left untouched; the "
            "paid-run-release manifest's stage_consumptions entry for this "
            "run permanently records CONSUMED_INCOMPLETE_NONREPORTABLE -- "
            "this recovered_summary.json is a separate, explicitly-labeled "
            "addendum, not a silent overwrite."
        ),
    }
    write_json(out_dir / "recovered_summary.json", recovered_summary)
    # The original (empty) action_labels.jsonl is never touched; the
    # recovered labels live in their own recovered_action_labels.jsonl file.

    print(
        f"RECOVERED: {len(recovery_detail)}/{len(recovery_detail)} recoverable "
        f"calls confirmed exactly-as-expected. completed_judge_pairs="
        f"{len(labels)}/{len(expected_pairs)}. quality_gate.status="
        f"{quality_gate.get('status')}, raw_family_quality_gate.status="
        f"{raw_family_quality_status}. See {out_dir / 'recovery_report.json'} "
        f"and {out_dir / 'recovered_summary.json'}."
    )


def ledger_started_attempts_bound(ledger_path: Path) -> int:
    """A generous, read-only-safe cap: recovery never reserves a new attempt."""

    return sum(1 for _ in iter_jsonl(ledger_path)) + 1


if __name__ == "__main__":
    main()
