#!/usr/bin/env python3
"""Run the exact full PM-v1.5 state-by-action sweep.

The fast track replaces PM-v2.2's human-only development gates with the
attested multi-family automated semantic review. A V1.5 run must explicitly
pass ``--v1-5-full-sweep-scope``; filtered/ad-hoc and legacy compatibility-
pilot sweeps are rejected so the resulting artifact is honestly labelled
``full`` and contains every legal state-action pair.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from metacom_pm.artifacts import require_artifact_attestation
from metacom_pm.attempt_ledger import (
    PersistentAttemptLedger,
    forbid_overwrite_of_spent_attempts,
)
from metacom_pm.bounded_retry import RETRYABLE_UP_TO_FULL_BUDGET
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import ActionOutcome, parse_action_id
from metacom_pm.evidence_filter import EvidenceFilterConfig
from metacom_pm.evidence_filter_model import require_evidence_filter_artifacts
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.sweep import plan_action_sweep, run_action_sweep
from metacom_pm.pm_v2_data import load_evaluator_context_index, load_states
from metacom_pm.pm_v2_audit import audit_deployable_feature_observability
from metacom_pm.pm_v2_development_gate import require_development_pilot_gate
from metacom_pm.pm_v2_judge_schema_smoke import (
    require_development_judge_schema_smoke_pass,
)
from metacom_pm.pm_v1_5_shortcut_audit import (
    require_step0_shortcut_audit_pass,
)
from metacom_pm.pm_v1_5_rule_router import RULE_GRID_DIAGNOSTIC_PROTOCOL
from metacom_pm.pm_v2_semantic_audit import (
    require_pmv2_runtime_state_lineage,
    require_semantic_sanity_pass,
)
from metacom_pm.paid_run_release import require_paid_run_release
from metacom_pm.response_mechanism_contract import build_response_mechanism_contract
from metacom_pm.v1_5_actual_corpus_review import (
    require_actual_corpus_semantic_review_pass,
)
from metacom_pm.v1_5_actual_corpus_qualification import (
    require_actual_corpus_posthoc_qualification,
)


ROOT = Path(__file__).resolve().parents[2]

LONGITUDINAL_TRANSPORT_EXECUTION_PROTOCOL = (
    "pm-v1.5-longitudinal-sweep-transport-execution-v1"
)
LONGITUDINAL_TRANSPORT_MAX_ATTEMPTS_PER_CALL = 4
LONGITUDINAL_TRANSPORT_BACKOFF_SECONDS = (10.0, 30.0, 60.0)
LONGITUDINAL_TRANSPORT_CIRCUIT_BREAKER = 5


def _longitudinal_transport_execution_contract() -> dict:
    """Bind execution resilience without changing the scientific treatment."""

    code_paths = {
        "runner": Path(__file__).resolve(),
        "sweep": ROOT / "src" / "metacom_pm" / "sweep.py",
        "api": ROOT / "src" / "metacom_pm" / "api.py",
        "attempt_ledger": ROOT / "src" / "metacom_pm" / "attempt_ledger.py",
        "bounded_retry": ROOT / "src" / "metacom_pm" / "bounded_retry.py",
    }
    code_manifest = {
        name: {
            "relative_path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        }
        for name, path in sorted(code_paths.items())
    }
    payload = {
        "protocol": LONGITUDINAL_TRANSPORT_EXECUTION_PROTOCOL,
        "transport_retry_policy": "bounded_transport",
        "transport_max_attempts_per_call": (
            LONGITUDINAL_TRANSPORT_MAX_ATTEMPTS_PER_CALL
        ),
        "transport_backoff_seconds": list(
            LONGITUDINAL_TRANSPORT_BACKOFF_SECONDS
        ),
        "consecutive_same_class_circuit_breaker": (
            LONGITUDINAL_TRANSPORT_CIRCUIT_BREAKER
        ),
        "continue_after_isolated_terminal_failure": True,
        "retryable_classes": sorted(RETRYABLE_UP_TO_FULL_BUDGET),
        "terminal_content_failures_are_not_blindly_retried": True,
        "legacy_config_request_retries": 1,
        "legacy_config_fail_fast": True,
        "scientific_treatment_unchanged": True,
        "code_manifest": code_manifest,
        "code_manifest_sha256": sha256_text(canonical_json(code_manifest)),
    }
    payload["contract_sha256"] = sha256_text(canonical_json(payload))
    return payload


def _budget_gate(
    estimate: dict,
    *,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
) -> dict:
    checks = {
        "physical_api_attempts": int(estimate["maximum_physical_api_attempts"])
        <= int(max_api_calls),
        "estimated_cost_usd": float(estimate["estimated_cost_usd"])
        <= float(max_estimated_usd),
        "max_input_tokens_per_call": int(estimate["max_input_tokens"])
        <= int(max_input_tokens_per_call),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "limits": {
            "max_physical_api_attempts": int(max_api_calls),
            "max_estimated_usd": float(max_estimated_usd),
            "max_input_tokens_per_call": int(max_input_tokens_per_call),
        },
    }


def _persist_or_validate_dry_run(
    out_dir: Path, current: dict, rows: list[dict]
) -> str:
    """Freeze the exact accepted estimate/plan once an HTTP attempt exists."""

    estimate_path = out_dir / "cost_estimate.json"
    plan_path = out_dir / "call_plan.jsonl"
    ledger_path = out_dir / "physical_attempt_ledger.jsonl"
    spent = ledger_path.is_file() and ledger_path.stat().st_size > 0
    if spent:
        if not estimate_path.is_file() or not plan_path.is_file():
            raise RuntimeError(
                "spent action-sweep ledger freezes the accepted dry-run, but its "
                "estimate or full call plan is missing"
            )
        if read_json(estimate_path) != current or list(iter_jsonl(plan_path)) != rows:
            raise RuntimeError(
                "spent action-sweep ledger freezes the original exact estimate "
                "and full call plan; current dry-run drift is rejected"
            )
        return "VALIDATED_EXISTING"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(estimate_path, current)
    write_jsonl(plan_path, rows)
    return "WRITTEN"


def _require_saved_dry_run(
    out_dir: Path, current: dict, rows: list[dict]
) -> None:
    estimate_path = out_dir / "cost_estimate.json"
    plan_path = out_dir / "call_plan.jsonl"
    if not estimate_path.is_file() or not plan_path.is_file():
        raise RuntimeError(
            "API mode requires a completed matching --dry-run in the same output "
            "directory"
        )
    saved = read_json(estimate_path)
    if saved != current:
        raise RuntimeError(
            "saved action-sweep dry-run does not match current inputs/configuration; "
            "run --dry-run again"
        )
    if list(iter_jsonl(plan_path)) != rows:
        raise RuntimeError("saved action-sweep full call plan is stale")
    if saved.get("budget_gate", {}).get("status") != "PASS":
        raise RuntimeError("saved action-sweep dry-run did not pass its budget gate")


ACTION_SWEEP_CARRY_FORWARD_PROTOCOL = (
    "pm-v1.5-action-sweep-exact-plan-carry-forward-v1"
)


def _load_action_sweep_carry_forward(
    *, carry_forward_dir: Path | None, call_plan: list[dict[str, Any]]
) -> dict[str, Any]:
    """Authenticate successful calls from an exact-plan prior sweep.

    The source directory is read-only.  A failed/exhausted call is never
    inherited; only a ledger SUCCEEDED event containing the complete raw call
    and ActionOutcome recovery payload is eligible.  Exact full-plan equality
    prevents a continuation from silently mixing scientific treatments.
    """

    if carry_forward_dir is None:
        return {
            "binding": None,
            "carried_call_keys": set(),
            "carried_terminal_rows": {},
        }
    required = {
        "call_plan": carry_forward_dir / "call_plan.jsonl",
        "cost_estimate": carry_forward_dir / "cost_estimate.json",
        "ledger": carry_forward_dir / "physical_attempt_ledger.jsonl",
        "outcomes": carry_forward_dir / "action_outcomes.jsonl",
        "raw_calls": carry_forward_dir / "raw_api_calls.jsonl",
        "summary": carry_forward_dir / "summary.json",
    }
    missing = sorted(name for name, path in required.items() if not path.is_file())
    if missing:
        raise RuntimeError(
            f"action-sweep carry-forward source lacks required files: {missing}"
        )
    source_plan = list(iter_jsonl(required["call_plan"]))
    if source_plan != call_plan:
        raise RuntimeError(
            "action-sweep carry-forward source plan differs from the current "
            "freshly computed full plan"
        )
    source_estimate = read_json(required["cost_estimate"])
    source_estimate_body = {
        key: value
        for key, value in source_estimate.items()
        if key not in {"cost_estimate_sha256", "budget_gate"}
    }
    if source_estimate.get("cost_estimate_sha256") != sha256_text(
        canonical_json(source_estimate_body)
    ):
        raise RuntimeError("action-sweep carry-forward source estimate self-hash failed")
    current_plan_sha256 = sha256_text(canonical_json(call_plan))
    if source_estimate.get("call_plan_sha256") != current_plan_sha256:
        raise RuntimeError("action-sweep carry-forward source plan hash mismatch")
    expected_calls = {
        str(row["call_key"]): int(row["max_http_attempts"])
        for row in call_plan
    }
    source_ledger = PersistentAttemptLedger(
        required["ledger"],
        stage="action_sweep_generation",
        expected_calls=expected_calls,
        maximum_total_attempts=10**9,
    )
    plan_by_call_key: dict[str, dict[str, Any]] = {}
    for row in call_plan:
        plan_by_call_key.setdefault(str(row["call_key"]), row)
    carried_terminal_rows: dict[str, dict[str, Any]] = {}
    for call_key, plan_row in plan_by_call_key.items():
        if not source_ledger.succeeded(call_key):
            continue
        terminal = source_ledger.terminal_row(call_key)
        result = (terminal or {}).get("result")
        if (
            not isinstance(terminal, dict)
            or not isinstance(result, dict)
            or not isinstance(result.get("action_outcome"), dict)
            or not isinstance(result.get("raw_call"), dict)
            or not isinstance(terminal.get("usage"), dict)
        ):
            raise RuntimeError(
                f"action-sweep carry-forward source lacks recovery data: {call_key}"
            )
        outcome = ActionOutcome.model_validate(result["action_outcome"])
        raw_call = dict(result["raw_call"])
        if (
            outcome.card_id != str(plan_row["card_id"])
            or outcome.prompt_hash != str(plan_row["prompt_sha256"])
            or (outcome.provenance or {}).get("physical_call_key") != call_key
            or raw_call.get("error") is not None
            or str(raw_call.get("physical_call_key") or "") != call_key
        ):
            raise RuntimeError(
                f"action-sweep carry-forward payload mismatches plan: {call_key}"
            )
        carried_terminal_rows[call_key] = dict(terminal)
    carried_call_keys = set(carried_terminal_rows)
    remaining_call_keys = set(plan_by_call_key) - carried_call_keys
    summary = read_json(required["summary"])
    carried_logical_outcomes = sum(
        1 for row in call_plan if str(row["call_key"]) in carried_call_keys
    )
    if int(summary.get("completed_outcomes", -1)) != carried_logical_outcomes:
        raise RuntimeError(
            "action-sweep carry-forward summary/ledger completion mismatch"
        )
    binding = {
        "protocol": ACTION_SWEEP_CARRY_FORWARD_PROTOCOL,
        "source_directory": str(carry_forward_dir),
        "source_call_plan_file_sha256": sha256_file(required["call_plan"]),
        "source_cost_estimate_file_sha256": sha256_file(required["cost_estimate"]),
        "source_cost_estimate_sha256": source_estimate["cost_estimate_sha256"],
        "source_ledger_sha256": sha256_file(required["ledger"]),
        "source_outcomes_sha256": sha256_file(required["outcomes"]),
        "source_raw_calls_sha256": sha256_file(required["raw_calls"]),
        "source_summary_sha256": sha256_file(required["summary"]),
        "source_physical_http_attempts": int(
            summary.get("total_physical_http_attempts", -1)
        ),
        "full_call_plan_sha256": current_plan_sha256,
        "carried_forward_call_count": len(carried_call_keys),
        "carried_forward_logical_outcome_count": carried_logical_outcomes,
        "carried_forward_call_keys_sha256": sha256_text(
            canonical_json(sorted(carried_call_keys))
        ),
        "remaining_new_call_count": len(remaining_call_keys),
        "remaining_new_call_keys_sha256": sha256_text(
            canonical_json(sorted(remaining_call_keys))
        ),
    }
    binding["binding_sha256"] = sha256_text(canonical_json(binding))
    return {
        "binding": binding,
        "carried_call_keys": carried_call_keys,
        "carried_terminal_rows": carried_terminal_rows,
    }


def _continuation_cost_estimate(
    estimate: dict[str, Any],
    rows: list[dict[str, Any]],
    carry_forward: dict[str, Any],
) -> dict[str, Any]:
    """Reprice an exact full plan for only the calls not carried forward."""

    binding = carry_forward.get("binding")
    if binding is None:
        return estimate
    carried = set(carry_forward["carried_call_keys"])
    physical_by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        physical_by_key.setdefault(str(row["call_key"]), row)
    remaining_keys = set(physical_by_key) - carried
    remaining_physical_rows = [physical_by_key[key] for key in sorted(remaining_keys)]
    remaining_logical_rows = [
        row for row in rows if str(row["call_key"]) in remaining_keys
    ]
    attempts_per_call = int(estimate["transport_max_attempts_per_call"])
    input_tokens = [
        int(row["estimated_input_tokens"]) for row in remaining_physical_rows
    ]
    logical_input = sum(input_tokens)
    logical_output = sum(
        int(row["maximum_output_tokens"]) for row in remaining_physical_rows
    )
    pricing = dict(estimate["pricing"])
    logical_cost = (
        logical_input / 1_000_000 * float(pricing["input_usd_per_mtok"])
        + logical_output / 1_000_000 * float(pricing["output_usd_per_mtok"])
    )
    ordered = sorted(input_tokens)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1) if ordered else 0
    payload = {
        key: value
        for key, value in estimate.items()
        if key != "cost_estimate_sha256"
    }
    contract_bindings = {
        **dict(payload.get("contract_bindings") or {}),
        "carry_forward": dict(binding),
    }
    payload.update(
        {
            "full_expected_api_calls": int(estimate["expected_api_calls"]),
            "expected_api_calls": len(remaining_physical_rows),
            "full_logical_api_calls": int(estimate["logical_api_calls"]),
            "logical_api_calls": len(rows),
            "historical_carried_forward_calls": len(carried),
            "remaining_new_logical_calls": len(remaining_physical_rows),
            "remaining_new_logical_outcomes": len(remaining_logical_rows),
            "full_maximum_physical_api_attempts": int(
                estimate["maximum_physical_api_attempts"]
            ),
            "maximum_physical_api_attempts": (
                len(remaining_physical_rows) * attempts_per_call
            ),
            "estimated_total_input_tokens": logical_input,
            "maximum_total_output_tokens": logical_output,
            "unduplicated_logical_total_input_tokens": sum(
                int(row["estimated_input_tokens"])
                for row in remaining_logical_rows
            ),
            "unduplicated_logical_maximum_output_tokens": sum(
                int(row["maximum_output_tokens"])
                for row in remaining_logical_rows
            ),
            "maximum_physical_total_input_tokens": (
                logical_input * attempts_per_call
            ),
            "maximum_physical_total_output_tokens": (
                logical_output * attempts_per_call
            ),
            "mean_input_tokens": (
                logical_input / len(input_tokens) if input_tokens else 0.0
            ),
            "p95_input_tokens": ordered[p95_index] if ordered else 0.0,
            "max_input_tokens": max(input_tokens, default=0),
            "logical_estimated_cost_usd": logical_cost,
            "estimated_cost_usd": logical_cost * attempts_per_call,
            "contract_bindings": contract_bindings,
        }
    )
    payload["cost_estimate_sha256"] = sha256_text(canonical_json(payload))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed action sweep: exact dry-run and accepted cost hash before API use."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "experiment.yaml"
    )
    parser.add_argument(
        "--pm-v2-config",
        type=Path,
        default=ROOT / "configs" / "pm_v1_5.yaml",
        help=(
            "Bind a PM-v2 sweep to its preregistered generator and retrieval "
            "settings. Required for data/pm_v1_5 runtime inputs."
        ),
    )
    parser.add_argument(
        "--evidence-filter-checkpoint",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_evidence_filter" / "evidence_filter.joblib",
    )
    parser.add_argument(
        "--evidence-filter-report",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_evidence_filter" / "training_report.json",
    )
    parser.add_argument(
        "--evidence-filter-attestation",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_evidence_filter" / "artifact_attestation.json",
    )
    parser.add_argument(
        "--endpoint",
        "--generator-endpoint",
        dest="endpoint",
        default=None,
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "runtime_states.jsonl",
    )
    parser.add_argument(
        "--backend",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "memory_backend.jsonl",
    )
    parser.add_argument(
        "--strategy-bank",
        type=Path,
        default=ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "outputs" / "pm_v1_5_sweep"
    )
    parser.add_argument(
        "--carry-forward-from",
        type=Path,
        help=(
            "Read-only exact-plan source directory. Only ledger-authenticated "
            "SUCCEEDED calls are inherited; failed calls receive a fresh, "
            "separately approved continuation identity in --out-dir."
        ),
    )
    parser.add_argument("--max-cards", type=int)
    parser.add_argument(
        "--v1-5-full-sweep-scope",
        action="store_true",
        help=(
            "Required PM-v1.5 protocol flag: sweep every state and every "
            "legal action under the automated-review gate and record the "
            "artifact with scope=full."
        ),
    )
    parser.add_argument(
        "--actions",
        help="Optional comma-separated pilot action IDs; full confirmatory runs omit this.",
    )
    parser.add_argument(
        "--pilot-plan",
        type=Path,
        help="Frozen balanced PM-v2 compatibility pilot plan from script 31.",
    )
    parser.add_argument(
        "--completed-pilot-plan",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_development_pilot" / "pilot_plan.json",
        help="Pilot plan whose completed action/judge chain gates a full PM-v2 sweep.",
    )
    parser.add_argument(
        "--pilot-sweep-summary",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_development_pilot" / "summary.json",
    )
    parser.add_argument(
        "--pilot-sweep-attestation",
        type=Path,
        default=(
            ROOT / "outputs" / "pm_v1_5_development_pilot" / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--judge-compatibility-summary",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_development_pilot_judging"
            / "summary.json"
        ),
    )
    parser.add_argument(
        "--judge-compatibility-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_development_pilot_judging"
            / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--pilot-human-spot-check-report",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_development_pilot_human_spot_check"
            / "pilot_human_spot_check_report.json"
        ),
    )
    parser.add_argument(
        "--pilot-human-spot-check-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_development_pilot_human_spot_check"
            / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--judge-schema-smoke-summary",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_development_judge_schema_smoke"
            / "summary.json"
        ),
    )
    parser.add_argument(
        "--judge-schema-smoke-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_development_judge_schema_smoke"
            / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--pm-v2-states",
        type=Path,
        help=(
            "PM-v2 states bound by the human semantic-sanity attestation; defaults "
            "to pm_v2_states.jsonl beside --runtime."
        ),
    )
    parser.add_argument(
        "--evaluator-contexts",
        type=Path,
        help=(
            "Evaluator-only PM-v2 contexts bound by the semantic-sanity "
            "attestation; defaults to evaluator_contexts.jsonl beside --runtime."
        ),
    )
    parser.add_argument(
        "--semantic-sanity-report",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v1_5_semantic_sanity"
        / "semantic_sanity_report.json",
    )
    parser.add_argument(
        "--semantic-sanity-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v1_5_semantic_sanity"
        / "artifact_attestation.json",
    )
    parser.add_argument(
        "--automated-semantic-review-report",
        type=Path,
        help=(
            "Deprecated V4 calibration artifact; forbidden for the formal "
            "V1.5 sweep, which binds actual-468 structured QA v3 instead."
        ),
    )
    parser.add_argument(
        "--automated-semantic-review-attestation",
        type=Path,
        help="Deprecated companion V4 attestation; forbidden.",
    )
    parser.add_argument(
        "--generation-pilot-attestation",
        type=Path,
        help="Deprecated direct input; generation lineage comes through the development-data attestation.",
    )
    parser.add_argument(
        "--actual-corpus-semantic-review-report",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_actual_corpus_semantic_review"
            / "gate_report.json"
        ),
    )
    parser.add_argument(
        "--actual-corpus-semantic-review-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_actual_corpus_semantic_review"
            / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--actual-corpus-qualification-report",
        type=Path,
        help=(
            "Frozen post-hoc instrument-qualification addendum. When supplied "
            "with its attestation, the original review remains FAIL and this "
            "narrow qualification is used instead of claiming gate PASS."
        ),
    )
    parser.add_argument(
        "--actual-corpus-qualification-attestation",
        type=Path,
        help="Companion attestation for --actual-corpus-qualification-report.",
    )
    parser.add_argument(
        "--step0-shortcut-audit-report",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_step0_shortcut_audit"
            / "step0_shortcut_audit.json"
        ),
    )
    parser.add_argument(
        "--step0-shortcut-audit-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_step0_shortcut_audit"
            / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--development-data-attestation",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "artifact_attestation.json",
    )
    parser.add_argument(
        "--rule-grid-preflight-report",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_rule_grid_preflight"
            / "rule_grid_report.json"
        ),
    )
    parser.add_argument(
        "--rule-grid-preflight-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_rule_grid_preflight"
            / "artifact_attestation.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--request-retries",
        type=int,
        help=(
            "Maximum physical attempts per logical generation call. PM-v2 binds "
            "this to development_sweep.request_retries=1."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=None,
        help="Stop the sweep after the first failed logical generation call.",
    )
    parser.add_argument("--strategy-top-k", type=int)
    parser.add_argument(
        "--memory-min-score",
        type=float,
        help="Exclusive lexical relevance threshold; 0 rejects zero-overlap memory items.",
    )
    parser.add_argument(
        "--strategy-min-score",
        type=float,
        help="Exclusive lexical relevance threshold; 0 rejects zero-overlap strategy cards.",
    )
    parser.add_argument(
        "--input-usd-per-mtok",
        type=float,
        help="Optional exact assertion against PM-v2 frozen sweep pricing.",
    )
    parser.add_argument(
        "--output-usd-per-mtok",
        type=float,
        help="Optional exact assertion against PM-v2 frozen sweep pricing.",
    )
    parser.add_argument(
        "--max-api-calls",
        type=int,
        default=6000,
        help="Maximum physical HTTP attempts, including retries.",
    )
    parser.add_argument("--max-estimated-usd", type=float, default=10.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=12000)
    parser.add_argument("--accept-cost-estimate-sha256")
    args = parser.parse_args()

    if args.run and args.overwrite:
        raise RuntimeError("paid PM-v1.5 action-sweep runs prohibit --overwrite")
    if (
        args.carry_forward_from is not None
        and args.carry_forward_from.resolve() == args.out_dir.resolve()
    ):
        raise RuntimeError("action-sweep carry-forward requires a new output directory")
    if args.pilot_plan is not None:
        raise RuntimeError(
            "PM-v1.5 does not run the PM-v2.2 compatibility-pilot branch; "
            "use --v1-5-full-sweep-scope"
        )
    if args.v1_5_full_sweep_scope and (
        args.max_cards is not None or bool(args.actions)
    ):
        raise RuntimeError(
            "--v1-5-full-sweep-scope cannot be combined with --max-cards "
            "or --actions"
        )

    if (
        args.input_usd_per_mtok is not None
        and args.input_usd_per_mtok < 0
    ) or (
        args.output_usd_per_mtok is not None
        and args.output_usd_per_mtok < 0
    ):
        raise ValueError("pricing must be non-negative")
    if args.max_api_calls <= 0 or args.max_estimated_usd < 0:
        raise ValueError("budget limits must be positive")

    experiment_config = load_config(args.config)
    pm_v2_root = (ROOT / "data" / "pm_v1_5").resolve()
    runtime_has_pm_v2_provenance = any(
        bool((row.get("provenance") or {}).get("pm_v2_state_id"))
        for row in iter_jsonl(args.runtime)
    )
    runtime_is_pm_v2 = (
        pm_v2_root in args.runtime.resolve().parents
        or runtime_has_pm_v2_provenance
    )
    if runtime_is_pm_v2 and not args.v1_5_full_sweep_scope:
        raise RuntimeError(
            "PM-v1.5 runtime requires --v1-5-full-sweep-scope; filtered and "
            "legacy compatibility-pilot sweeps are outside this protocol"
        )
    if runtime_is_pm_v2:
        forbid_overwrite_of_spent_attempts(
            args.out_dir / "physical_attempt_ledger.jsonl",
            overwrite=args.overwrite,
            stage="PM-v2 action sweep",
        )
        if args.pilot_plan is None and (
            args.max_cards is not None or bool(args.actions)
        ):
            raise RuntimeError(
                "PM-v2 filtered sweeps require the exact frozen --pilot-plan; "
                "ad-hoc --actions/--max-cards cannot bypass the full-sweep "
                "development pilot gate"
            )
    if runtime_is_pm_v2 and args.pm_v2_config is None:
        raise RuntimeError(
            "PM-v2 action sweeps must pass --pm-v2-config so generator and "
            "retrieval settings are immutable and included in the cost hash"
        )
    if args.pilot_plan is not None and args.pm_v2_config is None:
        raise RuntimeError("--pilot-plan requires --pm-v2-config")
    if (args.pm_v2_states is not None or args.evaluator_contexts is not None) and (
        args.pm_v2_config is None
    ):
        raise RuntimeError(
            "--pm-v2-states/--evaluator-contexts require --pm-v2-config"
        )
    if args.pilot_plan is not None and (args.max_cards is not None or args.actions):
        raise RuntimeError("--pilot-plan cannot be combined with ad-hoc pilot filters")
    pilot_plan = read_json(args.pilot_plan) if args.pilot_plan is not None else None
    formal_v1_5_full_sweep = bool(
        runtime_is_pm_v2
        and args.v1_5_full_sweep_scope
        and pilot_plan is None
        and args.max_cards is None
        and not args.actions
    )

    contract_bindings: dict = {}
    pm_v2_states_path = None
    evaluator_contexts_path = None
    deployable_feature_observability = None
    judge_schema_smoke = None
    evidence_filter_config = None
    memory_helpfulness_model = None
    evidence_filter_model_binding = None
    supporter_generation_contract = None
    transport_max_attempts_per_call: int | None = None
    transport_retry_policy = "single_attempt"
    transport_backoff_seconds: tuple[float, ...] = ()
    consecutive_same_class_circuit_breaker: int | None = None
    longitudinal_transport_execution_contract: dict | None = None
    if args.pm_v2_config is not None:
        pm_config = load_config(args.pm_v2_config)
        require_paid_run_release(
            pm_config,
            config_path=args.pm_v2_config,
            stage="development_action_sweep_generation",
            run=bool(args.run),
            run_identity=args.accept_cost_estimate_sha256,
        )
        if pm_config.get("version") != "pm-v1.5":
            raise RuntimeError("PM-v1.5 action sweep requires a pm-v1.5 config")
        supporter_generation_contract = SupporterGenerationContract.from_config(
            pm_config
        )
        sweep_config = dict(pm_config["development_sweep"])
        if set(sweep_config) != {
            "pricing_usd_per_mtok",
            "seed",
            "request_retries",
            "fail_fast",
        }:
            raise RuntimeError(
                "development_sweep must not duplicate the PM-v2.2 supporter "
                "generation treatment"
            )
        retrieval_config = dict(pm_config["retrieval"])
        # PM-v1.5: Evidence Filter is out of scope and must be disabled
        # identically here and in the external-generation fork
        # (scripts/v1_5/24_run_pm_v2_evoemo_v1_5.py). Leaving this branch's
        # original "must be enabled" requirement in place while the external
        # fork force-disables it would silently make the "ME+RS" trained here
        # a different mechanism (post-retrieval-filtered) from the "ME+RS"
        # evaluated externally (raw retrieval) -- the same class of
        # treatment mismatch this whole V1.5 effort exists to fix.
        evidence_filter_config = EvidenceFilterConfig.from_mapping(
            {**pm_config["evidence_filter"], "enabled": False}
        )
        memory_helpfulness_model = None
        evidence_filter_model_binding = {
            "mode": "disabled_for_pm_v1_5",
            "reason": "Evidence Filter is out of scope for PM-v1.5; PM is a pure pre-retrieval router.",
        }
        sweep_pricing = dict(sweep_config["pricing_usd_per_mtok"])
        if set(sweep_pricing) != {"input", "output"}:
            raise RuntimeError(
                "development_sweep pricing must contain exactly input/output"
            )
        frozen_input_price = float(sweep_pricing["input"])
        frozen_output_price = float(sweep_pricing["output"])
        if frozen_input_price <= 0.0 or frozen_output_price <= 0.0:
            raise RuntimeError("PM-v2 sweep budget-accounting prices must be positive")
        semantic_sanity = None
        runtime_state_lineage = None
        if runtime_is_pm_v2 or pilot_plan is not None:
            pm_v2_states_path = args.pm_v2_states or (
                args.runtime.parent / "pm_v2_states.jsonl"
            )
            evaluator_contexts_path = args.evaluator_contexts or (
                args.runtime.parent / "evaluator_contexts.jsonl"
            )
            audited_states = load_states(pm_v2_states_path)
            runtime_state_lineage = require_pmv2_runtime_state_lineage(
                args.runtime, audited_states
            )
            if (
                args.automated_semantic_review_report is not None
                or args.automated_semantic_review_attestation is not None
                or args.generation_pilot_attestation is not None
            ):
                raise RuntimeError(
                    "formal sweep refuses direct legacy V4/pilot inputs; use the "
                    "attested development corpus plus actual structured QA v3"
                )
            qualification_args = (
                args.actual_corpus_qualification_report,
                args.actual_corpus_qualification_attestation,
            )
            if any(value is not None for value in qualification_args) and not all(
                value is not None for value in qualification_args
            ):
                raise RuntimeError(
                    "actual-corpus qualification report and attestation must be supplied together"
                )
            if all(value is not None for value in qualification_args):
                actual_corpus_verification = require_actual_corpus_posthoc_qualification(
                    args.actual_corpus_qualification_report,
                    args.actual_corpus_qualification_attestation,
                    expected_experiment_config_path=args.config,
                    expected_states_path=pm_v2_states_path,
                    expected_evaluator_contexts_path=evaluator_contexts_path,
                    expected_backend_path=args.backend,
                    expected_strategy_bank_path=args.strategy_bank,
                    expected_pm_config_path=args.pm_v2_config,
                )
            else:
                actual_corpus_verification = require_actual_corpus_semantic_review_pass(
                    args.actual_corpus_semantic_review_report,
                    args.actual_corpus_semantic_review_attestation,
                    expected_experiment_config_path=args.config,
                    expected_states_path=pm_v2_states_path,
                    expected_evaluator_contexts_path=evaluator_contexts_path,
                    expected_backend_path=args.backend,
                    expected_strategy_bank_path=args.strategy_bank,
                    expected_pm_config_path=args.pm_v2_config,
                )
            shortcut_audit_verification = require_step0_shortcut_audit_pass(
                args.step0_shortcut_audit_report,
                args.step0_shortcut_audit_attestation,
                expected_states_path=pm_v2_states_path,
                expected_evaluator_contexts_path=evaluator_contexts_path,
                expected_pm_config_path=args.pm_v2_config,
                expected_generation_attestation_path=(
                    args.development_data_attestation
                ),
            )
            rule_grid_attestation = require_artifact_attestation(
                args.rule_grid_preflight_attestation,
                required_stage="pm_v1_5_pre_training_rule_grid_diagnostic",
                required_output_paths={
                    "rule_grid_report": args.rule_grid_preflight_report
                },
            )
            rule_grid_report = read_json(args.rule_grid_preflight_report)
            if (
                rule_grid_report.get("protocol")
                != RULE_GRID_DIAGNOSTIC_PROTOCOL
                or rule_grid_report.get("status") != "PASS"
                or rule_grid_report.get("outcome_labels_used") is not False
                or rule_grid_report.get("internal_states_used") is not False
                or rule_grid_report.get("selection_or_retuning_authorized")
                is not False
                or rule_grid_report.get("pm_v1_5_config_sha256")
                != sha256_file(args.pm_v2_config)
                or rule_grid_report.get("states_sha256")
                != sha256_file(pm_v2_states_path)
            ):
                raise RuntimeError(
                    "action sweep requires exact PASS outcome-free rule-grid preflight"
                )
            semantic_sanity = {
                "protocol": (
                    "pm-v1.5-actual-corpus-and-shortcut-gate-v3"
                ),
                "status": actual_corpus_verification["status"],
                "actual_corpus_admission_mode": actual_corpus_verification.get(
                    "mode", "ORIGINAL_GATE_PASS"
                ),
                "human_calibration_performed": False,
                "actual_corpus_review_report_sha256": (
                    actual_corpus_verification["report_sha256"]
                ),
                "actual_corpus_review_attestation_sha256": (
                    actual_corpus_verification["attestation_sha256"]
                ),
                "step0_shortcut_audit_report_sha256": (
                    shortcut_audit_verification["report_sha256"]
                ),
                "step0_shortcut_audit_attestation_sha256": (
                    shortcut_audit_verification["attestation_sha256"]
                ),
                "rule_grid_preflight_report_sha256": sha256_file(
                    args.rule_grid_preflight_report
                ),
                "rule_grid_preflight_attestation_sha256": (
                    rule_grid_attestation["attestation_sha256"]
                ),
            }
            if pilot_plan is not None:
                evaluator_index = load_evaluator_context_index(
                    evaluator_contexts_path,
                    states=audited_states,
                    require_exact=True,
                )
                deployable_feature_observability = (
                    audit_deployable_feature_observability(
                        audited_states,
                        evaluator_contexts=evaluator_index,
                        settings=dict(
                            pm_config["development_judging"][
                                "compatibility_pilot"
                            ]["deployable_feature_observability"]
                        ),
                    )
                )
                if (
                    deployable_feature_observability.get("status") != "PASS"
                    or pilot_plan.get("deployable_feature_observability")
                    != deployable_feature_observability
                ):
                    raise RuntimeError(
                        "PM-v2 pilot deployable-feature observability gate "
                        "is absent, stale, or failed"
                    )
                judge_schema_smoke = (
                    require_development_judge_schema_smoke_pass(
                        summary_path=args.judge_schema_smoke_summary,
                        attestation_path=args.judge_schema_smoke_attestation,
                        experiment_config_path=args.config,
                        pm_v2_config_path=args.pm_v2_config,
                        states_path=pm_v2_states_path,
                        backend_path=args.backend,
                        evaluator_contexts_path=evaluator_contexts_path,
                        pilot_plan_path=args.pilot_plan,
                        semantic_sanity_report_path=args.semantic_sanity_report,
                        semantic_sanity_attestation_path=(
                            args.semantic_sanity_attestation
                        ),
                    )
                )
        api_cost_config = dict(pm_config["api_cost_planning"])
        if set(api_cost_config) != {
            "input_token_safety_factor",
            "fail_on_reported_input_overrun",
        }:
            raise RuntimeError("api_cost_planning keys do not match PM-v2")
        input_token_safety_factor = float(
            api_cost_config["input_token_safety_factor"]
        )
        fail_on_reported_input_overrun = bool(
            api_cost_config["fail_on_reported_input_overrun"]
        )
        if input_token_safety_factor < 1.0 or not fail_on_reported_input_overrun:
            raise RuntimeError(
                "PM-v2 requires input token safety factor >=1 and fail-on-overrun"
            )

        frozen_values = {
            "endpoint": supporter_generation_contract.generator_endpoint,
            "temperature": supporter_generation_contract.temperature,
            "max_output_tokens": (
                supporter_generation_contract.max_output_tokens
            ),
            "seed": int(sweep_config["seed"]),
            "request_retries": int(sweep_config["request_retries"]),
            "fail_fast": bool(sweep_config["fail_fast"]),
            "strategy_top_k": int(retrieval_config["strategy_top_k"]),
            "memory_min_score": float(retrieval_config["memory_min_score"]),
            "strategy_min_score": float(retrieval_config["strategy_min_score"]),
        }
        requested_values = {
            "endpoint": args.endpoint,
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
            "seed": args.seed,
            "request_retries": args.request_retries,
            "fail_fast": args.fail_fast,
            "strategy_top_k": args.strategy_top_k,
            "memory_min_score": args.memory_min_score,
            "strategy_min_score": args.strategy_min_score,
        }
        for name, requested in requested_values.items():
            if requested is not None and requested != frozen_values[name]:
                raise RuntimeError(
                    f"{name} override differs from the PM-v2 config; edit and "
                    "review the preregistration before generating any labels"
                )
        endpoint_name = frozen_values["endpoint"]
        temperature = frozen_values["temperature"]
        max_output_tokens = frozen_values["max_output_tokens"]
        seed = frozen_values["seed"]
        request_retries = frozen_values["request_retries"]
        if request_retries != 1:
            raise RuntimeError(
                "PM-v2 development_sweep.request_retries must equal 1 so the "
                "dry-run is a physical-attempt cap"
            )
        fail_fast = frozen_values["fail_fast"]
        if not fail_fast:
            raise RuntimeError(
                "PM-v2 development_sweep.fail_fast must be true so an exact "
                "action-matrix failure cannot spend the remaining budget"
            )
        if formal_v1_5_full_sweep:
            # The old config values remain frozen so the scientific corpus and
            # response-mechanism identities do not drift.  Formal V1.5 uses a
            # separately hashed execution-only resilience contract: each
            # physical HTTP call still asks for exactly the same model,
            # prompt, seed and output cap, while transient transport failures
            # receive a bounded fresh attempt recorded in the durable ledger.
            longitudinal_transport_execution_contract = (
                _longitudinal_transport_execution_contract()
            )
            transport_max_attempts_per_call = (
                LONGITUDINAL_TRANSPORT_MAX_ATTEMPTS_PER_CALL
            )
            transport_retry_policy = "bounded_transport"
            transport_backoff_seconds = (
                LONGITUDINAL_TRANSPORT_BACKOFF_SECONDS
            )
            consecutive_same_class_circuit_breaker = (
                LONGITUDINAL_TRANSPORT_CIRCUIT_BREAKER
            )
            fail_fast = False
        strategy_top_k = frozen_values["strategy_top_k"]
        memory_min_score = frozen_values["memory_min_score"]
        strategy_min_score = frozen_values["strategy_min_score"]
        if (
            args.input_usd_per_mtok is not None
            and float(args.input_usd_per_mtok) != frozen_input_price
        ):
            raise RuntimeError("input pricing override differs from PM-v2 YAML")
        if (
            args.output_usd_per_mtok is not None
            and float(args.output_usd_per_mtok) != frozen_output_price
        ):
            raise RuntimeError("output pricing override differs from PM-v2 YAML")
        args.input_usd_per_mtok = frozen_input_price
        args.output_usd_per_mtok = frozen_output_price
        generator_endpoint = endpoint_from_config(
            experiment_config, supporter_generation_contract.generator_endpoint
        )
        generator_endpoint_sha256 = sha256_text(
            canonical_json(
                {
                    "model": generator_endpoint.model,
                    "family": generator_endpoint.family,
                    "base_url": generator_endpoint.base_url,
                }
            )
        )
        response_mechanism_contract = build_response_mechanism_contract(
            project_root=ROOT,
            supporter_generation_contract=supporter_generation_contract,
            generator_endpoint_sha256=generator_endpoint_sha256,
            strategy_bank_sha256=sha256_file(args.strategy_bank),
            memory_min_score=float(retrieval_config["memory_min_score"]),
            strategy_min_score=float(retrieval_config["strategy_min_score"]),
            strategy_top_k=int(retrieval_config["strategy_top_k"]),
            evidence_filter_enabled=bool(evidence_filter_config.enabled),
        )
        contract_bindings = {
            "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
            "pm_v2_version": str(pm_config["version"]),
            "response_mechanism_contract": response_mechanism_contract,
            "supporter_generation_treatment": (
                supporter_generation_contract.payload()
            ),
            "supporter_generation_treatment_sha256": (
                supporter_generation_contract.digest()
            ),
            "development_sweep": sweep_config,
            "retrieval": retrieval_config,
            "evidence_filter": evidence_filter_config.payload(),
            "evidence_filter_config_sha256": evidence_filter_config.digest(),
            "evidence_filter_model": evidence_filter_model_binding,
            **(
                {
                    "semantic_sanity": semantic_sanity,
                    "runtime_state_lineage": runtime_state_lineage,
                }
                if semantic_sanity is not None
                else {}
            ),
            **(
                {
                    "deployable_feature_observability": (
                        deployable_feature_observability
                    ),
                    "judge_schema_smoke": judge_schema_smoke,
                }
                if pilot_plan is not None
                else {}
            ),
            "api_cost_planning": api_cost_config,
            "scope": (
                "compatibility_pilot"
                if pilot_plan is not None
                else "pilot"
                if args.max_cards is not None or args.actions
                else "full"
            ),
        }
        if longitudinal_transport_execution_contract is not None:
            contract_bindings["transport_execution_contract"] = (
                longitudinal_transport_execution_contract
            )
    else:
        if args.input_usd_per_mtok is None or args.output_usd_per_mtok is None:
            raise RuntimeError(
                "non-PM-v2 sweeps require explicit input/output pricing"
            )
        endpoint_name = args.endpoint or "generator"
        temperature = 0.0 if args.temperature is None else float(args.temperature)
        max_output_tokens = (
            300 if args.max_output_tokens is None else int(args.max_output_tokens)
        )
        seed = 4311 if args.seed is None else int(args.seed)
        request_retries = (
            3 if args.request_retries is None else int(args.request_retries)
        )
        fail_fast = bool(args.fail_fast) if args.fail_fast is not None else False
        strategy_top_k = 3 if args.strategy_top_k is None else int(args.strategy_top_k)
        memory_min_score = args.memory_min_score
        strategy_min_score = args.strategy_min_score
        evidence_filter_config = None
        memory_helpfulness_model = None
        input_token_safety_factor = 1.0
        fail_on_reported_input_overrun = False

    endpoint = endpoint_from_config(experiment_config, endpoint_name)
    action_filter = None
    card_filter = None
    if pilot_plan is not None:
        assert supporter_generation_contract is not None
        if pilot_plan.get("status") != "READY":
            raise RuntimeError("PM-v2 compatibility pilot plan is not READY")
        if pilot_plan.get("protocol") != (
            "pm_v2_development_compatibility_pilot_v2_treatment_bound"
        ):
            raise RuntimeError("PM-v2 compatibility pilot protocol is stale")
        if (
            pilot_plan.get("supporter_generation_treatment")
            != supporter_generation_contract.payload()
            or pilot_plan.get("supporter_generation_treatment_sha256")
            != supporter_generation_contract.digest()
        ):
            raise RuntimeError(
                "PM-v2 pilot plan supporter-generation treatment mismatch"
            )
        expected_hashes = {
            "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
            "runtime_sha256": sha256_file(args.runtime),
            "backend_sha256": sha256_file(args.backend),
            "strategy_bank_sha256": sha256_file(args.strategy_bank),
        }
        mismatches = {
            name: {"expected": expected, "observed": pilot_plan.get(name)}
            for name, expected in expected_hashes.items()
            if pilot_plan.get(name) != expected
        }
        if mismatches:
            raise RuntimeError(f"PM-v2 pilot plan lineage mismatch: {mismatches}")
        if pilot_plan.get("semantic_sanity") != contract_bindings.get(
            "semantic_sanity"
        ):
            raise RuntimeError(
                "PM-v2 pilot plan semantic-sanity attestation mismatch"
            )
        if pilot_plan.get("runtime_state_lineage") != contract_bindings.get(
            "runtime_state_lineage"
        ):
            raise RuntimeError("PM-v2 pilot plan runtime/state lineage mismatch")
        pilot_payload = {
            key: value for key, value in pilot_plan.items() if key != "pilot_plan_sha256"
        }
        if pilot_plan.get("pilot_plan_sha256") != sha256_text(
            canonical_json(pilot_payload)
        ):
            raise RuntimeError("PM-v2 pilot plan self-hash mismatch")
        action_filter = {str(value) for value in pilot_plan["actions"]}
        card_filter = {
            str(row["card_id"]) for row in pilot_plan["selected_states"]
        }
        contract_bindings["pilot_plan_sha256"] = str(
            pilot_plan["pilot_plan_sha256"]
        )
        contract_bindings["pilot_expected_keys_sha256"] = str(
            pilot_plan["expected_keys_sha256"]
        )
    elif args.actions:
        action_filter = {value.strip() for value in args.actions.split(",") if value.strip()}
        if not action_filter:
            raise ValueError("--actions did not contain any action IDs")
        for action_id in action_filter:
            parse_action_id(action_id)
    elif runtime_is_pm_v2 and args.max_cards is None:
        if (
            not args.v1_5_full_sweep_scope
            or semantic_sanity is None
            or runtime_state_lineage is None
        ):
            raise RuntimeError("PM-v1.5 full sweep lacks its frozen review gate")
        contract_bindings["v1_5_full_sweep_gate"] = {
            "protocol": "pm-v1.5-full-sweep-gate-v3",
            "status": "PASS",
            "scope": "full",
            "human_calibration_performed": False,
            "actual_corpus_review_attestation_sha256": semantic_sanity[
                "actual_corpus_review_attestation_sha256"
            ],
            "actual_corpus_review_report_sha256": semantic_sanity[
                "actual_corpus_review_report_sha256"
            ],
            "step0_shortcut_audit_attestation_sha256": semantic_sanity[
                "step0_shortcut_audit_attestation_sha256"
            ],
            "step0_shortcut_audit_report_sha256": semantic_sanity[
                "step0_shortcut_audit_report_sha256"
            ],
            **(
                {
                    "actual_corpus_admission_mode": semantic_sanity[
                        "actual_corpus_admission_mode"
                    ],
                    "actual_corpus_admission_status": semantic_sanity["status"],
                    "original_actual_corpus_gate_status": "FAIL",
                }
                if semantic_sanity.get("actual_corpus_admission_mode")
                == "POSTHOC_INSTRUMENT_QUALIFICATION"
                else {}
            ),
        }
    estimate, rows = plan_action_sweep(
        args.runtime,
        args.backend,
        args.strategy_bank,
        endpoint=endpoint,
        max_cards=args.max_cards,
        card_filter=card_filter,
        action_filter=action_filter,
        temperature=temperature,
        max_tokens=max_output_tokens,
        seed=seed,
        request_retries=request_retries,
        transport_max_attempts_per_call=transport_max_attempts_per_call,
        fail_fast=fail_fast,
        input_token_safety_factor=input_token_safety_factor,
        fail_on_reported_input_overrun=fail_on_reported_input_overrun,
        strategy_top_k=strategy_top_k,
        memory_min_score=memory_min_score,
        strategy_min_score=strategy_min_score,
        evidence_filter_config=evidence_filter_config,
        memory_helpfulness_model=memory_helpfulness_model,
        supporter_generation_contract=supporter_generation_contract,
        input_usd_per_mtok=args.input_usd_per_mtok,
        output_usd_per_mtok=args.output_usd_per_mtok,
        contract_bindings=contract_bindings,
    )
    carry_forward = _load_action_sweep_carry_forward(
        carry_forward_dir=args.carry_forward_from,
        call_plan=rows,
    )
    estimate = _continuation_cost_estimate(estimate, rows, carry_forward)
    contract_bindings = dict(estimate["contract_bindings"])
    gate = _budget_gate(
        estimate,
        max_api_calls=args.max_api_calls,
        max_estimated_usd=args.max_estimated_usd,
        max_input_tokens_per_call=args.max_input_tokens_per_call,
    )
    result = {**estimate, "budget_gate": gate}

    if args.dry_run:
        _persist_or_validate_dry_run(args.out_dir, result, rows)
        print(result)
        if gate["status"] != "PASS":
            raise RuntimeError("action-sweep dry-run failed the frozen budget gate")
        return

    if gate["status"] != "PASS":
        raise RuntimeError("action-sweep API run blocked by budget gate")
    _require_saved_dry_run(args.out_dir, result, rows)
    expected_hash = str(result["cost_estimate_sha256"])
    if not args.accept_cost_estimate_sha256:
        raise RuntimeError(
            "API mode is fail-closed: pass --accept-cost-estimate-sha256 "
            f"{expected_hash} from the matching dry-run"
        )
    if args.accept_cost_estimate_sha256 != expected_hash:
        raise RuntimeError(
            "accepted cost estimate hash does not match current action-sweep plan"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = run_action_sweep(
        args.runtime,
        args.backend,
        args.strategy_bank,
        args.out_dir / "action_outcomes.jsonl",
        args.out_dir / "raw_api_calls.jsonl",
        args.out_dir / "summary.json",
        endpoint=endpoint,
        max_cards=args.max_cards,
        card_filter=card_filter,
        action_filter=action_filter,
        temperature=temperature,
        max_tokens=max_output_tokens,
        seed=seed,
        request_retries=request_retries,
        transport_max_attempts_per_call=transport_max_attempts_per_call,
        fail_fast=fail_fast,
        input_token_safety_factor=input_token_safety_factor,
        fail_on_reported_input_overrun=fail_on_reported_input_overrun,
        strategy_top_k=strategy_top_k,
        memory_min_score=memory_min_score,
        strategy_min_score=strategy_min_score,
        evidence_filter_config=evidence_filter_config,
        memory_helpfulness_model=memory_helpfulness_model,
        supporter_generation_contract=supporter_generation_contract,
        overwrite=args.overwrite,
        max_physical_api_attempts=args.max_api_calls,
        transport_retry_policy=transport_retry_policy,
        transport_backoff_seconds=transport_backoff_seconds,
        consecutive_same_class_circuit_breaker=(
            consecutive_same_class_circuit_breaker
        ),
        contract_bindings={
            **contract_bindings,
            "accepted_cost_estimate_sha256": expected_hash,
            "pricing": result["pricing"],
        },
        carry_forward_terminal_rows=carry_forward["carried_terminal_rows"],
        carry_forward_binding=carry_forward["binding"],
    )
    print(summary)


if __name__ == "__main__":
    main()
