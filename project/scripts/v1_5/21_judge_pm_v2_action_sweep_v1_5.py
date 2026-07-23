#!/usr/bin/env python3
"""Judge the complete PM-v1.5 action sweep with two model families.

The input sweep must carry the PM-v1.5 full-scope gate and the same attested
automated semantic review required here. The legacy PM-v2.2 compatibility-
pilot branch is intentionally rejected on this fast track.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from metacom_pm.api import (
    ProviderRequestError,
    RetryableProviderError,
    StructuredOutputValidationError,
    chat_request_payload,
    make_client,
    request_payload_has_schema,
)
from metacom_pm.attempt_ledger import (
    PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
    PersistentAttemptLedger,
    forbid_overwrite_of_spent_attempts,
    physical_call_key as make_physical_call_key,
    reported_prompt_token_error,
)
from metacom_pm.bounded_retry import (
    BOUNDED_PROVIDER_OUTPUT_RETRY_CLASSES,
    RETRYABLE_UP_TO_FULL_BUDGET,
    RETRY_CONTRACT_PROTOCOL,
    TERMINAL_DISPOSITION,
    call_retry_blocker,
    execute_with_bounded_retry,
    failure_metadata,
    retry_ledger_summary,
)
from metacom_pm.artifacts import create_artifact_attestation, require_artifact_attestation
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import ActionOutcome
from metacom_pm.io import (
    append_jsonl,
    canonical_json,
    ensure_run_manifest,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.pm_v2_data import load_evaluator_context_index, load_states
from metacom_pm.pm_v2_development_gate import require_development_pilot_gate
from metacom_pm.pm_v2_judge_schema_smoke import (
    require_development_judge_schema_smoke_pass,
)
from metacom_pm.pm_v2_audit import audit_pilot_label_value_feasibility
from metacom_pm.pm_v2_judging import (
    JudgeResult,
    ResponseJudgeOutput,
    RiskJudgeOutput,
    build_action_label,
    build_response_messages,
    build_risk_messages,
    composite_spec_from_config,
    composite_weights_hash,
    judge_one,
    labeling_settings_from_config,
    prompt_contract_hash,
    validate_raw_judge_family_health,
    validate_raw_judge_family_subgroup_health,
    validate_action_applicable_risk_signal,
    validate_judge_table,
)
from metacom_pm.pm_v2_semantic_audit import (
    require_pmv2_runtime_state_lineage,
    require_semantic_sanity_pass,
)
from metacom_pm.pm_v1_5_shortcut_audit import (
    require_step0_shortcut_audit_pass,
)
from metacom_pm.paid_run_release import require_paid_run_release
from metacom_pm.v1_5_actual_corpus_review import (
    require_actual_corpus_semantic_review_pass,
)
from metacom_pm.v1_5_actual_corpus_qualification import (
    require_actual_corpus_posthoc_qualification,
)
from metacom_pm.v1_5_judge_isolation import require_judge_role_isolation
from metacom_pm.v1_5_deterministic_sharding import (
    load_validated_sharding_contract,
)
from metacom_pm.internal_holdout import seal_internal_label_bundle
from metacom_pm.text import conservative_token_bound, estimate_tokens

ROOT = Path(__file__).resolve().parents[2]

DEVELOPMENT_JUDGING_TRANSPORT_PROTOCOL = (
    "pm-v1.5-development-sweep-judging-transport-execution-v1"
)
DEVELOPMENT_JUDGING_MAXIMUM_PHYSICAL_ATTEMPTS = 4
DEVELOPMENT_JUDGING_BACKOFF_SECONDS = (10.0, 30.0, 60.0)
DEVELOPMENT_JUDGING_CONSECUTIVE_FAILURE_BREAKER = 5
TRAIN_CALIBRATION_SCOPE = "train_calibration"
SEALED_INTERNAL_TEST_SCOPE = "sealed_internal_test"
FORMAL_JUDGING_SCOPES = (TRAIN_CALIBRATION_SCOPE, SEALED_INTERNAL_TEST_SCOPE)
DEVELOPMENT_JUDGING_ISOLATABLE_FAILURE_CLASSES = frozenset(
    set(RETRYABLE_UP_TO_FULL_BUDGET)
    | set(BOUNDED_PROVIDER_OUTPUT_RETRY_CLASSES)
    | {"structured_output_validation_error", "output_token_limit"}
)


def development_judging_transport_contract(
    *, provider_output_attempts_by_family: Mapping[str, int]
) -> dict[str, Any]:
    """Bind execution resilience without changing the scientific judge contract."""

    if not provider_output_attempts_by_family or any(
        int(value) < 1
        or int(value) > DEVELOPMENT_JUDGING_MAXIMUM_PHYSICAL_ATTEMPTS
        for value in provider_output_attempts_by_family.values()
    ):
        raise ValueError(
            "per-family provider-output attempts must fit the physical-attempt bound"
        )
    code_paths = {
        "runner": Path(__file__).resolve(),
        "api": ROOT / "src" / "metacom_pm" / "api.py",
        "attempt_ledger": ROOT / "src" / "metacom_pm" / "attempt_ledger.py",
        "bounded_retry": ROOT / "src" / "metacom_pm" / "bounded_retry.py",
        "judging": ROOT / "src" / "metacom_pm" / "pm_v2_judging.py",
    }
    code_manifest = {
        name: {
            "relative_path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        }
        for name, path in sorted(code_paths.items())
    }
    payload: dict[str, Any] = {
        "protocol": DEVELOPMENT_JUDGING_TRANSPORT_PROTOCOL,
        "retry_contract_protocol": RETRY_CONTRACT_PROTOCOL,
        "maximum_physical_attempts_per_logical_call": (
            DEVELOPMENT_JUDGING_MAXIMUM_PHYSICAL_ATTEMPTS
        ),
        "transport_backoff_seconds": list(DEVELOPMENT_JUDGING_BACKOFF_SECONDS),
        "retryable_transport_classes": sorted(RETRYABLE_UP_TO_FULL_BUDGET),
        "isolatable_provider_failure_classes": sorted(
            DEVELOPMENT_JUDGING_ISOLATABLE_FAILURE_CLASSES
        ),
        "provider_output_maximum_attempts_by_family": {
            str(family): int(value)
            for family, value in sorted(provider_output_attempts_by_family.items())
        },
        "terminal_content_or_schema_failure_is_not_blindly_retried": True,
        "continue_after_isolated_provider_failure": True,
        "consecutive_same_class_circuit_breaker": (
            DEVELOPMENT_JUDGING_CONSECUTIVE_FAILURE_BREAKER
        ),
        "client_internal_retries": 1,
        "scientific_judge_contract_unchanged": True,
        "code_manifest": code_manifest,
        "code_manifest_sha256": sha256_text(canonical_json(code_manifest)),
    }
    payload["contract_sha256"] = sha256_text(canonical_json(payload))
    return payload


def _isolatable_provider_failure_class(exc: Exception) -> str | None:
    """Return only provider-surface failures safe to isolate to one matrix row."""

    if isinstance(exc, RetryableProviderError):
        retry_class = str(exc.last_retry_class)
        return (
            retry_class
            if retry_class in DEVELOPMENT_JUDGING_ISOLATABLE_FAILURE_CLASSES
            else None
        )
    if isinstance(exc, StructuredOutputValidationError):
        return "structured_output_validation_error"
    # A non-429 4xx generally means the request contract or credential is
    # wrong for every following row.  It must stop the run, not be diluted as
    # one isolated observation in a very large matrix.
    if isinstance(exc, ProviderRequestError):
        return None
    return None


def _persisted_isolatable_failure_class(
    ledger: PersistentAttemptLedger, call_key: str
) -> str | None:
    terminal = ledger.terminal_row(call_key) or {}
    metadata = terminal.get("metadata") or {}
    retry_class = str(metadata.get("retry_class") or "")
    return (
        retry_class
        if retry_class in DEVELOPMENT_JUDGING_ISOLATABLE_FAILURE_CLASSES
        else None
    )


def development_judging_cost_bounds(
    cost_rows: list[Mapping[str, Any]],
) -> dict[str, int | float]:
    """Return logical and all-attempt bounds using stable float summation."""

    logical_input_tokens = sum(int(row["input_tokens_est"]) for row in cost_rows)
    logical_output_tokens = sum(int(row["max_output_tokens"]) for row in cost_rows)
    logical_cost_usd = math.fsum(float(row["maximum_cost_usd"]) for row in cost_rows)
    attempts = DEVELOPMENT_JUDGING_MAXIMUM_PHYSICAL_ATTEMPTS
    return {
        "logical_input_tokens": logical_input_tokens,
        "logical_output_tokens": logical_output_tokens,
        "logical_cost_usd": logical_cost_usd,
        "maximum_physical_attempts": len(cost_rows) * attempts,
        "maximum_input_tokens": logical_input_tokens * attempts,
        "maximum_output_tokens": logical_output_tokens * attempts,
        "maximum_cost_usd": logical_cost_usd * attempts,
    }


def select_formal_judging_outcomes(
    outcomes: list[ActionOutcome],
    *,
    state_by_card: Mapping[str, Any],
    label_scope: str,
) -> list[ActionOutcome]:
    """Select exactly one pre-registered split scope without peeking at labels."""

    if label_scope not in FORMAL_JUDGING_SCOPES:
        raise ValueError(f"unsupported formal judging scope: {label_scope}")
    allowed_splits = (
        {"train", "calibration"}
        if label_scope == TRAIN_CALIBRATION_SCOPE
        else {"internal_test"}
    )
    selected = [
        outcome
        for outcome in outcomes
        if str(state_by_card[outcome.card_id].split.value) in allowed_splits
    ]
    observed_splits = {
        str(state_by_card[outcome.card_id].split.value) for outcome in selected
    }
    if observed_splits != allowed_splits:
        raise RuntimeError(
            f"formal judging scope {label_scope} is incomplete: "
            f"{sorted(observed_splits)} != {sorted(allowed_splits)}"
        )
    if not selected:
        raise RuntimeError(f"formal judging scope {label_scope} selected no outcomes")
    return selected


def evaluate_raw_judge_gates(
    canonical_raw_rows: list[Mapping[str, Any]],
    *,
    outcomes: list[ActionOutcome],
    endpoints: list[Any],
    labeling: Mapping[str, Any],
    composite_spec: Any,
    compatibility_pilot: bool,
) -> dict[str, Any]:
    """Evaluate development-label health only on train/calibration rows."""

    expected_families = [str(endpoint.family) for endpoint in endpoints]
    common = {
        "expected_families": expected_families,
        "duplicate_exact_match_rate": labeling["duplicate_exact_match_rate"],
        "maximum_absolute_dimension_correlation": labeling[
            "maximum_absolute_dimension_correlation"
        ],
        "composite_support_exact_match_rate": labeling[
            "composite_support_exact_match_rate"
        ],
        "maximum_absolute_composite_support_correlation": labeling[
            "maximum_absolute_composite_support_correlation"
        ],
        "reject_constant_response_dimensions": labeling[
            "reject_constant_response_dimensions"
        ],
        "reject_constant_risk_dimensions": labeling[
            "reject_constant_risk_dimensions"
        ],
        "composite_spec": composite_spec,
        "raise_on_failure": not compatibility_pilot,
    }
    global_gate = validate_raw_judge_family_health(canonical_raw_rows, **common)
    action_gate = validate_raw_judge_family_subgroup_health(
        canonical_raw_rows,
        subgroup_key="action_id",
        expected_subgroups=sorted({row.action_id for row in outcomes}),
        **common,
    )
    risk_gate = validate_action_applicable_risk_signal(
        canonical_raw_rows,
        expected_actions=sorted({row.action_id for row in outcomes}),
        expected_families=expected_families,
        minimum_signal_rate=labeling[
            "minimum_action_applicable_risk_signal_rate"
        ],
        minimum_distinct_values=labeling[
            "minimum_action_applicable_risk_distinct_values"
        ],
        raise_on_failure=not compatibility_pilot,
    )
    return {
        "status": (
            "PASS"
            if global_gate.get("status") == "PASS"
            and action_gate.get("status") == "PASS"
            and (compatibility_pilot or risk_gate.get("status") == "PASS")
            else "FAIL"
        ),
        "global": global_gate,
        "family_by_action": action_gate,
        "action_applicable_risk_signal": {
            **risk_gate,
            "enforced": not compatibility_pilot,
        },
    }


def advance_development_judging_failure_streak(
    *,
    previous_class: str | None,
    previous_count: int,
    retry_class: str,
) -> tuple[str, int]:
    """Advance the cross-call breaker and stop on a systemic-looking streak."""

    count = previous_count + 1 if retry_class == previous_class else 1
    if count >= DEVELOPMENT_JUDGING_CONSECUTIVE_FAILURE_BREAKER:
        raise RuntimeError(
            f"circuit breaker: {retry_class} recurred {count} times in a row "
            "across different development-judging calls"
        )
    return retry_class, count


def raw_key(row):
    return str(row["state_id"]), str(row["action_id"]), str(row["judge_family"])


def raw_row_succeeded(row) -> bool:
    return bool(row.get("schema_success")) and row.get("status") == "SUCCESS"


def preflight_development_judge_clients(
    endpoints: Mapping[str, Any], *, client_factory=make_client
) -> dict[str, Any]:
    """Resolve every family credential, then build every client before reserve."""

    normalized = {str(family): endpoint for family, endpoint in endpoints.items()}
    if len(normalized) != len(endpoints) or not normalized:
        raise RuntimeError("development judge endpoints have invalid family keys")
    for endpoint in normalized.values():
        endpoint.api_key
    clients: dict[str, Any] = {}
    try:
        for family, endpoint in normalized.items():
            clients[family] = client_factory(endpoint)
    except Exception:
        for client in clients.values():
            client.close()
        raise
    return clients


def persist_or_validate_judge_dry_run(
    *,
    ledger_path: str | Path,
    estimate_path: str | Path,
    call_plan_path: str | Path,
    cost_estimate: Mapping[str, Any],
    budget_gate: Mapping[str, Any],
    call_plan: list[dict[str, Any]],
) -> str:
    """Never mutate the accepted full plan after a physical attempt exists."""

    ledger_path = Path(ledger_path)
    estimate_path = Path(estimate_path)
    call_plan_path = Path(call_plan_path)
    expected_estimate = {**dict(cost_estimate), "budget_gate": dict(budget_gate)}
    spent = ledger_path.is_file() and ledger_path.stat().st_size > 0
    if spent:
        if not estimate_path.is_file() or not call_plan_path.is_file():
            raise RuntimeError(
                "spent development-judge ledger freezes the accepted dry-run, "
                "but its estimate or full call plan is missing"
            )
        if read_json(estimate_path) != expected_estimate or list(
            iter_jsonl(call_plan_path)
        ) != call_plan:
            raise RuntimeError(
                "spent development-judge ledger freezes the original exact "
                "estimate and full call plan; current drift is rejected"
            )
        return "VALIDATED_EXISTING"
    write_json(estimate_path, expected_estimate)
    write_jsonl(call_plan_path, call_plan)
    return "WRITTEN"


def load_development_judging_carry_forward(
    *,
    carry_forward_dir: Path | None,
    call_plan: list[dict[str, Any]],
    stage: str,
) -> dict[str, Any]:
    """Load only exact-plan successful calls from one immutable prior ledger."""

    if carry_forward_dir is None:
        return {
            "source_directory": None,
            "source_ledger_sha256": None,
            "carried_call_keys": set(),
            "terminal_rows": {},
        }
    old_plan_path = carry_forward_dir / "call_plan.jsonl"
    old_ledger_path = carry_forward_dir / "judge_call_ledger.jsonl"
    if not old_plan_path.is_file() or not old_ledger_path.is_file():
        raise RuntimeError(
            "development-judging carry-forward source lacks call plan or ledger"
        )
    if list(iter_jsonl(old_plan_path)) != call_plan:
        raise RuntimeError(
            "development-judging carry-forward call plan is not byte-equivalent"
        )
    plan_by_key = {str(row["physical_call_key"]): row for row in call_plan}
    if len(plan_by_key) != len(call_plan):
        raise RuntimeError("development-judging call plan has duplicate physical keys")
    old_ledger = PersistentAttemptLedger(
        old_ledger_path,
        stage=stage,
        expected_calls={
            key: DEVELOPMENT_JUDGING_MAXIMUM_PHYSICAL_ATTEMPTS
            for key in plan_by_key
        },
        maximum_total_attempts=10**9,
    )
    carried: set[str] = set()
    terminals: dict[str, dict[str, Any]] = {}
    for physical_key, plan in plan_by_key.items():
        if not old_ledger.succeeded(physical_key):
            continue
        terminal = old_ledger.terminal_row(physical_key) or {}
        expected_record_ids = {
            "state_id": str(plan["state_id"]),
            "action_id": str(plan["action_id"]),
            "judge_family": str(plan["judge_family"]),
            "judge_type": str(plan["judge_type"]),
        }
        if (
            terminal.get("record_ids") != expected_record_ids
            or terminal.get("prompt_sha256") != plan["prompt_hash"]
            or not isinstance((terminal.get("result") or {}).get("parsed"), Mapping)
            or not terminal.get("request_hash")
        ):
            raise RuntimeError(
                "development-judging carry-forward success lacks exact provenance: "
                f"{physical_key}"
            )
        carried.add(physical_key)
        terminals[physical_key] = terminal
    return {
        "source_directory": str(carry_forward_dir),
        "source_ledger_sha256": sha256_file(old_ledger_path),
        "carried_call_keys": carried,
        "terminal_rows": terminals,
    }


def require_exact_saved_judge_dry_run(
    *,
    estimate_path: str | Path,
    call_plan_path: str | Path,
    cost_estimate: Mapping[str, Any],
    budget_gate: Mapping[str, Any],
    call_plan: list[dict[str, Any]],
) -> None:
    estimate_path = Path(estimate_path)
    call_plan_path = Path(call_plan_path)
    if not estimate_path.is_file() or not call_plan_path.is_file():
        raise RuntimeError("judge API run requires a matching saved --dry-run")
    expected_estimate = {**dict(cost_estimate), "budget_gate": dict(budget_gate)}
    if read_json(estimate_path) != expected_estimate:
        raise RuntimeError("saved PM-v2 judge dry-run exact estimate is stale")
    if list(iter_jsonl(call_plan_path)) != call_plan:
        raise RuntimeError("saved PM-v2 judge full call plan is stale")


def _attested_path(
    attestation: Mapping[str, Any], section: str, logical_name: str
) -> Path:
    record = (attestation.get(section) or {}).get(logical_name)
    if not isinstance(record, Mapping) or not record.get("path"):
        raise RuntimeError(
            f"action sweep attestation lacks {section}.{logical_name}"
        )
    return Path(str(record["path"])).resolve()


def require_action_sweep_source_chain(
    *,
    outcomes_path: str | Path,
    summary_path: str | Path,
    manifest_path: str | Path,
    attestation_path: str | Path,
) -> dict[str, Any]:
    """Content-address every paid sweep artifact before judging it."""

    outcomes_path = Path(outcomes_path).resolve()
    summary_path = Path(summary_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    attestation_path = Path(attestation_path).resolve()
    verification = require_artifact_attestation(
        attestation_path,
        required_stage="action_sweep",
        required_output_paths={
            "action_outcomes": outcomes_path,
            "summary": summary_path,
        },
    )
    attestation = read_json(attestation_path)
    if _attested_path(attestation, "inputs", "run_manifest") != manifest_path:
        raise RuntimeError("action sweep attestation manifest path mismatch")
    for name in ("raw_calls", "physical_attempt_ledger"):
        _attested_path(attestation, "outputs", name)
    manifest = read_json(manifest_path)
    manifest_payload = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if (
        manifest.get("stage") != "action_sweep"
        or manifest.get("manifest_sha256")
        != sha256_text(canonical_json(manifest_payload))
    ):
        raise RuntimeError("action sweep manifest is stale or self-inconsistent")
    summary = read_json(summary_path)
    outcomes = list(iter_jsonl(outcomes_path))
    expected = int(summary.get("expected_outcomes") or -1)
    if (
        summary.get("status") != "COMPLETE"
        or int(summary.get("completed_outcomes") or -1) != expected
        or len(outcomes) != expected
        or len(
            {
                (str(row.get("card_id")), str(row.get("action_id")))
                for row in outcomes
            }
        )
        != expected
    ):
        raise RuntimeError("action sweep source matrix is not exact COMPLETE")
    manifest_bindings = manifest.get("contract_bindings") or {}
    if (
        summary.get("contract_bindings") != manifest_bindings
        or (attestation.get("parameters") or {}).get("contract_bindings")
        != manifest_bindings
        or int((attestation.get("expected") or {}).get("outcomes") or -1)
        != expected
    ):
        raise RuntimeError("action sweep contract bindings are inconsistent")
    return {
        "status": "PASS",
        "attestation_sha256": verification["attestation_sha256"],
        "summary_sha256": sha256_file(summary_path),
        "manifest_sha256": sha256_file(manifest_path),
        "outcomes_sha256": sha256_file(outcomes_path),
        "contract_bindings": manifest_bindings,
        "expected_outcomes": expected,
    }


def require_full_sweep_development_binding(
    sweep_source_chain: Mapping[str, Any],
    development_pilot_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject full judging when the paid sweep bypassed the current gate."""

    observed = (sweep_source_chain.get("contract_bindings") or {}).get(
        "development_pilot_gate"
    )
    expected = dict(development_pilot_gate)
    if (
        observed != expected
        or str((observed or {}).get("binding_sha256") or "")
        != str(expected.get("binding_sha256") or "")
        or len(str(expected.get("binding_sha256") or "")) != 64
    ):
        raise RuntimeError(
            "full action sweep is not bound to the current complete "
            "development pilot/human gate"
        )
    return expected


def require_v1_5_full_sweep_binding(
    sweep_source_chain: Mapping[str, Any],
    *,
    actual_corpus_review_report_sha256: str,
    actual_corpus_review_attestation_sha256: str,
    step0_shortcut_audit_report_sha256: str,
    step0_shortcut_audit_attestation_sha256: str,
    actual_corpus_admission_mode: str | None = None,
    actual_corpus_admission_status: str | None = None,
) -> dict[str, Any]:
    """Require an honestly full V1.5 matrix bound to the current review."""

    bindings = sweep_source_chain.get("contract_bindings") or {}
    observed = bindings.get("v1_5_full_sweep_gate") or {}
    expected = {
        "protocol": "pm-v1.5-full-sweep-gate-v3",
        "status": "PASS",
        "scope": "full",
        "human_calibration_performed": False,
        "actual_corpus_review_attestation_sha256": (
            actual_corpus_review_attestation_sha256
        ),
        "actual_corpus_review_report_sha256": actual_corpus_review_report_sha256,
        "step0_shortcut_audit_attestation_sha256": (
            step0_shortcut_audit_attestation_sha256
        ),
        "step0_shortcut_audit_report_sha256": (
            step0_shortcut_audit_report_sha256
        ),
        **(
            {
                "actual_corpus_admission_mode": actual_corpus_admission_mode,
                "actual_corpus_admission_status": actual_corpus_admission_status,
                "original_actual_corpus_gate_status": "FAIL",
            }
            if actual_corpus_admission_mode
            == "POSTHOC_INSTRUMENT_QUALIFICATION"
            else {}
        ),
    }
    if bindings.get("scope") != "full" or observed != expected:
        raise RuntimeError(
            "action sweep is not the exact PM-v1.5 full matrix bound to the "
            "current automated semantic review"
        )
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument(
        "--aggregate-only",
        action="store_true",
        help=(
            "Zero-API aggregation from one exact, complete merged shard ledger. "
            "Requires --carry-forward-from and never opens provider clients."
        ),
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument(
        "--pm-v2-config", type=Path, default=ROOT / "configs" / "pm_v1_5.yaml"
    )
    parser.add_argument("--states", type=Path, default=ROOT / "data" / "pm_v1_5" / "pm_v2_states.jsonl")
    parser.add_argument("--outcomes", type=Path)
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
        "--evaluator-contexts",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "evaluator_contexts.jsonl",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--carry-forward-from",
        type=Path,
        help=(
            "Prior incomplete judging directory. Only successful calls from an "
            "exactly identical call plan are copied into a fresh identity."
        ),
    )
    parser.add_argument(
        "--execution-sharding-contract",
        type=Path,
        help=(
            "Content-addressed deterministic partition of the full call plan. "
            "Requires --execution-shard-index; each shard has an independent "
            "paid identity and only emits call-level ledger evidence."
        ),
    )
    parser.add_argument("--execution-shard-index", type=int)
    parser.add_argument("--compatibility-pilot", action="store_true")
    parser.add_argument(
        "--label-scope",
        choices=FORMAL_JUDGING_SCOPES,
        required=True,
        help=(
            "Formal holdout boundary. Run train_calibration before candidate "
            "selection; sealed_internal_test only creates the opaque held-out "
            "label bundle and must not compute outcome-quality summaries."
        ),
    )
    parser.add_argument(
        "--pilot-plan",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_development_pilot" / "pilot_plan.json",
    )
    parser.add_argument("--sweep-manifest", type=Path)
    parser.add_argument("--sweep-summary", type=Path)
    parser.add_argument("--sweep-attestation", type=Path)
    parser.add_argument(
        "--pilot-sweep-summary",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_development_pilot" / "summary.json",
    )
    parser.add_argument(
        "--pilot-sweep-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_development_pilot"
            / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--compatibility-summary",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v1_5_development_pilot_judging"
        / "summary.json",
    )
    parser.add_argument(
        "--compatibility-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v1_5_development_pilot_judging"
        / "artifact_attestation.json",
    )
    parser.add_argument(
        "--semantic-sanity-report",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_semantic_sanity"
            / "semantic_sanity_report.json"
        ),
    )
    parser.add_argument(
        "--semantic-sanity-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_semantic_sanity"
            / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--automated-semantic-review-report",
        type=Path,
        help=(
            "Deprecated V4 calibration artifact; forbidden for formal judging."
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
        help="Deprecated direct input; generation lineage is inherited from the full sweep.",
    )
    parser.add_argument(
        "--actual-corpus-semantic-review-report",
        type=Path,
        default=(
            ROOT / "outputs" / "pm_v1_5_actual_corpus_semantic_review" / "gate_report.json"
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
        help="Frozen post-hoc actual-468 instrument-qualification report.",
    )
    parser.add_argument(
        "--actual-corpus-qualification-attestation",
        type=Path,
        help="Companion attestation for the post-hoc qualification report.",
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
    parser.add_argument("--max-api-calls", type=int, default=50000)
    parser.add_argument("--max-estimated-usd", type=float, default=50.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=12000)
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.run and args.overwrite:
        raise RuntimeError(
            "paid API runs prohibit --overwrite; use a new output directory"
        )
    if (args.execution_sharding_contract is None) != (
        args.execution_shard_index is None
    ):
        raise RuntimeError(
            "--execution-sharding-contract and --execution-shard-index "
            "must be provided together"
        )
    if args.aggregate_only and args.carry_forward_from is None:
        raise RuntimeError("--aggregate-only requires --carry-forward-from")
    if args.aggregate_only and args.execution_sharding_contract is not None:
        raise RuntimeError("--aggregate-only consumes the merged ledger, not one shard")
    if args.compatibility_pilot:
        raise RuntimeError(
            "PM-v1.5 does not run the PM-v2.2 compatibility-pilot judging "
            "branch; judge the complete V1.5 sweep instead"
        )

    compatibility_pilot = bool(args.compatibility_pilot)
    outcomes_path = args.outcomes or (
        ROOT
        / "outputs"
        / ("pm_v1_5_development_pilot" if compatibility_pilot else "pm_v1_5_sweep")
        / "action_outcomes.jsonl"
    )
    out_dir = args.out_dir or (
        ROOT
        / "outputs"
        / (
            "pm_v1_5_development_pilot_judging"
            if compatibility_pilot
            else "pm_v1_5_judging"
        )
    )
    sweep_manifest_path = args.sweep_manifest or outcomes_path.parent / "run_manifest.json"
    sweep_summary_path = args.sweep_summary or outcomes_path.parent / "summary.json"
    sweep_attestation_path = (
        args.sweep_attestation
        or outcomes_path.parent / "artifact_attestation.json"
    )
    sweep_source_chain = require_action_sweep_source_chain(
        outcomes_path=outcomes_path,
        summary_path=sweep_summary_path,
        manifest_path=sweep_manifest_path,
        attestation_path=sweep_attestation_path,
    )

    states = load_states(args.states)
    state_by_card = {state.card_id: state for state in states}
    evaluator_index = load_evaluator_context_index(
        args.evaluator_contexts, states=states, require_exact=True
    )
    evaluator_by_state = evaluator_index.by_state
    all_outcomes = [ActionOutcome.model_validate(row) for row in iter_jsonl(outcomes_path)]
    unknown_cards = sorted({row.card_id for row in all_outcomes} - set(state_by_card))
    if unknown_cards:
        raise RuntimeError(f"outcomes contain unknown PM-v2 cards: {unknown_cards[:10]}")
    if len({(row.card_id, row.action_id) for row in all_outcomes}) != len(all_outcomes):
        raise RuntimeError("duplicate state-action outcomes")
    outcomes = select_formal_judging_outcomes(
        all_outcomes,
        state_by_card=state_by_card,
        label_scope=args.label_scope,
    )
    config = load_config(args.config)
    pm_v2_config = load_config(args.pm_v2_config)
    paid_release_stage = (
        "development_action_judging_train_calibration"
        if args.label_scope == TRAIN_CALIBRATION_SCOPE
        else "development_action_judging_internal_test"
    )
    if args.execution_shard_index is not None:
        paid_release_stage = (
            f"{paid_release_stage}_shard_{int(args.execution_shard_index):02d}"
        )
    require_paid_run_release(
        pm_v2_config,
        config_path=args.pm_v2_config,
        stage=paid_release_stage,
        run=bool(args.run),
        run_identity=args.accept_cost_estimate_sha256,
    )
    api_cost_planning = dict(pm_v2_config["api_cost_planning"])
    if set(api_cost_planning) != {
        "input_token_safety_factor",
        "fail_on_reported_input_overrun",
    }:
        raise ValueError("api_cost_planning must contain exactly the frozen keys")
    input_token_safety_factor = float(
        api_cost_planning["input_token_safety_factor"]
    )
    if input_token_safety_factor < 1.0 or not bool(
        api_cost_planning["fail_on_reported_input_overrun"]
    ):
        raise ValueError("PM-v2 requires conservative fail-closed API cost planning")
    composite_spec = composite_spec_from_config(pm_v2_config)
    composite_weights_sha256 = composite_weights_hash(composite_spec)
    labeling = labeling_settings_from_config(pm_v2_config)
    judging_config = dict(pm_v2_config["development_judging"])
    pilot_config = dict(judging_config["compatibility_pilot"])
    endpoint_names = [str(value) for value in judging_config["judge_endpoints"]]
    judge_role_isolation = require_judge_role_isolation(
        config,
        pm_v2_config,
        development_endpoint_names=endpoint_names,
    )
    judge_seed = int(judging_config["seed"])
    response_max_output_tokens = int(
        judging_config["response_max_output_tokens"]
    )
    risk_max_output_tokens = int(judging_config["risk_max_output_tokens"])
    if response_max_output_tokens <= 0 or risk_max_output_tokens <= 0:
        raise ValueError("development judge max-output-token limits must be positive")
    endpoints = [endpoint_from_config(config, name) for name in endpoint_names]
    families = {endpoint.family for endpoint in endpoints}
    if (
        None in families
        or len(families) < labeling["minimum_families"]
        or len(families) != len(endpoints)
    ):
        raise ValueError(
            "PM-v2 requires at least two endpoints from distinct, declared judge families"
        )
    endpoint_descriptors = [
        {
            "name": name,
            "model": endpoint.model,
            "family": endpoint.family,
            "base_url": endpoint.base_url,
        }
        for name, endpoint in zip(endpoint_names, endpoints)
    ]
    pricing_by_family = {
        str(family): {
            "input": float(values["input"]),
            "output": float(values["output"]),
        }
        for family, values in dict(
            judging_config["pricing_usd_per_mtok"]
        ).items()
    }
    if set(pricing_by_family) != {str(endpoint.family) for endpoint in endpoints}:
        raise RuntimeError(
            "development judge pricing must exactly cover frozen endpoint families"
        )
    if any(
        set(values) != {"input", "output"}
        or any(value < 0.0 for value in values.values())
        for values in pricing_by_family.values()
    ):
        raise ValueError("development judge family pricing is invalid")
    provider_output_attempts_by_family = {
        str(family): int(value)
        for family, value in dict(
            judging_config.get("maximum_provider_output_attempts_by_family") or {}
        ).items()
    }
    expected_families = {str(endpoint.family) for endpoint in endpoints}
    if set(provider_output_attempts_by_family) != expected_families:
        raise RuntimeError(
            "development judging provider-output retry limits must exactly cover "
            "the frozen endpoint families"
        )
    transport_execution_contract = development_judging_transport_contract(
        provider_output_attempts_by_family=provider_output_attempts_by_family
    )
    pilot_plan_sha256 = None
    pilot_expected_keys_sha256 = None
    compatibility_attestation_sha256 = None
    development_pilot_gate = None
    v1_5_full_sweep_gate = None
    if compatibility_pilot:
        pilot_plan = read_json(args.pilot_plan)
        pilot_payload = {
            key: value for key, value in pilot_plan.items() if key != "pilot_plan_sha256"
        }
        if (
            pilot_plan.get("status") != "READY"
            or pilot_plan.get("pilot_plan_sha256")
            != sha256_text(canonical_json(pilot_payload))
        ):
            raise RuntimeError("PM-v2 compatibility pilot plan is invalid or stale")
        expected_plan_values = {
            "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
            "states_sha256": sha256_file(args.states),
            "evaluator_contexts_sha256": evaluator_index.source_sha256,
            "evaluator_contexts_map_sha256": evaluator_index.map_sha256,
            "sample_seed": int(pilot_config["sample_seed"]),
            "states_per_regime": int(pilot_config["states_per_regime"]),
            "actions": [str(value) for value in pilot_config["actions"]],
        }
        for key, expected in expected_plan_values.items():
            if pilot_plan.get(key) != expected:
                raise RuntimeError(f"compatibility pilot plan mismatch: {key}")
        expected_outcome_keys = sorted(
            [str(row["card_id"]), str(action)]
            for row in pilot_plan["selected_states"]
            for action in row["actions"]
        )
        actual_outcome_keys = sorted([row.card_id, row.action_id] for row in outcomes)
        if actual_outcome_keys != expected_outcome_keys:
            raise RuntimeError("compatibility pilot outcomes do not exactly match plan")
        if pilot_plan.get("expected_keys_sha256") != sha256_text(
            canonical_json(expected_outcome_keys)
        ):
            raise RuntimeError("compatibility pilot expected-key hash mismatch")
        sweep_bindings = sweep_source_chain["contract_bindings"]
        expected_sweep_bindings = {
            "scope": "compatibility_pilot",
            "pilot_plan_sha256": pilot_plan["pilot_plan_sha256"],
            "pilot_expected_keys_sha256": pilot_plan["expected_keys_sha256"],
            "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
        }
        for key, expected in expected_sweep_bindings.items():
            if sweep_bindings.get(key) != expected:
                raise RuntimeError(f"compatibility sweep manifest mismatch: {key}")
        judge_schema_smoke = require_development_judge_schema_smoke_pass(
            summary_path=args.judge_schema_smoke_summary,
            attestation_path=args.judge_schema_smoke_attestation,
            experiment_config_path=args.config,
            pm_v2_config_path=args.pm_v2_config,
            states_path=args.states,
            backend_path=args.backend,
            evaluator_contexts_path=args.evaluator_contexts,
            pilot_plan_path=args.pilot_plan,
            semantic_sanity_report_path=args.semantic_sanity_report,
            semantic_sanity_attestation_path=args.semantic_sanity_attestation,
        )
        if (
            sweep_bindings.get("judge_schema_smoke") != judge_schema_smoke
            or sweep_bindings.get("deployable_feature_observability")
            != pilot_plan.get("deployable_feature_observability")
        ):
            raise RuntimeError(
                "compatibility action sweep bypassed the current schema-smoke "
                "or deployable-feature observability gate"
            )
        pilot_plan_sha256 = str(pilot_plan["pilot_plan_sha256"])
        pilot_expected_keys_sha256 = str(pilot_plan["expected_keys_sha256"])
    else:
        # V1.5's fast track does not run V2.2's 180-generation/360-judge
        # compatibility pilot. Full judging itself is fail-closed on the first
        # unsuccessful physical call and subsequently enforces the complete
        # two-family matrix/quality gates. The actual-468 structured QA and
        # Step-0 shortcut audit are content-bound inputs inherited from the
        # full sweep; external key claims additionally require the frozen
        # forced-swap sensitivity canary.
        if (
            args.automated_semantic_review_report is not None
            or args.automated_semantic_review_attestation is not None
            or args.generation_pilot_attestation is not None
        ):
            raise RuntimeError(
                "formal judging refuses direct legacy V4/pilot inputs; it must "
                "inherit current actual-QA lineage from the full sweep"
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
                expected_states_path=args.states,
                expected_evaluator_contexts_path=args.evaluator_contexts,
                expected_backend_path=args.backend,
                expected_strategy_bank_path=args.strategy_bank,
                expected_pm_config_path=args.pm_v2_config,
            )
        else:
            actual_corpus_verification = require_actual_corpus_semantic_review_pass(
                args.actual_corpus_semantic_review_report,
                args.actual_corpus_semantic_review_attestation,
                expected_experiment_config_path=args.config,
                expected_states_path=args.states,
                expected_evaluator_contexts_path=args.evaluator_contexts,
                expected_backend_path=args.backend,
                expected_strategy_bank_path=args.strategy_bank,
                expected_pm_config_path=args.pm_v2_config,
            )
        shortcut_audit_verification = require_step0_shortcut_audit_pass(
            args.step0_shortcut_audit_report,
            args.step0_shortcut_audit_attestation,
            expected_states_path=args.states,
            expected_evaluator_contexts_path=args.evaluator_contexts,
            expected_pm_config_path=args.pm_v2_config,
            expected_generation_attestation_path=args.development_data_attestation,
        )
        semantic_sanity = {
            "protocol": "pm-v1.5-actual-corpus-and-shortcut-gate-v3",
            "status": actual_corpus_verification["status"],
            "actual_corpus_admission_mode": actual_corpus_verification.get(
                "mode", "ORIGINAL_GATE_PASS"
            ),
            "human_calibration_performed": False,
            "actual_corpus_review_report_sha256": actual_corpus_verification[
                "report_sha256"
            ],
            "actual_corpus_review_attestation_sha256": actual_corpus_verification[
                "attestation_sha256"
            ],
            "step0_shortcut_audit_report_sha256": shortcut_audit_verification[
                "report_sha256"
            ],
            "step0_shortcut_audit_attestation_sha256": (
                shortcut_audit_verification["attestation_sha256"]
            ),
        }
        v1_5_full_sweep_gate = require_v1_5_full_sweep_binding(
            sweep_source_chain,
            actual_corpus_review_report_sha256=semantic_sanity[
                "actual_corpus_review_report_sha256"
            ],
            actual_corpus_review_attestation_sha256=semantic_sanity[
                "actual_corpus_review_attestation_sha256"
            ],
            step0_shortcut_audit_report_sha256=semantic_sanity[
                "step0_shortcut_audit_report_sha256"
            ],
            step0_shortcut_audit_attestation_sha256=semantic_sanity[
                "step0_shortcut_audit_attestation_sha256"
            ],
            actual_corpus_admission_mode=semantic_sanity.get(
                "actual_corpus_admission_mode"
            ),
            actual_corpus_admission_status=semantic_sanity.get("status"),
        )
        compatibility_attestation_sha256 = None
        runtime_state_lineage = require_pmv2_runtime_state_lineage(
            args.runtime, states
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    sealed_holdout_scope = args.label_scope == SEALED_INTERNAL_TEST_SCOPE
    train_calibration_labels_path = (
        out_dir / "action_labels_train_calibration.jsonl"
    )
    internal_test_labels_path = out_dir / "action_labels_internal_test.jsonl"
    labels_path = (
        internal_test_labels_path
        if sealed_holdout_scope
        else out_dir / "action_labels.jsonl"
    )
    raw_path = out_dir / "judge_results.jsonl"
    ledger_path = out_dir / "judge_call_ledger.jsonl"
    manifest_path = out_dir / "run_manifest.json"
    cost_estimate_path = out_dir / "cost_estimate.json"
    call_plan_path = out_dir / "call_plan.jsonl"
    attestation_path = out_dir / "artifact_attestation.json"
    forbid_overwrite_of_spent_attempts(
        ledger_path,
        overwrite=args.overwrite,
        stage="PM-v2 development judging",
    )
    if args.overwrite:
        paths = (
            labels_path,
            train_calibration_labels_path,
            internal_test_labels_path,
            raw_path,
            out_dir / "summary.json",
            manifest_path,
            attestation_path,
        )
        if args.dry_run:
            paths = (*paths, cost_estimate_path, call_plan_path)
        for path in paths:
            if path.exists():
                path.unlink()
    stage = (
        "pm_v2_development_judge_compatibility"
        if compatibility_pilot
        else f"pm_v2_action_judging_{args.label_scope}"
    )
    manifest = ensure_run_manifest(
        manifest_path,
        {
            "stage": stage,
            "experiment_config_sha256": sha256_file(args.config),
            "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
            "states_sha256": sha256_file(args.states),
            "outcomes_sha256": sha256_file(outcomes_path),
            "sweep_manifest_sha256": sha256_file(sweep_manifest_path),
            "evaluator_contexts_sha256": sha256_file(args.evaluator_contexts),
            "evaluator_context_map_sha256": evaluator_index.map_sha256,
            "judge_endpoints": endpoint_descriptors,
            "judge_role_isolation": judge_role_isolation,
            "prompt_contract_hash": prompt_contract_hash(),
            "composite_spec": composite_spec.model_dump(mode="json"),
            "composite_weights_sha256": composite_weights_sha256,
            "labeling": labeling,
            "development_judging": judging_config,
            "seed": judge_seed,
            "response_max_output_tokens": response_max_output_tokens,
            "risk_max_output_tokens": risk_max_output_tokens,
            "scope": "compatibility_pilot" if compatibility_pilot else "full",
            "label_scope": args.label_scope,
            "source_full_outcomes_sha256": sha256_file(outcomes_path),
            "selected_outcome_keys_sha256": sha256_text(
                canonical_json(
                    sorted([row.card_id, row.action_id] for row in outcomes)
                )
            ),
            "max_outcomes": None,
            "pilot_plan_sha256": pilot_plan_sha256,
            "pilot_expected_keys_sha256": pilot_expected_keys_sha256,
            "compatibility_attestation_sha256": compatibility_attestation_sha256,
            "sweep_source_chain": sweep_source_chain,
            "development_pilot_gate": development_pilot_gate,
            "v1_5_full_sweep_gate": v1_5_full_sweep_gate,
            "judge_retries": 1,
            "transport_execution_contract": transport_execution_contract,
            "pricing_usd_per_mtok": pricing_by_family,
            "api_cost_planning": api_cost_planning,
            "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
            "execution_sharding_contract_path": (
                str(args.execution_sharding_contract)
                if args.execution_sharding_contract is not None
                else None
            ),
            "execution_sharding_contract_file_sha256": (
                sha256_file(args.execution_sharding_contract)
                if args.execution_sharding_contract is not None
                else None
            ),
            "execution_shard_index": args.execution_shard_index,
            "aggregate_only": bool(args.aggregate_only),
        },
    )
    endpoint_by_family = {str(endpoint.family): endpoint for endpoint in endpoints}
    raw_rows = list(iter_jsonl(raw_path)) if raw_path.exists() else []
    raw_by_key = {}
    for row in raw_rows:
        key = raw_key(row)
        if key in raw_by_key:
            raise RuntimeError(f"duplicate judge result key: {key}")
        endpoint = endpoint_by_family.get(key[2])
        if endpoint is None or str(row.get("judge_model")) != endpoint.model:
            raise RuntimeError(f"stale judge endpoint provenance: {key}")
        if row.get("prompt_contract_hash") != prompt_contract_hash():
            raise RuntimeError(f"stale judge prompt provenance: {key}")
        if raw_row_succeeded(row):
            ResponseJudgeOutput.model_validate(row.get("response"))
            RiskJudgeOutput.model_validate(row.get("risk"))
            if not row.get("response_request_hash") or not row.get(
                "risk_request_hash"
            ):
                raise RuntimeError(f"successful judge row lacks request hashes: {key}")
        elif not compatibility_pilot:
            raise RuntimeError(
                "full development judging cannot resume from a failed schema row; "
                "rerun the compatibility pilot or use --overwrite"
            )
        raw_by_key[key] = row
    required_keys = {
        (state_by_card[outcome.card_id].state_id, outcome.action_id, str(endpoint.family))
        for outcome in outcomes
        for endpoint in endpoints
    }
    unexpected = sorted(set(raw_by_key) - required_keys)
    if unexpected:
        raise RuntimeError(
            "existing raw judge file contains rows outside this frozen run: "
            + str(unexpected[:10])
        )
    missing_keys = required_keys - set(raw_by_key)
    remaining_calls = len(missing_keys) * 2
    full_calls = len(required_keys) * 2
    cost_rows = []
    execution_by_call_key = {}
    for outcome_index_for_plan, outcome in enumerate(outcomes):
        state = state_by_card[outcome.card_id]
        authorized_context = str(
            evaluator_by_state[state.state_id].get("authorized_user_context") or ""
        )
        if not authorized_context:
            raise RuntimeError(f"state {state.card_id} lacks authorized evaluator context")
        selected_context = "\n".join(
            [f"MEMORY[{item.source.value}]: {item.text}" for item in outcome.memory_view]
            + [
                f"STRATEGY[{card.strategy_label}]: {card.guidance_text}"
                for card in outcome.strategy_view
            ]
        )
        response_messages = build_response_messages(
            state=state,
            authorized_user_context=authorized_context,
            candidate_response=outcome.response,
        )
        risk_messages = build_risk_messages(
            state=state,
            authorized_user_context=authorized_context,
            selected_context=selected_context,
            candidate_response=outcome.response,
        )
        for endpoint_index, endpoint in enumerate(endpoints):
            base_call_seed = (
                judge_seed + outcome_index_for_plan * 10 + endpoint_index
            )
            for judge_type, messages, max_output_tokens, schema, call_seed in (
                (
                    "response",
                    response_messages,
                    response_max_output_tokens,
                    ResponseJudgeOutput,
                    base_call_seed,
                ),
                (
                    "risk",
                    risk_messages,
                    risk_max_output_tokens,
                    RiskJudgeOutput,
                    base_call_seed + 1,
                ),
            ):
                request_payload = chat_request_payload(
                    endpoint,
                    messages,
                    temperature=0.0,
                    max_tokens=max_output_tokens,
                    seed=call_seed,
                    response_schema=schema,
                )
                if not request_payload_has_schema(request_payload):
                    raise RuntimeError(
                        "development judge request payload lacks structured schema"
                    )
                payload_text = canonical_json(request_payload)
                base_input_tokens_est = estimate_tokens(payload_text)
                input_tokens_est = conservative_token_bound(
                    payload_text, safety_factor=input_token_safety_factor
                )
                pricing = pricing_by_family[str(endpoint.family)]
                plan_row = {
                        "state_id": state.state_id,
                        "action_id": outcome.action_id,
                        "judge_family": endpoint.family,
                        "judge_model": endpoint.model,
                        "judge_type": judge_type,
                        "seed": call_seed,
                        "input_tokens_est": input_tokens_est,
                        "base_input_tokens_est": base_input_tokens_est,
                        "max_output_tokens": max_output_tokens,
                        "max_http_attempts": (
                            DEVELOPMENT_JUDGING_MAXIMUM_PHYSICAL_ATTEMPTS
                        ),
                        "prompt_hash": sha256_text(canonical_json(messages)),
                        "request_payload_sha256": sha256_text(
                            canonical_json(request_payload)
                        ),
                        "request_payload_includes_schema": (
                            request_payload_has_schema(request_payload)
                        ),
                        "pricing_usd_per_mtok": pricing,
                        "maximum_cost_usd": input_tokens_est / 1_000_000
                        * pricing["input"]
                        + max_output_tokens / 1_000_000 * pricing["output"],
                }
                plan_row["physical_call_key"] = make_physical_call_key(
                    stage=stage,
                    record_ids={
                        "state_id": state.state_id,
                        "action_id": outcome.action_id,
                        "judge_family": str(endpoint.family),
                        "judge_type": judge_type,
                    },
                    prompt_sha256=str(plan_row["prompt_hash"]),
                    endpoint=endpoint,
                    request_parameters={
                        "temperature": 0.0,
                        "max_tokens": int(max_output_tokens),
                        "seed": int(call_seed),
                        "response_schema": schema.__name__,
                        "request_payload_sha256": plan_row[
                            "request_payload_sha256"
                        ],
                        "retries": 1,
                    },
                )
                cost_rows.append(plan_row)
                execution_by_call_key[
                    (
                        state.state_id,
                        outcome.action_id,
                        str(endpoint.family),
                        judge_type,
                    )
                ] = {
                    "messages": messages,
                    "schema": schema,
                    "card_id": state.card_id,
                }
    def call_key(row):
        return (
            str(row["state_id"]),
            str(row["action_id"]),
            str(row["judge_family"]),
            str(row["judge_type"]),
        )

    plan_by_call_key = {call_key(row): row for row in cost_rows}
    if len(plan_by_call_key) != len(cost_rows):
        raise RuntimeError("duplicate development judge call-plan key")
    plan_by_physical_key = {
        str(row["physical_call_key"]): row for row in cost_rows
    }
    if len(plan_by_physical_key) != len(cost_rows):
        raise RuntimeError("duplicate development judge physical-call key")
    execution_sharding_contract = None
    execution_shard_record = None
    execution_cost_rows = cost_rows
    if args.execution_sharding_contract is not None:
        execution_sharding_contract, shard_plans = (
            load_validated_sharding_contract(
                full_rows=cost_rows,
                contract_path=args.execution_sharding_contract,
            )
        )
        shard_index = int(args.execution_shard_index)
        if not 0 <= shard_index < len(shard_plans):
            raise RuntimeError(
                f"execution shard index {shard_index} is outside "
                f"[0,{len(shard_plans)})"
            )
        execution_cost_rows = shard_plans[shard_index]
        execution_shard_record = dict(
            execution_sharding_contract["shards"][shard_index]
        )
        if not execution_cost_rows:
            raise RuntimeError("development judging execution shard is empty")
    execution_physical_keys = {
        str(row["physical_call_key"]) for row in execution_cost_rows
    }
    if len(execution_physical_keys) != len(execution_cost_rows):
        raise RuntimeError("execution shard contains duplicate physical-call keys")
    if args.carry_forward_from is not None and (
        args.carry_forward_from.resolve() == out_dir.resolve()
    ):
        raise RuntimeError("carry-forward source must differ from the new output directory")
    carry_forward = load_development_judging_carry_forward(
        carry_forward_dir=args.carry_forward_from,
        call_plan=cost_rows,
        stage=stage,
    )
    carried_call_keys = set(carry_forward["carried_call_keys"])
    if args.execution_sharding_contract is not None and not carried_call_keys.issubset(
        execution_physical_keys
    ):
        raise RuntimeError(
            "shard continuation contains successful calls outside this shard"
        )
    if args.aggregate_only and carried_call_keys != set(plan_by_physical_key):
        missing = sorted(set(plan_by_physical_key) - carried_call_keys)
        raise RuntimeError(
            "aggregate-only requires a complete exact merged successful ledger; "
            f"missing={missing[:3]}"
        )
    attempt_ledger = PersistentAttemptLedger(
        ledger_path,
        stage=stage,
        expected_calls={
            key: DEVELOPMENT_JUDGING_MAXIMUM_PHYSICAL_ATTEMPTS
            for key in execution_physical_keys
        },
        maximum_total_attempts=int(args.max_api_calls) + len(carried_call_keys),
    )
    ledger_rows = attempt_ledger.event_rows
    successful_call_rows: dict[tuple[str, str, str, str], dict] = {}
    for ledger_row in ledger_rows:
        physical_key = str(ledger_row["call_key"])
        plan = plan_by_physical_key[physical_key]
        key = call_key(plan)
        expected_record_ids = {
            "state_id": key[0],
            "action_id": key[1],
            "judge_family": key[2],
            "judge_type": key[3],
        }
        if (
            ledger_row.get("record_ids") != expected_record_ids
            or ledger_row.get("prompt_sha256") != plan["prompt_hash"]
        ):
            raise RuntimeError(f"stale judge attempt-ledger provenance: {key}")
        if ledger_row.get("event") == "SUCCEEDED":
            if key in successful_call_rows:
                raise RuntimeError(f"duplicate successful judge HTTP call: {key}")
            result_payload = ledger_row.get("result") or {}
            schema = (
                ResponseJudgeOutput if key[3] == "response" else RiskJudgeOutput
            )
            parsed = schema.model_validate(result_payload.get("parsed"))
            if not ledger_row.get("request_hash"):
                raise RuntimeError(f"successful judge ledger row lacks request hash: {key}")
            usage_error = reported_prompt_token_error(
                ledger_row.get("usage"),
                maximum_prompt_tokens=int(plan["input_tokens_est"]),
                stage="persisted development judge",
                require_positive=True,
            )
            if usage_error is not None:
                raise RuntimeError(
                    f"successful development judge usage is invalid for {key}: "
                    f"{usage_error}"
                )
            successful_call_rows[key] = {
                **plan,
                "parsed": parsed.model_dump(mode="json"),
                "request_hash": str(ledger_row["request_hash"]),
            }
    for physical_key in sorted(carried_call_keys):
        plan = plan_by_physical_key[physical_key]
        key = call_key(plan)
        if key in successful_call_rows:
            continue
        terminal = carry_forward["terminal_rows"][physical_key]
        schema = ResponseJudgeOutput if key[3] == "response" else RiskJudgeOutput
        parsed = schema.model_validate((terminal.get("result") or {}).get("parsed"))
        successful_call_rows[key] = {
            **plan,
            "parsed": parsed.model_dump(mode="json"),
            "request_hash": str(terminal["request_hash"]),
        }
    pending_cost_rows = [
        row
        for row in execution_cost_rows
        if call_key(row) not in successful_call_rows
    ]
    completed_pair_keys = {
        (state_id, action_id, family)
        for state_id, action_id, family in required_keys
        if (state_id, action_id, family, "response") in successful_call_rows
        and (state_id, action_id, family, "risk") in successful_call_rows
    }
    missing_keys = required_keys - completed_pair_keys
    remaining_calls = len(pending_cost_rows)
    historical_attempts = attempt_ledger.started_attempts
    # The accepted estimate is immutable and always describes the complete
    # pre-attempt matrix.  Mutable resume state belongs in summary diagnostics,
    # never in the approval hash or saved call plan.
    newly_costed_rows = [
        row
        for row in execution_cost_rows
        if str(row["physical_call_key"]) not in carried_call_keys
    ]
    cost_bounds = development_judging_cost_bounds(newly_costed_rows)
    maximum_physical_attempts = int(cost_bounds["maximum_physical_attempts"])
    input_counts = [int(row["input_tokens_est"]) for row in execution_cost_rows]
    logical_input_tokens = int(cost_bounds["logical_input_tokens"])
    logical_output_tokens = int(cost_bounds["logical_output_tokens"])
    logical_cost_usd = float(cost_bounds["logical_cost_usd"])
    total_input_tokens = int(cost_bounds["maximum_input_tokens"])
    total_output_tokens = int(cost_bounds["maximum_output_tokens"])
    cost_payload = {
        "stage": stage,
        "label_scope": args.label_scope,
        "full_logical_api_calls": len(cost_rows),
        "execution_logical_api_calls": len(execution_cost_rows),
        "historical_carried_forward_calls": len(carried_call_keys),
        "remaining_new_logical_calls": len(newly_costed_rows),
        "carry_forward_source_directory": carry_forward["source_directory"],
        "carry_forward_source_ledger_sha256": carry_forward[
            "source_ledger_sha256"
        ],
        "historical_physical_http_attempts": len(carried_call_keys),
        "planned_new_api_calls": maximum_physical_attempts,
        "maximum_physical_http_attempts": maximum_physical_attempts,
        "maximum_physical_attempts_per_logical_call": (
            DEVELOPMENT_JUDGING_MAXIMUM_PHYSICAL_ATTEMPTS
        ),
        "expected_judge_pairs": len(required_keys),
        "logical_input_tokens_est": logical_input_tokens,
        "logical_output_tokens_est": logical_output_tokens,
        "total_input_tokens_est": total_input_tokens,
        "max_input_tokens_per_call_est": max(input_counts, default=0),
        "total_output_tokens_est": total_output_tokens,
        "logical_single_attempt_estimated_cost_usd": logical_cost_usd,
        "estimated_cost_usd": float(cost_bounds["maximum_cost_usd"]),
        "pricing_usd_per_mtok": pricing_by_family,
        "api_cost_planning": api_cost_planning,
        "judge_role_isolation": judge_role_isolation,
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        "transport_execution_contract": transport_execution_contract,
        "call_plan_sha256": sha256_text(canonical_json(cost_rows)),
        "execution_call_plan_sha256": sha256_text(
            canonical_json(execution_cost_rows)
        ),
        "execution_sharding_contract": execution_sharding_contract,
        "execution_shard_record": execution_shard_record,
        "execution_shard_index": args.execution_shard_index,
        "aggregate_only": bool(args.aggregate_only),
        "ledger_sha256": sha256_text(canonical_json([])),
        "run_manifest_sha256": manifest["manifest_sha256"],
        "scope": "compatibility_pilot" if compatibility_pilot else "full",
        "pilot_plan_sha256": pilot_plan_sha256,
        "pilot_expected_keys_sha256": pilot_expected_keys_sha256,
        "compatibility_attestation_sha256": compatibility_attestation_sha256,
        "budget_limits": {
            "max_api_calls": int(args.max_api_calls),
            "max_estimated_usd": float(args.max_estimated_usd),
            "max_input_tokens_per_call": int(args.max_input_tokens_per_call),
        },
    }
    cost_estimate = {
        **cost_payload,
        "cost_estimate_sha256": sha256_text(canonical_json(cost_payload)),
    }
    budget_checks = {
        "api_calls": maximum_physical_attempts <= int(args.max_api_calls),
        "estimated_cost_usd": cost_estimate["estimated_cost_usd"]
        <= float(args.max_estimated_usd),
        "max_input_tokens_per_call": cost_estimate["max_input_tokens_per_call_est"]
        <= int(args.max_input_tokens_per_call),
    }
    budget_gate = {
        "status": "PASS" if all(budget_checks.values()) else "FAIL",
        "checks": budget_checks,
        "limits": {
            "max_api_calls": int(args.max_api_calls),
            "max_estimated_usd": float(args.max_estimated_usd),
            "max_input_tokens_per_call": int(args.max_input_tokens_per_call),
        },
    }
    summary = {
        "status": "DRY_RUN_COMPLETE" if args.dry_run else "STARTING",
        "label_scope": args.label_scope,
        "outcomes": len(outcomes),
        "judge_endpoints": endpoint_names,
        "judge_families": sorted(str(value) for value in families),
        "full_expected_api_calls": full_calls,
        "execution_expected_api_calls": len(execution_cost_rows),
        "completed_judge_pairs": len(completed_pair_keys),
        "remaining_judge_pairs": len(missing_keys),
        "remaining_api_calls": remaining_calls,
        "historical_physical_http_attempts": historical_attempts,
        "planned_new_logical_calls": len(pending_cost_rows),
        "planned_new_api_calls": sum(
            DEVELOPMENT_JUDGING_MAXIMUM_PHYSICAL_ATTEMPTS
            - attempt_ledger.attempts_for(str(row["physical_call_key"]))
            for row in pending_cost_rows
        ),
        "prompt_contract_hash": prompt_contract_hash(),
        "run_manifest_sha256": manifest["manifest_sha256"],
        "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
        "composite_spec": composite_spec.model_dump(mode="json"),
        "composite_weights_sha256": composite_weights_sha256,
        "labeling_gates": labeling,
        "development_judging": judging_config,
        "api_cost_planning": api_cost_planning,
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        "transport_execution_contract": transport_execution_contract,
        "execution_sharding_contract": execution_sharding_contract,
        "execution_shard_record": execution_shard_record,
        "execution_shard_index": args.execution_shard_index,
        "aggregate_only": bool(args.aggregate_only),
        "judge_endpoint_descriptors": endpoint_descriptors,
        "judge_role_isolation": judge_role_isolation,
        "scope": "compatibility_pilot" if compatibility_pilot else "full",
        "reportability_status": (
            "SHARD_EXECUTION_ONLY_NO_AGGREGATE"
            if args.execution_sharding_contract is not None
            else (
                "COMPATIBILITY_GATE_PENDING"
                if compatibility_pilot
                else "REPORTABLE"
            )
        ),
        "pilot_plan_sha256": pilot_plan_sha256,
        "pilot_expected_keys_sha256": pilot_expected_keys_sha256,
        "compatibility_attestation_sha256": compatibility_attestation_sha256,
        "resumable": True,
        "cost_estimate": cost_estimate,
        "budget_gate": budget_gate,
    }
    print(summary)
    if args.dry_run or args.aggregate_only or budget_gate["status"] != "PASS":
        persist_or_validate_judge_dry_run(
            ledger_path=ledger_path,
            estimate_path=cost_estimate_path,
            call_plan_path=call_plan_path,
            cost_estimate=cost_estimate,
            budget_gate=budget_gate,
            call_plan=cost_rows,
        )
    if budget_gate["status"] != "PASS":
        raise RuntimeError("PM-v2 judge budget gate failed before API calls")
    if args.dry_run:
        return
    if not args.aggregate_only:
        require_exact_saved_judge_dry_run(
            estimate_path=cost_estimate_path,
            call_plan_path=call_plan_path,
            cost_estimate=cost_estimate,
            budget_gate=budget_gate,
            call_plan=cost_rows,
        )
        if args.accept_cost_estimate_sha256 != cost_estimate["cost_estimate_sha256"]:
            raise RuntimeError(
                "judge API run requires exact --accept-cost-estimate-sha256 from dry-run"
            )

    # Seed the fresh ledger only after the exact dry-run identity and approval
    # have been validated. These rows represent immutable, already-paid calls;
    # their source ledger hash is part of this run's cost identity above.
    for physical_key in sorted(carried_call_keys):
        if attempt_ledger.succeeded(physical_key):
            continue
        plan = plan_by_physical_key[physical_key]
        terminal = carry_forward["terminal_rows"][physical_key]
        reservation = attempt_ledger.reserve(
            physical_key,
            record_ids={
                "state_id": str(plan["state_id"]),
                "action_id": str(plan["action_id"]),
                "judge_family": str(plan["judge_family"]),
                "judge_type": str(plan["judge_type"]),
            },
            prompt_sha256=str(plan["prompt_hash"]),
        )
        attempt_ledger.finish(
            reservation,
            succeeded=True,
            request_hash=str(terminal["request_hash"]),
            usage=terminal.get("usage"),
            error=None,
            result=terminal.get("result"),
            metadata={
                "carried_forward": True,
                "source_directory": carry_forward["source_directory"],
                "source_ledger_sha256": carry_forward["source_ledger_sha256"],
            },
        )
    historical_attempts = attempt_ledger.started_attempts
    summary["historical_physical_http_attempts"] = historical_attempts
    summary["carried_forward_calls"] = len(carried_call_keys)

    allowed_schema_failures = math.floor(
        (1.0 - float(pilot_config["minimum_schema_success_rate"]))
        * len(required_keys)
        + 1e-12
    )
    blocked_unsuccessful_calls = {
        call_key(plan)
        for plan in cost_rows
        if call_key(plan) not in successful_call_rows
        and call_retry_blocker(
            attempt_ledger,
            str(plan["physical_call_key"]),
            max_provider_output_attempts=provider_output_attempts_by_family[
                str(plan["judge_family"])
            ],
        )
        is not None
    }
    failed_pair_keys = {key[:3] for key in blocked_unsuccessful_calls}
    schema_failures_seen = len(failed_pair_keys)
    pilot_futility_reason = None
    if compatibility_pilot and schema_failures_seen > allowed_schema_failures:
        pilot_futility_reason = (
            "schema success threshold is mathematically unreachable from existing "
            "failed pilot rows"
        )
    endpoint_by_family = {str(endpoint.family): endpoint for endpoint in endpoints}
    clients = (
        preflight_development_judge_clients(endpoint_by_family)
        if pending_cost_rows and pilot_futility_reason is None
        else {}
    )
    isolated_failures: dict[str, str] = {}
    last_isolated_retry_class: str | None = None
    consecutive_same_class_count = 0
    try:
        # Preserve the frozen full-plan adjacency. Already-successful calls
        # reset the breaker even during resume; iterating only pending rows
        # would falsely compress failures that were far apart into a streak.
        for plan in execution_cost_rows:
            if pilot_futility_reason is not None:
                break
            key = call_key(plan)
            physical_key = str(plan["physical_call_key"])
            if key in successful_call_rows or attempt_ledger.succeeded(physical_key):
                last_isolated_retry_class = None
                consecutive_same_class_count = 0
                continue
            execution = execution_by_call_key[key]
            endpoint = endpoint_by_family[key[2]]
            record_ids = {
                "state_id": key[0],
                "action_id": key[1],
                "judge_family": key[2],
                "judge_type": key[3],
            }
            blocker = call_retry_blocker(
                attempt_ledger,
                physical_key,
                max_provider_output_attempts=provider_output_attempts_by_family[
                    key[2]
                ],
            )
            if blocker is not None:
                retry_class = _persisted_isolatable_failure_class(
                    attempt_ledger, physical_key
                )
                if retry_class is None:
                    raise RuntimeError(
                        "development judging cannot continue past a persisted "
                        f"non-isolatable failure: {physical_key}: {blocker}"
                    )
                isolated_failures[physical_key] = f"{retry_class}: {blocker}"
                (
                    last_isolated_retry_class,
                    consecutive_same_class_count,
                ) = advance_development_judging_failure_streak(
                    previous_class=last_isolated_retry_class,
                    previous_count=consecutive_same_class_count,
                    retry_class=retry_class,
                )
                continue

            def call_fn(
                execution=execution,
                endpoint=endpoint,
                plan=plan,
            ):
                return clients[str(endpoint.family)].chat(
                    execution["messages"],
                    temperature=0.0,
                    max_tokens=int(plan["max_output_tokens"]),
                    seed=int(plan["seed"]),
                    response_schema=execution["schema"],
                    retries=1,
                )

            reservation = None
            result = None
            parsed = None
            try:
                reservation, result, parsed = execute_with_bounded_retry(
                    attempt_ledger,
                    physical_key,
                    record_ids=record_ids,
                    prompt_sha256=str(plan["prompt_hash"]),
                    call_fn=call_fn,
                    max_provider_output_attempts=(
                        provider_output_attempts_by_family[key[2]]
                    ),
                    backoff_seconds=DEVELOPMENT_JUDGING_BACKOFF_SECONDS,
                )
            except Exception as exc:
                retry_class = _isolatable_provider_failure_class(exc)
                if retry_class is None:
                    raise
                isolated_failures[physical_key] = (
                    f"{retry_class}: {type(exc).__name__}: {exc}"
                )
                try:
                    (
                        last_isolated_retry_class,
                        consecutive_same_class_count,
                    ) = advance_development_judging_failure_streak(
                        previous_class=last_isolated_retry_class,
                        previous_count=consecutive_same_class_count,
                        retry_class=retry_class,
                    )
                except RuntimeError as breaker_error:
                    raise breaker_error from exc
                failed_pair_keys.add(key[:3])
                schema_failures_seen = len(failed_pair_keys)
                continue
            else:
                assert parsed is not None
                assert reservation is not None and result is not None
                parsed_payload = parsed.model_dump(mode="json")
                usage_error = reported_prompt_token_error(
                    result.usage,
                    maximum_prompt_tokens=int(plan["input_tokens_est"]),
                    stage=f"development {key[3]} judge",
                    require_positive=bool(
                        api_cost_planning["fail_on_reported_input_overrun"]
                    ),
                )
                if usage_error is not None:
                    attempt_ledger.finish(
                        reservation,
                        succeeded=False,
                        request_hash=result.request_hash,
                        usage=result.usage,
                        error=usage_error,
                        result={"parsed": parsed_payload},
                        metadata={
                            **failure_metadata(
                                retry_class="stage_postcondition_failure",
                                retry_disposition=TERMINAL_DISPOSITION,
                            ),
                            "plan_sha256": sha256_text(canonical_json(plan)),
                            "prompt_contract_hash": prompt_contract_hash(),
                        },
                    )
                    raise RuntimeError(
                        "development judge reported invalid token usage: "
                        + str(key)
                    )
                else:
                    attempt_ledger.finish(
                        reservation,
                        succeeded=True,
                        request_hash=result.request_hash,
                        usage=result.usage,
                        error=None,
                        result={"parsed": parsed_payload},
                        metadata={
                            "plan_sha256": sha256_text(canonical_json(plan)),
                            "prompt_contract_hash": prompt_contract_hash(),
                        },
                    )
                    successful_call_rows[key] = {
                        **plan,
                        "parsed": parsed_payload,
                        "request_hash": result.request_hash,
                    }
            last_isolated_retry_class = None
            consecutive_same_class_count = 0
    finally:
        for client in clients.values():
            client.close()
    ledger_rows = attempt_ledger.event_rows

    if args.execution_sharding_contract is not None:
        missing_shard_calls = [
            row
            for row in execution_cost_rows
            if call_key(row) not in successful_call_rows
        ]
        shard_results_path = out_dir / "shard_call_results.jsonl"
        write_jsonl(
            shard_results_path,
            [
                successful_call_rows[call_key(row)]
                for row in execution_cost_rows
                if call_key(row) in successful_call_rows
            ],
        )
        shard_status = (
            "SHARD_COMPLETE_NO_AGGREGATE"
            if not missing_shard_calls
            else "SHARD_INCOMPLETE_NONREPORTABLE"
        )
        shard_report = {
            **summary,
            "status": shard_status,
            "reportability_status": "SHARD_EXECUTION_ONLY_NO_AGGREGATE",
            "completed_shard_calls": len(execution_cost_rows)
            - len(missing_shard_calls),
            "missing_shard_calls": [
                str(row["physical_call_key"]) for row in missing_shard_calls[:50]
            ],
            "physical_http_attempts": attempt_ledger.started_attempts,
            "transport_retry_summary": retry_ledger_summary(
                attempt_ledger, sorted(execution_physical_keys)
            ),
            "final_ledger_sha256": sha256_text(canonical_json(ledger_rows)),
            "training_labels_created": False,
        }
        summary_path = out_dir / "summary.json"
        write_json(summary_path, shard_report)
        create_artifact_attestation(
            attestation_path,
            stage=stage,
            inputs={
                "experiment_config": args.config,
                "pm_v2_config": args.pm_v2_config,
                "states": args.states,
                "outcomes": outcomes_path,
                "evaluator_contexts": args.evaluator_contexts,
                "sweep_manifest": sweep_manifest_path,
                "sweep_summary": sweep_summary_path,
                "sweep_attestation": sweep_attestation_path,
                "cost_estimate": cost_estimate_path,
                "full_call_plan": call_plan_path,
                "sharding_contract": args.execution_sharding_contract,
            },
            outputs={
                "summary": (summary_path, False),
                "shard_call_results": (shard_results_path, True),
                "call_ledger": (ledger_path, True),
            },
            parameters={
                "status": shard_status,
                "label_scope": args.label_scope,
                "execution_sharding_contract": execution_sharding_contract,
                "execution_shard_record": execution_shard_record,
                "execution_shard_index": args.execution_shard_index,
                "accepted_cost_estimate_sha256": cost_estimate[
                    "cost_estimate_sha256"
                ],
                "transport_execution_contract": transport_execution_contract,
                "training_labels_created": False,
            },
        )
        if missing_shard_calls:
            raise RuntimeError(
                "development judging shard incomplete: "
                f"{len(missing_shard_calls)} missing calls"
            )
        print(shard_report)
        return

    raw_by_key = {}
    state_card = {state.state_id: state.card_id for state in states}
    for pair_key in sorted(required_keys):
        state_id, action_id, family = pair_key
        response_row = successful_call_rows.get(
            (state_id, action_id, family, "response")
        )
        risk_row = successful_call_rows.get((state_id, action_id, family, "risk"))
        if response_row is None or risk_row is None:
            continue
        pair_row = {
            "status": "SUCCESS",
            "schema_success": True,
            "state_id": state_id,
            "card_id": state_card[state_id],
            "action_id": action_id,
            "judge_family": family,
            "judge_model": response_row["judge_model"],
            "prompt_contract_hash": prompt_contract_hash(),
            "response": response_row["parsed"],
            "risk": risk_row["parsed"],
            "response_request_hash": response_row["request_hash"],
            "risk_request_hash": risk_row["request_hash"],
        }
        raw_by_key[pair_key] = pair_row
    write_jsonl(raw_path, [raw_by_key[key] for key in sorted(raw_by_key)])
    missing_after = sorted(required_keys - set(raw_by_key))
    if missing_after and pilot_futility_reason is None:
        report = {
            **summary,
            "status": "INCOMPLETE",
            "reportability_status": "NONREPORTABLE_INCOMPLETE_MATRIX",
            "missing_keys": missing_after[:50],
            "isolated_failures": isolated_failures,
            "transport_retry_summary": retry_ledger_summary(
                attempt_ledger, sorted(plan_by_physical_key)
            ),
        }
        write_json(out_dir / "summary.json", report)
        raise RuntimeError(f"judge run incomplete: {len(missing_after)} missing pairs")

    if sealed_holdout_scope:
        # Do not calculate any outcome-dependent aggregate before candidate,
        # thresholds, comparators, and uncertainty rules are frozen. The raw
        # rows are schema-validated above and immediately sealed below.
        raw_family_quality_gate = {
            "status": "NOT_EVALUATED_SEALED_HOLDOUT",
            "reason": "internal-test outcome aggregates are forbidden before consumption",
        }
    else:
        canonical_raw_rows = [
            {
                "judge_family": row["judge_family"],
                "action_id": row["action_id"],
                "response": row["response"],
                "risk": row["risk"],
            }
            for row in raw_by_key.values()
        ]
        raw_family_quality_gate = evaluate_raw_judge_gates(
            canonical_raw_rows,
            outcomes=outcomes,
            endpoints=endpoints,
            labeling=labeling,
            composite_spec=composite_spec,
            compatibility_pilot=compatibility_pilot,
        )

    labels = []
    labels_path.write_text("", encoding="utf-8")
    selected_labels_path = (
        internal_test_labels_path
        if sealed_holdout_scope
        else train_calibration_labels_path
    )
    selected_labels_path.write_text("", encoding="utf-8")
    prompt_equivalence_class_sizes: dict[tuple[str, str], int] = {}
    for outcome in outcomes:
        state = state_by_card[outcome.card_id]
        key = (state.state_id, str(outcome.prompt_equivalence_id))
        prompt_equivalence_class_sizes[key] = (
            prompt_equivalence_class_sizes.get(key, 0) + 1
        )
    for outcome in outcomes:
        state = state_by_card[outcome.card_id]
        results = []
        for endpoint in endpoints:
            row = raw_by_key.get(
                (state.state_id, outcome.action_id, str(endpoint.family))
            )
            if row is None:
                results = []
                break
            if not raw_row_succeeded(row):
                results = []
                break
            if str(row["judge_model"]) != endpoint.model:
                raise RuntimeError(
                    f"judge model changed for {state.state_id}/{outcome.action_id}/"
                    f"{endpoint.family}: {row['judge_model']} != {endpoint.model}"
                )
            results.append(
                JudgeResult(
                    family=str(row["judge_family"]),
                    model=str(row["judge_model"]),
                    response=ResponseJudgeOutput.model_validate(row["response"]),
                    risk=RiskJudgeOutput.model_validate(row["risk"]),
                    response_request_hash=str(row["response_request_hash"]),
                    risk_request_hash=str(row["risk_request_hash"]),
                )
            )
        if not results:
            continue
        label = build_action_label(
            state=state,
            action_id=outcome.action_id,
            observed_input_tokens=outcome.cost.total_input_tokens,
            retrieval_calls=outcome.cost.retrieval_calls,
            results=results,
            composite_spec=composite_spec,
            minimum_families=labeling["minimum_families"],
            reliable_mad_threshold=labeling["reliable_mad_threshold"],
            provenance={
                "outcome_request_hash": outcome.request_hash,
                "outcome_prompt_hash": outcome.prompt_hash,
                "requested_action_id": outcome.requested_action_id,
                "realized_action_id": outcome.realized_action_id,
                "prompt_equivalence_id": outcome.prompt_equivalence_id,
                "label_lineage_id": outcome.label_lineage_id,
                "prompt_equivalence_class_size": (
                    prompt_equivalence_class_sizes[
                        (state.state_id, str(outcome.prompt_equivalence_id))
                    ]
                ),
            },
        )
        labels.append(label)
        append_jsonl(labels_path, label.model_dump(mode="json"))
        if selected_labels_path != labels_path:
            append_jsonl(selected_labels_path, label.model_dump(mode="json"))
    pilot_reliable_threshold = float(pilot_config["minimum_reliable_label_rate"])
    if sealed_holdout_scope:
        quality_gate = {
            "status": "NOT_EVALUATED_SEALED_HOLDOUT",
            "reason": "internal-test outcome aggregates are forbidden before consumption",
        }
    elif labels:
        quality_gate = validate_judge_table(
            labels,
            minimum_families=labeling["minimum_families"],
            minimum_reliable_rate=(
                pilot_reliable_threshold
                if compatibility_pilot
                else labeling["minimum_reliable_rate"]
            ),
            reliable_mad_threshold=labeling["reliable_mad_threshold"],
            minimum_low_mad_coverage_per_dimension=(
                float(pilot_config["minimum_low_mad_coverage_per_dimension"])
                if compatibility_pilot
                else labeling["minimum_low_mad_coverage_per_dimension"]
            ),
            minimum_low_mad_coverage_per_action_dimension=(
                float(
                    pilot_config[
                        "minimum_low_mad_coverage_per_action_dimension"
                    ]
                )
                if compatibility_pilot
                else labeling["minimum_low_mad_coverage_per_action_dimension"]
            ),
            duplicate_exact_match_rate=labeling["duplicate_exact_match_rate"],
            maximum_absolute_dimension_correlation=labeling[
                "maximum_absolute_dimension_correlation"
            ],
            composite_support_exact_match_rate=labeling[
                "composite_support_exact_match_rate"
            ],
            maximum_absolute_composite_support_correlation=labeling[
                "maximum_absolute_composite_support_correlation"
            ],
            reject_constant_response_dimensions=labeling[
                "reject_constant_response_dimensions"
            ],
            reject_constant_risk_dimensions=labeling[
                "reject_constant_risk_dimensions"
            ],
            composite_spec=composite_spec,
            raise_on_failure=not compatibility_pilot,
        )
    else:
        quality_gate = {
            "status": "FAIL",
            "n": 0,
            "reason": "no complete two-family label rows",
        }
    successful_pairs = sum(raw_row_succeeded(row) for row in raw_by_key.values())
    schema_success_rate = successful_pairs / len(required_keys) if required_keys else 0.0
    reliable_rate = (
        None
        if sealed_holdout_scope
        else (sum(label.label_reliable for label in labels) / len(labels) if labels else 0.0)
    )
    compatibility_thresholds = {
        "minimum_schema_success_rate": float(
            pilot_config["minimum_schema_success_rate"]
        ),
        "minimum_reliable_label_rate": pilot_reliable_threshold,
        "minimum_low_mad_coverage_per_dimension": float(
            pilot_config["minimum_low_mad_coverage_per_dimension"]
        ),
        "minimum_low_mad_coverage_per_action_dimension": float(
            pilot_config["minimum_low_mad_coverage_per_action_dimension"]
        ),
    }
    compatibility_checks = None
    compatibility_gate = None
    if compatibility_pilot:
        compatibility_checks = {
            "exact_raw_matrix": set(raw_by_key) == required_keys,
            "schema_success_rate": schema_success_rate
            >= compatibility_thresholds["minimum_schema_success_rate"],
            "complete_two_family_labels": len(labels) == len(outcomes),
            "dimension_quality_gate": quality_gate.get("status") == "PASS",
            "raw_family_dimension_quality_gate": (
                raw_family_quality_gate.get("status") == "PASS"
            ),
        }
        compatibility_gate = {
            "status": "PASS" if all(compatibility_checks.values()) else "NONREPORTABLE",
            "checks": compatibility_checks,
            "thresholds": compatibility_thresholds,
            "schema_success_rate": schema_success_rate,
            "reliable_label_rate": reliable_rate,
            "joint_reliable_rate_is_diagnostic_only": True,
            "futility_triggered": pilot_futility_reason is not None,
            "futility_reason": pilot_futility_reason,
            "allowed_schema_failures": allowed_schema_failures,
        }
    label_value_feasibility = None
    if compatibility_pilot:
        selected_state_by_id = {
            state_by_card[outcome.card_id].state_id: state_by_card[outcome.card_id]
            for outcome in outcomes
        }
        label_value_feasibility = audit_pilot_label_value_feasibility(
            [selected_state_by_id[key] for key in sorted(selected_state_by_id)],
            labels,
            evaluator_contexts=evaluator_index,
            composite_spec=composite_spec,
            pilot_actions=[str(value) for value in pilot_config["actions"]],
            thresholds=dict(pilot_config["label_value_feasibility"]),
            risk_weight=float(pm_v2_config["selection"]["risk_weight"]),
            cost_weight=float(pm_v2_config["selection"]["cost_weight"]),
        )
        assert compatibility_gate is not None
        compatibility_gate["checks"]["label_value_feasibility"] = (
            label_value_feasibility["status"] == "PASS"
        )
        compatibility_gate["status"] = (
            "PASS"
            if all(compatibility_gate["checks"].values())
            else "NONREPORTABLE"
        )
    final_status = (
        compatibility_gate["status"]
        if compatibility_pilot and compatibility_gate is not None
        else (
            "SEALED_INTERNAL_TEST_COMPLETE"
            if sealed_holdout_scope
            else "COMPLETE"
        )
    )
    final_missing_api_calls = sum(
        call_key(row) not in successful_call_rows for row in cost_rows
    )
    sealed_internal_bundle_path = out_dir / "sealed_internal_bundle_manifest.json"
    sealed_internal_bundle = None
    if sealed_holdout_scope:
        sealed_internal_bundle = seal_internal_label_bundle(
            sealed_internal_bundle_path,
            internal_labels_path=internal_test_labels_path,
        )
    report = {
        **summary,
        "status": final_status,
        "reportability_status": (
            "SEALED_HOLDOUT_NOT_YET_CONSUMED"
            if sealed_holdout_scope
            else ("COMPATIBILITY_GATE_ONLY" if compatibility_pilot else "REPORTABLE")
        ),
        "label_scope": args.label_scope,
        "completed_judge_pairs": len(raw_by_key),
        "remaining_judge_pairs": len(required_keys - set(raw_by_key)),
        "remaining_api_calls": final_missing_api_calls,
        "label_rows": len(labels),
        "reliable_rows": (
            None if sealed_holdout_scope else sum(label.label_reliable for label in labels)
        ),
        "schema_success_pairs": successful_pairs,
        "schema_success_rate": schema_success_rate,
        "quality_gate": quality_gate,
        "raw_family_quality_gate": raw_family_quality_gate,
        "compatibility_gate": compatibility_gate if compatibility_pilot else None,
        "label_value_feasibility": label_value_feasibility,
        "labels_path": str(labels_path),
        "train_calibration_labels_path": (
            str(train_calibration_labels_path) if not sealed_holdout_scope else None
        ),
        "internal_test_labels_path": (
            str(internal_test_labels_path) if sealed_holdout_scope else None
        ),
        "sealed_internal_bundle_path": (
            str(sealed_internal_bundle_path) if sealed_holdout_scope else None
        ),
        "sealed_internal_bundle": sealed_internal_bundle,
        "raw_path": str(raw_path),
        "ledger_path": str(ledger_path),
        "physical_http_attempts": attempt_ledger.started_attempts,
        "carry_forward_source_directory": carry_forward["source_directory"],
        "carry_forward_source_ledger_sha256": carry_forward[
            "source_ledger_sha256"
        ],
        "carried_forward_calls": len(carried_call_keys),
        "transport_execution_contract": transport_execution_contract,
        "transport_retry_summary": retry_ledger_summary(
            attempt_ledger, sorted(plan_by_physical_key)
        ),
        "final_ledger_sha256": sha256_text(canonical_json(ledger_rows)),
    }
    summary_path = out_dir / "summary.json"
    write_json(summary_path, report)
    attestation_inputs = {
        "experiment_config": args.config,
        "pm_v2_config": args.pm_v2_config,
        "states": args.states,
        "outcomes": outcomes_path,
        "evaluator_contexts": args.evaluator_contexts,
        "sweep_manifest": sweep_manifest_path,
        "sweep_summary": sweep_summary_path,
        "sweep_attestation": sweep_attestation_path,
        "cost_estimate": cost_estimate_path,
        "call_plan": call_plan_path,
    }
    if compatibility_pilot:
        attestation_inputs["pilot_plan"] = args.pilot_plan
    else:
        attestation_inputs["actual_corpus_semantic_review"] = (
            args.actual_corpus_semantic_review_report
        )
        attestation_inputs["actual_corpus_semantic_review_attestation"] = (
            args.actual_corpus_semantic_review_attestation
        )
    if args.carry_forward_from is not None:
        attestation_inputs["carry_forward_call_plan"] = (
            args.carry_forward_from / "call_plan.jsonl"
        )
        attestation_inputs["carry_forward_ledger"] = (
            args.carry_forward_from / "judge_call_ledger.jsonl"
        )
    attestation_outputs = {
        "summary": (summary_path, False),
        "raw_results": (raw_path, True),
        "call_ledger": (ledger_path, True),
    }
    if sealed_holdout_scope:
        attestation_outputs.update(
            {
                "sealed_internal_bundle": (sealed_internal_bundle_path, False),
                "internal_test_labels": (internal_test_labels_path, True),
            }
        )
    else:
        attestation_outputs.update(
            {
                "labels": (labels_path, True),
                "train_calibration_labels": (train_calibration_labels_path, True),
            }
        )
    create_artifact_attestation(
        attestation_path,
        stage=stage,
        inputs=attestation_inputs,
        outputs=attestation_outputs,
        parameters={
            "status": final_status,
            "scope": "compatibility_pilot" if compatibility_pilot else "full",
            "label_scope": args.label_scope,
            "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
            "prompt_contract_hash": prompt_contract_hash(),
            "composite_spec": composite_spec.model_dump(mode="json"),
            "composite_weights_sha256": composite_weights_sha256,
            "labeling_gates": labeling,
            "development_judging": judging_config,
            "judge_endpoint_descriptors": endpoint_descriptors,
            "judge_role_isolation": judge_role_isolation,
            "pilot_plan_sha256": pilot_plan_sha256,
            "pilot_expected_keys_sha256": pilot_expected_keys_sha256,
            "compatibility_attestation_sha256": compatibility_attestation_sha256,
            "compatibility_gate": (
                compatibility_gate if compatibility_pilot else None
            ),
            "label_value_feasibility": label_value_feasibility,
            "raw_family_quality_gate": raw_family_quality_gate,
            "accepted_cost_estimate_sha256": cost_estimate[
                "cost_estimate_sha256"
            ],
            "judge_retries": 1,
            "transport_execution_contract": transport_execution_contract,
            "transport_retry_summary": retry_ledger_summary(
                attempt_ledger, sorted(plan_by_physical_key)
            ),
            "carry_forward_source_directory": carry_forward["source_directory"],
            "carry_forward_source_ledger_sha256": carry_forward[
                "source_ledger_sha256"
            ],
            "carried_forward_calls": len(carried_call_keys),
            "api_cost_planning": api_cost_planning,
            "pricing_usd_per_mtok": pricing_by_family,
            "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
            "physical_http_attempts": attempt_ledger.started_attempts,
            "final_ledger_sha256": sha256_text(canonical_json(ledger_rows)),
        },
        expected={
            "outcomes": len(outcomes),
            "judge_role_isolation_status": judge_role_isolation["status"],
            "judge_pairs": len(required_keys),
            "full_logical_api_calls": len(cost_rows),
            "new_physical_http_attempts": (
                attempt_ledger.started_attempts - historical_attempts
            ),
            "physical_http_attempts": attempt_ledger.started_attempts,
        },
    )
    print(report)


if __name__ == "__main__":
    main()
