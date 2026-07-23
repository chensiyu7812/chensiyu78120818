#!/usr/bin/env python3
"""Dual-family (gemini + deepseek) quality/risk judging for ESConv-auxiliary
generation outcomes.

Produces ``action_labels.jsonl`` in the exact ``ActionLabel`` shape PM-v2
training already consumes (``build_action_label``/``pm_v2_contracts.
ActionLabel``), from the real generated M0+R0/M0+RS responses in
``outputs/esconv_auxiliary_generation_v1_5_{split}/action_outcomes.jsonl``
(built by ``13b_run_esconv_auxiliary_generation_v1_5.py``).

Judge prompts (``build_response_messages``/``build_risk_messages``,
``pm_v2_judging.py``) never receive the action identity -- "The resource
policy and action name are hidden" is stated directly in the risk prompt --
so action-blinding is inherited for free from the existing, already-tested
prompt builders, not reimplemented here.

``authorized_user_context`` is always empty for ESConv states: memory is
structurally unavailable (MP/MS/ME all False), so there is no cross-session
memory bank to authorize beyond what is already visible in
``current_session_history``/``current_session_summary``.

Uses ``execute_with_bounded_retry`` (``bounded_retry.py``) with a separately
content-addressed execution contract.  Both historical NVIDIA-hosted judges
and the current official/provider-native judge surfaces have produced real
5xx, timeout, malformed-output, and truncation failures in this project.
Every physical attempt is therefore ledger-visible and budgeted; this does
not change the scientific prompt, schema, endpoint, or scoring contract.

The four physical calls per (state, action) -- gemini response, gemini risk,
deepseek response, deepseek risk -- are scheduled in a seeded-shuffled order
(``call_plan_shuffle_seed``, frozen and recorded in the cost estimate) rather
than natural (state, action, family, type) order. This governs only physical
HTTP attempt *scheduling*; ``judge_one``'s pointwise design judges one
candidate response at a time and never shows a judge both R0 and RS
together, so there is no pairwise presentation order to control -- this
shuffle exists so a sustained provider outage clusters less predictably
against any single (state, action) subgroup.

Deliberately not coupled to the internal 7,488-action sweep's provenance
gates (``require_action_sweep_source_chain``, ``v1_5_full_sweep_gate``,
actual-corpus review, Step-0 shortcut audit): those gates authenticate that
sweep's *generation* source, which does not apply here (ESConv-auxiliary
generation is its own disclosed, self-contained track, verified directly by
``13b_run_esconv_auxiliary_generation_v1_5.py``'s own contract bindings).
The judge *output*-quality gates (``validate_raw_judge_family_health`` and
friends) are retained in full: they check judge behavior, not sweep
provenance, and apply exactly the same way here.

``--pilot`` mode: at small N (the 24-pair pilot), several risk dimensions
can show zero measured variance across an entire family purely from sample
size -- this is the exact situation
``21_judge_pm_v2_action_sweep_v1_5.py``'s ``compatibility_pilot`` branch
already exists to handle for the internal training sweep. This script
mirrors that precedent exactly rather than inventing a new policy: with
``--pilot``, the raw-family-health/subgroup-health/action-applicable-risk
gates and the final judge-table gate are still computed and reported in
full, at the *same* thresholds as a formal run, but do not raise (mirroring
``raise_on_failure=not compatibility_pilot``); the judge-table gate
additionally uses the frozen, pre-registered
``development_judging.compatibility_pilot`` thresholds
(``minimum_reliable_label_rate``, ``minimum_low_mad_coverage_per_dimension``,
``minimum_low_mad_coverage_per_action_dimension``) instead of the full-scale
``labeling`` ones. A pilot run's ``summary.json`` is always tagged
``reportability_status: "PILOT_DIAGNOSTIC_ONLY"`` -- its labels are never
implicitly promoted to formal training data. Only a non-pilot invocation can
produce ``reportability_status: "REPORTABLE"``, and only then are the full
thresholds enforced as fatal.

``--carry-forward-from PRIOR_OUT_DIR``: since ``--pilot`` changes only
post-hoc validation policy (not any judge prompt, endpoint, or token
parameter), a prior run's call_plan.jsonl is byte-identical to a fresh
non-pilot-vs-pilot recomputation, so every physical call it already
succeeded can be carried forward at zero new cost rather than re-spent --
mirroring ``scripts/v1_5_run_automated_semantic_review.py``'s carry-forward
mechanism exactly. Refuses (rather than silently reusing a stale subset)
unless the prior directory's call plan is byte-identical to this run's own
freshly computed one.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any, Mapping

from metacom_pm.api import (
    ProviderRequestError,
    RetryableProviderError,
    StructuredOutputValidationError,
    make_client,
)
from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.attempt_ledger import (
    PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
    forbid_overwrite_of_spent_attempts,
    physical_call_key as make_physical_call_key,
)
from metacom_pm.bounded_retry import (
    BOUNDED_PROVIDER_OUTPUT_RETRY_CLASSES,
    DEFAULT_MAX_PROVIDER_OUTPUT_ATTEMPTS,
    RETRYABLE_UP_TO_FULL_BUDGET,
    RETRY_CONTRACT_PROTOCOL,
    call_retry_blocker,
    execute_with_bounded_retry,
    retry_ledger_summary,
)
from metacom_pm.attempt_ledger import PersistentAttemptLedger
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import ActionOutcome
from metacom_pm.internal_holdout import seal_internal_label_bundle
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
from metacom_pm.paid_run_release import (
    require_paid_run_release,
    resolve_first_unconsumed_output_directory,
)
from metacom_pm.pm_v2_data import load_states
from metacom_pm.pm_v2_judging import (
    DIMENSION_APPLICABILITY_CONTRACT_PROTOCOL,
    JudgeResult,
    ResponseJudgeOutput,
    RiskJudgeOutput,
    build_action_label,
    build_response_messages,
    build_risk_messages,
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
from metacom_pm.pm_v2_model import (
    MAD_ADJUSTED_CONSERVATIVE_UTILITY_LAMBDA,
    MAD_ADJUSTED_CONSERVATIVE_UTILITY_PROTOCOL,
    RESPONSE_DIMENSION_CLAMP_RANGE,
    RISK_DIMENSION_CLAMP_RANGE,
)
from metacom_pm.text import conservative_token_bound, estimate_tokens


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "calibration", "internal_test")
# Frozen, disclosed constant; changing it changes call-plan order (and
# therefore the cost-estimate hash) but never scoring.
CALL_PLAN_SHUFFLE_SEED = 913171
# Matches the durable retry budget already exercised by the actual-corpus
# semantic audit.  The bound is execution resilience, not a scientific model
# or rubric setting, and every authorized attempt is included in dry-run cost.
MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL = 10
# Two sparse, mostly-zero risk dimensions can reach a high exact-match rate
# purely from shared zero-zero rows; below this many informative
# (at-least-one-side-nonzero) rows, a duplicate/high-correlation claim is
# reported as insufficient evidence rather than a false positive.
MINIMUM_NONZERO_OBSERVATIONS_FOR_DUPLICATE_CHECK = 10
TRANSPORT_BACKOFF_SECONDS: tuple[float, ...] = (
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
    300.0,
    600.0,
    600.0,
    900.0,
)
ESCONV_AUXILIARY_JUDGING_TRANSPORT_PROTOCOL = (
    "pm-v1.5-esconv-auxiliary-judging-transport-v1"
)
CONSECUTIVE_SAME_CLASS_CIRCUIT_BREAKER = 5
ISOLATABLE_PROVIDER_FAILURE_CLASSES = frozenset(
    set(RETRYABLE_UP_TO_FULL_BUDGET)
    | set(BOUNDED_PROVIDER_OUTPUT_RETRY_CLASSES)
    | {"structured_output_validation_error", "output_token_limit"}
)


def judging_cost_bounds(
    rows: list[Mapping[str, Any]],
) -> dict[str, int | float]:
    """Bound every authorized physical attempt, not only first attempts."""

    logical_input_tokens = sum(int(row["input_tokens_est"]) for row in rows)
    logical_output_tokens = sum(int(row["max_output_tokens"]) for row in rows)
    logical_cost_usd = math.fsum(float(row["maximum_cost_usd"]) for row in rows)
    attempts = MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL
    return {
        "logical_input_tokens": logical_input_tokens,
        "logical_output_tokens": logical_output_tokens,
        "logical_cost_usd": logical_cost_usd,
        "maximum_physical_attempts": len(rows) * attempts,
        "maximum_input_tokens": logical_input_tokens * attempts,
        "maximum_output_tokens": logical_output_tokens * attempts,
        "maximum_cost_usd": logical_cost_usd * attempts,
    }


def judging_transport_contract(
    *, provider_output_attempts_by_family: Mapping[str, int]
) -> dict[str, Any]:
    """Content-address execution resilience separately from judge science."""

    if not provider_output_attempts_by_family or any(
        int(value) < 1
        or int(value) > MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL
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
        "protocol": ESCONV_AUXILIARY_JUDGING_TRANSPORT_PROTOCOL,
        "retry_contract_protocol": RETRY_CONTRACT_PROTOCOL,
        "maximum_physical_attempts_per_logical_call": (
            MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL
        ),
        "transport_backoff_seconds": list(TRANSPORT_BACKOFF_SECONDS),
        "retryable_transport_classes": sorted(RETRYABLE_UP_TO_FULL_BUDGET),
        "isolatable_provider_failure_classes": sorted(
            ISOLATABLE_PROVIDER_FAILURE_CLASSES
        ),
        "provider_output_maximum_attempts_by_family": {
            str(family): int(value)
            for family, value in sorted(provider_output_attempts_by_family.items())
        },
        "terminal_content_or_schema_failure_is_not_blindly_retried": True,
        "continue_after_isolated_provider_failure": True,
        "consecutive_same_class_circuit_breaker": (
            CONSECUTIVE_SAME_CLASS_CIRCUIT_BREAKER
        ),
        "client_internal_retries": 1,
        "scientific_judge_contract_unchanged": True,
        "code_manifest": code_manifest,
        "code_manifest_sha256": sha256_text(canonical_json(code_manifest)),
    }
    payload["contract_sha256"] = sha256_text(canonical_json(payload))
    return payload


MEASUREMENT_CONTRACT_PROTOCOL = (
    "pm-v1.5-judge-uncertainty-robust-measurement-contract-v1"
)


def measurement_contract_record(
    *,
    action_ids_present: list[str],
    inapplicable_risk_dimensions: frozenset[str],
    inapplicable_risk_dimensions_by_action: Mapping[str, frozenset[str]],
) -> dict[str, Any]:
    """Single, hashable freeze record for the applicability + conservative-
    utility measurement contract in force for this labeling run.

    Bound here (the train-split judging attestation) so
    21a_preflight_dual_domain_training_v1_5.py can require an exact match
    before calibration/internal-test judging is allowed to proceed --
    per the explicit freeze rule, none of these values may be tuned from
    calibration/internal-test results once a train-split run attests them.
    """

    code_paths = {
        "judging_runner": Path(__file__).resolve(),
        "pm_v2_contracts": ROOT / "src" / "metacom_pm" / "pm_v2_contracts.py",
        "pm_v2_judging": ROOT / "src" / "metacom_pm" / "pm_v2_judging.py",
        "pm_v2_model": ROOT / "src" / "metacom_pm" / "pm_v2_model.py",
        "v1_5_dual_domain_training": (
            ROOT / "src" / "metacom_pm" / "v1_5_dual_domain_training.py"
        ),
    }
    code_manifest = {
        name: {
            "relative_path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        }
        for name, path in sorted(code_paths.items())
    }
    payload: dict[str, Any] = {
        "protocol": MEASUREMENT_CONTRACT_PROTOCOL,
        "action_ids_present": sorted(action_ids_present),
        "dimension_applicability_contract_protocol": (
            DIMENSION_APPLICABILITY_CONTRACT_PROTOCOL
        ),
        "dimension_applicability_contract_sha256": (
            dimension_applicability_contract_sha256(action_ids_present)
        ),
        "inapplicable_risk_dimensions": sorted(inapplicable_risk_dimensions),
        "inapplicable_risk_dimensions_by_action": {
            action_id: sorted(dims)
            for action_id, dims in sorted(inapplicable_risk_dimensions_by_action.items())
        },
        "conservative_utility_protocol": MAD_ADJUSTED_CONSERVATIVE_UTILITY_PROTOCOL,
        "conservative_utility_lambda": MAD_ADJUSTED_CONSERVATIVE_UTILITY_LAMBDA,
        "conservative_utility_lambda_is_fixed_never_tuned": True,
        "response_dimension_clamp_range": list(RESPONSE_DIMENSION_CLAMP_RANGE),
        "risk_dimension_clamp_range": list(RISK_DIMENSION_CLAMP_RANGE),
        "code_manifest": code_manifest,
        "code_manifest_sha256": sha256_text(canonical_json(code_manifest)),
    }
    payload["contract_sha256"] = sha256_text(canonical_json(payload))
    return payload


def _isolatable_provider_failure_class(exc: Exception) -> str | None:
    if isinstance(exc, RetryableProviderError):
        retry_class = str(exc.last_retry_class)
        return retry_class if retry_class in ISOLATABLE_PROVIDER_FAILURE_CLASSES else None
    if isinstance(exc, StructuredOutputValidationError):
        return "structured_output_validation_error"
    if isinstance(exc, ProviderRequestError):
        return None
    return None


def _persisted_isolatable_failure_class(
    ledger: PersistentAttemptLedger, physical_key: str
) -> str | None:
    terminal = ledger.terminal_row(physical_key) or {}
    metadata = terminal.get("metadata") or {}
    retry_class = str(metadata.get("retry_class") or "")
    return retry_class if retry_class in ISOLATABLE_PROVIDER_FAILURE_CLASSES else None


def advance_failure_streak(
    *, previous_class: str | None, previous_count: int, retry_class: str
) -> tuple[str, int]:
    count = previous_count + 1 if retry_class == previous_class else 1
    if count >= CONSECUTIVE_SAME_CLASS_CIRCUIT_BREAKER:
        raise RuntimeError(
            f"circuit breaker: {retry_class} recurred {count} times in a row "
            "across different ESConv-auxiliary judging calls"
        )
    return retry_class, count


def outcome_key(outcome: ActionOutcome) -> tuple[str, str]:
    return (outcome.state_id, outcome.action_id)


def call_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["state_id"]),
        str(row["action_id"]),
        str(row["judge_family"]),
        str(row["judge_type"]),
    )


def _persist_or_validate_dry_run(
    *,
    ledger_path: Path,
    estimate_path: Path,
    call_plan_path: Path,
    cost_estimate: dict[str, Any],
    budget_gate: dict[str, Any],
    call_plan: list[dict[str, Any]],
) -> str:
    expected_estimate = {**cost_estimate, "budget_gate": budget_gate}
    spent = ledger_path.is_file() and ledger_path.stat().st_size > 0
    if spent:
        if not estimate_path.is_file() or not call_plan_path.is_file():
            raise RuntimeError(
                "spent ESConv-auxiliary-judging ledger freezes the accepted "
                "dry-run, but its estimate or full call plan is missing"
            )
        if read_json(estimate_path) != expected_estimate or list(
            iter_jsonl(call_plan_path)
        ) != call_plan:
            raise RuntimeError(
                "spent ESConv-auxiliary-judging ledger freezes the original "
                "exact estimate and full call plan; current drift is rejected"
            )
        return "VALIDATED_EXISTING"
    write_json(estimate_path, expected_estimate)
    write_jsonl(call_plan_path, call_plan)
    return "WRITTEN"


def _load_carry_forward_state(
    *,
    carry_forward_dir: Path | None,
    call_plan: list[dict[str, Any]],
    stage: str,
) -> dict[str, Any]:
    """Read-only: find which of this run's own call-plan rows already
    succeeded, with a complete parsed result, in a prior run's ledger.

    Mirrors scripts/v1_5_run_automated_semantic_review.py's carry-forward
    mechanism exactly. Refuses rather than silently carrying forward a stale
    subset unless the prior directory's call_plan.jsonl is byte-identical to
    the plan this run just freshly computed for itself. Never touches the
    prior directory's own ledger file; only reads it.
    """

    if carry_forward_dir is None:
        return {
            "carry_forward_source_directory": None,
            "carry_forward_source_ledger_sha256": None,
            "carried_call_keys": set(),
            "carried_terminal_rows": {},
        }
    old_call_plan_path = carry_forward_dir / "call_plan.jsonl"
    old_ledger_path = carry_forward_dir / "physical_attempt_ledger.jsonl"
    if not old_call_plan_path.is_file() or not old_ledger_path.is_file():
        raise RuntimeError(
            "carry-forward source directory lacks a call plan or ledger"
        )
    if list(iter_jsonl(old_call_plan_path)) != call_plan:
        raise RuntimeError(
            "carry-forward source call plan differs from this run's own "
            "freshly-computed plan -- refusing to trust its ledger's call keys"
        )
    expected_calls = {
        str(row["physical_call_key"]): MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL
        for row in call_plan
    }
    old_ledger = PersistentAttemptLedger(
        old_ledger_path,
        stage=stage,
        expected_calls=expected_calls,
        maximum_total_attempts=10**9,
    )
    carried_call_keys: set[str] = set()
    carried_terminal_rows: dict[str, dict[str, Any]] = {}
    for row in call_plan:
        physical_key = str(row["physical_call_key"])
        if not old_ledger.succeeded(physical_key):
            continue
        terminal = old_ledger.terminal_row(physical_key)
        result = (terminal or {}).get("result")
        if not isinstance(result, dict) or not result.get("parsed"):
            raise RuntimeError(
                "carry-forward source lacks a complete parsed result for "
                f"{physical_key}"
            )
        carried_call_keys.add(physical_key)
        carried_terminal_rows[physical_key] = terminal
    return {
        "carry_forward_source_directory": str(carry_forward_dir),
        "carry_forward_source_ledger_sha256": sha256_file(old_ledger_path),
        "carried_call_keys": carried_call_keys,
        "carried_terminal_rows": carried_terminal_rows,
    }


def _require_saved_dry_run(
    *,
    estimate_path: Path,
    call_plan_path: Path,
    cost_estimate: dict[str, Any],
    budget_gate: dict[str, Any],
    call_plan: list[dict[str, Any]],
) -> None:
    if not estimate_path.is_file() or not call_plan_path.is_file():
        raise RuntimeError(
            "ESConv-auxiliary judge API run requires a matching saved --dry-run"
        )
    expected_estimate = {**cost_estimate, "budget_gate": budget_gate}
    if read_json(estimate_path) != expected_estimate:
        raise RuntimeError("saved ESConv-auxiliary judge dry-run estimate is stale")
    if list(iter_jsonl(call_plan_path)) != call_plan:
        raise RuntimeError("saved ESConv-auxiliary judge full call plan is stale")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "experiment.yaml"
    )
    parser.add_argument(
        "--pm-v1-5-config", type=Path, default=ROOT / "configs" / "pm_v1_5.yaml"
    )
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument(
        "--scope",
        required=True,
        choices=("pilot", "full"),
        help=(
            "Folded into the stage name and output directory so a pilot-scale "
            "and full-scale run for the same split can never collide on the "
            "same default directory. Independent of --pilot (which only "
            "relaxes quality-gate strictness)."
        ),
    )
    parser.add_argument(
        "--auxiliary-dir", type=Path, default=ROOT / "data" / "esconv_auxiliary_v1_5"
    )
    parser.add_argument(
        "--generation-root", type=Path, default=ROOT / "outputs"
    )
    parser.add_argument(
        "--generation-dir",
        type=Path,
        default=None,
        help=(
            "Explicit generation output directory to judge, overriding the "
            "default esconv_auxiliary_generation_v1_5_{scope}_{split} guess. "
            "Needed when the generation run that produced these outcomes "
            "used a __retryN sibling directory (see "
            "resolve_first_unconsumed_output_directory in "
            "paid_run_release.py) because the plain directory was already "
            "consumed by an earlier attempt."
        ),
    )
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--max-api-calls", type=int, default=50000)
    parser.add_argument("--max-estimated-usd", type=float, default=50.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=12000)
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "Small-N pilot: quality gates are computed and reported at the "
            "same thresholds but do not raise (mirrors "
            "21_judge_pm_v2_action_sweep_v1_5.py's compatibility_pilot). "
            "Output is tagged reportability_status=PILOT_DIAGNOSTIC_ONLY and "
            "must never be treated as formal training data."
        ),
    )
    parser.add_argument(
        "--carry-forward-from",
        type=Path,
        default=None,
        help=(
            "Prior output directory whose already-succeeded physical calls "
            "(byte-identical call plan required) are carried forward at "
            "zero new cost."
        ),
    )
    parser.add_argument(
        "--sealed-holdout",
        action="store_true",
        help=(
            "Held-out scope (internal_test): never compute quality_gate, "
            "raw_family_quality_gate, or any other reliability/aggregate "
            "metric over the judge labels -- not merely 'do not raise' or "
            "'do not report' (--pilot's relaxation), but do not call those "
            "functions at all. Writes action_labels.jsonl, then immediately "
            "seals it (seal_internal_label_bundle, internal_holdout.py) and "
            "writes a summary containing only completeness/row-count/ledger/"
            "seal SHA256s. Incompatible with --pilot (a pilot run is "
            "diagnostic-only and is never the sealed holdout)."
        ),
    )
    args = parser.parse_args()

    if args.run and args.overwrite:
        raise RuntimeError(
            "paid ESConv-auxiliary judging runs prohibit --overwrite; use a "
            "new output directory"
        )
    if args.sealed_holdout and args.pilot:
        raise RuntimeError("--sealed-holdout and --pilot are mutually exclusive")

    split = str(args.split)
    scope = str(args.scope)
    stage = f"esconv_auxiliary_judging_{scope}_{split}"
    config = load_config(args.config)
    pm_v1_5_config = load_config(args.pm_v1_5_config)
    if pm_v1_5_config.get("version") != "pm-v1.5":
        raise RuntimeError("ESConv-auxiliary judging requires a pm-v1.5 config")
    require_paid_run_release(
        pm_v1_5_config,
        config_path=args.pm_v1_5_config,
        stage=stage,
        run=bool(args.run),
        run_identity=args.accept_cost_estimate_sha256,
    )
    out_dir = resolve_first_unconsumed_output_directory(
        args.out_root / f"esconv_auxiliary_judging_v1_5_{scope}_{split}",
        config=pm_v1_5_config,
        config_path=args.pm_v1_5_config,
    )

    # Judging always reads the generation run of the same scope (a pilot
    # judging run reads a pilot generation run; a full judging run reads a
    # full generation run) -- see 13b_run_esconv_auxiliary_generation_v1_5.py.
    # --generation-dir overrides this when the generation run actually used a
    # __retryN sibling (see resolve_first_unconsumed_output_directory).
    generation_dir = args.generation_dir or (
        args.generation_root / f"esconv_auxiliary_generation_v1_5_{scope}_{split}"
    )
    outcomes_path = generation_dir / "action_outcomes.jsonl"
    generation_summary_path = generation_dir / "summary.json"
    if not outcomes_path.is_file():
        raise RuntimeError(
            f"missing ESConv-auxiliary generation outcomes for split {split!r}: "
            f"{outcomes_path}; run 13b_run_esconv_auxiliary_generation_v1_5.py first"
        )
    generation_summary = read_json(generation_summary_path)
    if generation_summary.get("status") != "COMPLETE":
        raise RuntimeError(
            f"ESConv-auxiliary generation for split {split!r} is not COMPLETE"
        )

    states_path = args.auxiliary_dir / split / "pm_v2_states.jsonl"
    states = load_states(states_path)
    state_by_id = {state.state_id: state for state in states}
    outcomes = [ActionOutcome.model_validate(row) for row in iter_jsonl(outcomes_path)]
    unknown_states = sorted({o.state_id for o in outcomes} - set(state_by_id))
    if unknown_states:
        raise RuntimeError(
            f"ESConv-auxiliary outcomes reference unknown states: {unknown_states[:10]}"
        )
    if len({outcome_key(o) for o in outcomes}) != len(outcomes):
        raise RuntimeError("duplicate ESConv-auxiliary state-action outcomes")

    composite_spec = composite_spec_from_config(pm_v1_5_config)
    composite_weights_sha256 = composite_weights_hash(composite_spec)
    labeling = labeling_settings_from_config(pm_v1_5_config)
    judging_config = dict(pm_v1_5_config["development_judging"])
    # Per-family, not a single shared scalar: a judge family switching to a
    # provider with more format/output noise (e.g. deepseek_official's loose
    # json_object mode) must not silently change another family's (e.g.
    # google_gemini's) retry behavior. Falls back to the shared default for
    # any family not explicitly listed, so existing configs need no changes.
    # Computed here (before cost_payload) and folded into it below, so
    # changing any family's budget changes cost_estimate_sha256 -- otherwise
    # a config-only change to real retry behavior would silently keep an
    # already-approved identity valid.
    provider_output_attempts_by_family = {
        str(family): int(value)
        for family, value in dict(
            judging_config.get("maximum_provider_output_attempts_by_family") or {}
        ).items()
    }

    def _max_provider_output_attempts_for(judge_family: str) -> int:
        return provider_output_attempts_by_family.get(
            judge_family, DEFAULT_MAX_PROVIDER_OUTPUT_ATTEMPTS
        )

    endpoint_names = [str(value) for value in judging_config["judge_endpoints"]]
    judge_seed = int(judging_config["seed"])
    response_max_output_tokens = int(judging_config["response_max_output_tokens"])
    risk_max_output_tokens = int(judging_config["risk_max_output_tokens"])
    if response_max_output_tokens <= 0 or risk_max_output_tokens <= 0:
        raise ValueError("judge max-output-token limits must be positive")
    endpoints = [endpoint_from_config(config, name) for name in endpoint_names]
    families = {endpoint.family for endpoint in endpoints}
    if (
        None in families
        or len(families) < labeling["minimum_families"]
        or len(families) != len(endpoints)
    ):
        raise ValueError(
            "ESConv-auxiliary judging requires at least two endpoints from "
            "distinct, declared judge families"
        )
    provider_output_attempts_by_family = {
        str(endpoint.family): _max_provider_output_attempts_for(str(endpoint.family))
        for endpoint in endpoints
    }
    transport_execution_contract = judging_transport_contract(
        provider_output_attempts_by_family=provider_output_attempts_by_family
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
        for family, values in dict(judging_config["pricing_usd_per_mtok"]).items()
    }
    if set(pricing_by_family) != {str(endpoint.family) for endpoint in endpoints}:
        raise RuntimeError(
            "ESConv-auxiliary judge pricing must exactly cover frozen endpoint "
            "families"
        )

    api_cost_planning = dict(pm_v1_5_config["api_cost_planning"])
    input_token_safety_factor = float(api_cost_planning["input_token_safety_factor"])

    labels_path = out_dir / "action_labels.jsonl"
    raw_path = out_dir / "judge_results.jsonl"
    ledger_path = out_dir / "physical_attempt_ledger.jsonl"
    manifest_path = out_dir / "run_manifest.json"
    cost_estimate_path = out_dir / "cost_estimate.json"
    call_plan_path = out_dir / "call_plan.jsonl"
    forbid_overwrite_of_spent_attempts(ledger_path, overwrite=args.overwrite, stage=stage)
    if args.overwrite:
        for path in (
            labels_path,
            raw_path,
            out_dir / "summary.json",
            manifest_path,
        ):
            if path.exists():
                path.unlink()

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = ensure_run_manifest(
        manifest_path,
        {
            "stage": stage,
            "experiment_config_sha256": sha256_file(args.config),
            "pm_v1_5_config_sha256": sha256_file(args.pm_v1_5_config),
            "states_sha256": sha256_file(states_path),
            "outcomes_sha256": sha256_file(outcomes_path),
            "generation_summary_sha256": sha256_file(generation_summary_path),
            "judge_endpoints": endpoint_descriptors,
            "prompt_contract_hash": prompt_contract_hash(),
            "composite_spec": composite_spec.model_dump(mode="json"),
            "composite_weights_sha256": composite_weights_sha256,
            "labeling": labeling,
            "development_judging": judging_config,
            "seed": judge_seed,
            "call_plan_shuffle_seed": CALL_PLAN_SHUFFLE_SEED,
            "response_max_output_tokens": response_max_output_tokens,
            "risk_max_output_tokens": risk_max_output_tokens,
            "pricing_usd_per_mtok": pricing_by_family,
            "api_cost_planning": api_cost_planning,
            "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
            "retry_contract_protocol": RETRY_CONTRACT_PROTOCOL,
            "transport_execution_contract": transport_execution_contract,
            "split": split,
            "pilot_mode": bool(args.pilot),
        },
    )

    # Build the full call plan in natural order, then apply a frozen, seeded
    # shuffle over the outer (state, action) iteration -- see module
    # docstring. Sorting outcomes first makes the shuffle a pure function of
    # the frozen seed, independent of the outcomes file's on-disk row order.
    ordered_outcomes = sorted(outcomes, key=outcome_key)
    shuffled_indices = list(range(len(ordered_outcomes)))
    random.Random(CALL_PLAN_SHUFFLE_SEED).shuffle(shuffled_indices)
    shuffled_outcomes = [ordered_outcomes[i] for i in shuffled_indices]

    cost_rows: list[dict[str, Any]] = []
    execution_by_call_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for outcome_index, outcome in enumerate(shuffled_outcomes):
        state = state_by_id[outcome.state_id]
        authorized_user_context = ""
        selected_context = "\n".join(
            [f"MEMORY[{item.source.value}]: {item.text}" for item in outcome.memory_view]
            + [
                f"STRATEGY[{card.strategy_label}]: {card.guidance_text}"
                for card in outcome.strategy_view
            ]
        )
        response_messages = build_response_messages(
            state=state,
            authorized_user_context=authorized_user_context,
            candidate_response=outcome.response,
        )
        risk_messages = build_risk_messages(
            state=state,
            authorized_user_context=authorized_user_context,
            selected_context=selected_context,
            candidate_response=outcome.response,
        )
        for endpoint_index, endpoint in enumerate(endpoints):
            base_call_seed = judge_seed + outcome_index * 10 + endpoint_index
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
                prompt_hash = sha256_text(canonical_json(messages))
                base_input_tokens_est = estimate_tokens(canonical_json(messages))
                input_tokens_est = conservative_token_bound(
                    canonical_json(messages), safety_factor=input_token_safety_factor
                )
                pricing = pricing_by_family[str(endpoint.family)]
                # Bind the schema's actual field constraints (not just its
                # class name) into the call identity: a bare Pydantic field
                # change (e.g. a length bound) does not change response_
                # schema.__name__, messages, or physical_call_key under the
                # old scheme, so it silently would not mint a fresh identity
                # even though the real request contract changed.
                response_json_schema = schema.model_json_schema()
                response_schema_sha256 = sha256_text(
                    canonical_json(response_json_schema)
                )
                request_payload_sha256 = sha256_text(
                    canonical_json(
                        {
                            "messages": messages,
                            "response_schema": response_json_schema,
                            "temperature": 0.0,
                            "max_tokens": int(max_output_tokens),
                            "seed": int(call_seed),
                        }
                    )
                )
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
                    "prompt_hash": prompt_hash,
                    "response_schema_sha256": response_schema_sha256,
                    "request_payload_sha256": request_payload_sha256,
                    "pricing_usd_per_mtok": pricing,
                    "maximum_cost_usd": input_tokens_est / 1_000_000 * pricing["input"]
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
                    prompt_sha256=prompt_hash,
                    endpoint=endpoint,
                    request_parameters={
                        "temperature": 0.0,
                        "max_tokens": int(max_output_tokens),
                        "seed": int(call_seed),
                        "response_schema": schema.__name__,
                        "response_schema_sha256": response_schema_sha256,
                    },
                )
                cost_rows.append(plan_row)
                execution_by_call_key[call_key(plan_row)] = {
                    "messages": messages,
                    "schema": schema,
                }

    if len({str(r["physical_call_key"]) for r in cost_rows}) != len(cost_rows):
        raise RuntimeError("duplicate ESConv-auxiliary judge physical-call key")
    required_keys = {call_key(row) for row in cost_rows}
    expected_pairs = {outcome_key(o) for o in outcomes}

    # Loaded here (before cost accounting) so the frozen dry-run estimate
    # reflects only the REAL remaining spend, not a misleading full-cost
    # hypothetical -- mirrors scripts/v1_5_run_automated_semantic_review.py's
    # carry-forward cost accounting exactly.
    carry_forward = _load_carry_forward_state(
        carry_forward_dir=args.carry_forward_from,
        call_plan=cost_rows,
        stage=stage,
    )
    carried_call_keys = carry_forward["carried_call_keys"]
    remaining_cost_rows = [
        row for row in cost_rows if str(row["physical_call_key"]) not in carried_call_keys
    ]

    cost_bounds = judging_cost_bounds(remaining_cost_rows)
    maximum_physical_attempts = int(cost_bounds["maximum_physical_attempts"])
    logical_input_tokens = int(cost_bounds["logical_input_tokens"])
    logical_output_tokens = int(cost_bounds["logical_output_tokens"])
    logical_cost_usd = float(cost_bounds["logical_cost_usd"])
    total_input_tokens = int(cost_bounds["maximum_input_tokens"])
    total_output_tokens = int(cost_bounds["maximum_output_tokens"])
    cost_payload = {
        "stage": stage,
        "split": split,
        "full_logical_api_calls": len(cost_rows),
        "historical_carried_forward_calls": len(carried_call_keys),
        "remaining_new_logical_calls": len(remaining_cost_rows),
        "planned_new_api_calls": maximum_physical_attempts,
        "maximum_physical_http_attempts": maximum_physical_attempts,
        "maximum_physical_attempts_per_logical_call": (
            MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL
        ),
        "carry_forward_source_directory": carry_forward["carry_forward_source_directory"],
        "carry_forward_source_ledger_sha256": carry_forward[
            "carry_forward_source_ledger_sha256"
        ],
        "expected_judge_pairs": len(expected_pairs),
        "logical_input_tokens_est": logical_input_tokens,
        "logical_output_tokens_est": logical_output_tokens,
        "total_input_tokens_est": total_input_tokens,
        "max_input_tokens_per_call_est": max(
            (int(r["input_tokens_est"]) for r in remaining_cost_rows), default=0
        ),
        "total_output_tokens_est": total_output_tokens,
        "logical_single_attempt_estimated_cost_usd": logical_cost_usd,
        "estimated_cost_usd": float(cost_bounds["maximum_cost_usd"]),
        "pricing_usd_per_mtok": pricing_by_family,
        "api_cost_planning": api_cost_planning,
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        "retry_contract_protocol": RETRY_CONTRACT_PROTOCOL,
        # Folded into the hashed payload (not just used at runtime) so
        # changing any family's format-retry budget changes cost_estimate_
        # sha256 -- otherwise a config-only change to real retry behavior
        # could silently keep an already-approved identity valid.
        "transport_execution_contract": transport_execution_contract,
        "call_plan_shuffle_seed": CALL_PLAN_SHUFFLE_SEED,
        "call_plan_sha256": sha256_text(canonical_json(cost_rows)),
        "run_manifest_sha256": manifest["manifest_sha256"],
        "pilot_mode": bool(args.pilot),
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

    if args.dry_run or budget_gate["status"] != "PASS":
        _persist_or_validate_dry_run(
            ledger_path=ledger_path,
            estimate_path=cost_estimate_path,
            call_plan_path=call_plan_path,
            cost_estimate=cost_estimate,
            budget_gate=budget_gate,
            call_plan=cost_rows,
        )
    result = {**cost_estimate, "budget_gate": budget_gate}
    print(result)
    if args.dry_run:
        if budget_gate["status"] != "PASS":
            raise RuntimeError(
                "ESConv-auxiliary judging dry-run failed the frozen budget gate"
            )
        return
    if budget_gate["status"] != "PASS":
        raise RuntimeError("ESConv-auxiliary judging API run blocked by budget gate")
    _require_saved_dry_run(
        estimate_path=cost_estimate_path,
        call_plan_path=call_plan_path,
        cost_estimate=cost_estimate,
        budget_gate=budget_gate,
        call_plan=cost_rows,
    )
    expected_hash = str(cost_estimate["cost_estimate_sha256"])
    if not args.accept_cost_estimate_sha256:
        raise RuntimeError(
            "API mode is fail-closed: pass --accept-cost-estimate-sha256 "
            f"{expected_hash} from the matching dry-run"
        )
    if args.accept_cost_estimate_sha256 != expected_hash:
        raise RuntimeError(
            "accepted cost estimate hash does not match the current "
            "ESConv-auxiliary judging plan"
        )

    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=stage,
        expected_calls={
            str(row["physical_call_key"]): MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL
            for row in cost_rows
        },
        # Carried-forward calls are already paid for and need no further
        # attempt budget, but each still consumes one ledger "attempt slot"
        # when seeded below.
        maximum_total_attempts=int(args.max_api_calls) + len(carried_call_keys),
    )

    for row in cost_rows:
        physical_key = str(row["physical_call_key"])
        if physical_key not in carry_forward["carried_call_keys"] or ledger.succeeded(
            physical_key
        ):
            continue
        terminal = carry_forward["carried_terminal_rows"][physical_key]
        reservation = ledger.reserve(
            physical_key,
            record_ids={
                "state_id": row["state_id"],
                "action_id": row["action_id"],
                "judge_family": row["judge_family"],
                "judge_type": row["judge_type"],
            },
            prompt_sha256=str(row["prompt_hash"]),
        )
        ledger.finish(
            reservation,
            succeeded=True,
            request_hash=terminal.get("request_hash"),
            usage=terminal.get("usage"),
            error=None,
            result=terminal.get("result"),
            metadata={
                "carried_forward": True,
                "carried_forward_source_directory": carry_forward[
                    "carry_forward_source_directory"
                ],
                "carried_forward_source_ledger_sha256": carry_forward[
                    "carry_forward_source_ledger_sha256"
                ],
            },
        )

    endpoint_by_family = {str(endpoint.family): endpoint for endpoint in endpoints}
    pending = [
        row for row in cost_rows if not ledger.succeeded(str(row["physical_call_key"]))
    ]
    clients = (
        {family: make_client(endpoint) for family, endpoint in endpoint_by_family.items()}
        if pending
        else {}
    )
    successful_call_rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    isolated_failures: dict[str, str] = {}
    last_isolated_retry_class: str | None = None
    consecutive_same_class_count = 0
    try:
        # Iterate the frozen full plan, not only pending rows. A prior success
        # resets the breaker, so separated failures cannot be compressed into
        # a false systemic streak during resume.
        for row in cost_rows:
            key = call_key(row)
            physical_key = str(row["physical_call_key"])
            if ledger.succeeded(physical_key):
                terminal = ledger.terminal_row(physical_key)
                result_payload = (terminal or {}).get("result") or {}
                schema = ResponseJudgeOutput if key[3] == "response" else RiskJudgeOutput
                successful_call_rows[key] = {
                    **row,
                    "parsed": schema.model_validate(result_payload.get("parsed")).model_dump(
                        mode="json"
                    ),
                    "request_hash": str(terminal.get("request_hash")),
                }
                last_isolated_retry_class = None
                consecutive_same_class_count = 0
                continue
            execution = execution_by_call_key[key]
            endpoint = endpoint_by_family[key[2]]
            blocker = call_retry_blocker(
                ledger,
                physical_key,
                max_provider_output_attempts=_max_provider_output_attempts_for(key[2]),
            )
            if blocker is not None:
                retry_class = _persisted_isolatable_failure_class(ledger, physical_key)
                if retry_class is None:
                    raise RuntimeError(
                        "ESConv-auxiliary judging cannot continue past a persisted "
                        f"non-isolatable failure: {physical_key}: {blocker}"
                    )
                isolated_failures[physical_key] = f"{retry_class}: {blocker}"
                last_isolated_retry_class, consecutive_same_class_count = (
                    advance_failure_streak(
                        previous_class=last_isolated_retry_class,
                        previous_count=consecutive_same_class_count,
                        retry_class=retry_class,
                    )
                )
                continue

            def call_fn(row=row, execution=execution, endpoint=endpoint):
                return clients[str(endpoint.family)].chat(
                    execution["messages"],
                    temperature=0.0,
                    max_tokens=int(row["max_output_tokens"]),
                    seed=int(row["seed"]),
                    response_schema=execution["schema"],
                    retries=1,
                )

            try:
                reservation, call_result, parsed = execute_with_bounded_retry(
                    ledger,
                    physical_key,
                    record_ids={
                        "state_id": key[0],
                        "action_id": key[1],
                        "judge_family": key[2],
                        "judge_type": key[3],
                    },
                    prompt_sha256=str(row["prompt_hash"]),
                    call_fn=call_fn,
                    max_provider_output_attempts=_max_provider_output_attempts_for(
                        key[2]
                    ),
                    backoff_seconds=TRANSPORT_BACKOFF_SECONDS,
                )
            except Exception as exc:
                retry_class = _isolatable_provider_failure_class(exc)
                if retry_class is None:
                    raise
                isolated_failures[physical_key] = (
                    f"{retry_class}: {type(exc).__name__}: {exc}"
                )
                try:
                    last_isolated_retry_class, consecutive_same_class_count = (
                        advance_failure_streak(
                            previous_class=last_isolated_retry_class,
                            previous_count=consecutive_same_class_count,
                            retry_class=retry_class,
                        )
                    )
                except RuntimeError as breaker_error:
                    raise breaker_error from exc
                continue
            assert parsed is not None
            parsed_payload = parsed.model_dump(mode="json")
            ledger.finish(
                reservation,
                succeeded=True,
                request_hash=call_result.request_hash,
                usage=call_result.usage,
                error=None,
                result={"parsed": parsed_payload},
            )
            successful_call_rows[key] = {
                **row,
                "parsed": parsed_payload,
                "request_hash": call_result.request_hash,
            }
            last_isolated_retry_class = None
            consecutive_same_class_count = 0
    finally:
        for client in clients.values():
            client.close()

    raw_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    state_id_for_card: dict[str, str] = {o.state_id: o.card_id for o in outcomes}
    for state_id, action_id in sorted(expected_pairs):
        pair_rows = {}
        for family in sorted(str(e.family) for e in endpoints):
            response_row = successful_call_rows.get((state_id, action_id, family, "response"))
            risk_row = successful_call_rows.get((state_id, action_id, family, "risk"))
            if response_row is None or risk_row is None:
                pair_rows = {}
                break
            pair_rows[family] = (response_row, risk_row)
        if not pair_rows:
            continue
        for family, (response_row, risk_row) in pair_rows.items():
            raw_by_key[(state_id, action_id, family)] = {
                "status": "SUCCESS",
                "schema_success": True,
                "state_id": state_id,
                "card_id": state_id_for_card[state_id],
                "action_id": action_id,
                "judge_family": family,
                "judge_model": response_row["judge_model"],
                "prompt_contract_hash": prompt_contract_hash(),
                "response": response_row["parsed"],
                "risk": risk_row["parsed"],
                "response_request_hash": response_row["request_hash"],
                "risk_request_hash": risk_row["request_hash"],
            }
    write_jsonl(raw_path, [raw_by_key[key] for key in sorted(raw_by_key)])

    missing_call_keys = sorted(required_keys - set(successful_call_rows))
    completed_label_pairs = {
        (state_id, action_id)
        for state_id, action_id in expected_pairs
        if all(
            (state_id, action_id, str(endpoint.family)) in raw_by_key
            for endpoint in endpoints
        )
    }
    actual_new_physical_attempts = max(
        0, ledger.started_attempts - len(carry_forward["carried_call_keys"])
    )
    if missing_call_keys:
        # Partial judgments must never become training labels. Preserve the
        # successful raw rows and ledger for a fresh exact-plan continuation,
        # but materialize an empty label file and a non-reportable summary.
        labels_path.write_text("", encoding="utf-8")
        incomplete_summary = {
            "status": "INCOMPLETE",
            "pilot_mode": bool(args.pilot),
            "reportability_status": "NONREPORTABLE_INCOMPLETE_MATRIX",
            "split": split,
            "expected_judge_pairs": len(expected_pairs),
            "completed_judge_pairs": len(completed_label_pairs),
            "missing_call_keys": [list(key) for key in missing_call_keys[:50]],
            "run_manifest_sha256": manifest["manifest_sha256"],
            "carry_forward_source_directory": carry_forward[
                "carry_forward_source_directory"
            ],
            "carry_forward_source_ledger_sha256": carry_forward[
                "carry_forward_source_ledger_sha256"
            ],
            "carried_forward_physical_calls": len(
                carry_forward["carried_call_keys"]
            ),
            "new_physical_calls": actual_new_physical_attempts,
            "isolated_failures": isolated_failures,
            "transport_retry_summary": retry_ledger_summary(
                ledger, [str(row["physical_call_key"]) for row in cost_rows]
            ),
            "cost_estimate": cost_estimate,
            "budget_gate": budget_gate,
        }
        write_json(out_dir / "summary.json", incomplete_summary)
        raise RuntimeError(
            f"ESConv-auxiliary judging incomplete for split {split!r}: "
            f"{len(missing_call_keys)} missing logical calls"
        )

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
    # Sealed holdout (internal_test): these three functions compute
    # cross-item/cross-family reliability statistics over the labels'
    # *values* -- exactly the "aggregate" the seal must never touch before
    # the candidate model/thresholds are frozen. Skip calling them entirely
    # rather than merely suppressing or hiding their result.
    if not args.sealed_holdout:
        # ESConv-auxiliary's legal actions are frozen to {M0+R0, M0+RS}: no
        # memory source is ever selected, so every memory-misuse risk
        # dimension (applicable_risk_fields, pm_v2_model.py) is structurally
        # a zero here, not a judge defect. Derived from the actual action set
        # present, not hand-maintained, so it can never silently drift from
        # applicable_risk_fields's own definition.
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
            reject_constant_response_dimensions=labeling["reject_constant_response_dimensions"],
            reject_constant_risk_dimensions=labeling["reject_constant_risk_dimensions"],
            inapplicable_risk_dimensions=inapplicable_risk_dimensions,
            inapplicable_risk_dimensions_by_action=inapplicable_risk_dimensions_by_action,
            minimum_nonzero_observations=MINIMUM_NONZERO_OBSERVATIONS_FOR_DUPLICATE_CHECK,
            split_correlation_by_sign=True,
            composite_spec=composite_spec,
            raise_on_failure=not args.pilot,
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
            raise_on_failure=not args.pilot,
        )
        action_applicable_risk_gate = validate_action_applicable_risk_signal(
            canonical_raw_rows,
            expected_actions=sorted({o.action_id for o in outcomes}),
            expected_families=[str(endpoint.family) for endpoint in endpoints],
            minimum_signal_rate=labeling["minimum_action_applicable_risk_signal_rate"],
            minimum_distinct_values=labeling["minimum_action_applicable_risk_distinct_values"],
            raise_on_failure=not args.pilot,
        )
        # ESConv-auxiliary's legal action set is frozen to exactly {M0+R0,
        # M0+RS}: whether independent judge families agree on which one is
        # *better* per state (not just per-dimension score agreement) is a
        # distinct, diagnostic-only reliability question -- never a hard gate.
        if len(action_ids_present) != 2:
            raise RuntimeError(
                "directional preference diagnostic requires exactly two "
                f"legal actions, got {action_ids_present}"
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
    labels_path.write_text("", encoding="utf-8")
    for state_id, action_id in sorted(expected_pairs):
        state = state_by_id[state_id]
        outcome = next(
            o for o in outcomes if o.state_id == state_id and o.action_id == action_id
        )
        results = []
        for endpoint in endpoints:
            row = raw_by_key.get((state_id, action_id, str(endpoint.family)))
            if row is None:
                results = []
                break
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
            action_id=action_id,
            observed_input_tokens=outcome.cost.total_input_tokens,
            retrieval_calls=outcome.cost.retrieval_calls,
            results=results,
            composite_spec=composite_spec,
            minimum_families=labeling["minimum_families"],
            reliable_mad_threshold=labeling["reliable_mad_threshold"],
            provenance={"esconv_auxiliary_split": split},
        )
        labels.append(label)
        append_jsonl(labels_path, label.model_dump(mode="json"))

    missing_after = sorted(expected_pairs - {(l.state_id, l.action_id) for l in labels})

    if args.sealed_holdout:
        if missing_after:
            raise RuntimeError(
                f"ESConv-auxiliary judging incomplete for split {split!r}: "
                f"{len(missing_after)} missing pairs -- a sealed holdout run "
                "must produce the complete label set before it may be sealed"
            )
        seal = seal_internal_label_bundle(
            out_dir / "sealed_internal_bundle.json",
            internal_labels_path=labels_path,
        )
        summary = {
            "status": "COMPLETE",
            "pilot_mode": False,
            "sealed_holdout": True,
            "reportability_status": "SEALED_HOLDOUT_NOT_YET_EVALUATED",
            "split": split,
            "expected_judge_pairs": len(expected_pairs),
            "completed_judge_pairs": len(labels),
            "run_manifest_sha256": manifest["manifest_sha256"],
            "physical_attempt_ledger_sha256": sha256_file(ledger_path),
            "sealed_internal_bundle_path": str(out_dir / "sealed_internal_bundle.json"),
            "sealed_internal_bundle_seal_sha256": seal["seal_sha256"],
            "carry_forward_source_directory": carry_forward["carry_forward_source_directory"],
            "carry_forward_source_ledger_sha256": carry_forward[
                "carry_forward_source_ledger_sha256"
            ],
            "carried_forward_physical_calls": len(carry_forward["carried_call_keys"]),
            "new_physical_calls": actual_new_physical_attempts,
            "isolated_failures": isolated_failures,
            "transport_retry_summary": retry_ledger_summary(
                ledger, [str(row["physical_call_key"]) for row in cost_rows]
            ),
            "cost_estimate": cost_estimate,
            "budget_gate": budget_gate,
            "note": (
                "Sealed holdout scope: no quality_gate, raw_family_quality_"
                "gate, or any other cross-label aggregate was computed. Only "
                "completeness, row/state counts, and content hashes are "
                "reported here. The candidate model and thresholds must be "
                "frozen before this seal is ever opened for evaluation."
            ),
        }
        write_json(out_dir / "summary.json", summary)
        print(summary)
        return

    pilot_config = dict(judging_config.get("compatibility_pilot") or {})
    quality_gate = (
        validate_judge_table(
            labels,
            minimum_families=labeling["minimum_families"],
            minimum_reliable_rate=(
                float(pilot_config["minimum_reliable_label_rate"])
                if args.pilot
                else labeling["minimum_reliable_rate"]
            ),
            reliable_mad_threshold=labeling["reliable_mad_threshold"],
            minimum_low_mad_coverage_per_dimension=(
                float(pilot_config["minimum_low_mad_coverage_per_dimension"])
                if args.pilot
                else labeling["minimum_low_mad_coverage_per_dimension"]
            ),
            minimum_low_mad_coverage_per_action_dimension=(
                float(pilot_config["minimum_low_mad_coverage_per_action_dimension"])
                if args.pilot
                else labeling["minimum_low_mad_coverage_per_action_dimension"]
            ),
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
            raise_on_failure=not args.pilot,
        )
        if labels
        else {"status": "FAIL", "reason": "no completed labels"}
    )
    raw_family_quality_status = (
        "PASS"
        if raw_family_global_gate.get("status") == "PASS"
        and raw_family_action_gate.get("status") == "PASS"
        and action_applicable_risk_gate.get("status") == "PASS"
        else "FAIL"
    )
    measurement_contract = measurement_contract_record(
        action_ids_present=action_ids_present,
        inapplicable_risk_dimensions=inapplicable_risk_dimensions,
        inapplicable_risk_dimensions_by_action=inapplicable_risk_dimensions_by_action,
    )
    summary = {
        "status": "COMPLETE" if not missing_after else "INCOMPLETE",
        "pilot_mode": bool(args.pilot),
        "reportability_status": (
            "PILOT_DIAGNOSTIC_ONLY"
            if args.pilot
            else (
                "REPORTABLE"
                if raw_family_quality_status == "PASS"
                and quality_gate.get("status") == "PASS"
                else "FORMAL_GATE_FAILED"
            )
        ),
        "split": split,
        "expected_judge_pairs": len(expected_pairs),
        "completed_judge_pairs": len(labels),
        "missing_keys": missing_after[:50],
        "run_manifest_sha256": manifest["manifest_sha256"],
        "carry_forward_source_directory": carry_forward["carry_forward_source_directory"],
        "carry_forward_source_ledger_sha256": carry_forward[
            "carry_forward_source_ledger_sha256"
        ],
        "carried_forward_physical_calls": len(carry_forward["carried_call_keys"]),
        "new_physical_calls": actual_new_physical_attempts,
        "isolated_failures": isolated_failures,
        "transport_retry_summary": retry_ledger_summary(
            ledger, [str(row["physical_call_key"]) for row in cost_rows]
        ),
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
        "measurement_contract": measurement_contract,
        "cost_estimate": cost_estimate,
        "budget_gate": budget_gate,
    }
    write_json(out_dir / "summary.json", summary)
    print(summary)
    if missing_after:
        raise RuntimeError(
            f"ESConv-auxiliary judging incomplete for split {split!r}: "
            f"{len(missing_after)} missing pairs"
        )
    # Note: in non-pilot mode every gate above already used
    # raise_on_failure=True, so reaching this point means reportability_status
    # is necessarily REPORTABLE -- a FAIL would have raised inside the gate
    # call itself, not fallen through to here.

    # Formal attestation, written only once the run is genuinely complete
    # (unreachable above if missing_after was non-empty). Binds the
    # measurement contract -- applicability + conservative-utility protocol,
    # fixed lambda, per-dimension clamp ranges, applicable risk dimensions,
    # and the exact code hashes that define them -- alongside the label
    # artifact itself, so a downstream consumer (the dual-domain preflight)
    # can require an exact match before calibration/internal-test judging is
    # allowed to proceed.
    create_artifact_attestation(
        out_dir / "attestation.json",
        stage=stage,
        inputs={
            "experiment_config": args.config,
            "pm_v1_5_config": args.pm_v1_5_config,
            "states": states_path,
            "generation_outcomes": outcomes_path,
            "generation_summary": generation_summary_path,
        },
        outputs={
            "summary": (out_dir / "summary.json", False),
            "labels": (labels_path, True),
            "raw_results": (raw_path, True),
            "call_ledger": (ledger_path, True),
        },
        parameters={
            "status": summary["status"],
            "reportability_status": summary["reportability_status"],
            "pilot_mode": bool(args.pilot),
            "scope": scope,
            "split": split,
            "quality_gate_status": quality_gate.get("status"),
            "raw_family_quality_gate_status": raw_family_quality_status,
            "measurement_contract": measurement_contract,
        },
    )


if __name__ == "__main__":
    main()
