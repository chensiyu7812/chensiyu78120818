#!/usr/bin/env python3
"""Run the PM-v1.5 automated multi-family semantic-review gate.

Generates the deterministic 27-case V8 validation set (no API calls; 9
regimes x 3 cases), adds the exact nine surfaces from the paid compatibility
pilot, builds deliberately mislabeled positive-control cases, and judges all
of it through independent LLM families using the same 12-question rubric.
The exact paid-pilot attestation is a required input, so a PASS on unrelated
fixtures cannot authorize the real 52-user generation.

Ad-hoc, not part of the PM-v2.2 gated pipeline. Explicitly does not claim
independent human validation.

Pre-execution protocol amendment (2026-07-17): the panel was originally
three families (training_judge_deepseek_flash, training_judge_qwen122,
final_judge). Two independent real --run attempts both failed on
training_judge_qwen122 specifically -- first an HTTP 500, then a malformed
response body missing message.content -- while every deepseek/gpt-4o call
in both attempts succeeded (see outputs/pm_v1_5_automated_semantic_review/
and .../pm_v1_5_automated_semantic_review_retry1/'s physical_attempt_ledger.
jsonl). Neither failure was an HTTP 429 (this client's only quota/rate-limit
signal, api.py's dedicated 429 branch), so this is not evidence of a quota
limit -- only of two independent infrastructure/compatibility failures on
that one endpoint. Qwen is dropped from the panel for this reason, decided
BEFORE any successful run of the amended two-family panel exists -- not
because any semantic score was seen and disliked. Neither prior partial
attempt's results are reused or spliced into the new run; both stay as
immutable spent-attempt history in their original directories.

Second protocol amendment (2026-07-17): the amended two-family panel's first
real --run attempt ALSO failed once (HTTP 503 on training_judge_deepseek_flash
this time, not qwen at all -- see outputs/pm_v1_5_automated_semantic_review_
2family/physical_attempt_ledger.jsonl), after 30/66 calls succeeded cleanly.
Three transient failures across three real attempts -- two model routes on
the same NVIDIA-hosted base URL (Qwen twice, DeepSeek once) -- made a single
physical attempt per logical call impractical. This is evidence of instability
on the observed NVIDIA-hosted path, not proof about NVIDIA's whole gateway.
Every physical HTTP attempt is now bounded-retried via
metacom_pm.bounded_retry.execute_with_bounded_retry: up to the call's full
attempt budget (3) for rate_limited_429/request_timeout_408/http_5xx/
network_timeout, up to one extra attempt for a 2xx response missing an expected
field, and never for a genuine 4xx client error, schema-validation failure, or
a successfully-parsed
but unfavorable score -- each physical attempt, retried or not, is still its
own immutable ledger row. Retry eligibility is reconstructed from those rows,
so restarting the process cannot reset a deterministic failure or missing-field
counter. This changes the call-plan/cost-estimate contract
(worst-case 3x physical attempts budgeted), so any prior accepted
cost-estimate hash for this stage is stale.

Judge-isolation amendment (2026-07-18): every prior panel containing
``final_judge`` is ineligible for development gating because that endpoint is
also the primary external judge. The only default development panel is now
Gemini Flash Lite plus DeepSeek Flash. Before planning any call, code resolves
all development and final endpoint aliases and fails closed on overlap in the
alias, declared family, model identifier, or resolved base-URL/model route.
Old partial judgments are not reused; this amendment requires a new output
directory, dry-run, and accepted cost-estimate hash.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.api import (
    chat_request_payload,
    make_client,
    request_payload_has_schema,
    require_reported_usage,
)
from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.attempt_ledger import (
    PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
    PersistentAttemptLedger,
    forbid_overwrite_of_spent_attempts,
    physical_call_key,
)
from metacom_pm.bounded_retry import (
    DEFAULT_BACKOFF_SECONDS,
    RETRY_CONTRACT_PROTOCOL,
    RETRYABLE_UP_TO_FULL_BUDGET,
    TERMINAL_DISPOSITION,
    call_retry_blocker,
    execute_with_bounded_retry,
    failure_metadata,
    retry_ledger_summary,
)
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.pm_v2_generation_review_v8 import (
    VALIDATION_CASES_PER_REGIME,
    generate_v8_review_cases,
)
from metacom_pm.v1_5_automated_semantic_review import (
    AUTOMATED_CONTROL_PROTOCOL,
    AUTOMATED_REVIEW_PROTOCOL,
    AutomatedSemanticReviewOutput,
    RATING_FIELDS,
    V1_5_REVIEW_STRATEGY_CARD_IDS,
    _render_case_text,
    aggregate_gate,
    build_control_manifest,
    build_judge_endpoint_descriptors,
    build_positive_controls,
    judge_messages,
)
from metacom_pm.v1_5_judge_isolation import require_judge_role_isolation
from metacom_pm.paid_run_release import require_paid_run_release
from metacom_pm.v1_5_actual_corpus_review import (
    ACTUAL_CORPUS_CONTROL_PROTOCOL,
    ACTUAL_CORPUS_REVIEW_PROTOCOL,
    ACTUAL_CORPUS_REVIEW_STAGE,
    ACTUAL_REVIEW_QUESTIONS_EN,
    build_generation_pilot_review_items,
    build_actual_corpus_review_items,
)
from metacom_pm.text import conservative_token_bound, estimate_tokens

ROOT = Path(__file__).resolve().parents[1]
# Development-only panel. Final-evaluation judges are rejected by resolved
# identity even if a caller supplies a different endpoint alias.
DEFAULT_JUDGE_ENDPOINTS = (
    "training_judge_gemini_flash_lite",
    "training_judge_deepseek_flash",
)
# Worst-case physical attempts per logical call: 1 initial + 2 bounded retries
# for a transient 408/429/5xx/network-timeout failure (see
# module docstring's second amendment and metacom_pm.bounded_retry).
MAX_PHYSICAL_ATTEMPTS_PER_CALL = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument(
        "--pm-v1-5-config",
        type=Path,
        default=ROOT / "configs" / "pm_v1_5.yaml",
    )
    parser.add_argument("--strategy-bank", type=Path, default=ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl")
    parser.add_argument(
        "--review-scope",
        choices=("pilot_27", "actual_468"),
        default="pilot_27",
    )
    parser.add_argument(
        "--generation-pilot-attestation",
        type=Path,
        help=(
            "Exact paid nine-case compatibility-pilot attestation; required "
            "for pilot_27 and forbidden for actual_468."
        ),
    )
    parser.add_argument(
        "--states",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "pm_v2_states.jsonl",
    )
    parser.add_argument(
        "--evaluator-contexts",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "evaluator_contexts.jsonl",
    )
    parser.add_argument(
        "--memory-backend",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "memory_backend.jsonl",
    )
    parser.add_argument(
        "--bundles",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "pm_v2_bundles.jsonl",
    )
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--judge-endpoints", nargs="+", default=list(DEFAULT_JUDGE_ENDPOINTS)
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_automated_semantic_review_v8_7_native_gemini_candidate"
        ),
    )
    parser.add_argument("--max-api-calls", type=int, required=True)
    parser.add_argument("--max-estimated-usd", type=float, required=True)
    parser.add_argument("--max-input-tokens-per-call", type=int, required=True)
    parser.add_argument("--input-usd-per-million-tokens", type=float, required=True)
    parser.add_argument("--output-usd-per-million-tokens", type=float, required=True)
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run and args.overwrite:
        raise RuntimeError("paid automated semantic-review runs prohibit --overwrite")
    experiment_config = load_config(args.config)
    pm_config = load_config(args.pm_v1_5_config)
    if pm_config.get("version") != "pm-v1.5":
        raise RuntimeError("automated semantic review requires PM-v1.5")
    review_stage = (
        ACTUAL_CORPUS_REVIEW_STAGE
        if args.review_scope == "actual_468"
        else "pm_v1_5_automated_semantic_review"
    )
    review_protocol = (
        ACTUAL_CORPUS_REVIEW_PROTOCOL
        if args.review_scope == "actual_468"
        else AUTOMATED_REVIEW_PROTOCOL
    )
    control_cfg = dict(
        pm_config[
            "actual_corpus_semantic_audit"
            if args.review_scope == "actual_468"
            else "automated_semantic_review"
        ]
    )
    expected_endpoint_names = list(control_cfg.get("judge_endpoints") or [])
    if list(args.judge_endpoints) != expected_endpoint_names:
        raise RuntimeError(
            f"{args.review_scope} judge endpoints differ from frozen config: "
            f"expected={expected_endpoint_names}, got={list(args.judge_endpoints)}"
        )
    require_paid_run_release(
        pm_config,
        config_path=args.pm_v1_5_config,
        stage=(
            "development_actual_corpus_semantic_review"
            if args.review_scope == "actual_468"
            else "development_pilot_semantic_review"
        ),
        run=bool(args.run),
        run_identity=args.accept_cost_estimate_sha256,
    )
    judge_role_isolation = require_judge_role_isolation(
        experiment_config,
        pm_config,
        development_endpoint_names=args.judge_endpoints,
    )
    endpoints = {
        name: endpoint_from_config(experiment_config, name) for name in args.judge_endpoints
    }
    judge_endpoint_descriptors = build_judge_endpoint_descriptors(
        experiment_config, args.judge_endpoints
    )
    families = [str(endpoint.family or "") for endpoint in endpoints.values()]
    if len(endpoints) < 2 or "" in families or len(set(families)) != len(families):
        raise RuntimeError("automated review requires distinct declared judge families")

    corpus_audit = None
    paid_pilot_audit = None
    if args.review_scope == "actual_468":
        if args.generation_pilot_attestation is not None:
            raise RuntimeError(
                "actual_468 review does not accept a compatibility-pilot input"
            )
        actual_cfg = control_cfg
        if (
            actual_cfg.get("protocol") != ACTUAL_CORPUS_REVIEW_PROTOCOL
            or actual_cfg.get("control_protocol")
            != ACTUAL_CORPUS_CONTROL_PROTOCOL
            or list(actual_cfg.get("required_control_fields") or [])
            != list(RATING_FIELDS)
            or int(actual_cfg.get("controls_per_field") or 0) != 2
            or int(actual_cfg.get("control_seed") or -1) != int(args.seed)
        ):
            raise RuntimeError("actual-corpus control contract/config drift")
        real_case_rows, controls, corpus_audit = build_actual_corpus_review_items(
            states_path=args.states,
            evaluator_contexts_path=args.evaluator_contexts,
            backend_path=args.memory_backend,
            bundles_path=args.bundles,
            strategy_bank_path=args.strategy_bank,
            strategy_top_k=int(pm_config["retrieval"]["strategy_top_k"]),
            strategy_min_score=float(pm_config["retrieval"]["strategy_min_score"]),
            maximum_fallback_rate_by_split=actual_cfg[
                "maximum_provider_surface_fallback_rate_by_split"
            ],
            control_seed=args.seed,
            required_control_fields=actual_cfg["required_control_fields"],
            controls_per_field=int(actual_cfg["controls_per_field"]),
        )
    else:
        if args.generation_pilot_attestation is None:
            raise RuntimeError(
                "pilot_27 review requires --generation-pilot-attestation"
            )
        pilot_cfg = control_cfg
        if (
            pilot_cfg.get("protocol") != AUTOMATED_REVIEW_PROTOCOL
            or pilot_cfg.get("control_protocol") != AUTOMATED_CONTROL_PROTOCOL
            or list(pilot_cfg.get("required_control_fields") or [])
            != list(RATING_FIELDS)
            or int(pilot_cfg.get("controls_per_field") or 0) != 2
            or int(pilot_cfg.get("control_seed") or -1) != int(args.seed)
        ):
            raise RuntimeError("pilot control contract/config drift")
        cases = generate_v8_review_cases(
            strategy_bank_path=args.strategy_bank,
            cases_per_regime=VALIDATION_CASES_PER_REGIME,
            seed=args.seed,
            strategy_card_ids=V1_5_REVIEW_STRATEGY_CARD_IDS,
        )
        controls = build_positive_controls(
            cases,
            seed=args.seed,
            required_fields=pilot_cfg["required_control_fields"],
            controls_per_field=int(pilot_cfg["controls_per_field"]),
        )
        deterministic_case_rows = [
            {"kind": "real", "item_id": case.item_id, "text": _render_case_text(case)}
            for case in cases
        ]
        paid_pilot_rows, paid_pilot_audit = build_generation_pilot_review_items(
            pilot_attestation_path=args.generation_pilot_attestation,
            experiment_config_path=args.config,
            pm_v1_5_config_path=args.pm_v1_5_config,
            strategy_bank_path=args.strategy_bank,
            strategy_top_k=int(pm_config["retrieval"]["strategy_top_k"]),
            strategy_min_score=float(pm_config["retrieval"]["strategy_min_score"]),
        )
        real_case_rows = deterministic_case_rows + paid_pilot_rows

    planning = dict(pm_config["api_cost_planning"])
    if set(planning) != {
        "input_token_safety_factor",
        "fail_on_reported_input_overrun",
    } or float(planning["input_token_safety_factor"]) < 1.0 or not bool(
        planning["fail_on_reported_input_overrun"]
    ):
        raise RuntimeError("automated review requires fail-closed token planning")
    prices = {
        "input": float(args.input_usd_per_million_tokens),
        "output": float(args.output_usd_per_million_tokens),
    }
    if any(value < 0.0 for value in prices.values()):
        raise ValueError("automated-review prices must be non-negative")
    case_rows = list(real_case_rows) + [
        {"kind": "control", "item_id": row["item_id"], "text": row["case_text"]}
        for row in controls
    ]
    call_plan = []
    execution = {}
    rating_questions = ACTUAL_REVIEW_QUESTIONS_EN
    for case_row in case_rows:
        messages = (
            judge_messages(
                str(case_row["text"]), rating_questions=rating_questions
            )
            if rating_questions is not None
            else judge_messages(str(case_row["text"]))
        )
        for endpoint_name, endpoint in endpoints.items():
            payload = chat_request_payload(
                endpoint,
                messages,
                temperature=0.0,
                max_tokens=500,
                seed=13,
                response_schema=AutomatedSemanticReviewOutput,
            )
            if not request_payload_has_schema(payload):
                raise RuntimeError("automated-review request lacks structured schema")
            payload_text = canonical_json(payload)
            prompt_sha256 = sha256_text(canonical_json(messages))
            record_ids = {
                "kind": case_row["kind"],
                "item_id": case_row["item_id"],
                "judge_family": str(endpoint.family),
            }
            call_key = physical_call_key(
                stage=review_stage,
                record_ids=record_ids,
                prompt_sha256=prompt_sha256,
                endpoint=endpoint,
                request_parameters={
                    "temperature": 0.0,
                    "max_tokens": 500,
                    "seed": 13,
                    "response_schema": AutomatedSemanticReviewOutput.__name__,
                    "retries": 1,
                },
            )
            bound = conservative_token_bound(
                payload_text,
                safety_factor=float(planning["input_token_safety_factor"]),
            )
            call_plan.append(
                {
                    **record_ids,
                    "endpoint_name": endpoint_name,
                    "judge_model": endpoint.model,
                    "physical_call_key": call_key,
                    "prompt_sha256": prompt_sha256,
                    "request_payload_sha256": sha256_text(payload_text),
                    "raw_estimated_input_tokens": estimate_tokens(payload_text),
                    "input_token_upper_bound": bound,
                    "maximum_output_tokens": 500,
                    "maximum_physical_attempts": MAX_PHYSICAL_ATTEMPTS_PER_CALL,
                    # Worst case: every physical attempt up to the retry budget
                    # is a real, separately-billed call before one finally
                    # succeeds or the call is abandoned.
                    "maximum_cost_usd": MAX_PHYSICAL_ATTEMPTS_PER_CALL
                    * (
                        bound / 1_000_000 * prices["input"]
                        + 500 / 1_000_000 * prices["output"]
                    ),
                }
            )
            execution[call_key] = {
                "endpoint": endpoint,
                "messages": messages,
                "record_ids": record_ids,
            }
    endpoint_order = {
        name: index for index, name in enumerate(args.judge_endpoints)
    }
    # The frozen panel lists Gemini first.  Preserve that order so the repaired
    # native transport is exercised by the first paid call; if compatibility
    # is still broken, the fail-closed run stops before spending on DeepSeek.
    # This affects transport safety only, never the complete matrix or gate.
    call_plan = sorted(
        call_plan,
        key=lambda row: (
            row["kind"],
            row["item_id"],
            endpoint_order[str(row["endpoint_name"])],
        ),
    )
    n_calls = len(call_plan)
    # Worst-case physical HTTP attempts across the whole batch if every
    # logical call needed its full bounded-retry budget before succeeding
    # (or being abandoned) -- the number actually authorized and cost-capped,
    # per the module docstring's second amendment.
    max_physical_attempts_worst_case = n_calls * MAX_PHYSICAL_ATTEMPTS_PER_CALL
    estimate_payload = {
        "protocol": review_protocol,
        "stage": review_stage,
        "review_scope": args.review_scope,
        "n_real_cases": len(real_case_rows),
        "n_controls": len(controls),
        "control_protocol": control_cfg["control_protocol"],
        "required_control_fields": list(control_cfg["required_control_fields"]),
        "controls_per_field": int(control_cfg["controls_per_field"]),
        "control_seed": int(control_cfg["control_seed"]),
        "control_matrix_sha256": sha256_text(
            canonical_json(build_control_manifest(controls))
        ),
        "n_judge_families": len(endpoints),
        "n_logical_calls": n_calls,
        "maximum_physical_attempts_per_call": MAX_PHYSICAL_ATTEMPTS_PER_CALL,
        "maximum_physical_api_attempts": max_physical_attempts_worst_case,
        "call_plan_sha256": sha256_text(canonical_json(call_plan)),
        "maximum_estimated_usd": sum(row["maximum_cost_usd"] for row in call_plan),
        "maximum_input_tokens_per_call": max(
            row["input_token_upper_bound"] for row in call_plan
        ),
        "pricing_usd_per_mtok": prices,
        "api_cost_planning": planning,
        "judge_role_isolation": judge_role_isolation,
        "judge_endpoint_descriptors": judge_endpoint_descriptors,
        "call_order_protocol": "frozen-endpoint-order-native-gemini-first-v1",
        "review_strategy_card_ids": (
            {}
            if args.review_scope == "actual_468"
            else dict(V1_5_REVIEW_STRATEGY_CARD_IDS)
        ),
        "review_strategy_card_ids_sha256": (
            None
            if args.review_scope == "actual_468"
            else sha256_text(canonical_json(V1_5_REVIEW_STRATEGY_CARD_IDS))
        ),
        "paid_pilot_audit": paid_pilot_audit,
        "retry_contract": {
            "protocol": RETRY_CONTRACT_PROTOCOL,
            "retryable_up_to_full_budget": sorted(
                RETRYABLE_UP_TO_FULL_BUDGET
            ),
            "missing_field_maximum_additional_physical_attempts": 1,
            "never_retried": [
                "provider_request_error_4xx",
                "structured_output_validation_error",
                "successfully_parsed_but_unfavorable_score",
                "stage_postcondition_failure",
            ],
            "backoff_seconds": list(DEFAULT_BACKOFF_SECONDS),
            "retry_after_header_is_respected": True,
            "cross_process_eligibility_source": "physical_attempt_ledger",
            "legacy_or_unclassified_failure_policy": "fail_closed",
        },
        "budget_limits": {
            "max_api_calls": int(args.max_api_calls),
            "max_estimated_usd": float(args.max_estimated_usd),
            "max_input_tokens_per_call": int(args.max_input_tokens_per_call),
        },
    }
    estimate = {
        **estimate_payload,
        "cost_estimate_sha256": sha256_text(canonical_json(estimate_payload)),
    }
    budget_gate = {
        "status": (
            "PASS"
            if max_physical_attempts_worst_case <= args.max_api_calls
            and estimate["maximum_estimated_usd"] <= args.max_estimated_usd
            and estimate["maximum_input_tokens_per_call"]
            <= args.max_input_tokens_per_call
            else "FAIL"
        ),
        "checks": {
            "physical_api_attempts": max_physical_attempts_worst_case
            <= args.max_api_calls,
            "estimated_cost_usd": estimate["maximum_estimated_usd"]
            <= args.max_estimated_usd,
            "max_input_tokens_per_call": estimate[
                "maximum_input_tokens_per_call"
            ]
            <= args.max_input_tokens_per_call,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    estimate_path = args.out_dir / "cost_estimate.json"
    plan_path = args.out_dir / "call_plan.jsonl"
    ledger_path = args.out_dir / "physical_attempt_ledger.jsonl"
    forbid_overwrite_of_spent_attempts(
        ledger_path,
        overwrite=args.overwrite,
        stage=f"PM-v1.5 {args.review_scope} semantic review",
    )
    if args.dry_run:
        if ledger_path.is_file() and ledger_path.stat().st_size:
            if read_json(estimate_path) != {**estimate, "budget_gate": budget_gate} or list(
                iter_jsonl(plan_path)
            ) != call_plan:
                raise RuntimeError("spent automated-review ledger has a stale dry run")
        else:
            write_json(estimate_path, {**estimate, "budget_gate": budget_gate})
            write_jsonl(plan_path, call_plan)
        print({**estimate, "budget_gate": budget_gate, "status": "DRY_RUN_COMPLETE"})
        if budget_gate["status"] != "PASS":
            raise RuntimeError("automated semantic-review budget gate failed")
        return
    if budget_gate["status"] != "PASS":
        raise RuntimeError("automated semantic-review budget gate failed")
    if not estimate_path.is_file() or not plan_path.is_file():
        raise RuntimeError("automated review requires a saved matching dry run")
    if read_json(estimate_path) != {**estimate, "budget_gate": budget_gate} or list(
        iter_jsonl(plan_path)
    ) != call_plan:
        raise RuntimeError("saved automated-review dry run is stale")
    if args.accept_cost_estimate_sha256 != estimate["cost_estimate_sha256"]:
        raise RuntimeError(
            "automated review --run requires exact --accept-cost-estimate-sha256"
        )
    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=review_stage,
        expected_calls={
            str(row["physical_call_key"]): MAX_PHYSICAL_ATTEMPTS_PER_CALL
            for row in call_plan
        },
        maximum_total_attempts=max_physical_attempts_worst_case,
    )
    blocked: dict[str, str] = {}
    for row in call_plan:
        call_key = str(row["physical_call_key"])
        if ledger.succeeded(call_key):
            continue
        blocker = call_retry_blocker(ledger, call_key)
        if blocker is not None:
            blocked[call_key] = blocker
    if blocked:
        first_call_key = sorted(blocked)[0]
        raise RuntimeError(
            "automated semantic-review PASS is unreachable from the persisted "
            f"ledger: {first_call_key}: {blocked[first_call_key]}"
        )
    pending = [
        row for row in call_plan if not ledger.succeeded(str(row["physical_call_key"]))
    ]
    clients = {}
    if pending:
        for endpoint in endpoints.values():
            endpoint.api_key
        clients = {
            str(endpoint.family): make_client(endpoint)
            for endpoint in endpoints.values()
        }
    try:
        for row in pending:
            call_key = str(row["physical_call_key"])
            item = execution[call_key]

            def call_fn(item=item):
                return clients[str(item["record_ids"]["judge_family"])].chat(
                    item["messages"],
                    temperature=0.0,
                    max_tokens=500,
                    seed=13,
                    response_schema=AutomatedSemanticReviewOutput,
                    retries=1,
                )

            reservation = None
            try:
                reservation, result, parsed = execute_with_bounded_retry(
                    ledger,
                    call_key,
                    record_ids=item["record_ids"],
                    prompt_sha256=str(row["prompt_sha256"]),
                    call_fn=call_fn,
                )
                # execute_with_bounded_retry has already ledgered every failed
                # physical attempt; this reservation is still open (STARTED
                # only) because only we know whether our own post-hoc checks
                # below also accept it.
                if parsed is None:
                    raise RuntimeError("automated review returned no parsed object")
                usage = require_reported_usage(
                    result.usage, stage=f"PM-v1.5 {args.review_scope} semantic review"
                )
                if usage["prompt_tokens"] > int(row["input_token_upper_bound"]):
                    raise RuntimeError("automated-review prompt tokens exceed bound")
                ledger.finish(
                    reservation,
                    succeeded=True,
                    request_hash=result.request_hash,
                    usage=usage,
                    error=None,
                    result={
                        "parsed": parsed.model_dump(mode="json"),
                        "raw_text": result.text,
                    },
                )
            except Exception as exc:
                if reservation is not None and ledger.terminal_event(
                    reservation.call_key, reservation.attempt_index
                ) is None:
                    # The physical attempt itself succeeded; only our own
                    # post-hoc validation rejected it. Never retried -- this
                    # is a deterministic accounting/schema problem, not a
                    # transient one.
                    ledger.finish(
                        reservation,
                        succeeded=False,
                        request_hash=result.request_hash,
                        usage=result.usage,
                        error=f"{type(exc).__name__}: {exc}",
                        result={
                            "provider_response": result.raw_response,
                            "parsed": (
                                parsed.model_dump(mode="json")
                                if parsed is not None
                                else None
                            ),
                            "raw_text": result.text,
                            "provider_finish_reason": (
                                result.provider_finish_reason
                            ),
                            "normalized_finish_reason": (
                                result.normalized_finish_reason
                            ),
                        },
                        metadata=failure_metadata(
                            retry_class="stage_postcondition_failure",
                            retry_disposition=TERMINAL_DISPOSITION,
                        ),
                    )
                raise
    finally:
        for client in clients.values():
            client.close()

    real_case_results: dict[str, dict] = {}
    control_results: dict[str, dict] = {}
    for row in call_plan:
        call_key = str(row["physical_call_key"])
        terminal = ledger.terminal_row(call_key)
        if terminal is None or not ledger.succeeded(call_key):
            raise RuntimeError("automated semantic-review matrix is incomplete")
        result = terminal.get("result") or {}
        parsed = AutomatedSemanticReviewOutput.model_validate(result.get("parsed"))
        payload = parsed.model_dump(mode="json")
        judgment = {
            "ratings": {
                key: int(payload[key])
                for key in AutomatedSemanticReviewOutput.model_fields
                if key != "notes"
            },
            "notes": payload["notes"],
            "raw_text": result.get("raw_text"),
            "usage": terminal.get("usage"),
            "request_hash": terminal.get("request_hash"),
        }
        destination = real_case_results if row["kind"] == "real" else control_results
        destination.setdefault(str(row["item_id"]), {})[str(row["endpoint_name"])] = judgment

    gate = {
        **aggregate_gate(
            real_case_results=real_case_results,
            control_results=control_results,
            controls=controls,
            judge_family_names=list(endpoints),
            control_protocol=str(control_cfg["control_protocol"]),
            required_control_fields=control_cfg["required_control_fields"],
            controls_per_field=int(control_cfg["controls_per_field"]),
            protocol=review_protocol,
        ),
        "review_scope": args.review_scope,
        "corpus_audit": corpus_audit,
        "paid_pilot_audit": paid_pilot_audit,
        "n_paid_pilot_cases": (
            int((paid_pilot_audit or {}).get("item_count") or 0)
        ),
        "judge_endpoint_descriptors": judge_endpoint_descriptors,
        "transport_retry_summary": retry_ledger_summary(
            ledger,
            [str(row["physical_call_key"]) for row in call_plan],
        ),
    }
    write_json(args.out_dir / "real_case_judgments.json", real_case_results)
    write_json(args.out_dir / "control_judgments.json", control_results)
    write_json(
        args.out_dir / "controls.json",
        gate["control_manifest"],
    )
    write_json(args.out_dir / "gate_report.json", gate)
    create_artifact_attestation(
        args.out_dir / "artifact_attestation.json",
        stage=review_stage,
        inputs={
            "experiment_config": args.config,
            "pm_v1_5_config": args.pm_v1_5_config,
            "strategy_bank": args.strategy_bank,
            **(
                {
                    "states": args.states,
                    "evaluator_contexts": args.evaluator_contexts,
                    "memory_backend": args.memory_backend,
                    "bundles": args.bundles,
                }
                if args.review_scope == "actual_468"
                else {
                    "generation_pilot_attestation": args.generation_pilot_attestation,
                    "generation_pilot_bundle": Path(
                        str(paid_pilot_audit["pilot_bundle_path"])
                    ),
                }
            ),
            "cost_estimate": estimate_path,
            "call_plan": plan_path,
        },
        outputs={
            "real_case_judgments": (
                args.out_dir / "real_case_judgments.json",
                False,
            ),
            "control_judgments": (
                args.out_dir / "control_judgments.json",
                False,
            ),
            "controls": (args.out_dir / "controls.json", False),
            "gate_report": (args.out_dir / "gate_report.json", False),
            "physical_attempt_ledger": (ledger_path, True),
        },
        parameters={
            "protocol": review_protocol,
            "review_scope": args.review_scope,
            "control_protocol": control_cfg["control_protocol"],
            "required_control_fields": list(control_cfg["required_control_fields"]),
            "controls_per_field": int(control_cfg["controls_per_field"]),
            "control_seed": int(control_cfg["control_seed"]),
            "control_matrix_sha256": gate["control_matrix_sha256"],
            "judge_role_isolation": judge_role_isolation,
            "judge_endpoint_descriptors": judge_endpoint_descriptors,
            "review_strategy_card_ids": (
                {}
                if args.review_scope == "actual_468"
                else dict(V1_5_REVIEW_STRATEGY_CARD_IDS)
            ),
            "paid_pilot_attestation_sha256": (
                None
                if paid_pilot_audit is None
                else paid_pilot_audit["pilot_attestation_sha256"]
            ),
            "paid_pilot_contract_sha256": (
                None
                if paid_pilot_audit is None
                else paid_pilot_audit["pilot_contract_sha256"]
            ),
            "accepted_cost_estimate_sha256": estimate["cost_estimate_sha256"],
            "provider_client_retries": 1,
            "retry_contract": estimate["retry_contract"],
            "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        },
        expected={
            "status": gate["status"],
            "judge_role_isolation_status": judge_role_isolation["status"],
            "logical_calls": n_calls,
            "physical_attempts": ledger.started_attempts,
        },
    )
    print(gate)
    if gate["status"] != "PASS":
        raise RuntimeError("PM-v1.5 automated semantic-review gate did not PASS")


if __name__ == "__main__":
    main()
