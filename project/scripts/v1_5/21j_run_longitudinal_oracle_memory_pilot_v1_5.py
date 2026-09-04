from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from metacom_pm.artifacts import (
    create_artifact_attestation,
    require_artifact_attestation,
)
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import ActionOutcome
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
from metacom_pm.paid_run_release import require_paid_run_release
from metacom_pm.sweep import plan_action_sweep, run_action_sweep
from metacom_pm.v1_5_oracle_memory_pilot import (
    CONTROL_ARM,
    HARMFUL_ARM,
    HELPFUL_ARM,
    ORACLE_MEMORY_PILOT_PROTOCOL,
)


EXECUTION_PROTOCOL = (
    "pm-v1.5-longitudinal-oracle-memory-upper-bound-execution-v1"
)
COST_PROTOCOL = "pm-v1.5-longitudinal-oracle-memory-upper-bound-cost-v1"
PILOT_STAGE = "longitudinal_oracle_memory_upper_bound_pilot"
TRANSPORT_MAX_ATTEMPTS = 4
TRANSPORT_BACKOFF_SECONDS = (10.0, 30.0, 60.0)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_CONTRACT_PATH = (
    PROJECT_ROOT
    / "data/pm_v1_5_contracts/"
    "longitudinal_oracle_memory_upper_bound_pilot_v1.json"
)
CANONICAL_RUNTIME_PATH = (
    PROJECT_ROOT
    / "data/pm_v1_5_contracts/"
    "longitudinal_oracle_memory_upper_bound_runtime_v1.jsonl"
)
CANONICAL_BACKEND_PATH = (
    PROJECT_ROOT
    / "data/pm_v1_5_contracts/"
    "longitudinal_oracle_memory_upper_bound_backend_v1.jsonl"
)
ANALYSIS_CODE_PATH = (
    PROJECT_ROOT / "src/metacom_pm/v1_5_oracle_memory_uptake.py"
)
ANALYSIS_RUNNER_PATH = (
    PROJECT_ROOT
    / "scripts/v1_5/"
    "21k_analyze_longitudinal_oracle_memory_pilot_v1_5.py"
)


def _validate_contract_and_inputs(
    *,
    contract_path: Path,
    runtime_path: Path,
    backend_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required = {
        contract_path.resolve(): CANONICAL_CONTRACT_PATH.resolve(),
        runtime_path.resolve(): CANONICAL_RUNTIME_PATH.resolve(),
        backend_path.resolve(): CANONICAL_BACKEND_PATH.resolve(),
    }
    if any(actual != expected for actual, expected in required.items()):
        raise RuntimeError("formal oracle-memory pilot requires canonical inputs")
    contract = read_json(contract_path)
    without_sha = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    if (
        contract.get("protocol") != ORACLE_MEMORY_PILOT_PROTOCOL
        or contract.get("status") != "READY_ZERO_API_DESIGN"
        or contract.get("scope") != "longitudinal_train_only"
        or contract.get("contract_sha256")
        != sha256_text(canonical_json(without_sha))
        or contract.get("api_judges_used") is not False
        or contract.get("training_labels_created") is not False
        or int(contract.get("planned_new_logical_calls", -1)) != 18
    ):
        raise RuntimeError("oracle-memory pilot contract is stale or invalid")
    lineage = dict(contract.get("source_lineage") or {})
    if lineage.get("oracle_runtime_sha256") != sha256_file(runtime_path):
        raise RuntimeError("oracle-memory runtime digest drifted")
    if lineage.get("oracle_backend_sha256") != sha256_file(backend_path):
        raise RuntimeError("oracle-memory backend digest drifted")
    selected = [dict(row) for row in contract.get("selected_states") or []]
    if len(selected) != 18:
        raise RuntimeError("oracle-memory pilot must select exactly 18 states")
    if len({str(row["state_id"]) for row in selected}) != 18:
        raise RuntimeError("oracle-memory pilot repeats a state")
    if len({str(row["user_id"]) for row in selected}) != 18:
        raise RuntimeError("oracle-memory pilot repeats a user")
    if {
        str(row["treatment_arm"]) for row in selected
    } != {HELPFUL_ARM, HARMFUL_ARM}:
        raise RuntimeError("oracle-memory treatment arms drifted")
    runtime_rows = [dict(row) for row in iter_jsonl(runtime_path)]
    backend_rows = [dict(row) for row in iter_jsonl(backend_path)]
    if {str(row["card_id"]) for row in runtime_rows} != {
        str(row["card_id"]) for row in backend_rows
    }:
        raise RuntimeError("oracle-memory runtime/backend cards differ")
    if {str(row["card_id"]) for row in runtime_rows} != {
        str(row["card_id"]) for row in selected
    }:
        raise RuntimeError("oracle-memory selected/runtime cards differ")
    backend_by_card = {str(row["card_id"]): row for row in backend_rows}
    for row in selected:
        actual_ids = {
            str(item["memory_id"])
            for item in backend_by_card[str(row["card_id"])]["items"]
        }
        if actual_ids != set(str(value) for value in row["target_memory_ids"]):
            raise RuntimeError(
                f"oracle-memory item set drifted: {row['state_id']}"
            )
    return contract, selected


def _select_controls(
    *,
    outcomes_path: Path,
    attestation_path: Path,
    selected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    verification = require_artifact_attestation(
        attestation_path,
        required_stage="action_sweep",
        required_output_paths={"action_outcomes": outcomes_path},
    )
    wanted = {str(row["state_id"]) for row in selected}
    found: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(outcomes_path):
        state_id = str(row["state_id"])
        if state_id not in wanted or str(row["action_id"]) != "M0+R0":
            continue
        if state_id in found:
            raise RuntimeError(f"duplicate M0 control: {state_id}")
        outcome = ActionOutcome.model_validate(row).model_dump(mode="json")
        if outcome["memory_view"] or outcome["strategy_view"]:
            raise RuntimeError(f"M0 control exposes evidence: {state_id}")
        outcome["provenance"] = {
            **dict(outcome["provenance"]),
            "oracle_memory_pilot_arm": CONTROL_ARM,
            "zero_api_control_reuse": True,
        }
        found[state_id] = outcome
    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(f"oracle-memory controls are incomplete: {missing}")
    return [found[state_id] for state_id in sorted(found)], {
        "control_outcomes_sha256": sha256_file(outcomes_path),
        "control_attestation_sha256": verification["attestation_sha256"],
    }


def _plan(
    *,
    config_path: Path,
    experiment_config_path: Path,
    contract_path: Path,
    runtime_path: Path,
    backend_path: Path,
    strategy_bank_path: Path,
    control_outcomes_path: Path,
    control_attestation_path: Path,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    contract, selected = _validate_contract_and_inputs(
        contract_path=contract_path,
        runtime_path=runtime_path,
        backend_path=backend_path,
    )
    controls, control_binding = _select_controls(
        outcomes_path=control_outcomes_path,
        attestation_path=control_attestation_path,
        selected=selected,
    )
    config = load_config(config_path)
    experiment = load_config(experiment_config_path)
    supporter = SupporterGenerationContract.from_config(config)
    endpoint = endpoint_from_config(experiment, supporter.generator_endpoint)
    pricing = dict(config["development_sweep"]["pricing_usd_per_mtok"])
    retrieval = dict(config["retrieval"])
    planning = dict(config["api_cost_planning"])
    cards_by_action: dict[str, set[str]] = defaultdict(set)
    selected_by_card = {str(row["card_id"]): row for row in selected}
    for row in selected:
        cards_by_action[str(row["target_action_id"])].add(str(row["card_id"]))

    contract_bindings = {
        "protocol": EXECUTION_PROTOCOL,
        "pilot_contract_file_sha256": sha256_file(contract_path),
        "pilot_contract_sha256": contract["contract_sha256"],
        "oracle_runtime_sha256": sha256_file(runtime_path),
        "oracle_backend_sha256": sha256_file(backend_path),
        "supporter_generation_treatment_sha256": supporter.digest(),
        "oracle_item_selection": "evaluator_item_utility_exact_set",
        "retrieval_score_gate": "bypassed_for_diagnostic_upper_bound",
        "scientific_interpretation": "report_only_no_training_authority",
    }
    group_costs: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    for action_id in sorted(cards_by_action):
        cards = cards_by_action[action_id]
        group_cost, rows = plan_action_sweep(
            runtime_path,
            backend_path,
            strategy_bank_path,
            endpoint=endpoint,
            card_filter=cards,
            action_filter={action_id},
            temperature=supporter.temperature,
            max_tokens=supporter.max_output_tokens,
            seed=int(config["development_sweep"]["seed"]),
            request_retries=int(config["development_sweep"]["request_retries"]),
            transport_max_attempts_per_call=TRANSPORT_MAX_ATTEMPTS,
            fail_fast=False,
            input_token_safety_factor=float(
                planning["input_token_safety_factor"]
            ),
            fail_on_reported_input_overrun=bool(
                planning["fail_on_reported_input_overrun"]
            ),
            strategy_top_k=int(retrieval["strategy_top_k"]),
            memory_min_score=None,
            strategy_min_score=float(retrieval["strategy_min_score"]),
            evidence_filter_config=None,
            supporter_generation_contract=supporter,
            input_usd_per_mtok=float(pricing["input"]),
            output_usd_per_mtok=float(pricing["output"]),
            contract_bindings={
                **contract_bindings,
                "target_action_id": action_id,
            },
        )
        if len(rows) != len(cards):
            raise RuntimeError(f"oracle-memory group cardinality drifted: {action_id}")
        for row in rows:
            selected_row = selected_by_card[str(row["card_id"])]
            combined_rows.append(
                {
                    **row,
                    "oracle_memory_pilot_arm": selected_row["treatment_arm"],
                    "target_memory_ids": selected_row["target_memory_ids"],
                }
            )
        group_costs.append(
            {
                "target_action_id": action_id,
                "card_ids": sorted(cards),
                "cost_estimate_sha256": group_cost["cost_estimate_sha256"],
                "call_plan_sha256": group_cost["call_plan_sha256"],
                "logical_api_calls": group_cost["logical_api_calls"],
                "expected_api_calls": group_cost["expected_api_calls"],
                "maximum_physical_api_attempts": group_cost[
                    "maximum_physical_api_attempts"
                ],
                "logical_estimated_cost_usd": group_cost[
                    "logical_estimated_cost_usd"
                ],
                "maximum_estimated_cost_usd": group_cost["estimated_cost_usd"],
            }
        )
    combined_rows.sort(
        key=lambda row: (
            str(row["state_id"]),
            str(row["action_id"]),
            str(row["call_key"]),
        )
    )
    if len(combined_rows) != 18:
        raise RuntimeError("oracle-memory call plan must contain 18 calls")
    for row in combined_rows:
        if set(str(value) for value in row["target_memory_ids"]) == set():
            raise RuntimeError("oracle-memory call lacks target memory")
        if int(row["kept_memory_count"]) != len(row["target_memory_ids"]):
            raise RuntimeError(
                f"oracle-memory planning did not expose the exact target set: "
                f"{row['state_id']}"
            )
        if int(row["kept_strategy_count"]) != 0:
            raise RuntimeError("oracle-memory R0 treatment exposed strategy")

    expected_calls = sum(int(row["expected_api_calls"]) for row in group_costs)
    maximum_attempts = sum(
        int(row["maximum_physical_api_attempts"]) for row in group_costs
    )
    expected_cost = sum(
        float(row["logical_estimated_cost_usd"]) for row in group_costs
    )
    maximum_cost = sum(
        float(row["maximum_estimated_cost_usd"]) for row in group_costs
    )
    maximum_input = max(
        (int(row["estimated_input_tokens"]) for row in combined_rows),
        default=0,
    )
    errors: list[str] = []
    if maximum_attempts > max_api_calls:
        errors.append("maximum physical attempts exceed max_api_calls")
    if maximum_cost > max_estimated_usd:
        errors.append("maximum estimated cost exceeds max_estimated_usd")
    if maximum_input > max_input_tokens_per_call:
        errors.append("per-call input estimate exceeds token ceiling")
    payload = {
        "protocol": COST_PROTOCOL,
        "stage": PILOT_STAGE,
        "execution_protocol": EXECUTION_PROTOCOL,
        "config_sha256": sha256_file(config_path),
        "experiment_config_sha256": sha256_file(experiment_config_path),
        "contract_file_sha256": sha256_file(contract_path),
        "contract_sha256": contract["contract_sha256"],
        "runner_sha256": sha256_file(__file__),
        "analysis_code_sha256": sha256_file(ANALYSIS_CODE_PATH),
        "analysis_runner_sha256": sha256_file(ANALYSIS_RUNNER_PATH),
        "shared_sweep_sha256": sha256_file(
            PROJECT_ROOT / "src/metacom_pm/sweep.py"
        ),
        "runtime_sha256": sha256_file(runtime_path),
        "backend_sha256": sha256_file(backend_path),
        "strategy_bank_sha256": sha256_file(strategy_bank_path),
        **control_binding,
        "supporter_generation_treatment_sha256": supporter.digest(),
        "endpoint": {
            "base_url": endpoint.base_url,
            "model": endpoint.model,
            "family": endpoint.family,
        },
        "temperature": supporter.temperature,
        "max_output_tokens": supporter.max_output_tokens,
        "seed": int(config["development_sweep"]["seed"]),
        "oracle_item_selection": "evaluator_item_utility_exact_set",
        "retrieval_score_gate": "bypassed_for_diagnostic_upper_bound",
        "transport_max_attempts_per_call": TRANSPORT_MAX_ATTEMPTS,
        "transport_backoff_seconds": list(TRANSPORT_BACKOFF_SECONDS),
        "planned_control_rows_reused_zero_api": len(controls),
        "planned_new_logical_calls": len(combined_rows),
        "expected_api_calls": expected_calls,
        "maximum_physical_api_attempts": maximum_attempts,
        "expected_estimated_cost_usd": expected_cost,
        "maximum_estimated_cost_usd": maximum_cost,
        "maximum_input_tokens_per_call": maximum_input,
        "pricing": {
            "input_usd_per_mtok": float(pricing["input"]),
            "output_usd_per_mtok": float(pricing["output"]),
        },
        "group_plans": group_costs,
        "call_plan_sha256": sha256_text(canonical_json(combined_rows)),
        "budget_gate": {
            "status": "PASS" if not errors else "FAIL",
            "errors": errors,
            "max_api_calls": max_api_calls,
            "max_estimated_usd": max_estimated_usd,
            "max_input_tokens_per_call": max_input_tokens_per_call,
        },
        "api_judges_used": False,
        "training_labels_created": False,
        "scientific_interpretation": "report_only_no_training_authority",
    }
    payload["cost_estimate_sha256"] = sha256_text(canonical_json(payload))
    return payload, combined_rows, controls


def _run_groups(
    *,
    out_dir: Path,
    cost: Mapping[str, Any],
    config_path: Path,
    experiment_config_path: Path,
    contract_path: Path,
    runtime_path: Path,
    backend_path: Path,
    strategy_bank_path: Path,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    experiment = load_config(experiment_config_path)
    supporter = SupporterGenerationContract.from_config(config)
    endpoint = endpoint_from_config(experiment, supporter.generator_endpoint)
    retrieval = dict(config["retrieval"])
    planning = dict(config["api_cost_planning"])
    selected = {
        str(row["card_id"]): row
        for row in read_json(contract_path)["selected_states"]
    }
    generated: list[dict[str, Any]] = []
    for group in cost["group_plans"]:
        action_id = str(group["target_action_id"])
        group_dir = out_dir / "groups" / action_id.replace("+", "_")
        summary = run_action_sweep(
            runtime_path,
            backend_path,
            strategy_bank_path,
            group_dir / "action_outcomes.jsonl",
            group_dir / "raw_api_calls.jsonl",
            group_dir / "summary.json",
            endpoint=endpoint,
            card_filter=set(group["card_ids"]),
            action_filter={action_id},
            temperature=supporter.temperature,
            max_tokens=supporter.max_output_tokens,
            seed=int(config["development_sweep"]["seed"]),
            request_retries=int(config["development_sweep"]["request_retries"]),
            transport_max_attempts_per_call=TRANSPORT_MAX_ATTEMPTS,
            fail_fast=False,
            input_token_safety_factor=float(
                planning["input_token_safety_factor"]
            ),
            fail_on_reported_input_overrun=bool(
                planning["fail_on_reported_input_overrun"]
            ),
            strategy_top_k=int(retrieval["strategy_top_k"]),
            memory_min_score=None,
            strategy_min_score=float(retrieval["strategy_min_score"]),
            evidence_filter_config=None,
            supporter_generation_contract=supporter,
            contract_bindings={
                "protocol": EXECUTION_PROTOCOL,
                "pilot_contract_file_sha256": sha256_file(contract_path),
                "pilot_contract_sha256": cost["contract_sha256"],
                "pilot_cost_estimate_sha256": cost["cost_estimate_sha256"],
                "analysis_code_sha256": cost["analysis_code_sha256"],
                "analysis_runner_sha256": cost["analysis_runner_sha256"],
                "oracle_runtime_sha256": cost["runtime_sha256"],
                "oracle_backend_sha256": cost["backend_sha256"],
                "target_action_id": action_id,
                "oracle_item_selection": (
                    "evaluator_item_utility_exact_set"
                ),
                "retrieval_score_gate": (
                    "bypassed_for_diagnostic_upper_bound"
                ),
                "scientific_interpretation": (
                    "report_only_no_training_authority"
                ),
            },
            max_physical_api_attempts=int(
                group["maximum_physical_api_attempts"]
            ),
            transport_retry_policy="bounded_transport",
            transport_backoff_seconds=TRANSPORT_BACKOFF_SECONDS,
            consecutive_same_class_circuit_breaker=5,
        )
        if summary.get("status") != "COMPLETE":
            raise RuntimeError(f"oracle-memory group did not complete: {action_id}")
        for row in iter_jsonl(group_dir / "action_outcomes.jsonl"):
            outcome = ActionOutcome.model_validate(row).model_dump(mode="json")
            selected_row = selected[str(outcome["card_id"])]
            if set(outcome["selected_memory_ids"]) != set(
                selected_row["target_memory_ids"]
            ):
                raise RuntimeError(
                    f"oracle-memory execution exposed wrong items: "
                    f"{outcome['state_id']}"
                )
            if outcome["strategy_view"]:
                raise RuntimeError("oracle-memory execution exposed strategy")
            outcome["provenance"] = {
                **dict(outcome["provenance"]),
                "oracle_memory_pilot_arm": selected_row["treatment_arm"],
                "zero_api_control_reuse": False,
            }
            generated.append(outcome)
    if len(generated) != 18:
        raise RuntimeError("oracle-memory generated outcome count changed")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--strategy-bank", type=Path, required=True)
    parser.add_argument("--control-outcomes", type=Path, required=True)
    parser.add_argument("--control-attestation", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-api-calls", type=int, default=100)
    parser.add_argument("--max-estimated-usd", type=float, default=0.1)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=4000)
    parser.add_argument("--accept-cost-estimate-sha256")
    args = parser.parse_args()

    cost, call_plan, controls = _plan(
        config_path=args.config,
        experiment_config_path=args.experiment_config,
        contract_path=args.contract,
        runtime_path=args.runtime,
        backend_path=args.backend,
        strategy_bank_path=args.strategy_bank,
        control_outcomes_path=args.control_outcomes,
        control_attestation_path=args.control_attestation,
        max_api_calls=args.max_api_calls,
        max_estimated_usd=args.max_estimated_usd,
        max_input_tokens_per_call=args.max_input_tokens_per_call,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "cost_estimate.json", cost)
    write_jsonl(args.out_dir / "call_plan.jsonl", call_plan)
    write_jsonl(args.out_dir / "control_outcomes.jsonl", controls)
    if cost["budget_gate"]["status"] != "PASS":
        raise RuntimeError(
            "oracle-memory pilot budget gate failed: "
            + "; ".join(cost["budget_gate"]["errors"])
        )
    config = load_config(args.config)
    require_paid_run_release(
        config,
        config_path=args.config,
        stage=PILOT_STAGE,
        run=bool(args.run),
        run_identity=args.accept_cost_estimate_sha256,
    )
    if args.dry_run:
        summary = {
            "protocol": EXECUTION_PROTOCOL,
            "status": "DRY_RUN_COMPLETE",
            "cost_estimate_sha256": cost["cost_estimate_sha256"],
            "call_plan_sha256": cost["call_plan_sha256"],
            "analysis_code_sha256": cost["analysis_code_sha256"],
            "analysis_runner_sha256": cost["analysis_runner_sha256"],
            "planned_new_logical_calls": cost["planned_new_logical_calls"],
            "maximum_physical_api_attempts": cost[
                "maximum_physical_api_attempts"
            ],
            "expected_estimated_cost_usd": cost[
                "expected_estimated_cost_usd"
            ],
            "maximum_estimated_cost_usd": cost[
                "maximum_estimated_cost_usd"
            ],
            "api_clients_created": 0,
            "api_judges_used": False,
            "training_labels_created": False,
        }
        write_json(args.out_dir / "summary.json", summary)
        print(summary)
        return
    if str(args.accept_cost_estimate_sha256 or "") != cost[
        "cost_estimate_sha256"
    ]:
        raise RuntimeError("accepted oracle-memory cost identity does not match")

    generated = _run_groups(
        out_dir=args.out_dir,
        cost=cost,
        config_path=args.config,
        experiment_config_path=args.experiment_config,
        contract_path=args.contract,
        runtime_path=args.runtime,
        backend_path=args.backend,
        strategy_bank_path=args.strategy_bank,
    )
    pilot_rows = sorted(
        [*controls, *generated],
        key=lambda row: (
            str(row["state_id"]),
            str(row["provenance"]["oracle_memory_pilot_arm"]),
        ),
    )
    write_jsonl(args.out_dir / "pilot_outcomes.jsonl", pilot_rows)
    total_input = sum(
        int(row["cost"]["total_input_tokens"]) for row in generated
    )
    total_output = sum(int(row["cost"]["output_tokens"]) for row in generated)
    pricing = cost["pricing"]
    actual_proxy_cost = (
        total_input / 1_000_000 * float(pricing["input_usd_per_mtok"])
        + total_output / 1_000_000 * float(pricing["output_usd_per_mtok"])
    )
    summary = {
        "protocol": EXECUTION_PROTOCOL,
        "status": "COMPLETE_REPORT_ONLY",
        "cost_estimate_sha256": cost["cost_estimate_sha256"],
        "call_plan_sha256": cost["call_plan_sha256"],
        "control_rows": len(controls),
        "new_generated_rows": len(generated),
        "pilot_rows": len(pilot_rows),
        "actual_input_tokens": total_input,
        "actual_output_tokens": total_output,
        "actual_proxy_cost_usd": actual_proxy_cost,
        "api_judges_used": False,
        "training_labels_created": False,
        "next_step": (
            "run the preregistered zero-API oracle-memory uptake analysis"
        ),
    }
    write_json(args.out_dir / "summary.json", summary)
    create_artifact_attestation(
        args.out_dir / "artifact_attestation.json",
        stage=PILOT_STAGE,
        inputs={
            "config": args.config,
            "experiment_config": args.experiment_config,
            "contract": args.contract,
            "runtime": args.runtime,
            "backend": args.backend,
            "strategy_bank": args.strategy_bank,
            "control_outcomes": args.control_outcomes,
            "control_attestation": args.control_attestation,
            "cost_estimate": args.out_dir / "cost_estimate.json",
            "call_plan": args.out_dir / "call_plan.jsonl",
        },
        outputs={
            "pilot_outcomes": (args.out_dir / "pilot_outcomes.jsonl", True),
            "summary": (args.out_dir / "summary.json", False),
        },
        parameters={
            "cost_estimate_sha256": cost["cost_estimate_sha256"],
            "call_plan_sha256": cost["call_plan_sha256"],
            "contract_sha256": cost["contract_sha256"],
            "analysis_code_sha256": cost["analysis_code_sha256"],
            "analysis_runner_sha256": cost["analysis_runner_sha256"],
            "transport_max_attempts_per_call": TRANSPORT_MAX_ATTEMPTS,
            "supporter_generation_treatment_sha256": cost[
                "supporter_generation_treatment_sha256"
            ],
            "api_judges_used": False,
            "training_labels_created": False,
        },
        expected={"control_rows": 18, "generated_rows": 18, "pilot_rows": 36},
    )
    print(summary)


if __name__ == "__main__":
    main()
