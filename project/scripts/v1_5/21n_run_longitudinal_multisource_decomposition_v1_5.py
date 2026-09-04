from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from metacom_pm.artifacts import create_artifact_attestation
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
from metacom_pm.paid_run_release import (
    require_output_directory_not_previously_consumed,
    require_paid_run_release,
)
from metacom_pm.sweep import plan_action_sweep, run_action_sweep
from metacom_pm.v1_5_multisource_decomposition import (
    MULTISOURCE_DECOMPOSITION_PROTOCOL,
    MULTISOURCE_DECOMPOSITION_STAGE,
    SINGLE_SOURCE_ACTIONS,
    validate_existing_multisource_outcomes,
)


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_CONTRACT = (
    ROOT
    / "data/pm_v1_5_contracts/"
    "longitudinal_multisource_decomposition_pilot_v1.json"
)
COST_PROTOCOL = (
    "pm-v1.5-longitudinal-multisource-decomposition-cost-v1"
)
EXECUTION_PROTOCOL = (
    "pm-v1.5-longitudinal-multisource-decomposition-execution-v1"
)
TRANSPORT_MAX_ATTEMPTS = 4
TRANSPORT_BACKOFF_SECONDS = (10.0, 30.0, 60.0)
CONTRACT_CODE_PATH = (
    ROOT / "src/metacom_pm/v1_5_multisource_decomposition.py"
)
CONTRACT_PREPARATION_RUNNER_PATH = (
    ROOT
    / "scripts/v1_5/"
    "21m_prepare_longitudinal_multisource_decomposition_v1_5.py"
)
ANALYSIS_CODE_PATH = (
    ROOT / "src/metacom_pm/v1_5_multisource_uptake.py"
)
ANALYSIS_RUNNER_PATH = (
    ROOT
    / "scripts/v1_5/"
    "21o_analyze_longitudinal_multisource_decomposition_v1_5.py"
)


def _load_contract(
    *,
    contract_path: Path,
    runtime_path: Path,
    backend_path: Path,
    oracle_outcomes_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if contract_path.resolve() != CANONICAL_CONTRACT.resolve():
        raise RuntimeError("formal multisource pilot requires canonical contract")
    contract = read_json(contract_path)
    without_sha = {
        key: value
        for key, value in contract.items()
        if key != "contract_sha256"
    }
    if (
        contract.get("protocol") != MULTISOURCE_DECOMPOSITION_PROTOCOL
        or contract.get("stage") != MULTISOURCE_DECOMPOSITION_STAGE
        or contract.get("status") != "READY_ZERO_API_DESIGN"
        or contract.get("contract_sha256")
        != sha256_text(canonical_json(without_sha))
        or int(contract.get("planned_new_logical_calls", -1)) != 9
        or contract.get("training_labels_created") is not False
    ):
        raise RuntimeError("multisource decomposition contract is invalid")
    lineage = dict(contract.get("source_lineage") or {})
    expected = {
        "oracle_runtime_sha256": sha256_file(runtime_path),
        "oracle_backend_sha256": sha256_file(backend_path),
        "oracle_outcomes_sha256": sha256_file(oracle_outcomes_path),
        "preparation_code_sha256": sha256_file(CONTRACT_CODE_PATH),
        "preparation_runner_sha256": sha256_file(
            CONTRACT_PREPARATION_RUNNER_PATH
        ),
    }
    for key, actual in expected.items():
        if lineage.get(key) != actual:
            raise RuntimeError(f"multisource decomposition {key} drifted")
    outcome_rows = [
        dict(row) for row in iter_jsonl(oracle_outcomes_path)
    ]
    reused = validate_existing_multisource_outcomes(
        contract=contract,
        outcome_rows=outcome_rows,
    )
    if len(reused) != 6:
        raise RuntimeError("multisource decomposition must reuse six rows")
    return contract, reused


def _plan(
    *,
    config_path: Path,
    experiment_config_path: Path,
    contract_path: Path,
    runtime_path: Path,
    backend_path: Path,
    strategy_bank_path: Path,
    oracle_outcomes_path: Path,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    contract, reused = _load_contract(
        contract_path=contract_path,
        runtime_path=runtime_path,
        backend_path=backend_path,
        oracle_outcomes_path=oracle_outcomes_path,
    )
    config = load_config(config_path)
    experiment = load_config(experiment_config_path)
    supporter = SupporterGenerationContract.from_config(config)
    endpoint = endpoint_from_config(experiment, supporter.generator_endpoint)
    pricing = dict(config["development_sweep"]["pricing_usd_per_mtok"])
    retrieval = dict(config["retrieval"])
    planning = dict(config["api_cost_planning"])
    states = [dict(row) for row in contract["states"]]
    card_ids = {str(row["card_id"]) for row in states}
    expected_by_card_action: dict[tuple[str, str], str] = {}
    for row in states:
        for arm in row["single_source_arms"]:
            expected_by_card_action[
                (str(row["card_id"]), str(arm["action_id"]))
            ] = str(arm["target_memory_id"])

    bindings = {
        "protocol": EXECUTION_PROTOCOL,
        "contract_file_sha256": sha256_file(contract_path),
        "contract_sha256": contract["contract_sha256"],
        "oracle_outcomes_sha256": sha256_file(oracle_outcomes_path),
        "supporter_generation_treatment_sha256": supporter.digest(),
        "retrieval_score_gate": "bypassed_for_diagnostic_upper_bound",
        "scientific_interpretation": "report_only_no_training_authority",
    }
    group_plans: list[dict[str, Any]] = []
    combined: list[dict[str, Any]] = []
    for source, action_id in sorted(SINGLE_SOURCE_ACTIONS.items()):
        group_cost, rows = plan_action_sweep(
            runtime_path,
            backend_path,
            strategy_bank_path,
            endpoint=endpoint,
            card_filter=card_ids,
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
                **bindings,
                "single_source": source,
                "target_action_id": action_id,
            },
        )
        if len(rows) != 3:
            raise RuntimeError(
                f"multisource decomposition group drifted: {action_id}"
            )
        for row in rows:
            expected_id = expected_by_card_action[
                (str(row["card_id"]), action_id)
            ]
            if int(row["kept_memory_count"]) != 1:
                raise RuntimeError(
                    f"single-source plan kept != 1 item: {row['state_id']}"
                )
            kept_ids = {
                str(value) for value in row.get("kept_memory_ids") or []
            }
            if kept_ids and kept_ids != {expected_id}:
                raise RuntimeError(
                    f"single-source plan selected wrong item: {row['state_id']}"
                )
            if int(row["kept_strategy_count"]) != 0:
                raise RuntimeError("single-source R0 plan exposed strategy")
            combined.append(
                {
                    **row,
                    "single_source": source,
                    "target_memory_id": expected_id,
                }
            )
        group_plans.append(
            {
                "single_source": source,
                "target_action_id": action_id,
                "card_ids": sorted(card_ids),
                "logical_api_calls": group_cost["logical_api_calls"],
                "expected_api_calls": group_cost["expected_api_calls"],
                "maximum_physical_api_attempts": group_cost[
                    "maximum_physical_api_attempts"
                ],
                "logical_estimated_cost_usd": group_cost[
                    "logical_estimated_cost_usd"
                ],
                "maximum_estimated_cost_usd": group_cost[
                    "estimated_cost_usd"
                ],
                "group_cost_estimate_sha256": group_cost[
                    "cost_estimate_sha256"
                ],
                "group_call_plan_sha256": group_cost["call_plan_sha256"],
            }
        )
    combined.sort(
        key=lambda row: (
            str(row["state_id"]),
            str(row["action_id"]),
            str(row["call_key"]),
        )
    )
    if len(combined) != 9:
        raise RuntimeError("multisource decomposition must plan nine calls")
    maximum_attempts = sum(
        int(row["maximum_physical_api_attempts"]) for row in group_plans
    )
    expected_calls = sum(
        int(row["expected_api_calls"]) for row in group_plans
    )
    expected_cost = sum(
        float(row["logical_estimated_cost_usd"]) for row in group_plans
    )
    maximum_cost = sum(
        float(row["maximum_estimated_cost_usd"]) for row in group_plans
    )
    maximum_input = max(int(row["estimated_input_tokens"]) for row in combined)
    errors: list[str] = []
    if maximum_attempts > max_api_calls:
        errors.append("maximum physical attempts exceed max_api_calls")
    if maximum_cost > max_estimated_usd:
        errors.append("maximum estimated cost exceeds max_estimated_usd")
    if maximum_input > max_input_tokens_per_call:
        errors.append("per-call input estimate exceeds token ceiling")
    payload = {
        "protocol": COST_PROTOCOL,
        "execution_protocol": EXECUTION_PROTOCOL,
        "stage": MULTISOURCE_DECOMPOSITION_STAGE,
        "config_sha256": sha256_file(config_path),
        "experiment_config_sha256": sha256_file(experiment_config_path),
        "contract_file_sha256": sha256_file(contract_path),
        "contract_sha256": contract["contract_sha256"],
        "runner_sha256": sha256_file(__file__),
        "contract_code_sha256": sha256_file(CONTRACT_CODE_PATH),
        "contract_preparation_runner_sha256": sha256_file(
            CONTRACT_PREPARATION_RUNNER_PATH
        ),
        "analysis_code_sha256": sha256_file(ANALYSIS_CODE_PATH),
        "analysis_runner_sha256": sha256_file(ANALYSIS_RUNNER_PATH),
        "shared_sweep_sha256": sha256_file(ROOT / "src/metacom_pm/sweep.py"),
        "runtime_sha256": sha256_file(runtime_path),
        "backend_sha256": sha256_file(backend_path),
        "strategy_bank_sha256": sha256_file(strategy_bank_path),
        "oracle_outcomes_sha256": sha256_file(oracle_outcomes_path),
        "supporter_generation_treatment_sha256": supporter.digest(),
        "endpoint": {
            "base_url": endpoint.base_url,
            "model": endpoint.model,
            "family": endpoint.family,
        },
        "temperature": supporter.temperature,
        "max_output_tokens": supporter.max_output_tokens,
        "seed": int(config["development_sweep"]["seed"]),
        "transport_max_attempts_per_call": TRANSPORT_MAX_ATTEMPTS,
        "transport_backoff_seconds": list(TRANSPORT_BACKOFF_SECONDS),
        "reused_zero_api_outcomes": len(reused),
        "planned_new_logical_calls": len(combined),
        "expected_api_calls": expected_calls,
        "maximum_physical_api_attempts": maximum_attempts,
        "expected_estimated_cost_usd": expected_cost,
        "maximum_estimated_cost_usd": maximum_cost,
        "maximum_input_tokens_per_call": maximum_input,
        "pricing": {
            "input_usd_per_mtok": float(pricing["input"]),
            "output_usd_per_mtok": float(pricing["output"]),
        },
        "group_plans": group_plans,
        "call_plan_sha256": sha256_text(canonical_json(combined)),
        "budget_gate": {
            "status": "PASS" if not errors else "FAIL",
            "errors": errors,
            "max_api_calls": max_api_calls,
            "max_estimated_usd": max_estimated_usd,
            "max_input_tokens_per_call": max_input_tokens_per_call,
        },
        "api_judges_used": False,
        "training_labels_created": False,
    }
    payload["cost_estimate_sha256"] = sha256_text(canonical_json(payload))
    return payload, combined, reused


def _run(
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
    contract = read_json(contract_path)
    expected: dict[tuple[str, str], str] = {}
    for state in contract["states"]:
        for arm in state["single_source_arms"]:
            expected[
                (str(state["card_id"]), str(arm["action_id"]))
            ] = str(arm["target_memory_id"])
    generated: list[dict[str, Any]] = []
    for group in cost["group_plans"]:
        source = str(group["single_source"])
        action_id = str(group["target_action_id"])
        group_dir = out_dir / "groups" / source
        summary = run_action_sweep(
            runtime_path,
            backend_path,
            strategy_bank_path,
            group_dir / "action_outcomes.jsonl",
            group_dir / "raw_api_calls.jsonl",
            group_dir / "summary.json",
            endpoint=endpoint,
            card_filter=set(str(value) for value in group["card_ids"]),
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
                "contract_sha256": cost["contract_sha256"],
                "cost_estimate_sha256": cost["cost_estimate_sha256"],
                "single_source": source,
                "target_action_id": action_id,
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
            raise RuntimeError(
                f"multisource decomposition group incomplete: {source}"
            )
        for row in iter_jsonl(group_dir / "action_outcomes.jsonl"):
            outcome = ActionOutcome.model_validate(row).model_dump(mode="json")
            target_id = expected[(str(outcome["card_id"]), action_id)]
            if set(outcome["selected_memory_ids"]) != {target_id}:
                raise RuntimeError(
                    f"single-source execution selected wrong item: "
                    f"{outcome['state_id']}"
                )
            if outcome["strategy_view"]:
                raise RuntimeError("single-source execution exposed strategy")
            outcome["provenance"] = {
                **dict(outcome["provenance"]),
                "multisource_decomposition_source": source,
                "multisource_decomposition_contract_sha256": cost[
                    "contract_sha256"
                ],
            }
            generated.append(outcome)
    if len(generated) != 9:
        raise RuntimeError("multisource execution must yield nine outcomes")
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
    parser.add_argument("--oracle-outcomes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-api-calls", type=int, default=100)
    parser.add_argument("--max-estimated-usd", type=float, default=0.1)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=4000)
    parser.add_argument("--accept-cost-estimate-sha256")
    args = parser.parse_args()

    config = load_config(args.config)
    require_output_directory_not_previously_consumed(
        args.out_dir,
        config=config,
        config_path=args.config,
    )
    cost, call_plan, reused = _plan(
        config_path=args.config,
        experiment_config_path=args.experiment_config,
        contract_path=args.contract,
        runtime_path=args.runtime,
        backend_path=args.backend,
        strategy_bank_path=args.strategy_bank,
        oracle_outcomes_path=args.oracle_outcomes,
        max_api_calls=args.max_api_calls,
        max_estimated_usd=args.max_estimated_usd,
        max_input_tokens_per_call=args.max_input_tokens_per_call,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "cost_estimate.json", cost)
    write_jsonl(args.out_dir / "call_plan.jsonl", call_plan)
    write_jsonl(args.out_dir / "reused_oracle_outcomes.jsonl", reused)
    if cost["budget_gate"]["status"] != "PASS":
        raise RuntimeError(
            "multisource decomposition budget gate failed: "
            + "; ".join(cost["budget_gate"]["errors"])
        )
    require_paid_run_release(
        config,
        config_path=args.config,
        stage=MULTISOURCE_DECOMPOSITION_STAGE,
        run=bool(args.run),
        run_identity=args.accept_cost_estimate_sha256,
    )
    if args.dry_run:
        summary = {
            "protocol": EXECUTION_PROTOCOL,
            "status": "DRY_RUN_COMPLETE",
            "cost_estimate_sha256": cost["cost_estimate_sha256"],
            "call_plan_sha256": cost["call_plan_sha256"],
            "reused_zero_api_outcomes": len(reused),
            "planned_new_logical_calls": len(call_plan),
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
        raise RuntimeError("accepted multisource identity does not match")

    generated = _run(
        out_dir=args.out_dir,
        cost=cost,
        config_path=args.config,
        experiment_config_path=args.experiment_config,
        contract_path=args.contract,
        runtime_path=args.runtime,
        backend_path=args.backend,
        strategy_bank_path=args.strategy_bank,
    )
    write_jsonl(
        args.out_dir / "single_source_outcomes.jsonl",
        sorted(
            generated,
            key=lambda row: (str(row["state_id"]), str(row["action_id"])),
        ),
    )
    total_input = sum(int(row["cost"]["total_input_tokens"]) for row in generated)
    total_output = sum(int(row["cost"]["output_tokens"]) for row in generated)
    pricing = cost["pricing"]
    summary = {
        "protocol": EXECUTION_PROTOCOL,
        "status": "COMPLETE_REPORT_ONLY",
        "cost_estimate_sha256": cost["cost_estimate_sha256"],
        "call_plan_sha256": cost["call_plan_sha256"],
        "reused_zero_api_outcomes": len(reused),
        "new_generated_outcomes": len(generated),
        "actual_input_tokens": total_input,
        "actual_output_tokens": total_output,
        "actual_proxy_cost_usd": (
            total_input
            / 1_000_000
            * float(pricing["input_usd_per_mtok"])
            + total_output
            / 1_000_000
            * float(pricing["output_usd_per_mtok"])
        ),
        "api_judges_used": False,
        "training_labels_created": False,
    }
    write_json(args.out_dir / "summary.json", summary)
    create_artifact_attestation(
        args.out_dir / "artifact_attestation.json",
        stage=MULTISOURCE_DECOMPOSITION_STAGE,
        inputs={
            "config": args.config,
            "experiment_config": args.experiment_config,
            "contract": args.contract,
            "runtime": args.runtime,
            "backend": args.backend,
            "strategy_bank": args.strategy_bank,
            "oracle_outcomes": args.oracle_outcomes,
            "cost_estimate": args.out_dir / "cost_estimate.json",
            "call_plan": args.out_dir / "call_plan.jsonl",
        },
        outputs={
            "single_source_outcomes": (
                args.out_dir / "single_source_outcomes.jsonl",
                True,
            ),
            "summary": (args.out_dir / "summary.json", False),
        },
        parameters={
            "contract_sha256": cost["contract_sha256"],
            "cost_estimate_sha256": cost["cost_estimate_sha256"],
            "call_plan_sha256": cost["call_plan_sha256"],
            "transport_max_attempts_per_call": (
                TRANSPORT_MAX_ATTEMPTS
            ),
            "api_judges_used": False,
            "training_labels_created": False,
        },
        expected={
            "reused_zero_api_outcomes": 6,
            "new_generated_outcomes": 9,
        },
    )
    print(summary)


if __name__ == "__main__":
    main()
