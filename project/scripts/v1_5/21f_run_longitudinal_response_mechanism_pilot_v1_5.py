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
from metacom_pm.contracts import ActionOutcome, MemorySource
from metacom_pm.evidence_filter import (
    EVIDENCE_FILTER_PROTOCOL,
    EvidenceFilterConfig,
    EvidenceRule,
)
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
from metacom_pm.pm_v1_5_semantic import semantic_encoder_spec_from_config
from metacom_pm.pm_v2_data import load_states as load_pmv2_states
from metacom_pm.sweep import load_states as load_runtime_states
from metacom_pm.sweep import plan_action_sweep, run_action_sweep
from metacom_pm.v1_5_response_mechanism_pilot import (
    RESPONSE_MECHANISM_PILOT_PROTOCOL,
)
from metacom_pm.v1_5_response_mechanism_uptake import (
    UPTAKE_MEASUREMENT_PROTOCOL,
)


PILOT_EXECUTION_PROTOCOL = (
    "pm-v1.5-longitudinal-response-mechanism-pilot-execution-v1"
)
PILOT_COST_PROTOCOL = (
    "pm-v1.5-longitudinal-response-mechanism-pilot-cost-v1"
)
PILOT_STAGE = "longitudinal_response_mechanism_pilot"
PILOT_ARM = "top1_evidence_surface_same_prompt"
CONTROL_ARM = "frozen_current_control"
TRANSPORT_MAX_ATTEMPTS = 4
TRANSPORT_BACKOFF_SECONDS = (10.0, 30.0, 60.0)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_PILOT_CONTRACT_PATH = (
    PROJECT_ROOT
    / "data/pm_v1_5_contracts/longitudinal_response_mechanism_pilot_v1.json"
)
CANONICAL_UPTAKE_MEASUREMENT_CONTRACT_PATH = (
    PROJECT_ROOT
    / "data/pm_v1_5_contracts/"
    "longitudinal_response_mechanism_uptake_measurement_v1.json"
)
UPTAKE_MEASUREMENT_CODE_PATH = (
    PROJECT_ROOT / "src/metacom_pm/v1_5_response_mechanism_uptake.py"
)
UPTAKE_ANALYZER_PATH = (
    PROJECT_ROOT
    / "scripts/v1_5/"
    "21h_analyze_longitudinal_response_mechanism_uptake_v1_5.py"
)


def _top1_evidence_surface() -> EvidenceFilterConfig:
    rule = EvidenceRule(
        minimum_current_score=0.0,
        minimum_context_score=0.0,
        maximum_items=1,
        long_item_token_threshold=None,
        long_item_minimum_current_score=None,
    )
    return EvidenceFilterConfig(
        protocol=EVIDENCE_FILTER_PROTOCOL,
        enabled=True,
        candidate_scope="post_retrieval_pre_generation",
        memory_rules={source: rule for source in MemorySource},
        strategy_rule=rule,
        maximum_total_memory_items=3,
        drop_exact_duplicate_text=False,
        allow_empty_memory=True,
        allow_empty_strategy=True,
    )


def _validate_contract(
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
    states_path: Path,
    runtime_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if contract_path.resolve() != CANONICAL_PILOT_CONTRACT_PATH.resolve():
        raise RuntimeError(
            "formal response-mechanism pilot requires the tracked canonical "
            f"contract: {CANONICAL_PILOT_CONTRACT_PATH}"
        )
    if contract.get("protocol") != RESPONSE_MECHANISM_PILOT_PROTOCOL:
        raise RuntimeError("unexpected response-mechanism pilot protocol")
    if contract.get("status") != "READY_ZERO_API_DESIGN":
        raise RuntimeError("response-mechanism pilot design is not frozen-ready")
    if contract.get("scope") != "longitudinal_train_only":
        raise RuntimeError("response-mechanism pilot must remain train-only")
    if contract.get("training_labels_created") is not False:
        raise RuntimeError("response-mechanism pilot cannot create training labels")
    if int(contract.get("planned_new_logical_calls", -1)) != 14:
        raise RuntimeError("response-mechanism pilot must plan exactly 14 new calls")
    if set(contract.get("arms") or {}) != {CONTROL_ARM, PILOT_ARM}:
        raise RuntimeError("response-mechanism pilot arm set drifted")
    if (contract.get("evaluation") or {}).get("api_judges") != "forbidden":
        raise RuntimeError("response-mechanism pilot must forbid API judges")

    source = dict(contract.get("source_lineage") or {})
    if source.get("states_sha256") != sha256_file(states_path):
        raise RuntimeError("pilot contract states hash is stale")
    state_rows = load_pmv2_states(states_path)
    state_by_id = {state.state_id: state for state in state_rows}
    runtime_by_card = load_runtime_states(runtime_path)
    selected = [dict(row) for row in contract.get("selected_states") or []]
    if len(selected) != 14:
        raise RuntimeError("pilot contract must select exactly 14 states")
    if len({str(row["state_id"]) for row in selected}) != 14:
        raise RuntimeError("pilot contract repeats a state")
    if len({str(row["user_id"]) for row in selected}) != 14:
        raise RuntimeError("pilot contract repeats a user")

    normalized: list[dict[str, Any]] = []
    for row in selected:
        state_id = str(row["state_id"])
        state = state_by_id.get(state_id)
        if state is None:
            raise RuntimeError(f"pilot state is absent: {state_id}")
        if state.split.value != "train":
            raise RuntimeError(f"pilot state is not train-only: {state_id}")
        if state.user_id != str(row["user_id"]):
            raise RuntimeError(f"pilot user lineage drifted: {state_id}")
        if state.card_id not in runtime_by_card:
            raise RuntimeError(f"pilot runtime card is absent: {state.card_id}")
        target_action = str(row["target_action_id"])
        runtime = runtime_by_card[state.card_id]
        if target_action not in runtime.allowed_actions:
            raise RuntimeError(
                f"pilot target action is not legal: {state_id}/{target_action}"
            )
        normalized.append(
            {
                **row,
                "card_id": state.card_id,
                "semantic_family": state.semantic_family,
            }
        )
    state_ids = sorted(str(row["state_id"]) for row in normalized)
    if contract.get("selected_state_ids_sha256") != sha256_text(
        canonical_json(state_ids)
    ):
        raise RuntimeError("pilot selected-state digest drifted")
    return normalized, {
        "pilot_contract_path": str(contract_path),
        "pilot_contract_file_sha256": sha256_file(contract_path),
        "pilot_contract_sha256": str(contract["contract_sha256"]),
    }


def _select_control_outcomes(
    *,
    outcomes_path: Path,
    attestation_path: Path,
    selected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    verification = require_artifact_attestation(
        attestation_path,
        required_stage="action_sweep",
        required_output_paths={"action_outcomes": outcomes_path},
    )
    wanted = {
        (str(row["state_id"]), str(row["target_action_id"])) for row in selected
    }
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for row in iter_jsonl(outcomes_path):
        key = (str(row["state_id"]), str(row["action_id"]))
        if key not in wanted:
            continue
        if key in found:
            raise RuntimeError(f"duplicate frozen control outcome: {key}")
        outcome = ActionOutcome.model_validate(row).model_dump(mode="json")
        outcome["provenance"] = {
            **dict(outcome["provenance"]),
            "response_mechanism_pilot_arm": CONTROL_ARM,
            "zero_api_control_reuse": True,
        }
        found[key] = outcome
    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(f"frozen control outcomes are incomplete: {missing}")
    return [found[key] for key in sorted(found)], {
        "control_outcomes_sha256": sha256_file(outcomes_path),
        "control_attestation_sha256": verification["attestation_sha256"],
    }


def _plan(
    *,
    config_path: Path,
    experiment_config_path: Path,
    contract_path: Path,
    uptake_measurement_contract_path: Path,
    states_path: Path,
    runtime_path: Path,
    backend_path: Path,
    strategy_bank_path: Path,
    control_outcomes_path: Path,
    control_attestation_path: Path,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config = load_config(config_path)
    experiment_config = load_config(experiment_config_path)
    contract = read_json(contract_path)
    if (
        uptake_measurement_contract_path.resolve()
        != CANONICAL_UPTAKE_MEASUREMENT_CONTRACT_PATH.resolve()
    ):
        raise RuntimeError(
            "formal response-mechanism pilot requires the tracked canonical "
            "uptake measurement contract"
        )
    uptake_contract = read_json(uptake_measurement_contract_path)
    uptake_without_sha = {
        key: value
        for key, value in uptake_contract.items()
        if key != "contract_sha256"
    }
    if (
        uptake_contract.get("protocol") != UPTAKE_MEASUREMENT_PROTOCOL
        or uptake_contract.get("status") != "FROZEN_BEFORE_PILOT_OUTCOMES"
        or uptake_contract.get("contract_sha256")
        != sha256_text(canonical_json(uptake_without_sha))
        or uptake_contract.get("pilot_contract_sha256")
        != contract.get("contract_sha256")
        or uptake_contract.get("api_judges_used") is not False
        or uptake_contract.get("training_labels_created") is not False
    ):
        raise RuntimeError("uptake measurement contract is stale or invalid")
    if uptake_contract.get(
        "semantic_encoder_spec_sha256"
    ) != semantic_encoder_spec_from_config(config).digest():
        raise RuntimeError("uptake measurement semantic encoder binding drifted")
    selected, contract_binding = _validate_contract(
        contract,
        contract_path=contract_path,
        states_path=states_path,
        runtime_path=runtime_path,
    )
    control_rows, control_binding = _select_control_outcomes(
        outcomes_path=control_outcomes_path,
        attestation_path=control_attestation_path,
        selected=selected,
    )
    supporter = SupporterGenerationContract.from_config(config)
    endpoint = endpoint_from_config(
        experiment_config, supporter.generator_endpoint
    )
    pricing = dict(config["development_sweep"]["pricing_usd_per_mtok"])
    retrieval = dict(config["retrieval"])
    planning = dict(config["api_cost_planning"])
    top1 = _top1_evidence_surface()
    contract_bindings = {
        **contract_binding,
        **control_binding,
        "protocol": PILOT_EXECUTION_PROTOCOL,
        "pm_v1_5_config_sha256": sha256_file(config_path),
        "experiment_config_sha256": sha256_file(experiment_config_path),
        "runtime_sha256": sha256_file(runtime_path),
        "backend_sha256": sha256_file(backend_path),
        "strategy_bank_sha256": sha256_file(strategy_bank_path),
        "supporter_generation_treatment_sha256": supporter.digest(),
        "evidence_surface_filter_sha256": top1.digest(),
        "uptake_measurement_contract_file_sha256": sha256_file(
            uptake_measurement_contract_path
        ),
        "uptake_measurement_contract_sha256": uptake_contract[
            "contract_sha256"
        ],
        "uptake_measurement_code_sha256": sha256_file(
            UPTAKE_MEASUREMENT_CODE_PATH
        ),
        "uptake_analyzer_sha256": sha256_file(UPTAKE_ANALYZER_PATH),
        "scientific_interpretation": "report_only_no_training_authority",
    }

    cards_by_action: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        cards_by_action[str(row["target_action_id"])].add(str(row["card_id"]))

    group_costs: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    for action_id in sorted(cards_by_action):
        group_cost, rows = plan_action_sweep(
            runtime_path,
            backend_path,
            strategy_bank_path,
            endpoint=endpoint,
            card_filter=cards_by_action[action_id],
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
            memory_min_score=float(retrieval["memory_min_score"]),
            strategy_min_score=float(retrieval["strategy_min_score"]),
            evidence_filter_config=top1,
            supporter_generation_contract=supporter,
            input_usd_per_mtok=float(pricing["input"]),
            output_usd_per_mtok=float(pricing["output"]),
            contract_bindings={
                **contract_bindings,
                "pilot_arm": PILOT_ARM,
                "target_action_id": action_id,
            },
        )
        if len(rows) != len(cards_by_action[action_id]):
            raise RuntimeError(
                f"pilot plan cardinality drifted for {action_id}: "
                f"{len(rows)} != {len(cards_by_action[action_id])}"
            )
        group_costs.append(
            {
                "target_action_id": action_id,
                "card_ids": sorted(cards_by_action[action_id]),
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
                "maximum_estimated_cost_usd": group_cost[
                    "estimated_cost_usd"
                ],
            }
        )
        combined_rows.extend(
            {
                **row,
                "response_mechanism_pilot_arm": PILOT_ARM,
                "target_action_group": action_id,
            }
            for row in rows
        )
    combined_rows.sort(
        key=lambda row: (
            str(row["state_id"]),
            str(row["action_id"]),
            str(row["call_key"]),
        )
    )
    if len(combined_rows) != 14:
        raise RuntimeError(f"pilot call plan changed: {len(combined_rows)} != 14")

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
    budget_errors: list[str] = []
    if maximum_attempts > int(max_api_calls):
        budget_errors.append("maximum physical attempts exceed max_api_calls")
    if maximum_cost > float(max_estimated_usd):
        budget_errors.append("maximum estimated cost exceeds max_estimated_usd")
    if maximum_input > int(max_input_tokens_per_call):
        budget_errors.append("per-call input estimate exceeds token ceiling")
    payload = {
        "protocol": PILOT_COST_PROTOCOL,
        "stage": PILOT_STAGE,
        "execution_protocol": PILOT_EXECUTION_PROTOCOL,
        "config_sha256": sha256_file(config_path),
        "experiment_config_sha256": sha256_file(experiment_config_path),
        "contract_file_sha256": sha256_file(contract_path),
        "contract_sha256": contract["contract_sha256"],
        "uptake_measurement_contract_file_sha256": sha256_file(
            uptake_measurement_contract_path
        ),
        "uptake_measurement_contract_sha256": uptake_contract[
            "contract_sha256"
        ],
        "uptake_measurement_code_sha256": sha256_file(
            UPTAKE_MEASUREMENT_CODE_PATH
        ),
        "uptake_analyzer_sha256": sha256_file(UPTAKE_ANALYZER_PATH),
        "runner_sha256": sha256_file(__file__),
        "shared_sweep_sha256": sha256_file(
            Path(__file__).resolve().parents[2] / "src/metacom_pm/sweep.py"
        ),
        "runtime_sha256": sha256_file(runtime_path),
        "backend_sha256": sha256_file(backend_path),
        "strategy_bank_sha256": sha256_file(strategy_bank_path),
        "control_outcomes_sha256": control_binding["control_outcomes_sha256"],
        "control_attestation_sha256": control_binding[
            "control_attestation_sha256"
        ],
        "supporter_generation_treatment_sha256": supporter.digest(),
        "evidence_surface_filter": top1.payload(),
        "evidence_surface_filter_sha256": top1.digest(),
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
        "planned_control_rows_reused_zero_api": len(control_rows),
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
            "status": "PASS" if not budget_errors else "FAIL",
            "errors": budget_errors,
            "max_api_calls": int(max_api_calls),
            "max_estimated_usd": float(max_estimated_usd),
            "max_input_tokens_per_call": int(max_input_tokens_per_call),
        },
        "api_judges_used": False,
        "training_labels_created": False,
        "scientific_interpretation": "report_only_no_training_authority",
    }
    payload["cost_estimate_sha256"] = sha256_text(canonical_json(payload))
    return payload, combined_rows, control_rows


def _run_groups(
    *,
    out_dir: Path,
    cost: Mapping[str, Any],
    config_path: Path,
    experiment_config_path: Path,
    contract_path: Path,
    uptake_measurement_contract_path: Path,
    runtime_path: Path,
    backend_path: Path,
    strategy_bank_path: Path,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    experiment_config = load_config(experiment_config_path)
    supporter = SupporterGenerationContract.from_config(config)
    endpoint = endpoint_from_config(
        experiment_config, supporter.generator_endpoint
    )
    retrieval = dict(config["retrieval"])
    planning = dict(config["api_cost_planning"])
    top1 = _top1_evidence_surface()
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
            memory_min_score=float(retrieval["memory_min_score"]),
            strategy_min_score=float(retrieval["strategy_min_score"]),
            evidence_filter_config=top1,
            supporter_generation_contract=supporter,
            contract_bindings={
                "protocol": PILOT_EXECUTION_PROTOCOL,
                "pilot_contract_file_sha256": sha256_file(contract_path),
                "pilot_contract_sha256": cost["contract_sha256"],
                "pilot_cost_estimate_sha256": cost["cost_estimate_sha256"],
                "uptake_measurement_contract_file_sha256": sha256_file(
                    uptake_measurement_contract_path
                ),
                "uptake_measurement_contract_sha256": cost[
                    "uptake_measurement_contract_sha256"
                ],
                "uptake_measurement_code_sha256": cost[
                    "uptake_measurement_code_sha256"
                ],
                "uptake_analyzer_sha256": cost["uptake_analyzer_sha256"],
                "pilot_arm": PILOT_ARM,
                "target_action_id": action_id,
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
            raise RuntimeError(f"pilot group did not complete: {action_id}")
        for row in iter_jsonl(group_dir / "action_outcomes.jsonl"):
            outcome = ActionOutcome.model_validate(row).model_dump(mode="json")
            outcome["provenance"] = {
                **dict(outcome["provenance"]),
                "response_mechanism_pilot_arm": PILOT_ARM,
                "zero_api_control_reuse": False,
            }
            generated.append(outcome)
    if len(generated) != 14:
        raise RuntimeError(f"pilot generated outcome count changed: {len(generated)}")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--uptake-measurement-contract", type=Path, required=True
    )
    parser.add_argument("--states", type=Path, required=True)
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
        uptake_measurement_contract_path=args.uptake_measurement_contract,
        states_path=args.states,
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
            "response-mechanism pilot budget gate failed: "
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
            "protocol": PILOT_EXECUTION_PROTOCOL,
            "status": "DRY_RUN_COMPLETE",
            "cost_estimate_sha256": cost["cost_estimate_sha256"],
            "call_plan_sha256": cost["call_plan_sha256"],
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
            "scientific_interpretation": "report_only_no_training_authority",
        }
        write_json(args.out_dir / "summary.json", summary)
        print(summary)
        return

    accepted = str(args.accept_cost_estimate_sha256 or "")
    if accepted != cost["cost_estimate_sha256"]:
        raise RuntimeError("accepted pilot cost identity does not match dry-run")
    generated = _run_groups(
        out_dir=args.out_dir,
        cost=cost,
        config_path=args.config,
        experiment_config_path=args.experiment_config,
        contract_path=args.contract,
        uptake_measurement_contract_path=args.uptake_measurement_contract,
        runtime_path=args.runtime,
        backend_path=args.backend,
        strategy_bank_path=args.strategy_bank,
    )
    pilot_rows = sorted(
        [*controls, *generated],
        key=lambda row: (
            str(row["state_id"]),
            str(row["provenance"]["response_mechanism_pilot_arm"]),
        ),
    )
    write_jsonl(args.out_dir / "pilot_outcomes.jsonl", pilot_rows)
    total_input = sum(int(row["cost"]["total_input_tokens"]) for row in generated)
    total_output = sum(int(row["cost"]["output_tokens"]) for row in generated)
    pricing = cost["pricing"]
    actual_proxy_cost = (
        total_input / 1_000_000 * float(pricing["input_usd_per_mtok"])
        + total_output / 1_000_000 * float(pricing["output_usd_per_mtok"])
    )
    summary = {
        "protocol": PILOT_EXECUTION_PROTOCOL,
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
        "scientific_interpretation": "report_only_no_training_authority",
        "next_step": (
            "run the preregistered zero-API resource-uptake diagnostic; "
            "do not authorize labels or training"
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
            "uptake_measurement_contract": args.uptake_measurement_contract,
            "uptake_measurement_code": UPTAKE_MEASUREMENT_CODE_PATH,
            "uptake_analyzer": UPTAKE_ANALYZER_PATH,
            "states": args.states,
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
            "pilot_arm": PILOT_ARM,
            "control_arm": CONTROL_ARM,
            "uptake_measurement_contract_sha256": cost[
                "uptake_measurement_contract_sha256"
            ],
            "uptake_measurement_code_sha256": cost[
                "uptake_measurement_code_sha256"
            ],
            "uptake_analyzer_sha256": cost["uptake_analyzer_sha256"],
            "transport_max_attempts_per_call": TRANSPORT_MAX_ATTEMPTS,
            "evidence_surface_filter_sha256": cost[
                "evidence_surface_filter_sha256"
            ],
            "supporter_generation_treatment_sha256": cost[
                "supporter_generation_treatment_sha256"
            ],
            "api_judges_used": False,
            "training_labels_created": False,
            "scientific_interpretation": "report_only_no_training_authority",
        },
        expected={"control_rows": 14, "generated_rows": 14, "pilot_rows": 28},
    )
    print(summary)


if __name__ == "__main__":
    main()
