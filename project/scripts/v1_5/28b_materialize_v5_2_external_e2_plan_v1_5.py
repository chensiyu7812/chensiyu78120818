#!/usr/bin/env python3
"""Materialize the zero-API E2 logical plans for external response and QA.

Retrieval-dependent messages are intentionally left unresolved.  E3 is the
only stage allowed to select Top-4 sessions, project question-specific typed
memory, fit the context window, and seal exact prompts.  Gold answers and
evidence are written to a separate evaluator-only artifact and never enter a
generation call row.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)
from metacom_pm.text import conservative_token_bound
from metacom_pm.v1_5_external_memory_adapter import (
    EXTERNAL_QA_PROMPT_PROTOCOL,
    EXTERNAL_RAW_MEMORY_ADAPTER_PROTOCOL,
    QA_SYSTEM_PROMPT,
    qa_messages,
    render_evoemo_session_document,
)
from metacom_pm.v1_5b_policy_runtime import action_component_bits


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5.2-external-e2-logical-plan-v1"
E1_SCRIPT = ROOT / "scripts/v1_5/28a_audit_v5_2_external_e1_v1_5.py"
E1_DIR = ROOT / "outputs/pm_v1_5_v5_2_external_e1_static_audit_v1"
E1_REPORT = E1_DIR / "audit_report.json"
SURFACES = E1_DIR / "external_v5_2_response_free_surfaces_private.jsonl"
CONTRACT = ROOT / "data/pm_v1_5_contracts/v5_2_external_e2_plan_v1.json"
V52_CONTRACT = ROOT / "data/pm_v1_5_contracts/v5_2_content_disjoint_confirmation_v1.json"
EVOEMO = ROOT / "data/external/evo_emo.json"
CARDS = ROOT / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl"
CORE_POLICIES = (
    "always_off",
    "fixed_high_eligible",
    "transparent_rule",
    "cost_matched_fixed",
    "learned_pm",
)
RAW_CONDITIONS = (
    "raw_session_top4_plus_frozen_strategy",
    "all_raw_sessions_plus_frozen_strategy",
)
QA_CONDITIONS = (
    "no_memory",
    "full_history",
    "official_session_rag_top4",
    "typed_memory_fixed_high",
    "typed_memory_learned_pm",
)
FORBIDDEN_GENERATION_KEYS = frozenset(
    {"answer", "answers", "evidence", "capability", "summaries", "question_group"}
)


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def e1_module() -> Any:
    spec = importlib.util.spec_from_file_location("v52_external_e1", E1_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen E1 implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_int(seed_hex: str) -> int:
    value = int(seed_hex, 16) % 2_147_483_647
    return value or 1


def has_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key) in FORBIDDEN_GENERATION_KEYS or has_forbidden_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(has_forbidden_key(item) for item in value)
    return False


def user_qa_rows(user: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group in user.get("questions") or []:
        group_id = str(group.get("id") or "")
        for question in group.get("questions") or []:
            output.append({"group_id": group_id, **dict(question)})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_2_external_e2_plan_v1",
    )
    args = parser.parse_args()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal E2 planning requires {FORMAL_PYTHON}; got {sys.executable}")

    e1_report = read_json(E1_REPORT)
    if e1_report.get("status") != "PASS_READY_FOR_E2_PLAN_MATERIALIZATION":
        raise RuntimeError("E2 requires the passed E1 report")
    plan_contract = read_json(CONTRACT)
    v52_contract = read_json(V52_CONTRACT)
    helper = e1_module()
    surface_rows = rows(SURFACES)
    cards = {str(row["card_id"]): row for row in rows(CARDS)}
    evo_users = {str(row["id"]): row for row in json.loads(EVOEMO.read_text(encoding="utf-8"))}

    pm_config = load_config(ROOT / "configs/pm_v1_5.yaml")
    experiment = load_config(ROOT / "configs/experiment.yaml")
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, generation.generator_endpoint)
    safety_factor = float(pm_config["api_cost_planning"]["input_token_safety_factor"])
    cost_actions = dict(e1_report["cost_matched_fixed"]["selected_actions"])

    core_calls_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    core_policy_bindings: list[dict[str, Any]] = []
    action_distribution: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {policy: Counter() for policy in CORE_POLICIES}
    )
    for row in surface_rows:
        partition = str(row["partition"])
        state_id = str(row["state_id"])
        for policy in CORE_POLICIES:
            if policy == "cost_matched_fixed":
                requested = action_component_bits(
                    cost_actions[helper.cost_selection_population(partition)]
                )
            else:
                requested = helper.policy_bits(row, policy)
            material = helper.action_material(
                row=row,
                requested_bits=requested,
                cards=cards,
                safety_factor=safety_factor,
            )
            action = str(material["action_id"])
            action_distribution[partition][policy][action] += 1
            for seed_label in ("seed-a", "seed-b"):
                key = (state_id, action, seed_label)
                seed_hex = stable_hex(PROTOCOL, state_id, action, seed_label, n=16)
                call = core_calls_by_key.setdefault(
                    key,
                    {
                        "protocol": PROTOCOL,
                        "call_id": "e2resp_" + stable_hex(*key, PROTOCOL, n=24),
                        "experiment": "external_response_core",
                        "partition": partition,
                        "state_id": state_id,
                        "user_id_private_analysis_only": str(
                            row["user_id_private_analysis_only"]
                        ),
                        "treatment_id": action,
                        "requested_action_id": action,
                        "seed_label": seed_label,
                        "seed_hex": seed_hex,
                        "seed": seed_int(seed_hex),
                        "messages": material["messages"],
                        "messages_sha256": sha256_text(
                            canonical_json(material["messages"])
                        ),
                        "input_token_upper_bound": int(
                            material["input_token_upper_bound"]
                        ),
                        "output_token_cap": int(generation.max_output_tokens),
                        "selected_candidate_ids": material["selected_candidate_ids"],
                        "composition_plan": material["composition_plan"],
                        "policy_aliases": [],
                        "retrieval_status": "NOT_APPLICABLE_ALREADY_MATERIALIZED_IN_E1",
                        "response_quality_risk_human_or_judge_read": False,
                    },
                )
                call["policy_aliases"].append(policy)
            core_policy_bindings.append(
                {
                    "partition": partition,
                    "state_id": state_id,
                    "policy": policy,
                    "realized_action_id": action,
                    "response_or_outcome_read": False,
                }
            )

    core_calls = sorted(core_calls_by_key.values(), key=lambda item: str(item["call_id"]))
    for call in core_calls:
        call["policy_aliases"] = sorted(set(call["policy_aliases"]))

    raw_requests: list[dict[str, Any]] = []
    evo_surfaces = [row for row in surface_rows if str(row["domain"]) == "EVOEMO"]
    for row in evo_surfaces:
        user_id = str(row["user_id_private_analysis_only"])
        user = evo_users[user_id]
        human_name = str(user["basic_info"]["name"])
        documents = [
            render_evoemo_session_document(session, human_name=human_name)
            for session in user.get("dialog_history") or []
        ]
        if not documents or any(not item["session_id"] or not item["text"] for item in documents):
            raise RuntimeError(f"empty raw session document for {user_id}")
        if len({item["session_id"] for item in documents}) != len(documents):
            raise RuntimeError(f"duplicate raw session id for {user_id}")
        rs = row["components"]["RS"]
        rs_candidate_id = (
            str(rs["typed_candidate"]["resource_id"])
            if bool(rs["structurally_executable"])
            else None
        )
        rs_instruction = ""
        if rs_candidate_id is not None:
            rs_instruction = str(cards[rs_candidate_id]["prompt_guidance"])
        for condition in RAW_CONDITIONS:
            raw_requests.append(
                {
                    "protocol": PROTOCOL,
                    "retrieval_request_id": "e2raw_"
                    + stable_hex(PROTOCOL, str(row["state_id"]), condition, n=24),
                    "experiment": "external_response_secondary_raw_memory",
                    "partition": str(row["partition"]),
                    "state_id": str(row["state_id"]),
                    "user_id_private_analysis_only": user_id,
                    "condition": condition,
                    "query": helper.current_context(row),
                    "same_user_session_documents": documents,
                    "candidate_session_count": len(documents),
                    "selection": (
                        "BGE_M3_TOP4_PENDING_E3"
                        if condition.startswith("raw_session_top4")
                        else "ALL_SAME_USER_SESSIONS_PENDING_E3_CONTEXT_FIT"
                    ),
                    "strategy_candidate_id": rs_candidate_id,
                    "strategy_instruction": rs_instruction,
                    "adapter_protocol": EXTERNAL_RAW_MEMORY_ADAPTER_PROTOCOL,
                    "generation_seed_labels": ["seed-a", "seed-b"],
                    "generator": endpoint.model,
                    "temperature": float(generation.temperature),
                    "output_token_cap": int(generation.max_output_tokens),
                    "response_or_outcome_read": False,
                }
            )

    qa_logical_calls: list[dict[str, Any]] = []
    qa_evaluator_rows: list[dict[str, Any]] = []
    qa_questions = 0
    for user_id in plan_contract["qa"]["main_users"]:
        user = evo_users[str(user_id)]
        human_name = str(user["basic_info"]["name"])
        documents = [
            render_evoemo_session_document(session, human_name=human_name)
            for session in user.get("dialog_history") or []
        ]
        for ordinal, question in enumerate(user_qa_rows(user), start=1):
            qa_questions += 1
            question_id = "qa_" + stable_hex(
                "es-memeval-v1.0.0",
                user_id,
                question["group_id"],
                question.get("idx"),
                question["question"],
                n=24,
            )
            qa_evaluator_rows.append(
                {
                    "protocol": PROTOCOL,
                    "question_id": question_id,
                    "user_id": str(user_id),
                    "group_id": str(question["group_id"]),
                    "released_question_idx": question.get("idx"),
                    "capability": str(question["capability"]),
                    "answer": str(question["answer"]),
                    "evidence": list(question.get("evidence") or []),
                    "evaluator_only": True,
                }
            )
            for condition in QA_CONDITIONS:
                seed_hex = stable_hex(PROTOCOL, question_id, condition, "seed-a", n=16)
                resolved_messages = (
                    qa_messages(question=str(question["question"]), memory_fragments=[])
                    if condition == "no_memory"
                    else None
                )
                qa_logical_calls.append(
                    {
                        "protocol": PROTOCOL,
                        "call_id": "e2qa_"
                        + stable_hex(PROTOCOL, question_id, condition, n=24),
                        "experiment": "esmemeval_qa_main_p13_p18",
                        "question_id": question_id,
                        "user_id_private_analysis_only": str(user_id),
                        "question_ordinal_within_user": ordinal,
                        "condition": condition,
                        "question": str(question["question"]),
                        "same_user_session_documents": documents,
                        "candidate_session_count": len(documents),
                        "messages": resolved_messages,
                        "messages_sha256": (
                            sha256_text(canonical_json(resolved_messages))
                            if resolved_messages is not None
                            else None
                        ),
                        "input_token_upper_bound": (
                            conservative_token_bound(
                                canonical_json(resolved_messages),
                                safety_factor=safety_factor,
                            )
                            if resolved_messages is not None
                            else None
                        ),
                        "retrieval_status": (
                            "NOT_APPLICABLE_NO_MEMORY"
                            if condition == "no_memory"
                            else "PENDING_E3_RETRIEVAL_OR_CONTEXT_FIT"
                        ),
                        "qa_prompt_protocol": EXTERNAL_QA_PROMPT_PROTOCOL,
                        "qa_system_prompt_sha256": sha256_text(QA_SYSTEM_PROMPT),
                        "generator": endpoint.model,
                        "temperature": float(generation.temperature),
                        "output_token_cap": int(plan_contract["qa"]["max_output_tokens"]),
                        "seed_hex": seed_hex,
                        "seed": seed_int(seed_hex),
                        "answer_evidence_capability_visible_to_generation": False,
                    }
                )

    raw_physical_calls_after_e3 = len(raw_requests) * 2
    checks = {
        "e1_passed": True,
        "e1_all_static_checks_passed": all(
            bool(value) for value in e1_report["checks"].values()
        ),
        "surface_rows_are_e1_reprojections": all(
            str(row.get("protocol"))
            == "pm-v1.5-v5.2-external-e1-static-audit-v1"
            and bool(row.get("source_response_free_surface_sha256"))
            for row in surface_rows
        ),
        "external_surface_count_260": len(surface_rows) == 260,
        "core_response_calls_1576": len(core_calls) == 1576,
        "core_response_policy_bindings_1300": len(core_policy_bindings) == 1300,
        "evoemo_raw_requests_276": len(raw_requests) == 276,
        "raw_physical_calls_after_e3_552": raw_physical_calls_after_e3 == 552,
        "raw_memory_owner_isolation": all(
            str(row["user_id_private_analysis_only"]) in evo_users for row in raw_requests
        ),
        "qa_questions_418": qa_questions == 418,
        "qa_logical_calls_2090": len(qa_logical_calls) == 2090,
        "qa_five_conditions_exact": set(row["condition"] for row in qa_logical_calls)
        == set(QA_CONDITIONS),
        "qa_generation_rows_exclude_gold_fields": not any(
            has_forbidden_key(row) for row in qa_logical_calls
        ),
        "qa_evaluator_rows_418": len(qa_evaluator_rows) == 418,
        "zero_api_calls": True,
        "zero_response_quality_risk_human_or_judge_outcomes_read": True,
    }
    status = "PASS_READY_FOR_E3_RETRIEVAL" if all(checks.values()) else "FAIL_E2"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "response_core_call_plan_private.jsonl", core_calls)
    write_jsonl(
        args.out_dir / "response_core_policy_bindings_private.jsonl",
        core_policy_bindings,
    )
    write_jsonl(args.out_dir / "response_raw_retrieval_requests_private.jsonl", raw_requests)
    write_jsonl(args.out_dir / "qa_logical_call_plan_private.jsonl", qa_logical_calls)
    write_jsonl(args.out_dir / "qa_evaluator_only_gold_private.jsonl", qa_evaluator_rows)
    manifest = {
        "protocol": PROTOCOL,
        "status": status,
        "checks": checks,
        "contract_sha256": sha256_file(CONTRACT),
        "input_sha256": {
            str(E1_REPORT.relative_to(ROOT)): sha256_file(E1_REPORT),
            str(SURFACES.relative_to(ROOT)): sha256_file(SURFACES),
            str(V52_CONTRACT.relative_to(ROOT)): sha256_file(V52_CONTRACT),
            str(EVOEMO.relative_to(ROOT)): sha256_file(EVOEMO),
            str(CARDS.relative_to(ROOT)): sha256_file(CARDS),
        },
        "implementation_sha256": {
            "e2_planner": sha256_file(Path(__file__)),
            "e1_reprojection": sha256_file(E1_SCRIPT),
            "external_memory_adapter": sha256_file(
                ROOT / "src/metacom_pm/v1_5_external_memory_adapter.py"
            ),
        },
        "generator": endpoint.model,
        "temperature": float(generation.temperature),
        "response": {
            "external_states": len(surface_rows),
            "core_policy_bindings": len(core_policy_bindings),
            "core_calls_after_state_action_policy_alias_dedup": len(core_calls),
            "secondary_raw_retrieval_requests": len(raw_requests),
            "secondary_raw_calls_after_e3": raw_physical_calls_after_e3,
            "total_planned_calls_after_e3": len(core_calls) + raw_physical_calls_after_e3,
            "core_input_token_upper_bound": sum(
                int(row["input_token_upper_bound"]) for row in core_calls
            ),
            "raw_input_token_bound_status": "PENDING_E3",
            "action_distribution": {
                partition: {
                    policy: dict(sorted(counter.items()))
                    for policy, counter in policies.items()
                }
                for partition, policies in action_distribution.items()
            },
        },
        "qa": {
            "questions": qa_questions,
            "conditions": list(QA_CONDITIONS),
            "logical_calls": len(qa_logical_calls),
            "no_memory_calls_with_exact_messages": sum(
                row["messages"] is not None for row in qa_logical_calls
            ),
            "retrieval_or_context_fit_pending_e3": sum(
                row["messages"] is None for row in qa_logical_calls
            ),
            "exact_total_token_budget_status": "PENDING_E3",
            "output_token_upper_bound": len(qa_logical_calls)
            * int(plan_contract["qa"]["max_output_tokens"]),
        },
        "stage_boundary": plan_contract["stage_boundary"],
        "old_23m_paid_guard": {
            "in_e2_e8_dependency_path": False,
            "modified": False,
            "reason": "historical runner; single edit would not repair the stale global static test, while bulk edits would alter sealed runners",
            "new_e4_runner_must_use_active_release_guard": True,
        },
        "api_calls": 0,
        "human_or_llm_outcomes_read": 0,
    }
    write_json(args.out_dir / "plan_manifest.json", manifest)
    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "status": status,
                "core_response_calls": len(core_calls),
                "raw_response_calls_after_e3": raw_physical_calls_after_e3,
                "qa_calls": len(qa_logical_calls),
                "api_calls": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
