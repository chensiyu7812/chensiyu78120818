#!/usr/bin/env python3
"""Run the zero-API E1 audit for the frozen V5.2 external evaluation.

This script deliberately reads no response-quality, grounding-risk, judge, or
human-review outcome.  It verifies the sealed implementation, profiles the
official ES-MemEval artifact, reprojects the already-frozen response-free
external candidate surfaces through the V5.2 heads/executor, selects a
response-free cost-matched comparator, detects policy aliases, and freezes
call-count/token upper bounds for the response experiment.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import asdict
from html import escape
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import ALL_ACTION_IDS
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
from metacom_pm.text import conservative_token_bound
from metacom_pm.v1_5_typed_resource_adapter import TypedResourceCandidate
from metacom_pm.v1_5_v5_2_atomic_memory import (
    compile_atomic_reusable_outcome,
    compile_atomic_session_observation,
)
from metacom_pm.v1_5_v5_2_locked_composer import (
    base_generation_messages,
    build_locked_composition_plan,
)
from metacom_pm.v1_5b_policy_runtime import (
    COMPONENTS,
    action_component_bits,
    compile_component_bits,
)


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5.2-external-e1-static-audit-v1"
SURFACE_DIR = ROOT / "outputs/pm_v1_5_v5_1_t5_surfaces_v1"
SURFACES = SURFACE_DIR / "t5_state_surfaces_private.jsonl"
SURFACE_REPORT = SURFACE_DIR / "surface_gate_report.json"
CONTRACT = ROOT / "data/pm_v1_5_contracts/v5_2_content_disjoint_confirmation_v1.json"
SEAL = ROOT / "outputs/pm_v1_5_v5_2_confirmation_plan_v1/execution_seal.json"
EVOEMO = ROOT / "data/external/evo_emo.json"
CARDS = ROOT / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl"
EXTERNAL_PARTITIONS = (
    "esconv_corrected_test_panel",
    "evoemo_qualification_p7_p12",
    "evoemo_lockbox_p13_p18",
)
POLICIES = (
    "always_off",
    "fixed_high_eligible",
    "transparent_rule",
    "cost_matched_fixed",
    "learned_pm",
)
QA_SENSITIVE_KEYS = frozenset(
    {"questions", "question", "answer", "answers", "evidence", "capability", "summaries"}
)


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def learned_probability(
    *, component: str, features: Mapping[str, float], contract: Mapping[str, Any]
) -> float:
    head = contract["frozen_method"]["heads"][component]
    names = list(head["feature_names"])
    if set(features) != set(names):
        raise RuntimeError(f"{component} feature schema differs from the frozen V5.2 head")
    total = float(head["intercept"])
    for name, mean, scale, coefficient in zip(
        names,
        head["scaler_mean"],
        head["scaler_scale"],
        head["coefficients"],
        strict=True,
    ):
        divisor = float(scale) or 1.0
        total += ((float(features[name]) - float(mean)) / divisor) * float(coefficient)
    return sigmoid(total)


def transparent_rule(component: str, f: Mapping[str, float]) -> bool:
    if component == "MP":
        preference = f["candidate_preference_scope_fit"] >= 0.5
        profile = (
            f["candidate_profile_relevance"] >= 0.5
            and f["candidate_profile_entity_scope_fit"] >= 0.5
        )
        return bool(
            f["candidate_incremental_information"] >= 0.5
            and f["candidate_current_scope_conflict"] < 0.5
            and (preference or profile)
        )
    if component == "MS":
        return bool(
            f["candidate_incremental_information"] >= 0.5
            and f["candidate_prior_issue_marked_resolved"] < 0.5
            and f["candidate_current_goal_fit"] >= 0.5
            and (
                f["candidate_prior_outcome_or_distinction"] >= 0.5
                or f["candidate_specific_issue_or_distinction"] >= 0.5
            )
        )
    if component == "ME":
        return bool(
            f["candidate_incremental_information"] >= 0.5
            and f["candidate_current_goal_fit"] >= 0.5
            and f["candidate_contains_action"] >= 0.5
            and (
                f["candidate_contains_result"] >= 0.5
                or f["candidate_contains_mechanism"] >= 0.5
            )
        )
    return all(
        f[name] >= 0.5
        for name in (
            "candidate_mode_fit",
            "candidate_goal_fit",
            "candidate_burden_fit",
            "candidate_boundary_fit",
            "candidate_nonredundancy",
        )
    )


def current_context(row: Mapping[str, Any]) -> str:
    lines = ["Visible current conversation:"]
    for turn in row["visible_dialogue"]:
        lines.append(f"{str(turn['role']).capitalize()}: {str(turn['content']).strip()}")
    lines.append(f"User: {str(row['current_user_text']).strip()}")
    return "\n".join(lines)


def v52_candidate(
    *, component: str, item: Mapping[str, Any], user_id: str
) -> tuple[TypedResourceCandidate | None, str | None]:
    surface = dict(item["surface"])
    if not bool(surface.get("candidate_present")):
        return None, "CANDIDATE_ABSENT"
    raw = item.get("typed_candidate")
    if raw is None:
        return None, "V51_TYPED_CANDIDATE_ABSENT"
    old = TypedResourceCandidate(**dict(raw))
    if component != "RS" and old.owner_id != user_id:
        return None, "OWNER_MISMATCH"
    if component == "MP":
        return old, None
    if component == "MS":
        atomic = compile_atomic_session_observation(old.prior_observation)
        if atomic is None:
            return None, "V52_MS_ATOMIC_COMPILATION_FAILED"
        return old, None
    if component == "ME":
        literal = str(surface.get("candidate_text") or "")
        atomic = compile_atomic_reusable_outcome(literal)
        if atomic is None:
            return None, "V52_ME_ATOMIC_COMPILATION_FAILED"
        return TypedResourceCandidate(
            component="ME",
            subtype="ME_REUSABLE_OUTCOME",
            resource_id=old.resource_id,
            candidate_version=sha256_text(atomic.literal_evidence_span),
            source_kind="event",
            owner_id=old.owner_id,
            active=True,
            strictly_prior=old.strictly_prior,
            age_sessions=old.age_sessions,
            past_action=atomic.past_action_span,
            observed_outcome=atomic.observed_outcome_span,
            mechanism="literal_span:" + atomic.literal_evidence_span,
        ), None
    return old, None


def reproject_surface(
    *, row: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    copied = dict(row)
    copied["protocol"] = PROTOCOL
    copied["source_response_free_surface_sha256"] = sha256_text(canonical_json(row))
    updated: dict[str, Any] = {}
    user_id = str(row["user_id_private_analysis_only"])
    for component in COMPONENTS:
        source = dict(row["components"][component])
        features = {key: float(value) for key, value in source["model_features"].items()}
        candidate, failure = v52_candidate(component=component, item=source, user_id=user_id)
        hard_gate_pass = not bool(source["deterministic_hard_off"])
        executable = bool(candidate is not None and hard_gate_pass)
        probability = learned_probability(
            component=component, features=features, contract=contract
        )
        updated[component] = {
            "surface": source["surface"],
            "model_features": features,
            "deterministic_hard_off": bool(source["deterministic_hard_off"]),
            "hard_off_reasons": list(source["hard_off_reasons"]),
            "typed_candidate": asdict(candidate) if candidate is not None else None,
            "v5_2_typed_failure": failure,
            "structurally_executable": executable,
            "learned_probability": probability,
            "learned_threshold": float(
                contract["frozen_method"]["heads"][component]["threshold"]
            ),
            "learned_requested_on_before_feasibility": probability
            >= float(contract["frozen_method"]["heads"][component]["threshold"]),
            "transparent_rule_requested_on_before_feasibility": transparent_rule(
                component, features
            ),
            "response_or_outcome_read": False,
        }
    copied["components"] = updated
    copied["outcomes_read"] = False
    return copied


def executable_candidates(row: Mapping[str, Any]) -> dict[str, TypedResourceCandidate]:
    result: dict[str, TypedResourceCandidate] = {}
    for component in COMPONENTS:
        item = row["components"][component]
        if bool(item["structurally_executable"]):
            result[component] = TypedResourceCandidate(**dict(item["typed_candidate"]))
    return result


def policy_bits(row: Mapping[str, Any], policy: str) -> dict[str, bool]:
    if policy == "always_off":
        return {component: False for component in COMPONENTS}
    if policy == "fixed_high_eligible":
        return {
            component: bool(row["components"][component]["structurally_executable"])
            for component in COMPONENTS
        }
    key = {
        "transparent_rule": "transparent_rule_requested_on_before_feasibility",
        "learned_pm": "learned_requested_on_before_feasibility",
    }.get(policy)
    if key is None:
        raise ValueError(f"unsupported policy: {policy}")
    return {
        component: bool(row["components"][component][key])
        for component in COMPONENTS
    }


def action_material(
    *,
    row: Mapping[str, Any],
    requested_bits: Mapping[str, bool],
    cards: Mapping[str, Mapping[str, Any]],
    safety_factor: float,
) -> dict[str, Any]:
    executable = executable_candidates(row)
    feasible = {
        component: bool(requested_bits[component] and component in executable)
        for component in COMPONENTS
    }
    action = compile_component_bits(feasible)
    selected = {
        component: executable[component]
        for component in COMPONENTS
        if feasible[component]
    }
    guidance = ""
    if "RS" in selected:
        guidance = str(cards[selected["RS"].resource_id]["prompt_guidance"])
    plan = build_locked_composition_plan(
        requested_action_id=action,
        candidates=selected,
        strategy_prompt_guidance=guidance,
    )
    messages = base_generation_messages(current_context=current_context(row), plan=plan)
    return {
        "action_id": action,
        "feasible_bits": feasible,
        "messages": messages,
        "input_token_upper_bound": conservative_token_bound(
            canonical_json(messages), safety_factor=safety_factor
        ),
        "selected_candidate_ids": {
            component: candidate.resource_id for component, candidate in selected.items()
        },
        "composition_plan": {
            "requested_action_id": plan.requested_action_id,
            "locked_clauses": [asdict(value) for value in plan.locked_clauses],
            "response_preference": plan.response_preference,
            "strategy_instruction": plan.strategy_instruction,
            "maximum_generator_support_moves": plan.maximum_generator_support_moves,
        },
    }


def cost_selection_population(partition: str) -> str:
    return {
        "esconv_corrected_test_panel": "esconv",
        "evoemo_qualification_p7_p12": "evoemo_qualification",
        "evoemo_lockbox_p13_p18": "evoemo_lockbox",
    }[partition]


def select_cost_matched(
    *,
    surfaces: list[dict[str, Any]],
    cards: Mapping[str, Mapping[str, Any]],
    safety_factor: float,
) -> tuple[dict[str, str], dict[str, Any]]:
    populations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in surfaces:
        name = cost_selection_population(str(row["partition"]))
        if name != "evoemo_lockbox":
            populations[name].append(row)
    selections: dict[str, str] = {}
    audit: dict[str, Any] = {}
    for population, selected in populations.items():
        learned_costs = [
            action_material(
                row=row,
                requested_bits=policy_bits(row, "learned_pm"),
                cards=cards,
                safety_factor=safety_factor,
            )["input_token_upper_bound"]
            for row in selected
        ]
        learned_mean = sum(learned_costs) / len(learned_costs)
        actions: Iterable[str] = (
            ("M0+R0", "M0+RS") if population == "esconv" else ALL_ACTION_IDS
        )
        summaries: dict[str, Any] = {}
        for fixed_action in actions:
            costs: list[int] = []
            realized: Counter[str] = Counter()
            for row in selected:
                material = action_material(
                    row=row,
                    requested_bits=action_component_bits(fixed_action),
                    cards=cards,
                    safety_factor=safety_factor,
                )
                costs.append(int(material["input_token_upper_bound"]))
                realized[str(material["action_id"])] += 1
            mean = sum(costs) / len(costs)
            summaries[fixed_action] = {
                "mean_input_token_upper_bound": mean,
                "absolute_difference_from_learned": abs(mean - learned_mean),
                "relative_difference_from_learned": abs(mean - learned_mean)
                / max(1.0, learned_mean),
                "realized_action_distribution": dict(sorted(realized.items())),
            }
        chosen = min(
            summaries,
            key=lambda action: (
                summaries[action]["absolute_difference_from_learned"],
                summaries[action]["mean_input_token_upper_bound"],
                action,
            ),
        )
        selections[population] = chosen
        audit[population] = {
            "selection_states": len(selected),
            "learned_mean_input_token_upper_bound": learned_mean,
            "selected_action": chosen,
            "selected_summary": summaries[chosen],
            "all_fixed_actions": summaries,
            "selection_read_response_quality_risk_or_judge": False,
        }
    selections["evoemo_lockbox"] = selections["evoemo_qualification"]
    audit["evoemo_lockbox"] = {
        "selected_action": selections["evoemo_lockbox"],
        "selection_source": "evoemo_qualification_p7_p12_only",
        "lockbox_surface_used_for_selection": False,
    }
    return selections, audit


def recursive_sensitive_keys(value: Any, *, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in QA_SENSITIVE_KEYS:
                found.append(path)
            found.extend(recursive_sensitive_keys(item, prefix=path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(recursive_sensitive_keys(item, prefix=f"{prefix}[{index}]"))
    return found


def ast_sensitive_reads(path: Path) -> list[dict[str, Any]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        key: str | None = None
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str):
                key = node.slice.value
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"get", "pop", "setdefault"}
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            key = node.args[0].value
        if key in QA_SENSITIVE_KEYS:
            result.append({"path": str(path.relative_to(ROOT)), "line": node.lineno, "key": key})
    return result


def qa_profile() -> dict[str, Any]:
    data = json.loads(EVOEMO.read_text(encoding="utf-8"))
    per_user: dict[str, Any] = {}
    capability: Counter[str] = Counter()
    question_texts: list[str] = []
    for user in data:
        question_groups = list(user.get("questions") or [])
        questions = [
            dict(item)
            for group in question_groups
            for item in (group.get("questions") or [])
        ]
        user_counter: Counter[str] = Counter()
        for item in questions:
            cap = str(item.get("capability") or "UNKNOWN")
            capability[cap] += 1
            user_counter[cap] += 1
            question_texts.append(str(item.get("question") or "").strip())
        per_user[str(user["id"])] = {
            "questions": len(questions),
            "capabilities": dict(sorted(user_counter.items())),
        }
    partitions = {
        "p1_p6": sum(per_user[f"p{i}"]["questions"] for i in range(1, 7)),
        "p7_p12": sum(per_user[f"p{i}"]["questions"] for i in range(7, 13)),
        "p13_p18": sum(per_user[f"p{i}"]["questions"] for i in range(13, 19)),
    }
    return {
        "sha256": sha256_file(EVOEMO),
        "users": len(data),
        "questions": len(question_texts),
        "capabilities": dict(sorted(capability.items())),
        "partition_question_counts": partitions,
        "per_user": per_user,
        "paper_reported_total_questions": 1209,
        "released_artifact_total_questions": len(question_texts),
        "paper_vs_release_difference": len(question_texts) - 1209,
        "released_schema": "user.questions[] question-group, then group.questions[] QA item",
        "question_texts_private_audit_only": question_texts,
    }


def render_html(report: Mapping[str, Any]) -> str:
    status = escape(str(report["status"]))
    findings = "".join(
        f"<tr><td>{escape(str(x['severity']))}</td><td>{escape(str(x['finding']))}</td>"
        f"<td>{escape(str(x['implication']))}</td></tr>"
        for x in report["findings"]
    )
    aliases = "".join(
        f"<tr><td>{escape(partition)}</td><td>{escape(json.dumps(value, ensure_ascii=False))}</td></tr>"
        for partition, value in report["policy_aliases"].items()
    )
    return f"""<!doctype html><meta charset='utf-8'><title>PM V1.5 V5.2 E1 audit</title>
<style>body{{font:15px/1.55 system-ui;margin:32px;max-width:1180px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:8px;vertical-align:top}}code{{background:#f2f2f2;padding:2px 4px}}</style>
<h1>PM V1.5 V5.2 外部实验 E1 静态审计</h1>
<p><strong>结论：</strong>{status}。本报告未读取回复质量、risk、judge 或人评结果，API 调用为 0。</p>
<h2>关键发现</h2><table><tr><th>级别</th><th>发现</th><th>影响</th></tr>{findings}</table>
<h2>基线别名</h2><table><tr><th>外部分区</th><th>结果</th></tr>{aliases}</table>
<h2>预算</h2><pre>{escape(json.dumps(report['budget'], ensure_ascii=False, indent=2))}</pre>
<h2>ES-MemEval 数据身份</h2><pre>{escape(json.dumps(report['qa_artifact_profile_public'], ensure_ascii=False, indent=2))}</pre>
<h2>检查</h2><pre>{escape(json.dumps(report['checks'], ensure_ascii=False, indent=2))}</pre>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_2_external_e1_static_audit_v1",
    )
    args = parser.parse_args()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal E1 audit requires {FORMAL_PYTHON}; got {sys.executable}")

    contract = read_json(CONTRACT)
    seal = read_json(SEAL)
    surface_report = read_json(SURFACE_REPORT)
    surface_rows = [
        row for row in rows(SURFACES) if str(row["partition"]) in EXTERNAL_PARTITIONS
    ]
    cards = {str(row["card_id"]): row for row in rows(CARDS)}
    pm_config = load_config(ROOT / "configs/pm_v1_5.yaml")
    experiment = load_config(ROOT / "configs/experiment.yaml")
    generation = SupporterGenerationContract.from_config(pm_config)
    generator = endpoint_from_config(experiment, generation.generator_endpoint)
    safety_factor = float(pm_config["api_cost_planning"]["input_token_safety_factor"])

    seal_hash_checks = {
        relative: sha256_file(ROOT / relative) == expected
        for relative, expected in seal["implementation_sha256"].items()
    }
    source_input_hash_checks = {
        relative: sha256_file(ROOT / relative) == expected
        for relative, expected in surface_report["input_sha256"].items()
    }
    v52_contract_input_hash_checks = {
        relative: sha256_file(ROOT / relative) == expected
        for relative, expected in contract["input_sha256"].items()
    }
    response_surface_implementation_paths = {
        "surface_builder": "scripts/v1_5/26l_materialize_v5_1_t5_surfaces_v1_5.py",
        "memory_compiler": "src/metacom_pm/v1_5_memory_transport.py",
        "memory_retriever": "src/metacom_pm/v1_5_candidate_discovery.py",
        "strategy_ranker": "src/metacom_pm/v1_5_strategy_rag_repair.py",
        "feature_contract": "src/metacom_pm/v1_5_final_candidate_contract.py",
        "typed_executor": "src/metacom_pm/v1_5_typed_resource_adapter.py",
    }
    response_surface_implementation_hash_checks = {
        name: sha256_file(ROOT / response_surface_implementation_paths[name]) == expected
        for name, expected in surface_report["implementation_sha256"].items()
    }
    response_free_checks = [
        not bool(row.get("outcomes_read"))
        and not bool(row.get("evaluator_only_fields_read_by_pm_or_generator"))
        and all(
            not bool(row["components"][component].get("response_or_outcome_read"))
            for component in COMPONENTS
        )
        for row in surface_rows
    ]

    reprojected = [reproject_surface(row=row, contract=contract) for row in surface_rows]
    cost_actions, cost_audit = select_cost_matched(
        surfaces=reprojected, cards=cards, safety_factor=safety_factor
    )

    policy_routes: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    action_distributions: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {policy: Counter() for policy in POLICIES}
    )
    unique_calls: dict[tuple[str, str], dict[str, Any]] = {}
    owner_checks: list[bool] = []
    coverage: dict[str, Any] = defaultdict(
        lambda: {component: Counter() for component in COMPONENTS}
    )
    for row in reprojected:
        partition = str(row["partition"])
        state_id = str(row["state_id"])
        for component in COMPONENTS:
            item = row["components"][component]
            coverage[partition][component]["states"] += 1
            coverage[partition][component]["candidate_present"] += int(
                bool(item["surface"]["candidate_present"])
            )
            coverage[partition][component]["executable"] += int(
                bool(item["structurally_executable"])
            )
            coverage[partition][component]["learned_on"] += int(
                bool(item["learned_requested_on_before_feasibility"])
                and bool(item["structurally_executable"])
            )
            typed = item.get("typed_candidate")
            if typed and component != "RS":
                owner_checks.append(
                    str(typed["owner_id"]) == str(row["user_id_private_analysis_only"])
                )
        routes: dict[str, dict[str, Any]] = {}
        for policy in POLICIES:
            if policy == "cost_matched_fixed":
                requested = action_component_bits(
                    cost_actions[cost_selection_population(partition)]
                )
            else:
                requested = policy_bits(row, policy)
            material = action_material(
                row=row,
                requested_bits=requested,
                cards=cards,
                safety_factor=safety_factor,
            )
            routes[policy] = material
            action_distributions[partition][policy][str(material["action_id"])] += 1
            unique_calls[(state_id, str(material["action_id"]))] = material
        policy_routes[partition][state_id] = routes

    aliases: dict[str, Any] = {}
    for partition, states in policy_routes.items():
        pairs: list[dict[str, Any]] = []
        for index, left in enumerate(POLICIES):
            for right in POLICIES[index + 1 :]:
                differences = sum(
                    routes[left]["action_id"] != routes[right]["action_id"]
                    for routes in states.values()
                )
                pairs.append(
                    {
                        "policy_a": left,
                        "policy_b": right,
                        "state_action_differences": differences,
                        "exact_alias": differences == 0,
                    }
                )
        aliases[partition] = pairs

    qa = qa_profile()
    question_texts = [text for text in qa.pop("question_texts_private_audit_only") if text]
    response_source_paths = [
        ROOT / "src/metacom_pm/evoemo.py",
        ROOT / "scripts/v1_5/26l_materialize_v5_1_t5_surfaces_v1_5.py",
        ROOT / "scripts/v1_5/26m_freeze_v5_1_t5_plan_v1_5.py",
        ROOT / "scripts/v1_5/26n_run_v5_1_t5_generation_v1_5.py",
    ]
    semantic_reads = [
        finding for path in response_source_paths for finding in ast_sensitive_reads(path)
    ]
    panel_sensitive_fields: dict[str, list[str]] = {}
    for relative in (
        "outputs/pm_v1_5b_corrected_external_split_v1/evoemo_qualification_panel_private.jsonl",
        "outputs/pm_v1_5b_corrected_external_split_v1/evoemo_lockbox_panel_private.jsonl",
    ):
        findings: set[str] = set()
        for row in rows(ROOT / relative):
            findings.update(recursive_sensitive_keys(row))
        panel_sensitive_fields[relative] = sorted(findings)
    prior_response_artifacts = [
        ROOT / "outputs/pm_v1_5_v5_1_t5_plan_v1/call_plan_private.jsonl",
        ROOT / "outputs/pm_v1_5_v5_1_t5_execution_v1/outcomes_private.jsonl",
    ]
    exact_question_hits: dict[str, list[str]] = {}
    for path in prior_response_artifacts:
        if not path.is_file():
            continue
        blob = path.read_text(encoding="utf-8")
        hits = [sha256_text(question) for question in question_texts if question in blob]
        exact_question_hits[str(path.relative_to(ROOT))] = sorted(set(hits))

    qa_exposure = {
        "artifact_full_json_parsed_by_response_loader": True,
        "technical_parse_exposure": True,
        "qa_sensitive_field_accesses_in_response_path": semantic_reads,
        "qa_sensitive_fields_in_external_response_panels": panel_sensitive_fields,
        "exact_qa_question_text_hits_in_prior_response_prompts_or_outcomes": exact_question_hits,
        "strict_pristine_lockbox": False,
        "allowed_interpretation": (
            "task-disjoint QA evaluation with declared technical parse exposure; "
            "not an untouched lockbox"
        ),
    }

    input_tokens_per_seed = sum(
        int(material["input_token_upper_bound"]) for material in unique_calls.values()
    )
    response_calls_per_seed = len(unique_calls)
    response_seeds = 2
    pricing = {"input_per_million_usd": 0.15, "output_per_million_usd": 0.60}
    response_budget = {
        "external_states": len(reprojected),
        "unique_state_action_calls_per_seed_after_policy_dedup": response_calls_per_seed,
        "generation_seeds": response_seeds,
        "planned_response_calls": response_calls_per_seed * response_seeds,
        "input_token_upper_bound": input_tokens_per_seed * response_seeds,
        "output_token_cap": int(generation.max_output_tokens),
        "output_token_upper_bound": response_calls_per_seed
        * response_seeds
        * int(generation.max_output_tokens),
    }
    response_budget["estimated_usd_upper_bound_proxy"] = (
        response_budget["input_token_upper_bound"] / 1_000_000 * pricing["input_per_million_usd"]
        + response_budget["output_token_upper_bound"] / 1_000_000 * pricing["output_per_million_usd"]
    )
    qa_budget = {
        "main_p13_p18_questions": qa["partition_question_counts"]["p13_p18"],
        "main_conditions": 5,
        "main_calls_if_one_deterministic_pass": qa["partition_question_counts"]["p13_p18"] * 5,
        "supplementary_all_questions": qa["questions"],
        "supplementary_calls_if_one_deterministic_pass": qa["questions"] * 5,
        "exact_prompt_token_budget_status": "DEFERRED_TO_E2_AFTER_QA_PROMPTS_AND_RETRIEVAL_ARE_MATERIALIZED",
    }

    checks = {
        "current_v5_2_sealed_implementation_hashes_match": all(seal_hash_checks.values()),
        "current_v5_2_contract_input_hashes_match": all(
            v52_contract_input_hash_checks.values()
        ),
        "response_free_surface_input_hashes_match": all(source_input_hash_checks.values()),
        "response_free_surface_implementation_hashes_match": all(
            response_surface_implementation_hash_checks.values()
        ),
        "response_free_surface_artifact_hash_matches": sha256_file(SURFACES)
        == surface_report["t5_state_surfaces_sha256"],
        "external_surface_count_260": len(surface_rows) == 260,
        "response_free_surface_read_no_outcomes": all(response_free_checks),
        "evoemo_memory_owner_isolation": bool(owner_checks) and all(owner_checks),
        "evoemo_release_sha256_expected": qa["sha256"]
        == "f30698e87fddaeff51270a666c654da604f487a3456ec60d2b6ae08a6fecd420",
        "qa_release_total_1427": qa["questions"] == 1427,
        "qa_partition_counts_519_490_418": qa["partition_question_counts"]
        == {"p1_p6": 519, "p7_p12": 490, "p13_p18": 418},
        "no_qa_sensitive_field_access_in_response_path": len(semantic_reads) == 0,
        "no_qa_fields_in_response_panels": not any(panel_sensitive_fields.values()),
        "no_exact_qa_question_text_in_prior_response_prompts_or_outcomes": not any(
            exact_question_hits.values()
        ),
        "zero_api_calls": True,
        "zero_human_or_llm_outcomes_read": True,
    }
    findings = [
        {
            "severity": "PASS",
            "finding": "V5.2 sealed stack and external response-free source hashes match.",
            "implication": "E2 can build a post-hoc external replication without silently changing the method.",
        },
        {
            "severity": "LIMITATION",
            "finding": "The ES-MemEval JSON was technically parsed by the prior response loader.",
            "implication": "p13-p18 QA is task-disjoint, not a strict untouched lockbox; no QA-field or exact-question leakage was found.",
        },
        {
            "severity": "METHOD",
            "finding": "V5.1 external route decisions were discarded and reprojected with V5.2 heads and atomic compiler.",
            "implication": "The external experiment now uses the same policy/executor identity as the final internal system.",
        },
        {
            "severity": "DEFERRED",
            "finding": "Raw-session baselines and exact QA prompt-token bounds do not yet exist.",
            "implication": "E2 must materialize them before any API call; E1 freezes only core response calls and QA call counts.",
        },
    ]
    hard_required = list(checks.values())
    status = "PASS_READY_FOR_E2_PLAN_MATERIALIZATION" if all(hard_required) else "FAIL_E1_BLOCK_EXTERNAL_PLAN"
    report = {
        "protocol": PROTOCOL,
        "status": status,
        "scope": "zero-API outcome-blind external preflight",
        "generator": generator.model,
        "temperature": generation.temperature,
        "checks": checks,
        "seal_hash_checks": seal_hash_checks,
        "v5_2_contract_input_hash_checks": v52_contract_input_hash_checks,
        "source_input_hash_checks": source_input_hash_checks,
        "response_surface_implementation_hash_checks": response_surface_implementation_hash_checks,
        "qa_exposure": qa_exposure,
        "qa_artifact_profile_public": qa,
        "coverage": {
            partition: {
                component: dict(counter)
                for component, counter in components.items()
            }
            for partition, components in coverage.items()
        },
        "cost_matched_fixed": {"selected_actions": cost_actions, "audit": cost_audit},
        "action_distributions": {
            partition: {
                policy: dict(sorted(counter.items()))
                for policy, counter in policies.items()
            }
            for partition, policies in action_distributions.items()
        },
        "policy_aliases": aliases,
        "budget": {
            "pricing_proxy": pricing,
            "response_experiment": response_budget,
            "qa_experiment": qa_budget,
            "secondary_raw_session_baselines": "DEFERRED_TO_E2_ADAPTER_MATERIALIZATION",
        },
        "findings": findings,
        "api_calls": 0,
        "human_or_llm_outcomes_read": 0,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "external_v5_2_response_free_surfaces_private.jsonl", reprojected)
    write_json(args.out_dir / "audit_report.json", report)
    write_json(args.out_dir / "qa_artifact_profile.json", qa)
    (args.out_dir / "report.html").write_text(render_html(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "status": status,
                "external_states": len(reprojected),
                "planned_core_response_calls": response_budget["planned_response_calls"],
                "planned_main_qa_calls": qa_budget["main_calls_if_one_deterministic_pass"],
                "api_calls": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
