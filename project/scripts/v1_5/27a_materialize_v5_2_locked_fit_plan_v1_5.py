#!/usr/bin/env python3
"""Build the outcome-blind V5.2 FIT plan for the backend-locked executor."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    sha256_file,
    sha256_text,
    stable_hex,
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
from metacom_pm.v1_5b_policy_runtime import COMPONENTS


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5.2-locked-fit-plan-v2"
BLUEPRINT = ROOT / "data/pm_v1_5_v3_effect_blueprint_v1/private/construction_blueprint.jsonl"
CANDIDATES = ROOT / "outputs/pm_v1_5_v3_p2_exact_rank1_v4/candidate_rows_private.jsonl"
STRATEGY_BANK = ROOT / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl"
CONTRACT = ROOT / "data/pm_v1_5_contracts/v5_2_locked_executor_repair_v1.json"
RUNNER = ROOT / "scripts/v1_5/27b_run_v5_2_locked_fit_generation_v1_5.py"


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def current_context(row: dict[str, Any]) -> str:
    lines = ["Visible current conversation:"]
    for turn in row["visible_dialogue"]:
        lines.append(f"{str(turn['role']).capitalize()}: {str(turn['content']).strip()}")
    lines.append(f"User: {str(row['current_user_text']).strip()}")
    return "\n".join(lines)


def semantic_group_id(row: dict[str, Any]) -> str:
    """Group candidate contrasts that share the same model-visible state."""

    surface = {
        "target_component": str(row["target_component_private_not_model_input"]),
        "visible_dialogue": row["visible_dialogue"],
        "current_user_text": str(row["current_user_text"]),
    }
    return "v52sem_" + stable_hex(PROTOCOL, canonical_json(surface), n=24)


def _strategy_fields(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            result[key.strip()] = value.strip()
    return result


def parse_candidate(
    row: dict[str, Any], *, strategy_by_id: dict[str, dict[str, Any]]
) -> tuple[TypedResourceCandidate, str]:
    exact = row["exact_rank1_candidate"]
    component = str(row["target_component_private_not_model_input"])
    hint = str(exact["compiler_subtype_hint"])
    text = str(exact["candidate_text"]).strip()
    resource_id = str(exact["candidate_id"])
    common = {
        "component": component,
        "resource_id": resource_id,
        "candidate_version": str(exact["candidate_text_sha256"]),
    }
    owner = str(row["structured_candidate_metadata"]["candidate_owner_id"])
    if component == "MP" and hint == "MP_PREFERENCE":
        return (
            TypedResourceCandidate(
                **common,
                subtype="MP_PREFERENCE",
                source_kind="profile",
                owner_id=owner,
                preference=text.split(":", 1)[-1].strip(),
            ),
            "",
        )
    if component == "MP" and hint == "MP_PROFILE":
        return (
            TypedResourceCandidate(
                **common,
                subtype="MP_PROFILE",
                source_kind="profile",
                owner_id=owner,
                profile_fact=text.split(":", 1)[-1].strip(),
            ),
            "",
        )
    if component == "MS" and hint == "MS_SESSION":
        atomic = compile_atomic_session_observation(text)
        if atomic is None:
            raise RuntimeError(f"non-atomic MS candidate: {row['state_id']}")
        return (
            TypedResourceCandidate(
                **common,
                subtype="MS_SESSION_OBSERVATION",
                source_kind="session",
                owner_id=owner,
                strictly_prior=True,
                age_sessions=int(exact["candidate_age_sessions"]),
                prior_observation=atomic.literal_past_note,
            ),
            "",
        )
    if component == "ME" and hint == "ME_REUSABLE_OUTCOME":
        atomic = compile_atomic_reusable_outcome(text)
        if atomic is None:
            raise RuntimeError(f"non-atomic ME candidate: {row['state_id']}")
        return (
            TypedResourceCandidate(
                **common,
                subtype="ME_REUSABLE_OUTCOME",
                source_kind="event",
                owner_id=owner,
                strictly_prior=True,
                age_sessions=int(exact["candidate_age_sessions"]),
                past_action=atomic.past_action_span,
                observed_outcome=atomic.observed_outcome_span,
                mechanism="literal_span:" + atomic.literal_evidence_span,
            ),
            "",
        )
    if component == "RS" and hint == "RS_ATOMIC_MOVE":
        fields = _strategy_fields(text)
        card = strategy_by_id.get(resource_id)
        if card is None:
            raise RuntimeError(f"strategy card absent from frozen bank: {resource_id}")
        if str(card["support_move"]).strip() != fields.get("support_move", ""):
            raise RuntimeError(f"strategy support_move mismatch: {resource_id}")
        return (
            TypedResourceCandidate(
                **common,
                subtype="RS_ATOMIC_MOVE",
                source_kind="strategy",
                support_move=fields.get("support_move", ""),
                when_to_use=fields.get("when_to_use", ""),
                when_not_to_use=fields.get("when_not_to_use", ""),
            ),
            str(card["prompt_guidance"]).strip(),
        )
    raise RuntimeError(f"unsupported component/subtype: {component}/{hint}")


def seed_int(seed_hex: str) -> int:
    value = int(seed_hex, 16) % 2_147_483_647
    return value or 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_2_locked_fit_plan_v2",
    )
    args = parser.parse_args()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V5.2 requires {FORMAL_PYTHON}; got {sys.executable}")
    for required in (BLUEPRINT, CANDIDATES, STRATEGY_BANK, CONTRACT, RUNNER):
        if not required.is_file():
            raise RuntimeError(f"required V5.2 input missing: {required}")

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["status"] != "FROZEN_BEFORE_V5_2_FIT_GENERATION":
        raise RuntimeError("V5.2 executor contract is not frozen")
    blueprint = {str(row["blueprint_row_id"]): row for row in rows(BLUEPRINT)}
    strategy_by_id = {str(row["card_id"]): row for row in rows(STRATEGY_BANK)}
    candidates = [
        row
        for row in rows(CANDIDATES)
        if row["track_private_not_model_input"] == "COMPONENT_EFFECT"
        and row["split_private_not_model_input"] == "EFFECT_FIT"
    ]
    observed = Counter(
        str(row["target_component_private_not_model_input"]) for row in candidates
    )
    if observed != Counter({component: 64 for component in COMPONENTS}):
        raise RuntimeError(f"V5.2 FIT candidate shape mismatch: {observed}")

    pm_config = load_config(ROOT / "configs/pm_v1_5.yaml")
    experiment = load_config(ROOT / "configs/experiment.yaml")
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, generation.generator_endpoint)
    safety_factor = float(pm_config["api_cost_planning"]["input_token_safety_factor"])
    pricing = {"input": 0.15, "output": 0.60}

    calls: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    compiler_counts: Counter[str] = Counter()
    for row in sorted(candidates, key=lambda item: str(item["state_id"])):
        state_id = str(row["state_id"])
        source = blueprint[state_id]
        treatment = source["effect_treatments"]
        target = str(row["target_component_private_not_model_input"])
        if treatment is None or str(treatment["control"]) != "M0+R0":
            raise RuntimeError(f"invalid one-bit treatment: {state_id}")
        candidate, strategy_guidance = parse_candidate(
            row, strategy_by_id=strategy_by_id
        )
        compiler_counts[candidate.subtype] += 1
        treatment_action = str(treatment["treatment"])
        owner = str(row["user_id_private_not_model_input"])
        context = current_context(row)
        semantic_group = semantic_group_id(row)
        on_plan = build_locked_composition_plan(
            requested_action_id=treatment_action,
            candidates={target: candidate},
            strategy_prompt_guidance=strategy_guidance,
        )
        off_plan = build_locked_composition_plan(
            requested_action_id="M0+R0", candidates={}
        )
        bindings.append(
            {
                "state_id": state_id,
                "source_group_id": str(row["group_id_private_not_model_input"]),
                "semantic_group_id": semantic_group,
                "target_component": target,
                "candidate": asdict(candidate),
                "control_action_id": "M0+R0",
                "treatment_action_id": treatment_action,
                "generation_seed_hex": list(treatment["generation_seeds"]),
            }
        )
        for seed_hex in treatment["generation_seeds"]:
            for arm, action, plan in (
                ("OFF", "M0+R0", off_plan),
                ("ON", treatment_action, on_plan),
            ):
                messages = base_generation_messages(current_context=context, plan=plan)
                message_blob = canonical_json(messages)
                if arm == "ON" and target in {"MS", "ME"}:
                    literal = plan.locked_clauses[0].literal_source_span
                    if literal and literal in message_blob:
                        raise RuntimeError(f"history exposed to generator: {state_id}")
                bound = conservative_token_bound(
                    message_blob, safety_factor=safety_factor
                )
                call_id = "v52fit_" + stable_hex(
                    PROTOCOL, state_id, target, arm, str(seed_hex), n=24
                )
                calls.append(
                    {
                        "protocol": PROTOCOL,
                        "call_id": call_id,
                        "state_id": state_id,
                        "group_id_private_analysis_only": semantic_group,
                        "source_group_id_private_audit_only": str(
                            row["group_id_private_not_model_input"]
                        ),
                        "target_component": target,
                        "arm": arm,
                        "requested_action_id": action,
                        "candidate_id_pre_action": str(
                            row["exact_rank1_candidate"]["candidate_id"]
                        ),
                        "candidate_version_pre_action": str(
                            row["exact_rank1_candidate"]["candidate_text_sha256"]
                        ),
                        "candidate_owner_id_pre_action": owner,
                        "resource_presented_to_executor": {
                            component: bool(arm == "ON" and component == target)
                            for component in COMPONENTS
                        },
                        "seed_hex": str(seed_hex),
                        "seed": seed_int(str(seed_hex)),
                        "current_user_text": str(row["current_user_text"]),
                        "messages": messages,
                        "messages_sha256": sha256_text(message_blob),
                        "composition_plan": asdict(plan),
                        "input_token_upper_bound": bound,
                        "output_token_cap": int(generation.max_output_tokens),
                        "assigned_executor_version": "v5.2-backend-locked-composer-v1",
                        "assigned_generator_version": endpoint.model,
                    }
                )

    calls.sort(key=lambda item: str(item["call_id"]))
    if len(calls) != 1024 or len({str(row["call_id"]) for row in calls}) != 1024:
        raise RuntimeError("V5.2 FIT must contain 1024 unique calls")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for call in calls:
        grouped.setdefault((str(call["state_id"]), str(call["seed_hex"])), []).append(call)
    if any({str(item["arm"]) for item in pair} != {"ON", "OFF"} for pair in grouped.values()):
        raise RuntimeError("every V5.2 state/seed requires paired ON and OFF calls")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.out_dir / "call_plan_private.jsonl"
    binding_path = args.out_dir / "state_candidate_bindings_private.jsonl"
    write_jsonl(plan_path, calls)
    write_jsonl(binding_path, bindings)
    total_input = sum(int(row["input_token_upper_bound"]) for row in calls)
    total_output = sum(int(row["output_token_cap"]) for row in calls)
    estimated = total_input / 1_000_000 * pricing["input"] + total_output / 1_000_000 * pricing["output"]
    implementation = {
        "atomic_memory": ROOT / "src/metacom_pm/v1_5_v5_2_atomic_memory.py",
        "locked_composer": ROOT / "src/metacom_pm/v1_5_v5_2_locked_composer.py",
        "itt_validator": ROOT / "src/metacom_pm/v1_5_itt_policy.py",
        "plan_builder": Path(__file__).resolve(),
        "generation_runner": RUNNER,
    }
    freeze = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_SINGLE_V5_2_LOCKED_FIT_EXECUTION",
        "method_version": "V1_5_V5_2_BACKEND_LOCKED_EXECUTOR",
        "generator": endpoint.model,
        "temperature": float(generation.temperature),
        "planned_calls": len(calls),
        "states": len(candidates),
        "semantic_groups": len({str(row["semantic_group_id"]) for row in bindings}),
        "states_per_component": dict(observed),
        "compiler_subtypes": dict(compiler_counts),
        "arms": dict(Counter(str(row["arm"]) for row in calls)),
        "input_token_upper_bound_total": total_input,
        "output_token_cap_total": total_output,
        "estimated_generation_usd_upper_bound_proxy": round(estimated, 8),
        "pricing_usd_per_mtok_proxy": pricing,
        "call_plan_sha256": sha256_file(plan_path),
        "state_candidate_bindings_sha256": sha256_file(binding_path),
        "input_sha256": {
            str(BLUEPRINT.relative_to(ROOT)): sha256_file(BLUEPRINT),
            str(CANDIDATES.relative_to(ROOT)): sha256_file(CANDIDATES),
            str(STRATEGY_BANK.relative_to(ROOT)): sha256_file(STRATEGY_BANK),
            str(CONTRACT.relative_to(ROOT)): sha256_file(CONTRACT),
        },
        "implementation_sha256": {
            key: sha256_file(path) for key, path in implementation.items()
        },
        "checks": {
            "256_states_64_per_component": True,
            "paired_same_state_same_seed_on_off": True,
            "same_visible_component_state_candidates_share_semantic_group": True,
            "m0_r0_is_valid": True,
            "all_internal_me_candidates_atomic": compiler_counts["ME_REUSABLE_OUTCOME"] == 64,
            "all_internal_ms_candidates_specific": compiler_counts["MS_SESSION_OBSERVATION"] == 64,
            "ms_me_text_absent_from_generator_messages": True,
            "locked_clauses_have_literal_provenance": True,
            "no_quality_risk_judge_or_external_outcome_read": True,
        },
        "api_calls": 0,
        "human_labels_read": 0,
        "external_outcomes_read": False,
        "formal_python_executable": str(FORMAL_PYTHON),
    }
    write_json(args.out_dir / "freeze_manifest.json", freeze)
    print(json.dumps({
        "status": freeze["status"],
        "planned_calls": freeze["planned_calls"],
        "compiler_subtypes": freeze["compiler_subtypes"],
        "estimated_usd_upper_bound_proxy": freeze["estimated_generation_usd_upper_bound_proxy"],
        "api_calls": 0,
    }))


if __name__ == "__main__":
    main()
