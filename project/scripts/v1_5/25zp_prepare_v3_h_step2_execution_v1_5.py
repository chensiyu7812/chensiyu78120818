#!/usr/bin/env python3
"""Prepare the single, versioned V3 H-Step2 execution qualification packet."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Iterable

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import MemoryItem, MemorySource, StrategyMode, canonical_action_id
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
from metacom_pm.prompts import BASE_SUPPORTER_SYSTEM
from metacom_pm.text import conservative_token_bound
from metacom_pm.v1_5_candidate_discovery import (
    discover_final_typed_memory_candidates,
    materialize_rank1_memory_for_execution,
)
from metacom_pm.v1_5_generator_alignment_audit import (
    build_resource_execution_plan,
    materialize_strategy_card_for_execution,
    resource_bundle_application_messages_v1_5b,
)
from metacom_pm.v1_5_memory_transport import compile_bounded_memory_with_metadata
from metacom_pm.retrieval import source_specific_memory_queries


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v3-h-step2-execution-qualification-plan-v2-exact-action-schema"
INPUT_USD_PER_MTOK = 0.15
OUTPUT_USD_PER_MTOK = 0.60
SAFETY_FACTOR = 1.5
COMPONENTS = ("MP", "MS", "ME", "RS")
MULTI_ACTIONS = (
    ("MP", "MS"),
    ("MP", "ME"),
    ("MS", "ME"),
    ("MP", "MS", "ME"),
    ("MP", "RS"),
    ("MS", "RS"),
    ("ME", "RS"),
    ("MP", "MS", "ME", "RS"),
)


def _require_formal_python() -> None:
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V3 requires {FORMAL_PYTHON}; got {sys.executable}")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _visible_context(dialogue: Iterable[dict[str, Any]], current: str) -> str:
    history = "\n".join(
        f"{str(turn['role'])}: {str(turn['content'])}" for turn in dialogue
    )
    return (
        "Current-session summary:\n(none)\n\n"
        f"Recent dialogue:\n{history or '(none)'}\n\n"
        f"Current user message:\n{current}"
    )


def _fallback_messages(visible_context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": BASE_SUPPORTER_SYSTEM},
        {
            "role": "user",
            "content": visible_context + "\n\nWrite only the counselor's next response.",
        },
    ]


def _resource_subtype(component: str, hint: str) -> str:
    if component == "MP":
        if hint not in {"MP_PROFILE", "MP_PREFERENCE"}:
            raise ValueError(f"invalid MP subtype {hint!r}")
        return hint
    if component == "ME":
        if hint not in {"ME_REUSABLE_OUTCOME", "ME_EVENT_OUTCOME"}:
            raise ValueError(f"invalid ME subtype {hint!r}")
        return "ME_REUSABLE_OUTCOME"
    return {"MS": "MS_SESSION", "RS": "RS_ATOMIC_MOVE"}[component]


def _action_id(components: Iterable[str]) -> str:
    chosen = set(components)
    sources = frozenset(MemorySource(value) for value in ("MP", "MS", "ME") if value in chosen)
    strategy = StrategyMode.RS if "RS" in chosen else StrategyMode.R0
    return canonical_action_id(sources, strategy)


def _usd(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * INPUT_USD_PER_MTOK / 1_000_000
        + output_tokens * OUTPUT_USD_PER_MTOK / 1_000_000
    )


def _call(
    *,
    call_kind: str,
    state_id: str,
    user_id: str,
    components: tuple[str, ...],
    subtype_by_component: dict[str, str],
    resources: dict[str, str],
    lineage: dict[str, Any],
    visible_dialogue: list[dict[str, Any]],
    current_user_text: str,
    generation: SupporterGenerationContract,
    protocol: str = PROTOCOL,
) -> dict[str, Any]:
    plans = {
        component: build_resource_execution_plan(
            component=component,
            resource_subtype=subtype_by_component[component],
            current_user_text=current_user_text,
        )
        for component in components
    }
    labels = {component: f"{component}1" for component in components}
    context = _visible_context(visible_dialogue, current_user_text)
    messages = resource_bundle_application_messages_v1_5b(
        plans=plans,
        visible_context=context,
        selected_resources=resources,
        resource_labels=labels,
        system_prompt=generation.system_prompt,
    )
    fallback = _fallback_messages(context)
    action_id = _action_id(components)
    return {
        "protocol": protocol,
        "call_id": "v3h2_" + stable_hex(protocol, call_kind, state_id, action_id, n=24),
        "panel_id": f"v3_h_step2_{state_id}",
        "domain": "InternalQualification",
        "state_id": state_id,
        "group_id_private_analysis_only": user_id,
        "requested_action_id": action_id,
        "policy_aliases": [f"v3_h_step2_{call_kind}"],
        "requested_components": list(components),
        "execution_plans": {
            component: plan.model_dump(mode="json") for component, plan in plans.items()
        },
        "resource_labels": labels,
        "selected_resources_private": resources,
        "resource_lineage_private": lineage,
        "current_user_text": current_user_text,
        "visible_dialogue": visible_dialogue,
        "messages": messages,
        "messages_sha256": sha256_text(canonical_json(messages)),
        "response_schema": "ResourceBundleApplicationOutput",
        "fallback_messages": fallback,
        "fallback_messages_sha256": sha256_text(canonical_json(fallback)),
        "input_token_upper_bound": conservative_token_bound(
            canonical_json(messages), safety_factor=SAFETY_FACTOR
        ),
        "primary_output_token_cap": max(500, int(generation.max_output_tokens) + 200),
        "maximum_fallback_input_token_upper_bound": conservative_token_bound(
            canonical_json(fallback), safety_factor=SAFETY_FACTOR
        ),
        "maximum_fallback_output_token_cap": int(generation.max_output_tokens),
        "seed": int(stable_hex(protocol, state_id, action_id, n=8), 16) % (2**31 - 1),
        "generation_deduplication_key": f"v3h2::{protocol}::{state_id}::{action_id}",
        "selection_or_prompt_uses_external_response_quality_risk_or_judge": False,
    }


def _single_calls(
    *,
    candidates: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    generation: SupporterGenerationContract,
) -> list[dict[str, Any]]:
    admissible = {
        (str(row["state_id"]), str(row["component"]))
        for row in policy_rows
        if row["track"] == "COMPONENT_EFFECT" and row["step1_candidate_admissible"]
    }
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        component = str(row["target_component_private_not_model_input"])
        if (
            row["track_private_not_model_input"] == "COMPONENT_EFFECT"
            and (str(row["state_id"]), component) in admissible
            and row["exact_rank1_candidate"]["candidate_present"]
        ):
            pools[component].append(row)
    calls: list[dict[str, Any]] = []
    for component in COMPONENTS:
        ordered = sorted(
            pools[component],
            key=lambda row: stable_hex(PROTOCOL, "single", component, str(row["state_id"]), n=32),
        )
        # MP deliberately contains both behavioral preference and profile facts.
        if component == "MP":
            by_hint: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in ordered:
                by_hint[str(row["exact_rank1_candidate"]["compiler_subtype_hint"])].append(row)
            selected = by_hint["MP_PREFERENCE"][:3] + by_hint["MP_PROFILE"][:2]
        elif component == "RS":
            # Greedily retain family diversity without looking at a response or label.
            selected = []
            seen: set[str] = set()
            for row in ordered:
                family = str(row["private_retrieval_audit_not_model_input"].get("actual_strategy_family") or "")
                if family not in seen:
                    selected.append(row)
                    seen.add(family)
                if len(selected) == 5:
                    break
            if len(selected) < 5:
                selected += [row for row in ordered if row not in selected][: 5 - len(selected)]
        else:
            selected = ordered[:5]
        if len(selected) != 5:
            raise RuntimeError(f"not enough {component} single-component candidates")
        for row in selected:
            exact = dict(row["exact_rank1_candidate"])
            subtype = _resource_subtype(component, str(exact["compiler_subtype_hint"]))
            calls.append(
                _call(
                    call_kind="single",
                    state_id=str(row["state_id"]),
                    user_id=str(row["user_id_private_not_model_input"]),
                    components=(component,),
                    subtype_by_component={component: subtype},
                    resources={component: str(exact["candidate_text"])},
                    lineage={
                        component: {
                            "candidate_id": str(exact["candidate_id"]),
                            "selected_rank": 1,
                            "source": "v3_exact_rank1_materialization_v4",
                        }
                    },
                    visible_dialogue=list(row["visible_dialogue"]),
                    current_user_text=str(row["current_user_text"]),
                    generation=generation,
                )
            )
    return calls


def _multi_user(index: int, topic: str) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    user_id = f"user_v3h2_{stable_hex(PROTOCOL, index, topic, n=20)}"
    user = {
        "id": user_id,
        "basic_info": {
            "stable_preference_response_format": (
                "prefers concise wording and at most one focused question"
            ),
            "practical_constraint": (
                f"can reflect on the {topic} situation only during a short private evening window"
            ),
        },
        "dialog_history": [
            {
                "summary": "An unrelated conversation concerned an ordinary household errand.",
                "dialogue": [
                    {"role": "seeker", "content": "The errand was tiring but is finished."},
                    {"role": "supporter", "content": "It sounds settled for now."},
                ],
            },
            {
                "summary": (
                    f"The earlier {topic} session distinguished emotional uncertainty "
                    "from practical workload instead of treating them as one problem."
                ),
                "dialogue": [
                    {
                        "role": "seeker",
                        "content": (
                            f"For the {topic} situation, I wrote one short note naming "
                            "the most reversible next step, and that made the next conversation easier."
                        ),
                    },
                    {
                        "role": "supporter",
                        "content": "You found a small way to reduce the immediate load.",
                    },
                ],
            },
        ],
    }
    dialogue = [
        {
            "role": "user",
            "content": f"The {topic} pressure has returned and several parts feel tangled.",
        },
        {
            "role": "assistant",
            "content": "I will keep the different parts separate and not assume the old situation is unchanged.",
        },
    ]
    current = (
        f"The {topic} issue is on my mind again. I cannot tell whether emotional "
        "uncertainty or practical workload is the main part. A brief reference to what "
        "helped before is welcome. I only have a short private window this evening, "
        "so please keep this concise with at most one focused question."
    )
    return user, dialogue, current


def _multi_calls(
    *, generation: SupporterGenerationContract, strategy_cards: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    topics = (
        "project deadline",
        "family conversation",
        "friendship repair",
        "moving adjustment",
        "study workload",
        "work schedule",
        "relationship transition",
        "community commitment",
    )
    calls: list[dict[str, Any]] = []
    rs_card = next(
        card
        for card in strategy_cards
        if card["core_submove_id"] == "question_choose_priority"
        and card["execution_profile"] == "minimal"
    )
    for index, (components, topic) in enumerate(zip(MULTI_ACTIONS, topics, strict=True), 1):
        user, dialogue, current = _multi_user(index, topic)
        items, _, metadata = compile_bounded_memory_with_metadata(user)
        queries = source_specific_memory_queries(current, dialogue, "")
        discovered = discover_final_typed_memory_candidates(
            queries=queries,
            items=items,
            source_metadata=metadata,
            session_index=3,
        )
        resources: dict[str, str] = {}
        subtype: dict[str, str] = {}
        lineage: dict[str, Any] = {}
        for component in components:
            if component == "RS":
                resources[component] = materialize_strategy_card_for_execution(rs_card)
                subtype[component] = "RS_ATOMIC_MOVE"
                lineage[component] = {
                    "card_id": str(rs_card["card_id"]),
                    "source": "frozen_v4_shared_strategy_bank",
                    "retrieval_assessed_separately": True,
                }
                continue
            source = MemorySource(component)
            candidate = discovered[source]
            if not candidate.selected_items:
                raise RuntimeError(f"multi state {index} lacks {component} Rank-1")
            resource, execution_item = materialize_rank1_memory_for_execution(
                source=source,
                selected_items=candidate.selected_items,
                session_index=3,
            )
            resources[component] = resource
            hint = str(metadata[execution_item.memory_id].get("mp_subtype") or "")
            subtype[component] = _resource_subtype(component, hint)
            lineage[component] = {
                "candidate_id": execution_item.memory_id,
                "selected_rank": 1,
                "compiler_protocol": "shared_bounded_memory_compiler",
                "retriever_protocol": str(candidate.descriptor["protocol"]),
            }
        state_id = "v3h2_multi_" + stable_hex(PROTOCOL, index, topic, n=20)
        calls.append(
            _call(
                call_kind="multi",
                state_id=state_id,
                user_id=str(user["id"]),
                components=tuple(components),
                subtype_by_component=subtype,
                resources=resources,
                lineage=lineage,
                visible_dialogue=dialogue,
                current_user_text=current,
                generation=generation,
            )
        )
    return calls


def build(*, out_dir: Path) -> dict[str, Any]:
    _require_formal_python()
    pm_config = load_config(ROOT / "configs/pm_v1_5.yaml")
    experiment = load_config(ROOT / "configs/experiment.yaml")
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, generation.generator_endpoint)
    candidates_path = ROOT / "outputs/pm_v1_5_v3_p2_exact_rank1_v4/candidate_rows_private.jsonl"
    policy_path = ROOT / "outputs/pm_v1_5_v3_structured_policy_boundary_v1/policy_observation_rows.jsonl"
    bank_path = ROOT / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl"
    singles = _single_calls(
        candidates=_rows(candidates_path),
        policy_rows=_rows(policy_path),
        generation=generation,
    )
    multis = _multi_calls(generation=generation, strategy_cards=_rows(bank_path))
    calls = singles + multis
    if len(calls) != 28 or len({row["call_id"] for row in calls}) != 28:
        raise RuntimeError("V3 H-Step2 must contain exactly 28 unique calls")
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "call_plan_private.jsonl"
    write_jsonl(plan_path, calls)
    primary_input = sum(int(row["input_token_upper_bound"]) for row in calls)
    primary_output = sum(int(row["primary_output_token_cap"]) for row in calls)
    fallback_input = sum(int(row["maximum_fallback_input_token_upper_bound"]) for row in calls)
    fallback_output = sum(int(row["maximum_fallback_output_token_cap"]) for row in calls)
    checks = {
        "20_single_calls": len(singles) == 20,
        "single_balance_5_each": Counter(
            row["requested_components"][0] for row in singles
        ) == Counter({component: 5 for component in COMPONENTS}),
        "8_multi_calls": len(multis) == 8,
        "8_distinct_multi_actions": len({row["requested_action_id"] for row in multis}) == 8,
        "all_requested_resources_nonempty": all(
            str(row["selected_resources_private"][component]).strip()
            for row in calls
            for component in row["requested_components"]
        ),
        "all_resources_bound_into_prompt": all(
            any(
                str(row["selected_resources_private"][component])
                in str(message.get("content") or "")
                for message in row["messages"]
            )
            for row in calls
            for component in row["requested_components"]
        ),
        "no_response_outcome_judge_or_external_selection": all(
            not row["selection_or_prompt_uses_external_response_quality_risk_or_judge"]
            for row in calls
        ),
    }
    report = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_FINAL_PAID_GENERATION_REVIEW" if all(checks.values()) else "FAIL",
        "planned_primary_calls": 28,
        "maximum_fallback_calls": 28,
        "generator": endpoint.model,
        "call_plan_sha256": sha256_file(plan_path),
        "estimated_primary_generation_usd": round(_usd(primary_input, primary_output), 6),
        "estimated_generation_usd_with_all_fallbacks": round(
            _usd(primary_input + fallback_input, primary_output + fallback_output), 6
        ),
        "checks": checks,
        "qualification_gates": {
            "single_component_functional_minimum": "4/5 for each of MP, MS, ME, RS",
            "multi_component_all_requested_functional_minimum": "6/8",
            "material_misuse_maximum": "1/28 overall",
            "fabricated_recall_maximum": 0,
            "explicit_boundary_violation_maximum": 0,
            "machine_fallback_adoptable": "all displayed fallbacks",
        },
        "scientific_scope": (
            "This packet qualifies Step2 execution only. It does not evaluate PM "
            "Step1 routing, candidate retrieval accuracy, or general response quality."
        ),
        "inputs": {
            "v3_exact_rank1_candidates_sha256": sha256_file(candidates_path),
            "v3_structured_policy_rows_sha256": sha256_file(policy_path),
            "frozen_strategy_bank_sha256": sha256_file(bank_path),
            "external_lockbox_read": False,
            "human_labels_read_for_selection": False,
            "response_outcomes_read_for_selection": False,
        },
    }
    write_json(out_dir / "generation_preflight.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_h_step2_execution_v2_candidate",
    )
    args = parser.parse_args()
    print(build(out_dir=args.out_dir))


if __name__ == "__main__":
    main()
