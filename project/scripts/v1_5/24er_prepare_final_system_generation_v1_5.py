#!/usr/bin/env python3
"""Prepare the deduplicated final V1.5b system-generation call plan."""

from __future__ import annotations

from collections import Counter, defaultdict
import os
from pathlib import Path
from typing import Any

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import MemorySource, RuntimeState
from metacom_pm.evoemo import load_evoemo
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import canonical_json, iter_jsonl, read_json, sha256_file, sha256_text, stable_hex, write_json, write_jsonl
from metacom_pm.prompts import common_context, generation_messages
from metacom_pm.retrieval import MemoryRetriever, source_specific_memory_queries
from metacom_pm.text import conservative_token_bound, estimate_tokens, normalize_space
from metacom_pm.v1_5_candidate_discovery import (
    discover_memory_candidates_by_source_query_v1_5b,
    materialize_rank1_memory_for_execution,
)
from metacom_pm.v1_5_generator_alignment_audit import (
    build_resource_execution_plan,
    materialize_strategy_card_for_execution,
    resource_bundle_application_messages_v1_5b,
)
from metacom_pm.v1_5_memory_transport import compile_bounded_memory_with_metadata
from metacom_pm.v1_5_strategy_rag_repair import repaired_observable_opportunity_flags, repaired_rank_applicable_v4_cards
from metacom_pm.v1_5b_policy_runtime import COMPONENTS, action_component_bits


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5b-final-system-generation-plan-v1"
INPUT_USD_PER_MTOK = 0.15
OUTPUT_USD_PER_MTOK = 0.60
SAFETY_FACTOR = 1.5


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _dialogue(state: RuntimeState) -> list[dict[str, str]]:
    rows = [
        {
            "speaker": "seeker" if str(getattr(turn.role, "value", turn.role)) == "user" else "supporter",
            "content": turn.content,
        }
        for turn in state.current_session_history
    ]
    rows.append({"speaker": "seeker", "content": state.current_user_text})
    return rows


def _query(dialogue: list[dict[str, str]], flags: dict[str, Any]) -> str:
    seekers = [row["content"] for row in dialogue if row["speaker"] == "seeker"]
    cues = [
        key.replace("_", " ")
        for key, value in flags.items()
        if isinstance(value, bool) and value and key not in {"substantive", "pure_phatic", "routine_closing", "active_high_stakes", "explicit_stop", "ordinary_rag_hard_off"}
    ]
    return (
        "Represent this sentence for searching relevant passages: Choose one safe, "
        "topic-agnostic emotional-support technique. Observable cues: "
        + (", ".join(cues) or "none")
        + ". Recent seeker context: "
        + " ".join(seekers[-3:])
    )


def _index_strategy_cards(cards: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for source in cards:
        card = dict(source)
        card_id = normalize_space(card.get("card_id"))
        if not card_id:
            raise RuntimeError("Strategy Bank contains an empty card_id")
        if card_id in by_id:
            raise RuntimeError(f"Strategy Bank contains duplicate card_id: {card_id}")
        by_id[card_id] = card
    return by_id


def _resolve_ranked_card(
    ranked: list[dict[str, Any]], cards_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    if not ranked:
        return None
    card_id = str(ranked[0]["card_id"])
    if card_id not in cards_by_id:
        raise RuntimeError(f"ranked Strategy Bank card is missing: {card_id}")
    return dict(cards_by_id[card_id])


def _strategy_card(
    state: RuntimeState,
    cards: list[dict[str, Any]],
    cards_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    dialogue = _dialogue(state)
    seekers = [row["content"] for row in dialogue if row["speaker"] == "seeker"]
    recent = " ".join(seekers[-3:])
    flags = repaired_observable_opportunity_flags(
        current_user_text=state.current_user_text,
        recent_user_text=recent,
        visible_dialogue=dialogue,
    )
    ranked = repaired_rank_applicable_v4_cards(
        query=_query(dialogue, flags),
        current_user_text=state.current_user_text,
        recent_user_text=recent,
        visible_dialogue=dialogue,
        cards=cards,
    )
    # The ranker intentionally returns a compact audit row, not the generator
    # card surface. Resolve the frozen card by ID before materializing RS.
    return _resolve_ranked_card(ranked, cards_by_id)


def _card_surface(card: dict[str, Any]) -> str:
    try:
        return materialize_strategy_card_for_execution(card)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _require_active_resources(
    *, action_id: str, active: list[str], resources: dict[str, str], state_id: str
) -> None:
    missing = [component for component in active if component not in resources]
    if missing:
        raise RuntimeError(
            f"requested action {action_id} lacks resources {missing} for {state_id}"
        )
    empty = [
        component
        for component in active
        if not normalize_space(resources.get(component))
    ]
    if empty:
        raise RuntimeError(
            f"requested action {action_id} has empty resources {empty} for {state_id}"
        )


def _memory_subtype(source: str, selected, metadata: dict[str, dict[str, Any]]) -> str:
    if source == "MP":
        subtypes = {
            str(metadata.get(item.memory_id, {}).get("mp_subtype") or "MP_PROFILE")
            for item in selected
        }
        return "MP_PREFERENCE" if subtypes == {"MP_PREFERENCE"} else "MP_PROFILE"
    return "MS_SESSION" if source == "MS" else "ME_EVENT_OUTCOME"


def _estimated_usd(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * INPUT_USD_PER_MTOK / 1_000_000 + output_tokens * OUTPUT_USD_PER_MTOK / 1_000_000


def build(
    *,
    root: Path = ROOT,
    panel_dir: Path | None = None,
    routes_dir: Path | None = None,
    out_dir: Path | None = None,
    protocol: str = PROTOCOL,
    seed_protocol: str | None = None,
) -> dict[str, Any]:
    panel_dir = panel_dir or root / "outputs/pm_v1_5b_final_external_panel_v1"
    routes_dir = routes_dir or root / "outputs/pm_v1_5b_final_outcome_blind_routes_v1"
    seed_protocol = seed_protocol or protocol
    esconv_path = panel_dir / "esconv_panel_private.jsonl"
    evoemo_path = panel_dir / "evoemo_panel_private.jsonl"
    routes_path = routes_dir / "policy_routes_private.jsonl"
    routing_report_path = routes_dir / "routing_freeze_report.json"
    bank_path = root / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl"
    evoemo_data_path = root / "data/external/evo_emo.json"
    pm_config_path = root / "configs/pm_v1_5.yaml"
    experiment_path = root / "configs/experiment.yaml"
    routing_report = read_json(routing_report_path)
    if routing_report["status"] not in {
        "PASS_FINAL_OUTCOME_BLIND_ROUTES_FROZEN",
        "PASS_FINAL_V2_OUTCOME_BLIND_ROUTES_FROZEN",
    }:
        raise RuntimeError("final outcome-blind routes are not frozen")

    panels = {row["panel_id"]: row for row in [*_rows(esconv_path), *_rows(evoemo_path)]}
    routes = _rows(routes_path)
    cards = _rows(bank_path)
    cards_by_id = _index_strategy_cards(cards)
    pm_config = load_config(pm_config_path)
    experiment = load_config(experiment_path)
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, generation.generator_endpoint)
    retriever = MemoryRetriever(minimum_score_by_source={source: float(pm_config["retrieval"]["memory_min_score"]) for source in MemorySource})
    evo_users = load_evoemo(evoemo_data_path)
    compiled = {
        str(user["id"]): compile_bounded_memory_with_metadata(user)
        for user in evo_users
    }

    calls: list[dict[str, Any]] = []
    for route_row in routes:
        panel = panels[str(route_row["panel_id"])]
        state = RuntimeState.model_validate(panel["runtime_state"])
        resources: dict[str, str] = {}
        subtypes: dict[str, str] = {}
        resource_private_lineage: dict[str, Any] = {}
        if route_row["domain"] == "EvoEmo":
            items, _docs, metadata = compiled[str(panel["user_id_private_analysis_only"])]
            visible = [turn.model_dump(mode="json") for turn in state.current_session_history]
            discoveries = discover_memory_candidates_by_source_query_v1_5b(
                queries=source_specific_memory_queries(state.current_user_text, visible, state.current_session_summary),
                items=items,
                retriever=retriever,
                session_index=state.session_index,
            )
            for source in MemorySource:
                selected = discoveries[source].selected_items
                if selected:
                    resources[source.value], execution_item = (
                        materialize_rank1_memory_for_execution(
                        source=source,
                        selected_items=selected,
                        session_index=state.session_index,
                        )
                    )
                    execution_selected = (execution_item,)
                    subtypes[source.value] = _memory_subtype(
                        source.value, execution_selected, metadata
                    )
                    resource_private_lineage[source.value] = {
                        "candidate_memory_ids": [item.memory_id for item in selected],
                        "candidate_created_sessions": [item.created_session for item in selected],
                        "candidate_descriptor": discoveries[source].descriptor,
                        "execution_memory_id": execution_item.memory_id,
                        "execution_created_session": execution_item.created_session,
                        "execution_item_count": 1,
                        "execution_rank": 1,
                        "execution_injected_tokens": estimate_tokens(
                            resources[source.value]
                        ),
                    }
        card = _strategy_card(state, cards, cards_by_id)
        if card is not None:
            resources["RS"] = _card_surface(card)
            subtypes["RS"] = str(card["strategy_family"])
            resource_private_lineage["RS"] = {"card_id": card["card_id"]}

        action_to_policies: dict[str, list[str]] = defaultdict(list)
        for policy, value in route_row["policy_routes"].items():
            action_to_policies[str(value["requested_action_id"])].append(str(policy))
        for action_id in sorted(action_to_policies):
            bits = action_component_bits(action_id)
            active = [component for component in COMPONENTS if bits[component]]
            _require_active_resources(
                action_id=action_id,
                active=active,
                resources=resources,
                state_id=state.state_id,
            )
            fallback_messages = generation_messages(state, [], [], system_prompt=generation.system_prompt)
            if active:
                plans = {
                    component: build_resource_execution_plan(component=component, resource_subtype=subtypes[component])
                    for component in active
                }
                labels = {component: component + "1" for component in active}
                messages = resource_bundle_application_messages_v1_5b(
                    plans=plans,
                    visible_context=common_context(state),
                    selected_resources={component: resources[component] for component in active},
                    resource_labels=labels,
                    system_prompt=generation.system_prompt,
                )
                schema = "ResourceBundleApplicationOutput"
                primary_output_cap = max(500, int(generation.max_output_tokens) + 200)
            else:
                plans = {}
                labels = {}
                messages = fallback_messages
                schema = "plain_text"
                primary_output_cap = int(generation.max_output_tokens)
            input_bound = conservative_token_bound(canonical_json(messages), safety_factor=SAFETY_FACTOR)
            fallback_input_bound = conservative_token_bound(canonical_json(fallback_messages), safety_factor=SAFETY_FACTOR)
            call_id = "final_" + stable_hex(protocol, panel["panel_id"], action_id, n=28)
            calls.append(
                {
                    "protocol": protocol,
                    "call_id": call_id,
                    "panel_id": panel["panel_id"],
                    "domain": route_row["domain"],
                    "state_id": state.state_id,
                    "group_id_private_analysis_only": route_row["group_id_private_analysis_only"],
                    "requested_action_id": action_id,
                    "policy_aliases": sorted(action_to_policies[action_id]),
                    "requested_components": active,
                    "execution_plans": {component: plans[component].model_dump(mode="json") for component in active},
                    "resource_labels": labels,
                    "selected_resources_private": {component: resources[component] for component in active},
                    "resource_lineage_private": {component: resource_private_lineage[component] for component in active},
                    "current_user_text": state.current_user_text,
                    "messages": messages,
                    "messages_sha256": sha256_text(canonical_json(messages)),
                    "response_schema": schema,
                    "fallback_messages": fallback_messages,
                    "fallback_messages_sha256": sha256_text(canonical_json(fallback_messages)),
                    "input_token_upper_bound": input_bound,
                    "primary_output_token_cap": primary_output_cap,
                    "maximum_fallback_input_token_upper_bound": fallback_input_bound if active else 0,
                    "maximum_fallback_output_token_cap": int(generation.max_output_tokens) if active else 0,
                    "seed": int(stable_hex(seed_protocol, panel["panel_id"], action_id, n=8), 16) % (2**31 - 1),
                    "generation_deduplication_key": f"{panel['panel_id']}::{action_id}",
                    "selection_or_prompt_uses_external_response_quality_risk_or_judge": False,
                }
            )

    out_dir = out_dir or root / "outputs/pm_v1_5b_final_system_generation_v1_candidate"
    out_dir.mkdir(parents=True, exist_ok=True)
    call_plan_path = out_dir / "call_plan_private.jsonl"
    write_jsonl(call_plan_path, calls)
    primary_input = sum(int(row["input_token_upper_bound"]) for row in calls)
    primary_output = sum(int(row["primary_output_token_cap"]) for row in calls)
    fallback_input = sum(int(row["maximum_fallback_input_token_upper_bound"]) for row in calls)
    fallback_output = sum(int(row["maximum_fallback_output_token_cap"]) for row in calls)
    primary_usd = _estimated_usd(primary_input, primary_output)
    maximum_usd = _estimated_usd(primary_input + fallback_input, primary_output + fallback_output)
    checks = {
        "one_call_per_unique_state_action": len(calls) == routing_report["unique_state_action_calls_after_exact_dedup"] and len({row["generation_deduplication_key"] for row in calls}) == len(calls),
        "all_six_policy_assignments_covered_by_aliases": sum(len(row["policy_aliases"]) for row in calls) == routing_report["policy_assignments"],
        "requested_resources_present_and_source_local": all(set(row["requested_components"]) == set(row["selected_resources_private"]) for row in calls),
        # M0+R0 legitimately has no resource. Every component that Step1 turns
        # on must, however, have a non-empty source-local treatment surface.
        "every_requested_resource_nonempty": all(
            all(
                normalize_space(row["selected_resources_private"].get(component))
                for component in row["requested_components"]
            )
            for row in calls
        ),
        "every_requested_resource_materialized_in_prompt": all(
            all(
                any(
                    row["selected_resources_private"][component]
                    in str(message.get("content") or "")
                    for message in row["messages"]
                )
                for component in row["requested_components"]
            )
            for row in calls
        ),
        "memory_execution_is_rank1_single_item": all(
            all(
                row["resource_lineage_private"][component]["execution_item_count"]
                == 1
                and row["resource_lineage_private"][component]["execution_rank"]
                == 1
                and row["resource_lineage_private"][component]["execution_memory_id"]
                == row["resource_lineage_private"][component]["candidate_memory_ids"][0]
                for component in row["requested_components"]
                if component in {"MP", "MS", "ME"}
            )
            for row in calls
        ),
        "context_only_has_no_bundle_schema": all((row["requested_action_id"] == "M0+R0") == (row["response_schema"] == "plain_text") for row in calls),
        "resource_actions_have_one_bounded_fallback_budget": all((not row["requested_components"]) or row["maximum_fallback_output_token_cap"] == int(generation.max_output_tokens) for row in calls),
        "no_external_response_quality_risk_or_judge_used": all(not row["selection_or_prompt_uses_external_response_quality_risk_or_judge"] for row in calls),
    }
    report = {
        "protocol": protocol,
        "status": "READY_FOR_FINAL_PAID_GENERATION_REVIEW" if all(checks.values()) else "FAIL_FINAL_GENERATION_PLAN",
        "generator": endpoint.model,
        "api_key_env": endpoint.api_key_env,
        "api_key_present_at_dry_run": bool(os.environ.get(endpoint.api_key_env, "")),
        "planned_primary_calls": len(calls),
        "maximum_fallback_calls": sum(bool(row["requested_components"]) for row in calls),
        "domain_primary_calls": dict(sorted(Counter(row["domain"] for row in calls).items())),
        "requested_action_distribution": dict(sorted(Counter(row["requested_action_id"] for row in calls).items())),
        "primary_input_token_upper_bound": primary_input,
        "primary_output_token_cap_total": primary_output,
        "maximum_fallback_input_token_upper_bound": fallback_input,
        "maximum_fallback_output_token_cap_total": fallback_output,
        "estimated_primary_generation_usd": round(primary_usd, 6),
        "estimated_generation_usd_with_all_fallbacks": round(maximum_usd, 6),
        "pricing_assumption_usd_per_mtok": {"input": INPUT_USD_PER_MTOK, "output": OUTPUT_USD_PER_MTOK},
        "checks": checks,
        "external_outcome_used": False,
        "call_plan_sha256": sha256_file(call_plan_path),
        "inputs": {
            str(path.relative_to(root)): sha256_file(path)
            for path in (esconv_path, evoemo_path, routes_path, routing_report_path, bank_path, evoemo_data_path, pm_config_path, experiment_path)
        },
    }
    write_json(out_dir / "generation_preflight.json", report)
    return report


def main() -> None:
    report = build()
    print(report)


if __name__ == "__main__":
    main()
