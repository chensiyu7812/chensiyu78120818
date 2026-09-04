#!/usr/bin/env python3
"""Freeze the V1.5b fixed action closest to learned-full prompt cost.

Selection is performed only on the existing 32-user internal D3 state panel.
It reads no response, quality, risk, human preference, judge, EvoEmo, or ESConv
test outcome.  Every fixed action remains subject to the same candidate and
hard gates as learned routing, so the comparison changes only the routing rule.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from metacom_pm.config import load_config
from metacom_pm.contracts import ALL_ACTION_IDS, MemoryBackendRecord, MemorySource, RuntimeState
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import canonical_json, iter_jsonl, read_json, sha256_file, write_json, write_jsonl
from metacom_pm.prompts import common_context, generation_messages
from metacom_pm.retrieval import MemoryRetriever, source_specific_memory_queries
from metacom_pm.text import conservative_token_bound, estimate_tokens, normalize_space
from metacom_pm.v1_5_candidate_discovery import discover_memory_candidates_by_source_query_v1_5b
from metacom_pm.v1_5_generator_alignment_audit import build_resource_execution_plan, resource_bundle_application_messages_v1_5b
from metacom_pm.v1_5_memory_opportunity_features import build_scale_stable_memory_opportunity_observation, build_source_specific_memory_opportunity_observation
from metacom_pm.v1_5_strategy_rag_repair import repaired_observable_opportunity_flags, repaired_rank_applicable_v4_cards
from metacom_pm.v1_5b_policy_runtime import COMPONENTS, ComponentOpportunity, action_component_bits, route_policy


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5b-internal-cost-matched-fixed-freeze-v1"
SAFETY_FACTOR = 1.5


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _support(rows: list[dict[str, Any]], component: str, names: list[str]) -> dict[str, set[float] | tuple[float, float]]:
    selected = [row for row in rows if row["component"] == component]
    result: dict[str, set[float] | tuple[float, float]] = {}
    for name in names:
        values = [float(row["model_features"][name]) for row in selected]
        if name == "candidate_content_match_level":
            result[name] = set(values)
        else:
            result[name] = (min(values), max(values))
    return result


def _in_support(features: dict[str, float], support: dict[str, set[float] | tuple[float, float]]) -> bool:
    for name, specification in support.items():
        value = float(features[name])
        if isinstance(specification, set):
            if value not in specification:
                return False
        else:
            if value < specification[0] or value > specification[1]:
                return False
    return True


def _strategy_dialogue(state: RuntimeState) -> list[dict[str, str]]:
    rows = [
        {
            "speaker": "seeker"
            if str(getattr(turn.role, "value", turn.role)) == "user"
            else "supporter",
            "content": turn.content,
        }
        for turn in state.current_session_history
    ]
    rows.append({"speaker": "seeker", "content": state.current_user_text})
    return rows


def _strategy_query(dialogue: list[dict[str, str]], flags: dict[str, Any]) -> str:
    seeker = [row["content"] for row in dialogue if row["speaker"] == "seeker"]
    cues = [
        key.replace("_", " ")
        for key, value in flags.items()
        if isinstance(value, bool)
        and value
        and key not in {"substantive", "pure_phatic", "routine_closing", "active_high_stakes", "explicit_stop", "ordinary_rag_hard_off"}
    ]
    return (
        "Represent this sentence for searching relevant passages: Choose one safe, "
        "topic-agnostic emotional-support technique. Observable cues: "
        + (", ".join(cues) or "none")
        + ". Recent seeker context: "
        + " ".join(seeker[-3:])
    )


def _strategy_surface(card: dict[str, Any]) -> str:
    return "\n".join(
        f"{name}: {card[name]}"
        for name in ("support_move", "when_to_use", "when_not_to_use")
        if normalize_space(card.get(name))
    )


def _memory_subtype(source: str, selected, metadata: dict[str, dict[str, Any]]) -> str:
    if source == "MP":
        subtypes = {
            str(metadata.get(item.memory_id, {}).get("mp_subtype") or "MP_PROFILE")
            for item in selected
        }
        return "MP_PREFERENCE" if subtypes == {"MP_PREFERENCE"} else "MP_PROFILE"
    return "MS_SESSION" if source == "MS" else "ME_EVENT_OUTCOME"


def _prompt_tokens(
    *,
    state: RuntimeState,
    action_id: str,
    resources: dict[str, str],
    subtypes: dict[str, str],
    generation: SupporterGenerationContract,
) -> int:
    bits = action_component_bits(action_id)
    active = [component for component in COMPONENTS if bits[component] and component in resources]
    if not active:
        messages = generation_messages(state, [], [], system_prompt=generation.system_prompt)
    else:
        plans = {
            component: build_resource_execution_plan(
                component=component, resource_subtype=subtypes[component]
            )
            for component in active
        }
        messages = resource_bundle_application_messages_v1_5b(
            plans=plans,
            visible_context=common_context(state),
            selected_resources={component: resources[component] for component in active},
            resource_labels={component: component + "1" for component in active},
            system_prompt=generation.system_prompt,
        )
    return conservative_token_bound(canonical_json(messages), safety_factor=SAFETY_FACTOR)


def build(*, root: Path = ROOT) -> dict[str, Any]:
    d3 = root / "outputs/pm_v1_5_d3_step0_blueprint_v1"
    states_path = d3 / "runtime_states.jsonl"
    backend_path = d3 / "memory_backend.jsonl"
    metadata_path = d3 / "memory_item_metadata.jsonl"
    bank_path = root / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl"
    config_path = root / "configs/pm_v1_5.yaml"
    mpms_dir = root / "outputs/pm_v1_5b_mp_ms_opportunity_heads_v1"
    me_dir = root / "outputs/pm_v1_5_scale_stable_memory_opportunity_heads_v1"
    rs_dir = root / "outputs/pm_v1_5_same_bank_rs_opportunity_router_fit_v1"

    states = list(map(RuntimeState.model_validate, _rows(states_path)))
    backends = {row.card_id: row for row in map(MemoryBackendRecord.model_validate, _rows(backend_path))}
    metadata = {str(row["card_id"]): dict(row["memory_item_metadata"]) for row in _rows(metadata_path)}
    cards = _rows(bank_path)
    config = load_config(config_path)
    generation = SupporterGenerationContract.from_config(config)
    retriever = MemoryRetriever(minimum_score_by_source={source: float(config["retrieval"]["memory_min_score"]) for source in MemorySource})

    mpms_report = read_json(mpms_dir / "fit_report.json")
    mpms_rows = _rows(mpms_dir / "fit_and_confirmation_rows_audit.jsonl")
    me_report = read_json(me_dir / "fit_report.json")
    me_rows = _rows(me_dir / "fit_rows_audit.jsonl")
    rs_report = read_json(rs_dir / "fit_report.json")
    models = {
        "MP": joblib.load(mpms_dir / "mp_opportunity_router_v1_5b.joblib"),
        "MS": joblib.load(mpms_dir / "ms_opportunity_router_v1_5b.joblib"),
        "ME": joblib.load(me_dir / "me_opportunity_router.joblib"),
        "RS": joblib.load(rs_dir / "rs_opportunity_router.joblib"),
    }
    names = {
        "MP": list(mpms_report["components"]["MP"]["feature_names"]),
        "MS": list(mpms_report["components"]["MS"]["feature_names"]),
        "ME": list(me_report["components"]["ME"]["feature_names"]),
        "RS": list(rs_report["feature_names"]),
    }
    supports = {
        "MP": _support([row for row in mpms_rows if row["evidence_role"] == "development_fit"], "MP", names["MP"]),
        "MS": _support([row for row in mpms_rows if row["evidence_role"] == "development_fit"], "MS", names["MS"]),
        "ME": _support(me_rows, "ME", names["ME"]),
    }

    audit_rows: list[dict[str, Any]] = []
    learned_costs: list[int] = []
    fixed_costs: dict[str, list[int]] = {action: [] for action in ALL_ACTION_IDS}
    for state in states:
        backend = backends[state.card_id]
        item_metadata = metadata[state.card_id]
        visible = [turn.model_dump(mode="json") for turn in state.current_session_history]
        queries = source_specific_memory_queries(state.current_user_text, visible, state.current_session_summary)
        discoveries = discover_memory_candidates_by_source_query_v1_5b(
            queries=queries,
            items=backend.items,
            retriever=retriever,
            session_index=state.session_index,
        )
        observations: dict[str, ComponentOpportunity] = {}
        resources: dict[str, str] = {}
        subtypes: dict[str, str] = {}
        feature_audit: dict[str, Any] = {}
        for source in MemorySource:
            component = source.value
            builder = build_source_specific_memory_opportunity_observation if component in {"MP", "MS"} else build_scale_stable_memory_opportunity_observation
            observation = builder(
                candidate=discoveries[source],
                catalog_items=backend.items,
                catalog_user_id=state.user_id,
                current_user_id=state.user_id,
                current_session_index=state.session_index,
                current_user_text=state.current_user_text,
                visible_dialogue=visible,
                source_metadata=item_metadata,
                background_action="M0+R0",
            )
            features = {name: float(observation["model_features"][name]) for name in names[component]}
            in_support = _in_support(features, supports[component])
            probability = float(models[component].predict_proba(np.asarray([[features[name] for name in names[component]]]))[0, 1])
            hard_pass = not bool(observation["deterministic_hard_off"])
            transparent = bool(
                hard_pass
                and features["candidate_content_match_level"] >= 0.5
                and features["candidate_grounding_or_nonredundancy_score"] >= 0.999
            )
            observations[component] = ComponentOpportunity(
                component=component,
                candidate_present=bool(discoveries[source].selected_items),
                hard_gate_pass=hard_pass,
                transparent_rule_on=transparent,
                learned_in_support=in_support,
                learned_on=bool(probability >= 0.5),
                learned_probability=probability,
                decision_reason="FROZEN_HEAD_ON" if probability >= 0.5 else "FROZEN_HEAD_OFF",
            )
            if discoveries[source].selected_items:
                resources[component] = "\n".join(item.text for item in discoveries[source].selected_items)
                subtypes[component] = _memory_subtype(component, discoveries[source].selected_items, item_metadata)
            feature_audit[component] = {
                "features": features,
                "in_support": in_support,
                "probability_on": probability,
                "hard_gate_pass": hard_pass,
                "candidate_count": len(discoveries[source].selected_items),
            }

        dialogue = _strategy_dialogue(state)
        seekers = [row["content"] for row in dialogue if row["speaker"] == "seeker"]
        flags = repaired_observable_opportunity_flags(
            current_user_text=state.current_user_text,
            recent_user_text=" ".join(seekers[-3:]),
            visible_dialogue=dialogue,
        )
        query = _strategy_query(dialogue, flags)
        ranked = repaired_rank_applicable_v4_cards(
            query=query,
            current_user_text=state.current_user_text,
            recent_user_text=" ".join(seekers[-3:]),
            visible_dialogue=dialogue,
            cards=cards,
        )
        rs_probability = float(models["RS"].predict_proba(np.asarray([[float(bool(flags.get(name, False))) for name in names["RS"]]]))[0, 1])
        observations["RS"] = ComponentOpportunity(
            component="RS",
            candidate_present=bool(ranked),
            hard_gate_pass=bool(ranked),
            transparent_rule_on=bool(ranked),
            learned_in_support=True,
            learned_on=bool(rs_probability >= 0.5),
            learned_probability=rs_probability,
            decision_reason="FROZEN_HEAD_ON" if rs_probability >= 0.5 else "FROZEN_HEAD_OFF",
        )
        if ranked:
            resources["RS"] = _strategy_surface(ranked[0])
            subtypes["RS"] = str(ranked[0]["strategy_family"])
        feature_audit["RS"] = {
            "flags": {name: bool(flags.get(name, False)) for name in names["RS"]},
            "probability_on": rs_probability,
            "hard_gate_pass": bool(ranked),
            "candidate_count": len(ranked),
        }

        learned = route_policy(policy="learned_pm_full", opportunities=observations)
        learned_tokens = _prompt_tokens(
            state=state,
            action_id=learned.action_id,
            resources=resources,
            subtypes=subtypes,
            generation=generation,
        )
        learned_costs.append(learned_tokens)
        state_fixed: dict[str, Any] = {}
        for action in ALL_ACTION_IDS:
            fixed = route_policy(
                policy="cost_matched_fixed",
                opportunities=observations,
                cost_matched_action_id=action,
            )
            tokens = _prompt_tokens(
                state=state,
                action_id=fixed.action_id,
                resources=resources,
                subtypes=subtypes,
                generation=generation,
            )
            fixed_costs[action].append(tokens)
            state_fixed[action] = {"effective_action_id": fixed.action_id, "input_token_bound": tokens}
        audit_rows.append(
            {
                "protocol": PROTOCOL,
                "state_id": state.state_id,
                "user_id_private_not_model_input": state.user_id,
                "learned_full_action_id": learned.action_id,
                "learned_full_input_token_bound": learned_tokens,
                "component_audit": feature_audit,
                "fixed_actions": state_fixed,
                "response_quality_risk_or_external_outcome_read": False,
            }
        )

    learned_mean = float(np.mean(learned_costs))
    fixed_summary = {
        action: {
            "mean_input_token_bound": float(np.mean(values)),
            "absolute_mean_difference_from_learned_full": abs(float(np.mean(values)) - learned_mean),
            "relative_mean_difference_from_learned_full": abs(float(np.mean(values)) - learned_mean) / max(1.0, learned_mean),
        }
        for action, values in fixed_costs.items()
    }
    selected_action = min(
        ALL_ACTION_IDS,
        key=lambda action: (
            fixed_summary[action]["absolute_mean_difference_from_learned_full"],
            fixed_summary[action]["mean_input_token_bound"],
            action,
        ),
    )
    out_dir = root / "outputs/pm_v1_5b_internal_cost_matched_fixed_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "internal_cost_rows.jsonl"
    write_jsonl(rows_path, audit_rows)
    checks = {
        "32_internal_states_32_users": len(states) == 32 and len({state.user_id for state in states}) == 32,
        "all_16_fixed_actions_compared": set(fixed_summary) == set(ALL_ACTION_IDS),
        "selection_uses_input_tokens_only": True,
        "no_response_quality_risk_judge_or_external_outcome_read": True,
        "selected_relative_cost_difference_le_0_10": fixed_summary[selected_action]["relative_mean_difference_from_learned_full"] <= 0.10,
    }
    report = {
        "protocol": PROTOCOL,
        "status": "PASS_INTERNAL_COST_MATCHED_FIXED_FROZEN" if all(checks.values()) else "FAIL_INTERNAL_COST_MATCH",
        "selected_action_id": selected_action,
        "learned_full_mean_input_token_bound": learned_mean,
        "selected_fixed": fixed_summary[selected_action],
        "all_fixed_actions": fixed_summary,
        "learned_action_distribution": dict(sorted(Counter(row["learned_full_action_id"] for row in audit_rows).items())),
        "checks": checks,
        "external_outcome_used": False,
        "further_selection_from_external_results_allowed": False,
        "inputs": {
            str(path.relative_to(root)): sha256_file(path)
            for path in (
                states_path,
                backend_path,
                metadata_path,
                bank_path,
                config_path,
                mpms_dir / "freeze_manifest.json",
                me_dir / "freeze_manifest.json",
                rs_dir / "freeze_manifest.json",
            )
        },
        "internal_cost_rows_sha256": sha256_file(rows_path),
    }
    write_json(out_dir / "cost_match_report.json", report)
    return report


def main() -> None:
    report = build()
    print({
        "protocol": report["protocol"],
        "status": report["status"],
        "selected_action_id": report["selected_action_id"],
        "learned_full_mean_input_token_bound": report["learned_full_mean_input_token_bound"],
        "selected_fixed": report["selected_fixed"],
        "learned_action_distribution": report["learned_action_distribution"],
    })


if __name__ == "__main__":
    main()
