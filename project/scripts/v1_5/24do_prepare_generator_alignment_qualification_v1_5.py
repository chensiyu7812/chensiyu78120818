#!/usr/bin/env python3
"""Freeze a 32-state, 64-call generator resource-alignment qualification."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import (
    DialogueTurn,
    MemoryItem,
    MemorySource,
    RuntimeState,
    SourceCatalog,
)
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
from metacom_pm.prompts import (
    RESOURCE_MATCHED_GENERATION_PROTOCOL,
    generation_messages,
    resource_matched_generation_messages,
)
from metacom_pm.text import estimate_tokens, normalize_space


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-generator-alignment-qualification-plan-v1"
STATUS = "FROZEN_READY_FOR_64_GENERATOR_ALIGNMENT_CALLS"
PRICING_USD_PER_MTOK = {"input": 0.15, "output": 0.60}
ZERO = [0.0] * 64


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _inventory(source: MemorySource | None, count: int = 1) -> dict[MemorySource, SourceCatalog]:
    if source is None:
        return {}
    return {
        source: SourceCatalog(
            available=True,
            count=count,
            min_age_sessions=1,
            max_age_sessions=4,
            estimated_tokens=48 * count,
            catalog_fingerprint=ZERO,
        )
    }


def _state(
    *,
    state_id: str,
    user_id: str,
    current: str,
    history: list[DialogueTurn],
    source: MemorySource | None,
    session_index: int = 5,
    count: int = 1,
) -> RuntimeState:
    inventory = _inventory(source, count=count)
    allowed = ["M0+R0", "M0+RS"]
    if source is not None:
        stem = source.value
        allowed.extend([f"{stem}+R0", f"{stem}+RS"])
    return RuntimeState(
        state_id=state_id,
        card_id="card_" + stable_hex(PROTOCOL, state_id, n=20),
        user_id=user_id,
        split="development",
        semantic_family="generator_alignment_qualification",
        current_user_text=current,
        current_session_history=history,
        current_session_summary="",
        session_index=session_index,
        inventory=inventory,
        allowed_actions=allowed,
        provenance={"protocol": PROTOCOL},
    )


def _rs_rows(root: Path, cards_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    packet_path = root / "outputs/pm_v1_5_h2_retrieval_qualification_v1_candidate/h2_review_packet.jsonl"
    private_path = root / "outputs/pm_v1_5_h2_retrieval_qualification_v1_candidate/private_ranker_audit.jsonl"
    annotation_path = root / "outputs/pm_v1_5_h2_retrieval_qualification_v1/h2_annotations_frozen.jsonl"
    packet = {str(row["review_item_id"]): row for row in _rows(packet_path)}
    private = {str(row["review_item_id"]): row for row in _rows(private_path)}
    annotations = {str(row["review_item_id"]): row for row in _rows(annotation_path)}
    qualified: list[dict[str, Any]] = []
    for review_id in sorted(packet):
        human = annotations[review_id]
        audit = private[review_id]
        ranked = list(audit["transparent_ranked_candidate_numbers"])
        acceptable = set(int(x) for x in human["acceptable_candidate_numbers"])
        hard = set(int(x) for x in human["hard_exclusion_candidate_numbers"])
        if (
            human["rs_opportunity"] != "yes"
            or not ranked
            or ranked[0] not in acceptable
            or ranked[0] in hard
        ):
            continue
        card_id = str(audit["candidate_number_to_card_id"][str(ranked[0])])
        card = cards_by_id[card_id]
        dialogue = list(packet[review_id]["visible_dialogue"])
        current = normalize_space(dialogue[-1]["content"])
        history = [
            DialogueTurn(
                role="user" if row["speaker"] == "seeker" else "assistant",
                content=normalize_space(row["content"]),
            )
            for row in dialogue[:-1]
        ]
        state_id = "genalign_rs_" + stable_hex(PROTOCOL, review_id, n=18)
        qualified.append(
            {
                "component": "RS",
                "state": _state(
                    state_id=state_id,
                    user_id="genalign_" + str(audit["source_dialogue_id"]),
                    current=current,
                    history=history,
                    source=None,
                    session_index=1,
                ),
                "memories": [],
                "strategies": [card],
                "source_selection_basis": "H2 human opportunity=yes and transparent Top-1 acceptable without hard exclusion",
                "source_lineage_id": review_id,
                "resource_subtype": str(card["strategy_family"]),
            }
        )
    qualified.sort(
        key=lambda row: stable_hex(
            PROTOCOL,
            "RS",
            row["resource_subtype"],
            row["source_lineage_id"],
            n=32,
        )
    )
    # Stable round-robin across strategy families.
    chosen: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in qualified:
        by_family.setdefault(row["resource_subtype"], []).append(row)
    while len(chosen) < 8:
        progressed = False
        for family in sorted(by_family):
            if by_family[family]:
                chosen.append(by_family[family].pop(0))
                progressed = True
                if len(chosen) == 8:
                    break
        if not progressed:
            break
    if len(chosen) != 8:
        raise RuntimeError(f"only {len(chosen)} qualified RS states")
    return chosen


MS_CASES = (
    (
        "workload",
        "Several deadlines feel tangled again, and I want help understanding which pressure matters most today.",
        "The prior session separated task volume from uncertainty about an input that another team had not delivered.",
    ),
    (
        "study",
        "My concentration is slipping again while I prepare for an exam, and I want to put the difficulty into words.",
        "The previous session focused on concentration dropping after a late work shift rather than on the study material itself.",
    ),
    (
        "family",
        "I need another conversation at home, but I am worried it will become heated. One small optional idea would help.",
        "In the last session, the user wanted first to explain that they needed a few uninterrupted minutes before discussing solutions.",
    ),
    (
        "friendship",
        "My friend's silence is hurting again, and I am unsure what I want from a conversation with them.",
        "The previous session distinguished wanting clarity about the silence from wanting an immediate apology.",
    ),
    (
        "relocation",
        "This quiet weekend in the new city feels unusually lonely, and I want to understand what is different today.",
        "The last session noted that Sundays felt easier when the user's regular local activity was running.",
    ),
    (
        "grief",
        "A reminder brought the loss back sharply today. Please help me say what I am feeling without turning it into advice.",
        "The prior session stayed with describing the reminder and the feeling it carried; the user did not want problem-solving then.",
    ),
    (
        "caregiving",
        "Caregiving feels heavy again, and I cannot tell which part is draining me most right now.",
        "The previous session identified guilt about taking rest as distinct from the practical caregiving workload.",
    ),
    (
        "shift_sleep",
        "My sleep is unsettled after another rota change, and I want help understanding why this transition feels so difficult.",
        "The last session focused on the abrupt switch from night shifts to early shifts, not on a general inability to sleep.",
    ),
)


def _ms_rows(root: Path) -> list[dict[str, Any]]:
    del root
    rows: list[dict[str, Any]] = []
    for theme, current, text in MS_CASES:
        user_id = "genalign_ms_" + stable_hex(PROTOCOL, "MS", theme, n=14)
        state_id = "genalign_ms_" + stable_hex(
            PROTOCOL, "MS", theme, "state", n=18
        )
        memory = MemoryItem(
            memory_id="mem_" + stable_hex(PROTOCOL, "MS", theme, "memory", n=20),
            source=MemorySource.MS,
            created_session=4,
            text=text,
        )
        rows.append(
            {
                "component": "MS",
                "state": _state(
                    state_id=state_id,
                    user_id=user_id,
                    current=current,
                    history=[
                        DialogueTurn(
                            role="assistant",
                            content="What feels most present for you today?",
                        )
                    ],
                    source=MemorySource.MS,
                ),
                "memories": [memory],
                "strategies": [],
                "source_selection_basis": "content-disjoint controlled positive opportunity fixed before generation",
                "source_lineage_id": theme,
                "resource_subtype": "MS_SESSION",
            }
        )
    return rows


MP_CASES = (
    ("workload", "I keep circling around my workload and want help sorting out what feels hardest.", "MP_PREFERENCE", "Support preference: acknowledge the pressure briefly, then ask no more than one focused question at a time."),
    ("study", "My concentration keeps slipping while I study, and I want to talk through what is getting in the way.", "MP_PREFERENCE", "Support preference: offer choices rather than directives and keep any next step optional."),
    ("family", "I need to talk about tension at home, but I am not sure where to begin.", "MP_PREFERENCE", "Support preference: reflect the concern before exploring it further."),
    ("social_confidence", "I am uncertain what to say to a friend and want help thinking it through.", "MP_PREFERENCE", "Support preference: ask permission before offering suggestions."),
    ("relocation", "Weekends here still feel lonely and unfamiliar, and today it is weighing on me again.", "MP_PROFILE", "Stable profile: the user recently moved to a new city and currently lives alone."),
    ("relationship_change", "The end of the relationship has disrupted my routine, and evenings feel especially empty.", "MP_PROFILE", "Stable profile: the user recently ended a long-term relationship and is rebuilding daily routines."),
    ("caregiving", "I am exhausted from trying to support my parent and still keep up with everything else.", "MP_PROFILE", "Stable profile: the user is the primary caregiver for an older parent."),
    ("shift_work", "My sleep routine is hard to protect, and I feel worn down before work.", "MP_PROFILE", "Stable profile: the user works rotating night shifts."),
)


ME_CASES = (
    ("workload", "Deadlines feel tangled again, and I want help identifying what is making them feel unmanageable.", "In a prior deadline episode, naming the one task with an external dependency made the pressure easier to describe."),
    ("study", "The same evening concentration problem is back while I prepare for an exam. One small optional idea would help.", "During the previous exam period, moving the hardest reading to an earlier hour reduced evening frustration."),
    ("family", "I need to speak with someone at home, but I am worried the conversation will become heated again.", "In the previous family conversation, trying to resolve every disagreement at once caused the discussion to escalate."),
    ("friendship", "I want to message my friend, but I am still unsure what I actually want to say.", "Previously, writing an unsent draft helped the user separate hurt feelings from the request they wanted to make."),
    ("relocation", "Another quiet weekend in the new city is bringing the loneliness back strongly.", "At an earlier check-in, attending one familiar local activity made the new city feel less unfamiliar."),
    ("grief", "A reminder brought the loss back sharply today, and I want help putting the feeling into words.", "During an earlier reminder, naming the specific memory that surfaced helped the user describe the grief without forcing a solution."),
    ("caregiving", "Caregiving feels heavy again today, and I cannot tell which part is draining me most.", "In a prior caregiving week, separating urgent tasks from tasks that could wait created a small amount of recovery time."),
    ("work_transition", "I have another difficult meeting coming up, and the same anxiety is returning.", "After the previous difficult meeting, preparing one opening sentence reduced the uncertainty beforehand."),
)


def _controlled_memory_rows(component: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if component == "MP":
        cases = [
            (theme, current, subtype, text)
            for theme, current, subtype, text in MP_CASES
        ]
        source = MemorySource.MP
    else:
        cases = [
            (theme, current, "ME_EVENT_OUTCOME", text)
            for theme, current, text in ME_CASES
        ]
        source = MemorySource.ME
    for theme, current, subtype, text in cases:
        user_id = "genalign_" + component.lower() + "_" + stable_hex(
            PROTOCOL, component, theme, n=14
        )
        state_id = "genalign_" + component.lower() + "_" + stable_hex(
            PROTOCOL, component, theme, "state", n=18
        )
        memory = MemoryItem(
            memory_id="mem_" + stable_hex(PROTOCOL, component, theme, "memory", n=20),
            source=source,
            created_session=1,
            text=text,
        )
        rows.append(
            {
                "component": component,
                "state": _state(
                    state_id=state_id,
                    user_id=user_id,
                    current=current,
                    history=[
                        DialogueTurn(
                            role="assistant",
                            content="What feels most present for you today?",
                        )
                    ],
                    source=source,
                ),
                "memories": [memory],
                "strategies": [],
                "source_selection_basis": "content-disjoint controlled positive opportunity fixed before generation",
                "source_lineage_id": theme,
                "resource_subtype": subtype,
            }
        )
    return rows


def prepare(
    *,
    root: Path = ROOT,
    pm_config_path: Path | None = None,
    experiment_config_path: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    pm_config_path = pm_config_path or root / "configs/pm_v1_5.yaml"
    experiment_config_path = experiment_config_path or root / "configs/experiment.yaml"
    out_dir = out_dir or root / "outputs/pm_v1_5_generator_alignment_qualification_v1"
    bank_path = root / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl"
    cards_by_id = {str(row["card_id"]): row for row in _rows(bank_path)}
    states = (
        _rs_rows(root, cards_by_id)
        + _controlled_memory_rows("MP")
        + _ms_rows(root)
        + _controlled_memory_rows("ME")
    )
    if Counter(row["component"] for row in states) != Counter(
        {"RS": 8, "MP": 8, "MS": 8, "ME": 8}
    ):
        raise RuntimeError("generator alignment requires 8 states per component")

    pm_config = load_config(pm_config_path)
    experiment = load_config(experiment_config_path)
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, generation.generator_endpoint)
    identity = {
        "base_url": endpoint.base_url,
        "model": endpoint.model,
        "family": endpoint.family,
        "transport": endpoint.transport,
    }

    calls: list[dict[str, Any]] = []
    public_states: list[dict[str, Any]] = []
    private_resources: list[dict[str, Any]] = []
    for row in states:
        state: RuntimeState = row["state"]
        memories: list[MemoryItem] = row["memories"]
        strategies: list[dict[str, Any]] = row["strategies"]
        seed = int(stable_hex(PROTOCOL, state.state_id, "seed", n=8), 16)
        public_states.append(
            {
                "protocol": PROTOCOL,
                "state_id": state.state_id,
                "user_id": state.user_id,
                "component": row["component"],
                "resource_subtype": row["resource_subtype"],
                "source_lineage_id": row["source_lineage_id"],
                "source_selection_basis": row["source_selection_basis"],
                "current_user_text": state.current_user_text,
                "current_session_history": [turn.model_dump(mode="json") for turn in state.current_session_history],
                "selection_uses_response_or_judge_outcome": False,
            }
        )
        private_resources.append(
            {
                "protocol": PROTOCOL,
                "state_id": state.state_id,
                "component": row["component"],
                "runtime_state": state.model_dump(mode="json"),
                "selected_memory_items": [item.model_dump(mode="json") for item in memories],
                "selected_strategy_cards": strategies,
            }
        )
        for variant in ("legacy_generic", "source_matched"):
            messages = (
                generation_messages(
                    state,
                    memories,
                    strategies,
                    system_prompt=generation.system_prompt,
                )
                if variant == "legacy_generic"
                else resource_matched_generation_messages(
                    state,
                    memories,
                    strategies,
                    system_prompt=generation.system_prompt,
                )
            )
            digest = sha256_text(canonical_json(messages))
            calls.append(
                {
                    "protocol": PROTOCOL,
                    "call_id": "genalign_call_" + stable_hex(
                        PROTOCOL, state.state_id, variant, n=22
                    ),
                    "state_id": state.state_id,
                    "user_id": state.user_id,
                    "component": row["component"],
                    "resource_subtype": row["resource_subtype"],
                    "prompt_variant": variant,
                    "messages": messages,
                    "messages_sha256": digest,
                    "prompt_sha256": digest,
                    "selected_memory_ids": [item.memory_id for item in memories],
                    "selected_strategy_card_ids": [str(card["card_id"]) for card in strategies],
                    "selection_uses_response_or_judge_outcome": False,
                    "estimated_input_tokens": estimate_tokens(canonical_json(messages)),
                    "generator_identity": identity,
                    "generation": {**generation.payload(), "seed": seed},
                    "effect_label": "UNKNOWN_GENERATOR_ALIGNMENT_BEFORE_EXECUTION",
                }
            )

    state_counts = Counter(row["state_id"] for row in calls)
    component_counts = Counter(row["component"] for row in public_states)
    input_tokens = sum(int(row["estimated_input_tokens"]) for row in calls)
    output_tokens = len(calls) * generation.max_output_tokens
    checks = {
        "32_independent_states": len(public_states) == 32 and len({row["state_id"] for row in public_states}) == 32,
        "8_states_per_component": component_counts == Counter({"RS": 8, "MP": 8, "MS": 8, "ME": 8}),
        "64_calls_two_per_state": len(calls) == 64 and set(state_counts.values()) == {2},
        "same_seed_within_state": all(len({row["generation"]["seed"] for row in calls if row["state_id"] == state_id}) == 1 for state_id in state_counts),
        "same_resources_within_state": all(len({canonical_json({"memory": row["selected_memory_ids"], "strategy": row["selected_strategy_card_ids"]}) for row in calls if row["state_id"] == state_id}) == 1 for state_id in state_counts),
        "distinct_prompts_within_state": all(len({row["prompt_sha256"] for row in calls if row["state_id"] == state_id}) == 2 for state_id in state_counts),
        "outcome_blind_selection": all(not row["selection_uses_response_or_judge_outcome"] for row in calls),
        "source_matched_protocol_bound": RESOURCE_MATCHED_GENERATION_PROTOCOL == "pm-v1.5-generator-resource-use-alignment-v1",
    }
    if not all(checks.values()):
        raise RuntimeError(f"generator alignment freeze failed: {checks}")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "qualified_states.jsonl", public_states)
    write_jsonl(out_dir / "private_selected_resources.jsonl", private_resources)
    write_jsonl(out_dir / "call_plan.jsonl", calls)
    cost_upper = (
        input_tokens * PRICING_USD_PER_MTOK["input"]
        + output_tokens * PRICING_USD_PER_MTOK["output"]
    ) / 1_000_000
    report = {
        "protocol": PROTOCOL,
        "status": STATUS,
        "api_calls_made": 0,
        "states": len(public_states),
        "planned_calls": len(calls),
        "component_state_counts": dict(sorted(component_counts.items())),
        "estimated_input_tokens": input_tokens,
        "maximum_output_tokens": output_tokens,
        "pricing_usd_per_mtok": PRICING_USD_PER_MTOK,
        "cost_upper_bound_usd": round(cost_upper, 6),
        "generator_identity": identity,
        "checks": checks,
        "inputs": {
            "strategy_bank": {"path": str(bank_path.relative_to(root)), "sha256": sha256_file(bank_path)},
            "generator_alignment_contract": {
                "path": "data/pm_v1_5_contracts/generator_resource_use_alignment_v1.json",
                "sha256": sha256_file(root / "data/pm_v1_5_contracts/generator_resource_use_alignment_v1.json"),
            },
        },
    }
    write_json(out_dir / "generation_plan_report.json", report)
    write_json(
        out_dir / "freeze_manifest.json",
        {
            "protocol": PROTOCOL,
            "status": STATUS,
            "generator_identity": identity,
            "supporter_generation_treatment_sha256": generation.digest(),
            "outputs": {
                name: sha256_file(out_dir / name)
                for name in (
                    "qualified_states.jsonl",
                    "private_selected_resources.jsonl",
                    "call_plan.jsonl",
                    "generation_plan_report.json",
                )
            },
            "outcome_blind": True,
            "api_calls_made": 0,
        },
    )
    return report


def main() -> None:
    report = prepare()
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "protocol",
                    "status",
                    "states",
                    "planned_calls",
                    "component_state_counts",
                    "cost_upper_bound_usd",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
