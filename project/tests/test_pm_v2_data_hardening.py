from __future__ import annotations

import json
import importlib.util
import inspect
import os
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from pydantic import ValidationError
import yaml

from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.api import CallResult, Endpoint, StructuredOutputValidationError
from metacom_pm import pm_v2_data as pm_v2_data_module
from metacom_pm.contracts import (
    DialogueTurn,
    MemoryItem,
    MemorySource,
    RuntimeState,
    StrategyCard,
)
from metacom_pm.attempt_ledger import PersistentAttemptLedger
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.evoemo import (
    _catalog,
    build_evo_memory,
    load_evoemo,
    make_evo_runtime_state,
)
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.pm_v2_contracts import (
    ActionLabel,
    CompositeSpec,
    PMV2Split,
    PMV2State,
    ResourceNeedRegime,
    ResponseDimensions,
    RiskDimensions,
)
from metacom_pm.pm_v2_audit import _regime_pass
from metacom_pm.pm_v2_data import (
    DATA_GENERATION_CONTRACT_VERSION,
    GENERATION_CASE_FIELDS,
    GENERATION_FAMILY_TOPICS,
    READINESS_SURFACE_PROTOCOL,
    GenerationDraftCompilationError,
    GeneratedBundleDraft,
    GeneratedMemory,
    GeneratedSurfaceOnlyCaseDraft,
    GeneratedStateCase,
    GeneratedUserBundle,
    advice_readiness_target_for_case,
    audit_cross_split_near_duplicates,
    bind_bundle_to_generation_run,
    build_evaluator_context_index,
    build_deployable_catalog_statistics,
    case_to_evaluator_context,
    case_to_memory_backend,
    case_to_state,
    compile_surface_only_user_bundle,
    compile_generation_draft,
    compiler_surface_from_provider,
    evaluator_context_payload_sha256,
    generate_user_bundle,
    generation_case_family_assignments,
    generation_case_messages,
    generation_distractor_family_assignments,
    generation_messages,
    lint_generation_surface_case,
    readiness_surface_clause_for_case,
    require_bundle_generation_binding,
    runtime_to_pmv2_state,
    state_to_v1_runtime,
    validate_generation_shortcut_controls,
    validate_split_manifests,
    write_development_dataset,
)
from metacom_pm.pm_v2_features import PMV2FeatureBuilder
from metacom_pm.pm_v1_5_semantic import (
    FrozenSemanticEncoderSpec,
    FrozenTransformerSemanticEncoder,
    SemanticEncoderBinding,
    prepare_visible_semantic_state,
    visible_dialogue_state_text,
)
from metacom_pm.pm_v1_5_required_hit import validate_required_hit_preflight
from metacom_pm.pm_v2_judging import prompt_contract_hash
from metacom_pm.pm_v2_generation_pilot import (
    CALIBRATION_SEMANTIC_FAMILIES,
    GENERATION_PILOT_CONTRACT_VERSION,
    GENERATION_PILOT_FAMILIES,
    GENERATION_PILOT_MAX_ATTEMPTS,
    GENERATION_PILOT_MINIMUM_CALLS,
    GENERATION_PILOT_STAGE,
    INTERNAL_TEST_SEMANTIC_FAMILIES,
    TRAIN_SEMANTIC_FAMILIES,
    build_generation_compatibility_contract,
    build_generation_compatibility_plan,
    generation_family_schedule,
    require_generation_compatibility_attestation,
    validate_generation_pilot_bundle,
)
from metacom_pm.pm_v2_generation_review import (
    PACKET_FIELDS as GENERATION_REVIEW_PACKET_FIELDS,
    RATING_FIELDS as GENERATION_REVIEW_RATING_FIELDS,
    SIMPLE_REVIEW_FIELDS as GENERATION_SIMPLE_REVIEW_FIELDS,
    analyze_generation_pilot_semantic_review,
    prepare_generation_pilot_semantic_review,
    require_generation_pilot_semantic_review,
)
from metacom_pm.text import conservative_token_bound, estimate_tokens


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_conservative_token_bound_applies_frozen_margin() -> None:
    assert estimate_tokens("x" * 8) == 2
    assert conservative_token_bound("x" * 8, safety_factor=1.5) == 3
    with pytest.raises(ValueError, match="safety factor"):
        conservative_token_bound("x", safety_factor=0.99)


def test_generate_user_bundle_defaults_to_single_physical_attempt() -> None:
    signature = inspect.signature(generate_user_bundle)
    assert signature.parameters["request_retries"].default == 1


def test_semantic_family_schedule_counterbalances_regime_positions() -> None:
    split_design = {
        "train": (24, TRAIN_SEMANTIC_FAMILIES),
        "calibration": (12, CALIBRATION_SEMANTIC_FAMILIES),
        "internal_test": (16, INTERNAL_TEST_SEMANTIC_FAMILIES),
    }
    for split, (user_count, expected_families) in split_design.items():
        schedule = generation_family_schedule(split, user_count)
        assert len(schedule) == user_count
        observed_positions = {family: set() for family in expected_families}
        observed_regimes = {family: set() for family in expected_families}
        for cohort in schedule:
            assert len(cohort) == 3
            assert len(set(cohort)) == 3
            for position, family in enumerate(cohort):
                observed_positions[family].add(position)
            for case_index, (_, regime) in enumerate(GENERATION_CASE_FIELDS):
                observed_regimes[cohort[case_index % 3]].add(regime)
        assert set(observed_positions) == set(expected_families)
        assert all(positions == {0, 1, 2} for positions in observed_positions.values())
        assert all(
            regimes == set(ResourceNeedRegime)
            for regimes in observed_regimes.values()
        )


def test_generation_pilot_uses_a_real_frozen_orthogonal_cohort() -> None:
    assert GENERATION_PILOT_CONTRACT_VERSION.startswith(
        "pm-v2-generation-compatibility-pilot-v8.7-"
    )
    assert GENERATION_PILOT_FAMILIES in (
        ("relocation_loneliness", "academic_pressure", "trust_rebuilding"),
        ("relocation_loneliness", "self_confidence", "sleep_disruption"),
        ("academic_pressure", "trust_rebuilding", "sleep_disruption"),
        ("self_confidence", "academic_pressure", "relocation_loneliness"),
    )


def _memory(
    source: MemorySource,
    index: int,
    *,
    created_session: int,
    text: str,
    stale: bool = False,
    conflict: bool = False,
    item_utility: str = "irrelevant",
) -> GeneratedMemory:
    return GeneratedMemory(
        memory_id=f"generated_{source.value}_{index}",
        source=source,
        text=text,
        created_session=created_session,
        stale=stale,
        conflicts_with_current_state=conflict,
        private_sensitivity="ordinary",
        item_utility=item_utility,
    )


def _case(
    *,
    case_id: str = "case_1",
    regime: ResourceNeedRegime = ResourceNeedRegime.CONTEXT_ONLY,
    current_user_text: str = "I feel torn about tomorrow's conversation.",
    needed_memory_sources: list[MemorySource] | None = None,
) -> GeneratedStateCase:
    default_needs = {
        ResourceNeedRegime.PROFILE_NEEDED: [MemorySource.MP],
        ResourceNeedRegime.SUMMARY_NEEDED: [MemorySource.MS],
        ResourceNeedRegime.EVENT_NEEDED: [MemorySource.ME],
        ResourceNeedRegime.MULTI_SOURCE_NEEDED: [MemorySource.MS, MemorySource.ME],
    }
    needed_sources = (
        needed_memory_sources
        if needed_memory_sources is not None
        else default_needs.get(regime, [])
    )
    return GeneratedStateCase(
        case_id=case_id,
        semantic_family="workplace_conflict",
        surface_form_id=f"surface_{case_id}",
        regime=regime,
        current_user_text=current_user_text,
        recent_dialogue=[
            DialogueTurn(role="user", content="The meeting was moved again."),
            DialogueTurn(role="assistant", content="That uncertainty sounds tiring."),
        ],
        session_summary="The user is preparing for a difficult workplace conversation.",
        session_index=6,
        profile_memories=[
            _memory(
                MemorySource.MP,
                1,
                created_session=1,
                text="The user prefers time to think before difficult conversations.",
                item_utility=(
                    "harmful"
                    if regime is ResourceNeedRegime.MEMORY_HARMFUL
                    else "helpful"
                    if MemorySource.MP in needed_sources
                    else "irrelevant"
                ),
            ),
            _memory(
                MemorySource.MP,
                2,
                created_session=2,
                text="The user values direct but calm communication.",
            ),
        ],
        summary_memories=[
            _memory(
                MemorySource.MS,
                1,
                created_session=3,
                text="Recent work uncertainty has made decision-making harder.",
                stale=MemorySource.MS not in needed_sources,
                item_utility=(
                    "helpful" if MemorySource.MS in needed_sources else "irrelevant"
                ),
            ),
            _memory(
                MemorySource.MS,
                2,
                created_session=2,
                text="The user once considered reorganizing a bookshelf.",
            ),
        ],
        event_memories=[
            _memory(
                MemorySource.ME,
                1,
                created_session=4,
                text="A manager postponed an important feedback meeting.",
                conflict=MemorySource.ME not in needed_sources,
                item_utility=(
                    "helpful" if MemorySource.ME in needed_sources else "irrelevant"
                ),
            ),
            _memory(
                MemorySource.ME,
                2,
                created_session=2,
                text="The user attended a neighborhood event last year.",
            ),
        ],
        needed_memory_sources=needed_sources,
        authorized_user_context="PRIVATE_AUTHORIZED_CONTEXT_SENTINEL",
        coverage_rationale="PRIVATE_COVERAGE_RATIONALE_SENTINEL",
    )


def _bundle() -> GeneratedUserBundle:
    regimes = [
        ResourceNeedRegime.CONTEXT_ONLY,
        ResourceNeedRegime.EVENT_NEEDED,
        ResourceNeedRegime.STRATEGY_HELPFUL,
        ResourceNeedRegime.STRATEGY_HARMFUL,
    ]
    return GeneratedUserBundle(
        user_id="user_1",
        profile_summary="A privacy-safe synthetic user.",
        stable_preferences=["calm communication"],
        boundaries=["no diagnosis"],
        cases=[
            _case(
                case_id=f"case_{index}",
                regime=regime,
                current_user_text=f"I feel torn about conversation number {index}.",
            )
            for index, regime in enumerate(regimes, 1)
        ],
        generator_seed_id="seed_1",
    )


def _full_regime_bundle(
    *,
    user_id: str = "full_regime_user",
    observed_families: tuple[str, ...] = ("family_a", "family_b", "family_c"),
) -> GeneratedUserBundle:
    cases = []
    for index, regime in enumerate(ResourceNeedRegime):
        case = _case(
            case_id=f"{user_id}_case_{index}",
            regime=regime,
            current_user_text=f"Unique current user text {user_id} number {index}.",
        )
        case.semantic_family = observed_families[index % len(observed_families)]
        cases.append(case)
    return GeneratedUserBundle(
        user_id=user_id,
        profile_summary="A privacy-safe synthetic full-regime user.",
        stable_preferences=["calm communication"],
        boundaries=["no diagnosis"],
        cases=cases,
        generator_seed_id=f"seed_{user_id}",
    )


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _write_unique_generation_seeds(path: Path, count: int = 53) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {"dialogue_text": f"user: unique held-out seed dialogue {index}."}
            )
            + "\n"
            for index in range(count)
        ),
        encoding="utf-8",
    )


def _role_slot_draft(
    families: list[str] | None = None,
) -> GeneratedBundleDraft:
    families = families or [
        "relocation_loneliness",
        "workload_burnout",
        "friendship_distance",
    ]
    regimes = list(ResourceNeedRegime)
    targets = generation_case_family_assignments(families, regimes)
    distractors = generation_distractor_family_assignments(families, regimes)
    topic = {family: GENERATION_FAMILY_TOPICS[family] for family in families}

    def item(source: MemorySource, family: str, label: str) -> dict:
        content = topic[family]
        common = {
            "relevance_explanation": f"This {label} fragment has the frozen topic.",
            "private_sensitivity": "ordinary",
        }
        if source is MemorySource.MP:
            return {"stable_user_fact": f"prefers calm reflection about {content}", **common}
        if source is MemorySource.MS:
            return {"cross_session_pattern": f"often reports strain around {content}", **common}
        return {"concrete_past_event": f"last month described one incident involving {content}", **common}

    needed_by_case = {
        "profile_needed": {MemorySource.MP},
        "summary_needed": {MemorySource.MS},
        "event_needed": {MemorySource.ME},
        "multi_source_needed": set(MemorySource),
    }
    payload: dict = {
        "profile_summary": "A user navigating "
        + ", ".join(topic[family] for family in families)
        + ".",
        "stable_preferences": [f"calm reflection about {topic[families[0]]}"],
        "boundaries": [
            "avoid assumptions about "
            + " or ".join(topic[family] for family in families[1:])
        ],
    }
    for index, (case_name, regime) in enumerate(
        (
            ("context_only", ResourceNeedRegime.CONTEXT_ONLY),
            ("profile_needed", ResourceNeedRegime.PROFILE_NEEDED),
            ("summary_needed", ResourceNeedRegime.SUMMARY_NEEDED),
            ("event_needed", ResourceNeedRegime.EVENT_NEEDED),
            ("multi_source_needed", ResourceNeedRegime.MULTI_SOURCE_NEEDED),
            ("memory_harmful", ResourceNeedRegime.MEMORY_HARMFUL),
            ("strategy_helpful", ResourceNeedRegime.STRATEGY_HELPFUL),
            ("strategy_harmful", ResourceNeedRegime.STRATEGY_HARMFUL),
            ("ambiguous", ResourceNeedRegime.AMBIGUOUS),
        ),
        1,
    ):
        target = targets[case_name]
        current_user_text = f"Concern {index} now feels tied to {topic[target]}."
        if regime is ResourceNeedRegime.STRATEGY_HELPFUL:
            current_user_text = (
                f"I want guidance about {topic[target]}; what should I do next?"
            )
        elif regime is ResourceNeedRegime.STRATEGY_HARMFUL:
            current_user_text = (
                f"Please just listen without advice while I describe {topic[target]}."
            )
        surface = {
            "current_user_text": current_user_text,
            "dialogue_before_current": [
                {"role": "user", "content": f"Earlier context number {index}."},
                {"role": "assistant", "content": f"Prior reflection number {index}."},
            ],
            "session_summary": f"The current concern centers on {topic[target]}.",
            "authorized_user_context": f"Authorized context about {topic[target]}.",
            "coverage_rationale": (
                "A structured plan and steps would help organize support."
                if regime is ResourceNeedRegime.STRATEGY_HELPFUL
                else "Directive advice would be premature; listening is safer."
                if regime is ResourceNeedRegime.STRATEGY_HARMFUL
                else f"Coverage is frozen for case {index}."
            ),
        }
        if regime is ResourceNeedRegime.MEMORY_HARMFUL:
            off_topic = distractors[case_name]["event_source"]
            surface.update(
                {
                    "profile_source": {
                        "outdated_user_fact": f"avoids talking about {topic[target]}",
                        "explicit_current_update": f"now openly discusses {topic[target]}",
                        "why_recall_is_harmful": "The old preference conflicts with the correction.",
                    },
                    "summary_source": {
                        "outdated_cross_session_pattern": f"repeatedly withdrew around {topic[target]}",
                        "explicit_current_update": f"now stays engaged around {topic[target]}",
                        "why_recall_is_harmful": "The old pattern conflicts with current context.",
                    },
                    "event_source": {
                        "unrelated_sensitive_past_event": f"a private incident involving {topic[off_topic]}",
                        "why_recall_is_intrusive": "It is unrelated and sensitive.",
                    },
                }
            )
        else:
            for source_field, source in (
                ("profile_source", MemorySource.MP),
                ("summary_source", MemorySource.MS),
                ("event_source", MemorySource.ME),
            ):
                source_payload = {
                    "distractor": item(
                        source,
                        distractors[case_name][source_field],
                        "distractor",
                    )
                }
                if source in needed_by_case.get(case_name, set()):
                    source_payload["helpful"] = item(source, target, "helpful")
                surface[source_field] = source_payload
        if regime is ResourceNeedRegime.MULTI_SOURCE_NEEDED:
            surface.update(
                {
                    "profile_unique_contribution": "A durable personal boundary.",
                    "summary_unique_contribution": "A recurring cross-session pattern.",
                    "event_unique_contribution": "A concrete prior incident.",
                }
            )
        payload[case_name] = surface
    return GeneratedBundleDraft.model_validate(payload)


def _source_grounded_bundle(
    *, user_id: str, families: list[str] | None = None
) -> GeneratedUserBundle:
    selected_families = families or [
        "relocation_loneliness",
        "workload_burnout",
        "friendship_distance",
    ]
    draft = _role_slot_draft(selected_families)
    bundle = compile_generation_draft(
        draft=draft,
        seed_dialogue="fixture held-out seed",
        user_id=user_id,
        semantic_families=selected_families,
        regimes=list(ResourceNeedRegime),
    )
    provider_draft = draft.model_dump(mode="json")
    provider_response = {
        "id": "fixture-source-grounded-provider-response",
        "choices": [
            {"message": {"content": canonical_json(provider_draft)}}
        ],
    }
    bundle.provenance.update(
        {
            "provider_draft": provider_draft,
            "provider_response": provider_response,
            "provider_response_sha256": sha256_text(
                canonical_json(provider_response)
            ),
        }
    )
    return bundle


def _source_grounded_pilot_bundle(
    contract: dict,
    *,
    request_hash: str,
) -> GeneratedUserBundle:
    bundle = _source_grounded_bundle(
        user_id=str(contract["pilot_user_id"]),
        families=[str(value) for value in contract["semantic_families"]],
    )
    bundle.provenance.update(
        {
            "generation_compatibility_contract_sha256": contract[
                "contract_sha256"
            ],
            "request_hash": request_hash,
        }
    )
    return bundle


def _surface_only_pilot_inputs(
    contract: dict,
) -> dict[str, GeneratedSurfaceOnlyCaseDraft]:
    """Project the legacy fixture onto the role-safe provider schema."""

    draft = _role_slot_draft(
        [str(value) for value in contract["semantic_families"]]
    )
    surfaces = {}
    for case_field, _ in GENERATION_CASE_FIELDS:
        compiled = getattr(draft, case_field)
        turns = compiled.dialogue_before_current
        assert len(turns) % 2 == 0
        assert all(
            turns[index].role == "user" and turns[index + 1].role == "assistant"
            for index in range(0, len(turns), 2)
        )
        surfaces[case_field] = GeneratedSurfaceOnlyCaseDraft.model_validate(
            {
                "current_user_text": compiled.current_user_text,
                "dialogue_exchanges_before_current": [
                    {
                        "user_text": turns[index].content,
                        "assistant_text": turns[index + 1].content,
                    }
                    for index in range(0, len(turns), 2)
                ],
                "session_summary": compiled.session_summary,
                "authorized_user_context": compiled.authorized_user_context,
            }
        )
    return surfaces


def test_surface_provider_schema_makes_role_order_a_compiler_invariant() -> None:
    surface = GeneratedSurfaceOnlyCaseDraft.model_validate(
        {
            "current_user_text": "The move still feels lonely today.",
            "dialogue_exchanges_before_current": [
                {
                    "user_text": "I have not met anyone in the new city yet.",
                    "assistant_text": "That sounds isolating; what feels hardest?",
                },
                {
                    "user_text": "Evenings feel especially quiet.",
                    "assistant_text": "The quiet evenings seem to make this sharper.",
                },
            ],
            "session_summary": "The user feels lonely after moving to a new city.",
            "authorized_user_context": "Use only the visible relocation details.",
        }
    )
    compiled = compiler_surface_from_provider(surface)
    assert [turn.role for turn in compiled.dialogue_before_current] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert compiled.dialogue_before_current[-1].role == "assistant"
    lint = lint_generation_surface_case(
        case_field="context_only",
        regime=ResourceNeedRegime.CONTEXT_ONLY,
        family="relocation_loneliness",
        forbidden_families=("self_confidence", "sleep_disruption"),
        surface=surface,
    )
    assert lint["status"] == "PASS"
    assert lint["errors"] == []
    schema = GeneratedSurfaceOnlyCaseDraft.model_json_schema()
    assert "dialogue_exchanges_before_current" in schema["properties"]
    assert "dialogue_before_current" not in schema["properties"]


def _surface_only_pilot_bundle(
    contract: dict,
    *,
    request_hash_prefix: str = "test-surface-request",
) -> GeneratedUserBundle:
    surfaces = _surface_only_pilot_inputs(contract)
    assignments = generation_case_family_assignments(
        [str(value) for value in contract["semantic_families"]],
        list(ResourceNeedRegime),
    )
    accepted_calls: dict[str, CallResult] = {}
    accepted_messages: dict[str, list[dict[str, str]]] = {}
    for index, (case_field, regime) in enumerate(GENERATION_CASE_FIELDS):
        surface_payload = surfaces[case_field].model_dump(mode="json")
        accepted_calls[case_field] = CallResult(
            text=canonical_json(surface_payload),
            raw_response={
                "choices": [
                    {"message": {"content": canonical_json(surface_payload)}}
                ]
            },
            usage={
                "prompt_tokens": 100 + index,
                "completion_tokens": 50,
                "total_tokens": 150 + index,
            },
            latency_ms=1.0,
            request_hash=f"{request_hash_prefix}-{case_field}",
        )
        accepted_messages[case_field] = generation_case_messages(
            seed_dialogue="fixture held-out seed",
            user_id=str(contract["pilot_user_id"]),
            case_field=case_field,
            regime=regime,
            semantic_family=assignments[case_field],
            forbidden_families=[
                str(value)
                for value in contract["semantic_families"]
                if str(value) != assignments[case_field]
            ],
        )
    bundle = compile_surface_only_user_bundle(
        surfaces=surfaces,
        accepted_calls=accepted_calls,
        accepted_messages=accepted_messages,
        accepted_attempt_kinds={
            case_field: "initial" for case_field, _ in GENERATION_CASE_FIELDS
        },
        seed_dialogue="fixture held-out seed",
        user_id=str(contract["pilot_user_id"]),
        semantic_families=[
            str(value) for value in contract["semantic_families"]
        ],
        regimes=list(ResourceNeedRegime),
        generator_model=str(contract["endpoint"]["model"]),
        generator_family=contract["endpoint"].get("family"),
    )
    bundle.provenance["generation_compatibility_contract_sha256"] = contract[
        "contract_sha256"
    ]
    return bundle


def test_role_slot_compiler_guarantees_structural_bundle_without_self_reported_labels() -> None:
    regimes = list(ResourceNeedRegime)
    families = [
        "relocation_loneliness",
        "workload_burnout",
        "friendship_distance",
    ]
    draft = _role_slot_draft()
    bundle = compile_generation_draft(
        draft=draft,
        seed_dialogue="held-out seed",
        user_id="compiled_user",
        semantic_families=families,
        regimes=regimes,
    )
    assert len(bundle.cases) == 9
    assert {case.regime for case in bundle.cases} == set(regimes)
    assert {case.semantic_family for case in bundle.cases} == {
        *families,
    }
    assert all(
        case.profile_memories and case.summary_memories and case.event_memories
        for case in bundle.cases
    )
    assert all(
        len(case.profile_memories)
        == len(case.summary_memories)
        == len(case.event_memories)
        == 2
        for case in bundle.cases
    )
    for case in bundle.cases:
        topic = GENERATION_FAMILY_TOPICS[case.semantic_family].casefold()
        pools = {
            MemorySource.MP: case.profile_memories,
            MemorySource.MS: case.summary_memories,
            MemorySource.ME: case.event_memories,
        }
        for source, memories in pools.items():
            if source in set(case.needed_memory_sources):
                continue
            assert any(
                memory.item_utility == "irrelevant"
                and topic in memory.text.casefold()
                for memory in memories
            ), (
                "every non-needed source must retain a same-topic semantic decoy; "
                f"missing for {case.case_id}/{source.value}"
            )
    assert all(
        memory.created_session < case.session_index
        for case in bundle.cases
        for memory in (
            *case.profile_memories,
            *case.summary_memories,
            *case.event_memories,
        )
    )
    by_regime = {case.regime: case for case in bundle.cases}
    assert by_regime[ResourceNeedRegime.PROFILE_NEEDED].needed_memory_sources == [
        MemorySource.MP
    ]
    assert by_regime[ResourceNeedRegime.MULTI_SOURCE_NEEDED].needed_memory_sources == [
        MemorySource.MP,
        MemorySource.MS,
        MemorySource.ME,
    ]
    harmful = by_regime[ResourceNeedRegime.MEMORY_HARMFUL]
    assert harmful.needed_memory_sources == []
    assert sum(
        memory.item_utility == "harmful"
        for memory in (
            *harmful.profile_memories,
            *harmful.summary_memories,
            *harmful.event_memories,
        )
    ) == 3
    strategy_use = by_regime[ResourceNeedRegime.STRATEGY_HELPFUL]
    strategy_skip = by_regime[ResourceNeedRegime.STRATEGY_HARMFUL]
    assert strategy_use.strategy_resource_target == "use"
    assert strategy_skip.strategy_resource_target == "skip"
    assert {
        strategy_use.advice_readiness_target,
        strategy_skip.advice_readiness_target,
    } == {"listen_only", "light_suggestion"}
    assert bundle.provenance["generation_structure"] == (
        DATA_GENERATION_CONTRACT_VERSION
    )
    assert bundle.provenance["readiness_surface_protocol"] == (
        READINESS_SURFACE_PROTOCOL
    )
    for case in (strategy_use, strategy_skip):
        expected_clause = readiness_surface_clause_for_case(
            user_id=bundle.user_id, regime=case.regime
        )
        assert expected_clause
        assert case.current_user_text.endswith(expected_clause)
    assert harmful.current_user_text.count(". I used to avoid") == 1
    assert bundle.provenance["provider_memory_slots_ignored"] is True
    assert bundle.provenance["provider_coverage_rationale_ignored"] is True
    assert bundle.provenance["surface_selection"]["compiled_surface_lint"][
        "status"
    ] == "PASS"
    assert bundle.provenance["deterministic_draft_lint"]["status"] == "PASS"
    assert len(
        {
            case.session_index - memory.created_session
            for case in bundle.cases
            for memory in (
                *case.profile_memories,
                *case.summary_memories,
                *case.event_memories,
            )
        }
    ) >= 6
    messages = generation_messages(
        seed_dialogue="held-out seed",
        user_id="compiled_user",
        semantic_families=families,
        regimes=regimes,
    )
    assert "GeneratedBundleDraft" in messages[-1]["content"]
    assert "do not output this ID" in messages[-1]["content"]


def test_visible_readiness_is_varied_and_counterbalanced_against_strategy_target() -> None:
    observations = set()
    assignments = set()
    for index in range(1, 53):
        user_id = f"pmv2_train_u{index:03d}"
        use_target = advice_readiness_target_for_case(
            user_id=user_id, regime=ResourceNeedRegime.STRATEGY_HELPFUL
        )
        skip_target = advice_readiness_target_for_case(
            user_id=user_id, regime=ResourceNeedRegime.STRATEGY_HARMFUL
        )
        assignments.add((use_target, skip_target))
        for regime in (
            ResourceNeedRegime.STRATEGY_HELPFUL,
            ResourceNeedRegime.STRATEGY_HARMFUL,
        ):
            clause = readiness_surface_clause_for_case(
                user_id=user_id, regime=regime
            )
            assert clause
            observations.add(clause)
    assert assignments == {
        ("listen_only", "light_suggestion"),
        ("light_suggestion", "listen_only"),
    }
    assert len(observations) >= 8

def test_generation_session_and_age_metadata_cannot_encode_fixed_regime() -> None:
    bundles = [
        _source_grounded_bundle(user_id=f"pmv2_train_u{index:03d}")
        for index in range(1, 10)
    ]
    split_by_user = {
        bundle.user_id: PMV2Split.TRAIN for bundle in bundles
    }
    report = validate_generation_shortcut_controls(
        bundles=bundles,
        split_by_user=split_by_user,
    )
    assert report["status"] == "PASS"
    assert report["case_ids_hide_regime"] is True
    assert report["memory_age_multiset_ignores_utility"] is True
    for counts in report["splits"]["train"].values():
        assert counts == [1] * 9

    tampered = bundles[0].model_copy(deep=True)
    case = tampered.cases[0]
    case.profile_memories[0].created_session -= 1
    with pytest.raises(ValueError, match="ages reveal item utility"):
        validate_generation_shortcut_controls(
            bundles=[tampered],
            split_by_user={tampered.user_id: PMV2Split.TRAIN},
        )


def test_paid_role_slot_surface_drift_uses_recorded_case_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _role_slot_draft().model_dump(mode="json")
    payload["ambiguous"]["current_user_text"] = payload["context_only"][
        "current_user_text"
    ]
    draft = GeneratedBundleDraft.model_validate(payload)
    paid_call = CallResult(
        text=canonical_json(payload),
        raw_response={"id": "paid-role-slot-response"},
        usage={
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
        },
        latency_ms=10.0,
        request_hash="paid-role-slot-request",
    )

    class FakeClient:
        def chat(self, *args, **kwargs):
            assert kwargs["response_schema"] is GeneratedBundleDraft
            return paid_call, draft

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        pm_v2_data_module, "make_client", lambda endpoint: FakeClient()
    )
    endpoint = Endpoint(
        base_url="https://api.openai.com",
        model="gpt-4o-mini",
        api_key_env="NOT_READ_BY_FAKE",
        family="openai_gpt4o",
    )
    bundle = generate_user_bundle(
        endpoint=endpoint,
        seed_dialogue="held-out seed",
        user_id="compiled_user",
        semantic_families=[
            "relocation_loneliness",
            "workload_burnout",
            "friendship_distance",
        ],
        regimes=list(ResourceNeedRegime),
        seed=7,
    )
    selection = bundle.provenance["surface_selection"]
    assert selection["provider_surface_lint"]["status"] == "FAIL"
    assert selection["compiled_surface_lint"]["status"] == "PASS"
    assert "ambiguous" in selection["fallback_cases"]
    assert bundle.provenance["provider_response"] == {
        "id": "paid-role-slot-response"
    }
    assert bundle.provenance["reported_usage"]["total_tokens"] == 300
    assert bundle.provenance["provider_memory_slots_ignored"] is True
    assert bundle.provenance["provider_coverage_rationale_ignored"] is True


def test_all_local_surface_fallbacks_fit_provider_transport_schema() -> None:
    for index, regime in enumerate(ResourceNeedRegime):
        surface = pm_v2_data_module._fallback_generation_surface(
            case_field=f"case_{index}",
            regime=regime,
            family="relocation_loneliness",
        )
        assert (
            surface.coverage_rationale
            == pm_v2_data_module.LOCAL_FALLBACK_COVERAGE_PLACEHOLDER
        )
        assert len(surface.coverage_rationale) <= 180

    # Evaluator-only rationales are compiled into GeneratedStateCase and are
    # intentionally independent from the provider transport field's size cap.
    assert len(
        pm_v2_data_module._deterministic_coverage_rationale(
            ResourceNeedRegime.CONTEXT_ONLY
        )
    ) > 180


def test_surface_lint_does_not_treat_move_forward_as_relocation() -> None:
    assert not pm_v2_data_module._family_anchor_hits(
        "I am not sure how to move forward after the disagreement.",
        "relocation_loneliness",
    )
    assert pm_v2_data_module._family_anchor_hits(
        "I moved to a new city last week.",
        "relocation_loneliness",
    )


def test_generated_case_rejects_current_or_future_memory() -> None:
    case = _case().model_dump()
    case["event_memories"][0]["created_session"] = case["session_index"]
    with pytest.raises(ValidationError, match="not strictly prior"):
        GeneratedStateCase.model_validate(case)

    with pytest.raises(ValueError, match="current or a future session"):
        build_deployable_catalog_statistics(
            texts=["future memory"],
            created_sessions=[6],
            session_index=6,
        )


def test_generated_case_requires_structured_regime_memory_need() -> None:
    assert "strategy_catalog_count" not in GeneratedStateCase.model_fields
    assert "strategy_estimated_tokens" not in GeneratedStateCase.model_fields
    profile = _case(regime=ResourceNeedRegime.PROFILE_NEEDED).model_dump()
    profile["needed_memory_sources"] = []
    with pytest.raises(ValidationError, match="profile_needed requires"):
        GeneratedStateCase.model_validate(profile)

    multi = _case(regime=ResourceNeedRegime.MULTI_SOURCE_NEEDED).model_dump()
    multi["needed_memory_sources"] = [MemorySource.ME]
    with pytest.raises(ValidationError, match="two or three complementary"):
        GeneratedStateCase.model_validate(multi)


def test_private_context_is_physically_separate_from_model_state() -> None:
    case = _case()
    state = case_to_state(
        user_id="user_1",
        case=case,
        split=PMV2Split.TRAIN,
        strategy_catalog_count=12429,
        strategy_estimated_tokens=260,
        bundle_provenance={
            "generator_model": "generator-model",
            "authorized_user_context": "SHOULD_BE_DROPPED",
            "memory_items": {"MP": ["SHOULD_BE_DROPPED"]},
        },
    )
    serialized_state = canonical_json(state.model_dump(mode="json"))
    for forbidden in (
        "PRIVATE_AUTHORIZED_CONTEXT_SENTINEL",
        "PRIVATE_COVERAGE_RATIONALE_SENTINEL",
        "SHOULD_BE_DROPPED",
        "authorized_user_context",
        "coverage_rationale",
        "memory_items",
        case.profile_memories[0].text,
        '"regime"',
    ):
        assert forbidden not in serialized_state

    backend = case_to_memory_backend(state, case)
    evaluator = case_to_evaluator_context(state, case)
    assert case.profile_memories[0].text in {item.text for item in backend.items}
    assert evaluator["authorized_user_context"] == case.authorized_user_context
    assert evaluator["coverage_rationale"] == case.coverage_rationale
    assert evaluator["regime"] == case.regime.value
    assert evaluator["needed_memory_sources"] == []
    assert {row["memory_id"] for row in evaluator["memory_annotations"]} == {
        item.memory_id for item in backend.items
    }
    assert all("text" not in row for row in evaluator["memory_annotations"])


def test_evaluator_context_index_rejects_hash_map_and_future_tampering() -> None:
    case = _case()
    state = case_to_state(
        user_id="user_1",
        case=case,
        split=PMV2Split.TRAIN,
        strategy_catalog_count=12429,
        strategy_estimated_tokens=260,
    )
    row = case_to_evaluator_context(state, case)
    index = build_evaluator_context_index([row], states=[state])
    assert index.require_states([state], exact=True)[state.state_id]["regime"] == (
        ResourceNeedRegime.CONTEXT_ONLY.value
    )
    assert len(index.map_sha256) == 64

    hash_tampered = json.loads(json.dumps(row))
    hash_tampered["authorized_user_context"] = "tampered"
    with pytest.raises(RuntimeError, match="payload hash mismatch"):
        build_evaluator_context_index([hash_tampered], states=[state])

    future = json.loads(json.dumps(row))
    future["memory_annotations"][0]["created_session"] = state.session_index
    future["context_payload_sha256"] = evaluator_context_payload_sha256(future)
    with pytest.raises(RuntimeError, match="non-prior memories"):
        build_evaluator_context_index([future], states=[state])

    with pytest.raises(RuntimeError, match="not exact"):
        build_evaluator_context_index([row], states=[], require_exact=True)


def test_development_and_external_catalogs_have_golden_feature_parity() -> None:
    case = _case()
    development = case_to_state(
        user_id="user_1",
        case=case,
        split=PMV2Split.TRAIN,
        strategy_catalog_count=12429,
        strategy_estimated_tokens=260,
    )
    generated_by_source = {
        MemorySource.MP: case.profile_memories,
        MemorySource.MS: case.summary_memories,
        MemorySource.ME: case.event_memories,
    }
    memory_items = [
        MemoryItem(
            memory_id=f"mem_{index:012x}",
            source=source,
            created_session=memory.created_session,
            text=memory.text,
        )
        for index, (source, memory) in enumerate(
            (
                (source, memory)
                for source, memories in generated_by_source.items()
                for memory in memories
            ),
            1,
        )
    ]
    runtime = RuntimeState(
        state_id=development.state_id,
        card_id=development.card_id,
        user_id=development.user_id,
        split="evoemo_test",
        semantic_family=development.semantic_family,
        current_user_text=development.current_user_text,
        current_session_history=development.current_session_history,
        current_session_summary=development.current_session_summary,
        session_index=development.session_index,
        inventory={
            source: _catalog(memory_items, source, development.session_index)
            for source in MemorySource
        },
        allowed_actions=development.allowed_actions,
        provenance={"surface_form_id": development.surface_form_id},
    )
    external = runtime_to_pmv2_state(
        runtime,
        strategy_catalog_count=development.strategy_catalog_count,
        strategy_estimated_tokens=development.strategy_estimated_tokens,
    )
    fixed_policy_state = runtime_to_pmv2_state(
        runtime,
        strategy_catalog_count=development.strategy_catalog_count,
        strategy_estimated_tokens=development.strategy_estimated_tokens,
        include_step0_observation=False,
    )

    assert fixed_policy_state.step0_observation is None
    assert all(
        summary.query_similarity_mean == 0.0
        and not summary.representation_valid
        and summary.catalog_embedding == []
        for summary in fixed_policy_state.inventory.values()
    )

    for source in MemorySource:
        assert development.inventory[source].model_dump() == external.inventory[
            source
        ].model_dump()
        expected_tokens = sum(
            estimate_tokens(memory.text) for memory in generated_by_source[source]
        )
        assert development.inventory[source].estimated_tokens == expected_tokens
        assert "stale_fraction" not in type(development.inventory[source]).model_fields
        assert "conflict_fraction" not in type(development.inventory[source]).model_fields

    assert development.strategy_catalog_count == external.strategy_catalog_count == 12429
    assert development.strategy_estimated_tokens == external.strategy_estimated_tokens == 260

    builder = PMV2FeatureBuilder(
        word_features=32,
        char_features=32,
        use_precomputed_embeddings=False,
    ).fit([development, external])
    action_id = "MPMSME+R0"
    development_vector, external_vector = builder.transform(
        [(development, action_id), (external, action_id)]
    )
    assert np.allclose(development_vector, external_vector)

    leaked = external.model_dump()
    leaked["provenance"] = {"regime": "event_needed"}
    with pytest.raises(ValidationError, match="operational references only"):
        PMV2State.model_validate(leaked)


class _FakeSemanticEncoder:
    spec = FrozenSemanticEncoderSpec(
        model_id="fixture/semantic",
        revision="1" * 40,
        snapshot_tree_sha256="2" * 64,
        max_length=128,
        output_dimension=16,
        current_user_state_token_budget=48,
        session_summary_token_budget=24,
    )
    binding = SemanticEncoderBinding(
        spec_sha256=spec.digest(),
        snapshot_tree_sha256="2" * 64,
        snapshot_file_count=1,
        implementation="transformers-auto-model-cls-float32",
    )

    def encode(self, texts):
        rows = []
        for text in texts:
            vector = np.zeros(self.spec.output_dimension, dtype=float)
            for token in str(text).casefold().split():
                vector[int(sha256_text(token)[:8], 16) % len(vector)] += 1.0
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            rows.append(vector)
        return np.vstack(rows)


class _ReversibleTokenizer:
    truncation_side = "right"

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def __call__(self, text, *, add_special_tokens=False, **kwargs):
        del kwargs
        ids = []
        for token in str(text).split():
            if token not in self._token_to_id:
                token_id = 1000 + len(self._token_to_id)
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            ids.append(self._token_to_id[token])
        if add_special_tokens:
            ids = [101, *ids, 102]
        return {"input_ids": ids}

    def decode(self, ids, **kwargs):
        del kwargs
        return " ".join(
            self._id_to_token[value] for value in ids if value >= 1000
        )


class _SectionAwareFakeSemanticEncoder(_FakeSemanticEncoder):
    def __init__(self) -> None:
        self.tokenizer = _ReversibleTokenizer()
        self.encoded_batches: list[tuple[str, ...]] = []

    def encode(self, texts):
        self.encoded_batches.append(tuple(str(text) for text in texts))
        return super().encode(texts)

    def assemble_visible_dialogue_state(self, **kwargs):
        return FrozenTransformerSemanticEncoder.assemble_visible_dialogue_state(
            self, **kwargs
        )

    def tokenization_telemetry(self, texts, *, view_names=None):
        return FrozenTransformerSemanticEncoder.tokenization_telemetry(
            self, texts, view_names=view_names
        )


def test_reportable_semantic_observation_has_development_external_parity() -> None:
    encoder = _FakeSemanticEncoder()
    development = case_to_state(
        user_id="semantic_parity_user",
        case=_case(case_id="semantic_parity"),
        split=PMV2Split.TRAIN,
        strategy_catalog_count=0,
        strategy_estimated_tokens=0,
        semantic_encoder=encoder,
    )
    runtime = state_to_v1_runtime(development)
    external = runtime_to_pmv2_state(
        runtime,
        strategy_catalog_count=0,
        strategy_estimated_tokens=0,
        semantic_encoder=encoder,
    )

    assert np.allclose(development.text_embedding, external.text_embedding)
    assert development.provenance["semantic_observation"][
        "encoder_spec_sha256"
    ] == encoder.binding.spec_sha256
    assert all(
        development.inventory[source].query_similarity_mean
        == external.inventory[source].query_similarity_mean
        for source in MemorySource
    )
    builder = PMV2FeatureBuilder(use_precomputed_embeddings=False)
    assert np.allclose(
        builder._metadata_raw(development), builder._metadata_raw(external)
    )


def test_long_development_and_external_states_share_one_bounded_semantic_query() -> None:
    encoder = _SectionAwareFakeSemanticEncoder()
    long_history = [
        DialogueTurn(
            role="user" if index % 2 == 0 else "assistant",
            content=" ".join(
                f"history_{index}_{token}" for token in range(30)
            ),
        )
        for index in range(10)
    ]
    case = _case(case_id="long_unified_semantic_query").model_copy(
        update={
            "recent_dialogue": long_history,
            "session_summary": " ".join(
                f"summary_{index}" for index in range(80)
            ),
        }
    )
    old_unbounded_query = visible_dialogue_state_text(
        current_user_text=case.current_user_text,
        current_session_history=case.recent_dialogue,
        current_session_summary=case.session_summary,
    )
    prepared = prepare_visible_semantic_state(
        encoder,
        current_user_text=case.current_user_text,
        current_session_history=case.recent_dialogue,
        current_session_summary=case.session_summary,
    )
    assert prepared.query_text != old_unbounded_query
    assert "history_9_29" in prepared.query_text
    assert "history_0_0" not in prepared.query_text

    encoder.encoded_batches.clear()
    development = case_to_state(
        user_id="long_semantic_user",
        case=case,
        split=PMV2Split.TRAIN,
        strategy_catalog_count=0,
        strategy_estimated_tokens=0,
        semantic_encoder=encoder,
    )
    audit = development.provenance["semantic_observation"]
    assert audit["step0_semantic_query_sha256"] == audit[
        "state_embedding_query_sha256"
    ]
    assert audit["step0_semantic_query_vector_sha256"] == audit[
        "state_embedding_query_vector_sha256"
    ]
    allocation = audit["tokenization"]["section_allocation"]
    assert allocation["final_visible_state_token_count"] <= encoder.spec.max_length
    assert allocation["sections"]["recent_dialogue"]["dropped_token_count"] > 0
    assert audit["tokenization"]["views"]["visible_dialogue_state"][
        "truncated"
    ] is False
    assert not any(old_unbounded_query in batch for batch in encoder.encoded_batches)

    runtime = state_to_v1_runtime(development)
    external = runtime_to_pmv2_state(
        runtime,
        strategy_catalog_count=0,
        strategy_estimated_tokens=0,
        semantic_encoder=encoder,
    )
    external_audit = external.provenance["semantic_observation"]
    assert external_audit["state_embedding_query_sha256"] == audit[
        "state_embedding_query_sha256"
    ]
    assert external_audit["state_embedding_query_vector_sha256"] == audit[
        "state_embedding_query_vector_sha256"
    ]
    assert np.allclose(external.text_embedding, development.text_embedding)

    tampered_runtime = runtime.model_copy(
        update={
            "provenance": {
                **runtime.provenance,
                "semantic_query_sha256": "0" * 64,
            }
        }
    )
    with pytest.raises(RuntimeError, match="lineage drifts"):
        runtime_to_pmv2_state(
            tampered_runtime,
            semantic_encoder=encoder,
        )
    missing_lineage_runtime = runtime.model_copy(
        update={
            "provenance": {
                key: value
                for key, value in runtime.provenance.items()
                if not key.startswith("semantic_query")
            }
        }
    )
    with pytest.raises(RuntimeError, match="lineage drifts"):
        runtime_to_pmv2_state(
            missing_lineage_runtime,
            semantic_encoder=encoder,
        )

    tampered_state = development.model_dump(mode="json")
    tampered_state["provenance"]["semantic_observation"][
        "step0_semantic_query_sha256"
    ] = "0" * 64
    with pytest.raises(ValidationError, match="semantic queries drift"):
        PMV2State.model_validate_json(json.dumps(tampered_state))


def test_evoemo_memory_step0_and_state_embedding_share_semantic_lineage() -> None:
    encoder = _SectionAwareFakeSemanticEncoder()
    conversation = [
        {
            "role": "seeker" if index % 2 == 0 else "supporter",
            "content": " ".join(
                f"external_history_{index}_{token}" for token in range(30)
            ),
        }
        for index in range(8)
    ]
    memory_items = [
        MemoryItem(
            memory_id=f"mem_{index:012x}",
            source=source,
            created_session=1,
            text=f"Relevant {source.value} memory about workplace uncertainty.",
        )
        for index, source in enumerate(MemorySource, 1)
    ]
    runtime = make_evo_runtime_state(
        user={"id": "external_long_user"},
        topic={"idx": 1},
        conversation=conversation,
        current_user_text="I am unsure how to handle tomorrow's meeting.",
        items=memory_items,
        turn_index=4,
        condition="PM",
        semantic_encoder=encoder,
    )
    assert runtime.provenance["semantic_query_sha256"]
    assert runtime.provenance["semantic_query_vector_sha256"]
    external = runtime_to_pmv2_state(runtime, semantic_encoder=encoder)
    audit = external.provenance["semantic_observation"]
    assert runtime.provenance["semantic_query_sha256"] == audit[
        "state_embedding_query_sha256"
    ]
    assert runtime.provenance["semantic_query_vector_sha256"] == audit[
        "state_embedding_query_vector_sha256"
    ]


def test_memory_metadata_uses_retrieval_capacity_not_catalog_tail() -> None:
    base = case_to_state(
        user_id="user_tail_invariance",
        case=_case(case_id="tail_invariance"),
        split=PMV2Split.TRAIN,
        strategy_catalog_count=12429,
        strategy_estimated_tokens=260,
    )
    small_payload = base.model_dump(mode="json")
    large_payload = base.model_dump(mode="json")
    for payload, count, tokens in (
        (small_payload, 2, 200),
        (large_payload, 20, 2000),
    ):
        payload["state_id"] = f"state_tail_{count}"
        payload["card_id"] = f"card_tail_{count}"
        payload["provenance"]["backend_record_id"] = payload["card_id"]
        payload["inventory"]["MP"].update(
            {
                "available": True,
                "count": count,
                "estimated_tokens": tokens,
                "min_age_sessions": 1,
                "median_age_sessions": 3.0,
                "max_age_sessions": 5,
            }
        )
        # This test deliberately mutates the legacy inventory scale. Drop the
        # now-stale formal Step-0 snapshot so the builder recomputes it.
        payload["step0_observation"] = None
    small = PMV2State.model_validate_json(json.dumps(small_payload))
    large = PMV2State.model_validate_json(json.dumps(large_payload))
    builder = PMV2FeatureBuilder(
        word_features=32,
        char_features=32,
        use_precomputed_embeddings=False,
    )

    # Both stores expose the same two retrievable MP items at the same average
    # size. Eighteen additional tail items must not change model-visible metadata,
    # action features, or the deployable cost estimate.
    assert np.allclose(builder._metadata_raw(small), builder._metadata_raw(large))
    assert np.allclose(
        builder._action_raw(small, "MP+R0"),
        builder._action_raw(large, "MP+R0"),
    )
    assert builder.estimate_action_cost(small, "MP+R0") == pytest.approx(224.0)
    assert builder.estimate_action_cost(large, "MP+R0") == pytest.approx(224.0)


def test_required_hit_preflight_runs_before_outcomes_and_fails_closed() -> None:
    case = _case(
        case_id="required_hit",
        regime=ResourceNeedRegime.EVENT_NEEDED,
    )
    state = case_to_state(
        user_id="required_hit_user",
        case=case,
        split=PMV2Split.TRAIN,
        strategy_catalog_count=1,
        strategy_estimated_tokens=40,
    )
    backend = case_to_memory_backend(state, case)
    strategy = StrategyCard(
        strategy_id="strat_aaaaaaaaaaaa",
        strategy_label="Reflection of feelings",
        retrieval_text="difficult workplace conversation and uncertainty",
        guidance_text="Reflect the uncertainty.",
        example_response="That uncertainty sounds tiring.",
        source_dialogue_id="d1",
        source_turn_index=1,
    )
    report = validate_required_hit_preflight(
        states=[state],
        cases_by_state={state.state_id: case},
        backends_by_state={state.state_id: backend},
        strategy_cards=[strategy],
        strategy_top_k=1,
        memory_min_score=0.0,
        strategy_min_score=0.0,
    )
    assert report["status"] == "PASS"
    assert report["outcome_fields_accessed"] is False
    with pytest.raises(RuntimeError, match="required-hit"):
        validate_required_hit_preflight(
            states=[state],
            cases_by_state={state.state_id: case},
            backends_by_state={state.state_id: backend},
            strategy_cards=[strategy],
            strategy_top_k=1,
            memory_min_score=1.0,
            strategy_min_score=0.0,
        )


def test_real_evoemo_inventory_scale_does_not_force_metadata_ood_fallback() -> None:
    """Use real external aggregates only as a no-label scale regression test."""

    base = case_to_state(
        user_id="synthetic_scale_reference",
        case=_case(case_id="scale_reference"),
        split=PMV2Split.TRAIN,
        strategy_catalog_count=12429,
        strategy_estimated_tokens=260,
    )
    synthetic_states: list[PMV2State] = []
    for count in (1, 2, 3):
        payload = base.model_dump(mode="json")
        payload.update(
            {
                "state_id": f"synthetic_scale_state_{count}",
                "card_id": f"synthetic_scale_card_{count}",
                "user_id": f"synthetic_scale_user_{count}",
            }
        )
        payload["provenance"]["backend_record_id"] = payload["card_id"]
        for source in ("MP", "MS", "ME"):
            payload["inventory"][source].update(
                {
                    "available": True,
                    "count": count,
                    "min_age_sessions": 1,
                    "median_age_sessions": 3.0,
                    "max_age_sessions": 6,
                    "estimated_tokens": count * (12 + 4 * count),
                }
            )
        payload["step0_observation"] = None
        synthetic_states.append(PMV2State.model_validate_json(json.dumps(payload)))

    builder = PMV2FeatureBuilder(
        word_features=32,
        char_features=32,
        use_precomputed_embeddings=False,
    ).fit(synthetic_states)

    users = load_evoemo(PROJECT_ROOT / "data/external/evo_emo.json")
    real_counts: dict[MemorySource, list[int]] = {
        source: [] for source in MemorySource
    }
    real_token_totals: list[int] = []
    reports = []
    external_scale_states: list[PMV2State] = []
    for index, user in enumerate(users):
        items, _ = build_evo_memory(user)
        session_index = max(item.created_session for item in items) + 1
        inventory = {
            source: _catalog(items, source, session_index)
            for source in MemorySource
        }
        for source, catalog in inventory.items():
            real_counts[source].append(catalog.count)
            real_token_totals.append(catalog.estimated_tokens)
        runtime = RuntimeState(
            state_id=f"real_scale_state_{index}",
            card_id=f"real_scale_card_{index}",
            user_id=f"real_scale_user_{index}",
            split="evoemo_test",
            semantic_family=base.semantic_family,
            current_user_text=base.current_user_text,
            current_session_history=base.current_session_history,
            current_session_summary=base.current_session_summary,
            session_index=session_index,
            inventory=inventory,
            allowed_actions=base.allowed_actions,
            provenance={"surface_form_id": base.surface_form_id},
        )
        external_scale_state = runtime_to_pmv2_state(
            runtime,
            strategy_catalog_count=12429,
            strategy_estimated_tokens=260,
        )
        external_scale_states.append(external_scale_state)
        reports.append(builder.ood_report(external_scale_state))

    # Pin the currently observed external scale so the regression cannot silently
    # degrade into a small synthetic fixture. No EvoEmo label enters fitting.
    assert set(real_counts[MemorySource.MP]) == {7}
    assert (min(real_counts[MemorySource.MS]), max(real_counts[MemorySource.MS])) == (
        13,
        33,
    )
    assert (min(real_counts[MemorySource.ME]), max(real_counts[MemorySource.ME])) == (
        13,
        33,
    )
    assert max(real_token_totals) == 9935
    assert reports
    assert not any(report["severe_metadata_ood"] for report in reports)
    assert not any(report["recommendation"] == "FALLBACK" for report in reports)
    external_vectors = builder.transform(
        [(state, "MPMSME+R0") for state in external_scale_states]
    )
    assert np.all(np.isfinite(external_vectors))
    assert float(np.max(np.abs(external_vectors))) < 10.0


def test_split_manifest_rejects_same_split_duplicate_current_text() -> None:
    first = case_to_state(
        user_id="duplicate_text_user_a",
        case=_case(
            case_id="duplicate_text_a",
            current_user_text="I feel uncertain about the meeting.",
        ),
        split=PMV2Split.TRAIN,
        strategy_catalog_count=12429,
        strategy_estimated_tokens=260,
    )
    second = case_to_state(
        user_id="duplicate_text_user_b",
        case=_case(
            case_id="duplicate_text_b",
            current_user_text="  I FEEL uncertain   about the meeting. ",
        ),
        split=PMV2Split.TRAIN,
        strategy_catalog_count=12429,
        strategy_estimated_tokens=260,
    )
    split_states = {
        PMV2Split.TRAIN: [first, second],
        PMV2Split.CALIBRATION: [],
        PMV2Split.INTERNAL_TEST: [],
    }
    with pytest.raises(ValidationError, match="globally unique"):
        validate_split_manifests(split_states)

    second.current_user_text = "This is a genuinely different current turn."
    manifest = validate_split_manifests(split_states)
    assert manifest.total_states == 2
    assert manifest.unique_normalized_current_user_texts == 2
    assert manifest.normalized_current_user_text_unique_rate == 1.0


def test_cross_split_near_duplicate_audit_rejects_paraphrase_like_texts() -> None:
    states = []
    specifications = [
        (
            PMV2Split.TRAIN,
            "near_train",
            "family_train",
            "I feel overwhelmed by work deadlines and cannot sleep tonight.",
        ),
        (
            PMV2Split.CALIBRATION,
            "near_calibration",
            "family_calibration",
            "I feel overwhelmed by work deadlines and cannot sleep today.",
        ),
        (
            PMV2Split.INTERNAL_TEST,
            "near_internal",
            "family_internal",
            "My neighbor invited me to a relaxed weekend picnic.",
        ),
    ]
    for split, case_id, family, text in specifications:
        case = _case(case_id=case_id, current_user_text=text)
        case.semantic_family = family
        states.append(
            case_to_state(
                user_id=f"user_{case_id}",
                case=case,
                split=split,
                strategy_catalog_count=12429,
                strategy_estimated_tokens=260,
            )
        )
    by_split = {
        split: [state for state in states if state.split == split]
        for split in (
            PMV2Split.TRAIN,
            PMV2Split.CALIBRATION,
            PMV2Split.INTERNAL_TEST,
        )
    }
    validate_split_manifests(by_split)
    with pytest.raises(ValueError, match="near-duplicate audit failed"):
        audit_cross_split_near_duplicates(
            by_split,
            maximum_word_hash_cosine=0.70,
            maximum_char_hash_cosine=0.80,
            report_top_pairs=5,
        )

    states[1].current_user_text = (
        "After moving cities, I am unsure how to make new friends."
    )
    report = audit_cross_split_near_duplicates(
        by_split,
        maximum_word_hash_cosine=0.92,
        maximum_char_hash_cosine=0.95,
        report_top_pairs=5,
    )
    assert report["status"] == "PASS"
    assert report["cross_split_pair_count"] == 3


def test_bundle_must_cover_every_frozen_assigned_family_and_report_union(
    tmp_path: Path,
) -> None:
    script = PROJECT_ROOT / "scripts/20_generate_pm_v2_development_data.py"
    spec = importlib.util.spec_from_file_location("pm_v2_generator_contract", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    families = [
        "relocation_loneliness",
        "workload_burnout",
        "friendship_distance",
    ]
    incomplete = _source_grounded_bundle(user_id="family_incomplete")
    for case in incomplete.cases:
        if case.semantic_family == "friendship_distance":
            case.semantic_family = "workload_burnout"
    with pytest.raises(ValueError, match="exactly cover.*friendship_distance"):
        module.strict_bundle_check(
            incomplete, families
        )

    complete = _source_grounded_bundle(user_id="family_complete")
    module.strict_bundle_check(complete, families)
    report = write_development_dataset(
        bundles=[complete],
        split_by_user={complete.user_id: PMV2Split.TRAIN},
        out_dir=tmp_path / "complete",
        strategy_catalog_count=12429,
        strategy_estimated_tokens=260,
        strategy_top_k=3,
        strategy_bank_sha256="a" * 64,
        expected_semantic_families_by_split={
            PMV2Split.TRAIN: families,
            PMV2Split.CALIBRATION: [],
            PMV2Split.INTERNAL_TEST: [],
        },
    )
    coverage = report["semantic_family_coverage"]
    assert coverage["status"] == "PASS"
    assert coverage["splits"]["train"]["observed_count"] == 3
    assert report["split_manifest"][
        "normalized_current_user_text_unique_rate"
    ] == 1.0
    with pytest.raises(ValueError, match="semantic-family union"):
        write_development_dataset(
            bundles=[complete],
            split_by_user={complete.user_id: PMV2Split.TRAIN},
            out_dir=tmp_path / "incomplete_union",
            strategy_catalog_count=12429,
            strategy_estimated_tokens=260,
            strategy_top_k=3,
            strategy_bank_sha256="a" * 64,
            expected_semantic_families_by_split={
                PMV2Split.TRAIN: [
        *families,
                    "family_d",
                ],
                PMV2Split.CALIBRATION: [],
                PMV2Split.INTERNAL_TEST: [],
            },
        )

    frozen_report = {
        "n_states": 468,
        "split_counts": {
            "train": 216,
            "calibration": 108,
            "internal_test": 144,
        },
        "split_manifest": {
            "total_states": 468,
            "unique_normalized_current_user_texts": 468,
            "normalized_current_user_text_unique_rate": 1.0,
        },
        "semantic_family_coverage": {
            "splits": {
                "train": {"observed_count": 14},
                "calibration": {"observed_count": 5},
                "internal_test": {"observed_count": 5},
            }
        },
    }
    design = module.enforce_full_state_design(
        frozen_report,
        train_users=24,
        calibration_users=12,
        internal_test_users=16,
    )
    assert design["expected_total_states"] == 468
    assert design["semantic_family_union_counts"] == {
        "train": 14,
        "calibration": 5,
        "internal_test": 5,
    }
    duplicate_report = json.loads(json.dumps(frozen_report))
    duplicate_report["split_manifest"]["unique_normalized_current_user_texts"] = 467
    with pytest.raises(RuntimeError, match="not globally unique"):
        module.enforce_full_state_design(
            duplicate_report,
            train_users=24,
            calibration_users=12,
            internal_test_users=16,
        )


def test_v1_5_data_generation_binds_exact_strategy_bank_and_seed_manifest() -> None:
    script = (
        PROJECT_ROOT
        / "scripts"
        / "v1_5"
        / "20_generate_pm_v2_development_data_v1_5.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pm_v1_5_frozen_strategy_binding", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = load_config(PROJECT_ROOT / "configs" / "pm_v1_5.yaml")
    bank = PROJECT_ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl"
    selected = (
        PROJECT_ROOT
        / "data"
        / "strategy"
        / "pm_v1_5_selected_seed_sources.jsonl"
    )
    cards = [StrategyCard.model_validate(row) for row in iter_jsonl(bank)]
    report = module.require_frozen_strategy_bank_binding(
        pm_config=config,
        strategy_bank_path=bank,
        selected_seed_sources_path=selected,
        strategy_cards=cards,
    )
    assert report["status"] == "PASS"
    assert report["card_count"] == 11590
    assert report["source_dialogue_count"] == 823
    assert report["selected_seed_source_count"] == 52

    tampered = json.loads(json.dumps(config))
    tampered["strategy_bank_contract"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="Strategy Bank hash mismatch"):
        module.require_frozen_strategy_bank_binding(
            pm_config=tampered,
            strategy_bank_path=bank,
            selected_seed_sources_path=selected,
            strategy_cards=cards,
        )


def test_generation_resume_binding_rejects_legacy_and_mismatch() -> None:
    binding = {
        "generator_config_sha256": "a" * 64,
        "seed_dialogues_sha256": "b" * 64,
        "prompt_contract_sha256": "c" * 64,
        "code_manifest_sha256": "d" * 64,
    }
    bundle = _bundle()
    bind_bundle_to_generation_run(bundle, binding)
    require_bundle_generation_binding(bundle, binding)

    with pytest.raises(RuntimeError, match="different generator/seed/config/prompt/code"):
        require_bundle_generation_binding(
            bundle,
            {**binding, "seed_dialogues_sha256": "e" * 64},
        )

    legacy = _bundle()
    with pytest.raises(RuntimeError, match="predates immutable generation-run binding"):
        require_bundle_generation_binding(legacy, binding)


def test_regime_checks_reject_constant_full_resource_policy() -> None:
    assert _regime_pass("profile_needed", "MP+R0", 0.2, ["MP"])
    assert _regime_pass("summary_needed", "MS+R0", 0.2, ["MS"])
    assert _regime_pass("event_needed", "ME+R0", 0.2, ["ME"])
    for regime, needed in (
        ("profile_needed", ["MP"]),
        ("summary_needed", ["MS"]),
        ("event_needed", ["ME"]),
        ("multi_source_needed", ["MP", "ME"]),
    ):
        assert not _regime_pass(regime, "MPMSME+R0", 0.2, needed)
    assert _regime_pass(
        "multi_source_needed", "MPE+R0", 0.2, ["MP", "ME"]
    )
    assert _regime_pass("strategy_helpful", "M0+RS", 0.2, [])
    assert not _regime_pass("strategy_helpful", "MPMSME+RS", 0.2, [])


def test_explicit_train_ids_cannot_override_declared_test_split(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "dialogue_id": "test_1",
                "split": "test",
                "dialogue_text": "user: This row belongs to test.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    train_ids = tmp_path / "train_ids.txt"
    train_ids.write_text("test_1\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "18_prepare_pm_v2_seed_dialogues.py"),
            "--inputs",
            str(source),
            "--out",
            str(tmp_path / "out.jsonl"),
            "--train-ids",
            str(train_ids),
            "--minimum-seeds",
            "1",
        ],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "declared non-train source split" in result.stderr


def test_seed_output_records_hashed_lineage_and_computed_split_audit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    rows = [
        {"dialogue_id": "train_1", "split": "train", "dialogue_text": "first"},
        {"dialogue_id": "train_2", "split": "train", "dialogue_text": "second"},
        {"dialogue_id": "test_1", "split": "test", "dialogue_text": "held out"},
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    train_ids = tmp_path / "train_ids.txt"
    train_ids.write_text("train_1\ntrain_2\n", encoding="utf-8")
    excluded_ids = tmp_path / "excluded_ids.txt"
    excluded_ids.write_text("train_2\n", encoding="utf-8")
    output = tmp_path / "seeds.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "18_prepare_pm_v2_seed_dialogues.py"),
            "--inputs",
            str(source),
            "--out",
            str(output),
            "--train-ids",
            str(train_ids),
            "--excluded-ids",
            str(excluded_ids),
            "--minimum-seeds",
            "1",
        ],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=True,
    )
    audit = json.loads(
        output.with_suffix(output.suffix + ".audit.json").read_text(encoding="utf-8")
    )
    assert audit["test_or_validation_rows_in_output"] == 0
    assert audit["declared_non_train_rows_in_output"] == 0
    hashes = audit["lineage_manifest_hashes"]
    assert {
        "code_manifest_sha256",
        "source_manifest_sha256",
        "train_manifest_sha256",
        "exclusion_manifest_sha256",
    } <= set(hashes)
    assert all(len(value) == 64 for value in hashes.values())
    assert (
        audit["seed_extraction_contract_version"]
        == "pm-v2-seed-extraction-v1-strict-lineage"
    )
    assert audit["code_manifest"]["contract_version"] == audit[
        "seed_extraction_contract_version"
    ]
    assert audit["code_manifest_sha256"] == hashes["code_manifest_sha256"]
    assert audit["code_manifest_sha256"] == sha256_text(
        canonical_json(audit["code_manifest"])
    )
    manifest_files = {
        row["path"]: row["sha256"] for row in audit["code_manifest"]["files"]
    }
    assert manifest_files == {
        "scripts/18_prepare_pm_v2_seed_dialogues.py": sha256_file(
            PROJECT_ROOT / "scripts/18_prepare_pm_v2_seed_dialogues.py"
        ),
        "src/metacom_pm/io.py": sha256_file(PROJECT_ROOT / "src/metacom_pm/io.py"),
    }
    output_rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["dialogue_id"] for row in output_rows] == ["train_1"]
    assert all(output_rows[0][key] == value for key, value in hashes.items())
    assert output_rows[0]["seed_extraction_contract_version"] == audit[
        "seed_extraction_contract_version"
    ]


def test_index_split_manifest_strictly_joins_and_excludes_overlap(
    tmp_path: Path,
) -> None:
    source = tmp_path / "esconv.json"
    source.write_text(
        json.dumps(
            [
                {"dialog": [{"speaker": "seeker", "content": "train seed"}]},
                {"dialog": [{"speaker": "seeker", "content": "test row"}]},
                {"dialog": [{"speaker": "seeker", "content": "overlap row"}]},
            ]
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "split.jsonl"
    manifest_rows = [
        {
            "index": 0,
            "dialogue_id": "esconv_0000",
            "split": "train",
            "excluded_for_evoemo_overlap": False,
        },
        {
            "index": 1,
            "dialogue_id": "esconv_0001",
            "split": "test",
            "excluded_for_evoemo_overlap": False,
        },
        {
            "index": 2,
            "dialogue_id": "esconv_0002",
            "split": "train",
            "excluded_for_evoemo_overlap": True,
        },
    ]
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    output = tmp_path / "seeds.jsonl"
    base_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "18_prepare_pm_v2_seed_dialogues.py"),
        "--inputs",
        str(source),
        "--split-manifest",
        str(manifest),
        "--out",
        str(output),
        "--minimum-seeds",
        "1",
    ]
    subprocess.run(
        base_command,
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=True,
    )
    output_rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["dialogue_id"] for row in output_rows] == ["esconv_0000"]
    assert output_rows[0]["split_manifest_index"] == 0
    audit = json.loads(
        output.with_suffix(output.suffix + ".audit.json").read_text(encoding="utf-8")
    )
    assert audit["split_manifest_join"]["status"] == "PASS"
    assert audit["split_manifest_join"]["covered_source_rows"] == 3
    assert audit["evoemo_overlap_excluded_train"] == 1
    assert audit["test_or_validation_rows_in_output"] == 0
    assert "split_join_manifest_sha256" in audit["lineage_manifest_hashes"]

    train_ids = tmp_path / "train_ids.txt"
    train_ids.write_text("esconv_0001\n", encoding="utf-8")
    override = subprocess.run(
        [*base_command, "--train-ids", str(train_ids), "--overwrite"],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert override.returncode != 0
    assert "cannot override" in override.stderr

    broken_manifest = tmp_path / "broken.jsonl"
    broken_rows = [dict(row) for row in manifest_rows]
    broken_rows[-1]["index"] = 3
    broken_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in broken_rows),
        encoding="utf-8",
    )
    broken = subprocess.run(
        [
            *base_command[: base_command.index("--split-manifest") + 1],
            str(broken_manifest),
            *base_command[base_command.index("--out") :],
            "--overwrite",
        ],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert broken.returncode != 0
    assert "continuous from zero" in broken.stderr


def test_generation_dry_run_is_keyless_and_saves_exact_accepted_plan(
    tmp_path: Path,
) -> None:
    seed_path = tmp_path / "seeds.jsonl"
    _write_unique_generation_seeds(seed_path)
    out_dir = tmp_path / "generation"
    config = PROJECT_ROOT / "configs" / "experiment.yaml"
    pm_config = PROJECT_ROOT / "configs" / "pm_v2.yaml"
    env = _subprocess_env()
    env.pop("NVIDIA_API_KEY", None)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "20_generate_pm_v2_development_data.py"),
        "--config",
        str(config),
        "--pm-v2-config",
        str(pm_config),
        "--seed-dialogues",
        str(seed_path),
        "--out-dir",
        str(out_dir),
        "--max-users",
        "1",
        "--max-api-calls",
        "18",
        "--max-estimated-usd",
        "1",
        "--max-input-tokens-per-call",
        "12000",
    ]
    subprocess.run(
        [*command, "--dry-run"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    estimate = json.loads(
        (out_dir / "generation_cost_estimate.json").read_text(encoding="utf-8")
    )
    assert estimate["expected_api_calls"] == 9
    assert estimate["maximum_api_calls"] == 18
    assert estimate["historical_physical_api_attempts"] == 0
    assert estimate["maximum_physical_api_attempts_including_history"] == 18
    assert estimate["input_token_safety_factor"] == 1.5
    assert estimate["fail_on_reported_input_overrun"] is True
    assert estimate["pricing"] == {
        "input_usd_per_mtok": 0.15,
        "output_usd_per_mtok": 0.60,
    }
    assert estimate["budget_gate"]["status"] == "PASS"
    assert len(estimate["cost_estimate_sha256"]) == 64
    strategy_catalog = estimate["generation_run_binding"]["strategy_catalog"]
    assert strategy_catalog["count"] == 12429
    assert strategy_catalog["estimated_action_tokens"] == 260
    assert strategy_catalog["top_k"] == 3
    assert len(strategy_catalog["sha256"]) == 64
    call_plan_rows = list(iter_jsonl(out_dir / "generation_call_plan.jsonl"))
    assert len(call_plan_rows) == 9
    assert all(row["remaining_attempts"] == 2 for row in call_plan_rows)
    assert all(
        row["estimated_input_tokens"] > row["raw_estimated_input_tokens"]
        for case_plan in call_plan_rows
        for row in case_plan["attempts"]
    )
    pricing_override = subprocess.run(
        [*command, "--dry-run", "--input-usd-per-mtok", "0"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert pricing_override.returncode != 0
    assert "pricing override differs" in pricing_override.stderr
    all_attempts = [
        row for case_plan in call_plan_rows for row in case_plan["attempts"]
    ]
    assert len({row["call_key"] for row in all_attempts}) == 18

    # Inject the single paid failure.  The next dry-run must fail closed rather
    # than silently grant a replacement seed.
    expected_calls = {
        row["call_key"]: 1 for row in all_attempts
    }
    ledger = PersistentAttemptLedger(
        out_dir / "_generation_physical_attempt_ledger.jsonl",
        stage="pm_v2_synthetic_surface_generation",
        expected_calls=expected_calls,
        maximum_total_attempts=18,
    )
    first = call_plan_rows[0]["attempts"][0]
    reservation = ledger.reserve(
        first["call_key"],
        record_ids={
            "user_id": call_plan_rows[0]["user_id"],
            "case_field": call_plan_rows[0]["case_field"],
            "attempt_kind": first["attempt_kind"],
            "generation_seed": first["seed"],
        },
        prompt_sha256=first["prompt_sha256"],
    )
    ledger.finish(
        reservation,
        succeeded=False,
        request_hash=None,
        usage=None,
        error="injected failure",
    )
    overwrite_blocked = subprocess.run(
        [*command, "--dry-run", "--overwrite"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert overwrite_blocked.returncode != 0
    assert "refuses --overwrite" in overwrite_blocked.stderr
    assert len(list(iter_jsonl(out_dir / "_generation_physical_attempt_ledger.jsonl"))) == 2
    repair_dry_run = subprocess.run(
        [*command, "--dry-run"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    repair_estimate = read_json(out_dir / "generation_cost_estimate.json")
    repair_plan = list(iter_jsonl(out_dir / "generation_call_plan.jsonl"))
    assert repair_estimate["historical_physical_api_attempts"] == 1
    assert repair_estimate["expected_api_calls"] == 9
    assert repair_estimate["maximum_api_calls"] == 17
    assert repair_estimate["blocked_pending_users"] == []
    repaired_case = next(
        row
        for row in repair_plan
        if row["case_field"] == call_plan_rows[0]["case_field"]
    )
    assert repaired_case["remaining_attempts"] == 1
    assert repaired_case["attempts"][0]["attempt_kind"] == "repair"

    repair = call_plan_rows[0]["attempts"][1]
    repair_reservation = ledger.reserve(
        repair["call_key"],
        record_ids={
            "user_id": call_plan_rows[0]["user_id"],
            "case_field": call_plan_rows[0]["case_field"],
            "attempt_kind": "repair",
            "generation_seed": repair["seed"],
        },
        prompt_sha256=repair["prompt_sha256"],
    )
    ledger.finish(
        repair_reservation,
        succeeded=False,
        request_hash=None,
        usage=None,
        error="injected repair failure",
    )
    exhausted = subprocess.run(
        [*command, "--dry-run"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert exhausted.returncode != 0
    exhausted_estimate = read_json(out_dir / "generation_cost_estimate.json")
    assert exhausted_estimate["historical_physical_api_attempts"] == 2
    assert exhausted_estimate["maximum_api_calls"] == 16
    assert exhausted_estimate["blocked_pending_users"] == [
        {
            "user_id": call_plan_rows[0]["user_id"],
            "case_field": call_plan_rows[0]["case_field"],
            "reason": "maximum_case_attempts_exhausted",
            "historical_attempts": 2,
        }
    ]
    assert exhausted_estimate["budget_gate"]["status"] == "FAIL"


def test_generation_compatibility_pilot_dry_run_freezes_casewise_plan(
    tmp_path: Path,
) -> None:
    seed_path = tmp_path / "seeds.jsonl"
    _write_unique_generation_seeds(seed_path)
    out_dir = tmp_path / "generation_pilot"
    env = _subprocess_env()
    env.pop("NVIDIA_API_KEY", None)
    command = [
        sys.executable,
        str(
            PROJECT_ROOT
            / "scripts"
            / "20a_run_pm_v2_generation_compatibility_pilot.py"
        ),
        "--seed-dialogues",
        str(seed_path),
        "--out-dir",
        str(out_dir),
        "--max-api-calls",
        str(GENERATION_PILOT_MAX_ATTEMPTS),
        "--max-estimated-usd",
        "1",
        "--max-input-tokens-per-call",
        "12000",
    ]
    subprocess.run(
        [*command, "--dry-run"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    estimate = read_json(out_dir / "cost_estimate.json")
    plan = list(iter_jsonl(out_dir / "call_plan.jsonl"))
    assert estimate["minimum_api_calls_if_successful"] == GENERATION_PILOT_MINIMUM_CALLS
    assert estimate["maximum_physical_api_attempts"] == GENERATION_PILOT_MAX_ATTEMPTS
    assert estimate["maximum_repairs_per_case"] == 1
    assert estimate["stop_after_each_case_success"] is True
    assert estimate["pricing"] == {
        "input_usd_per_mtok": 0.15,
        "output_usd_per_mtok": 0.60,
    }
    assert estimate["input_token_safety_factor"] == 1.5
    assert estimate["maximum_input_token_upper_bound_per_call"] > 0
    assert estimate["maximum_output_tokens_per_call"] == 900
    assert estimate["budget_gate"]["status"] == "PASS"
    assert len(plan) == GENERATION_PILOT_MAX_ATTEMPTS
    assert len({row["physical_call_key"] for row in plan}) == len(plan)
    assert len({row["generation_seed"] for row in plan}) == len(plan)
    assert {row["attempt_kind"] for row in plan} == {"initial", "repair"}
    assert {
        row["case_field"] for row in plan
    } == {field for field, _ in GENERATION_CASE_FIELDS}
    assert all(
        sum(candidate["case_field"] == row["case_field"] for candidate in plan)
        == 2
        for row in plan
    )
    assert all(row["maximum_physical_attempts"] == 1 for row in plan)

    ledger_path = out_dir / "physical_attempt_ledger.jsonl"
    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=GENERATION_PILOT_STAGE,
        expected_calls={str(row["physical_call_key"]): 1 for row in plan},
        maximum_total_attempts=len(plan),
    )
    reservation = ledger.reserve(
        str(plan[0]["physical_call_key"]),
        record_ids={
            "user_id": plan[0]["user_id"],
            "generation_seed": plan[0]["generation_seed"],
            "case_field": plan[0]["case_field"],
            "attempt_kind": plan[0]["attempt_kind"],
        },
        prompt_sha256=str(plan[0]["prompt_sha256"]),
    )
    ledger.finish(
        reservation,
        succeeded=False,
        request_hash=None,
        usage=None,
        error="injected pilot failure",
    )
    overwrite_blocked = subprocess.run(
        [*command, "--dry-run", "--overwrite"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert overwrite_blocked.returncode != 0
    assert "refuses --overwrite" in overwrite_blocked.stderr
    assert len(list(iter_jsonl(ledger_path))) == 2

    saved_estimate_bytes = (out_dir / "cost_estimate.json").read_bytes()
    saved_plan_bytes = (out_dir / "call_plan.jsonl").read_bytes()
    exact_spent_dry_run = subprocess.run(
        [*command, "--dry-run"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert exact_spent_dry_run.returncode == 0
    assert (out_dir / "cost_estimate.json").read_bytes() == saved_estimate_bytes
    assert (out_dir / "call_plan.jsonl").read_bytes() == saved_plan_bytes
    stale_spent_dry_run = subprocess.run(
        [*command, "--dry-run", "--max-estimated-usd", "2"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert stale_spent_dry_run.returncode != 0
    assert "spent pilot ledger freezes" in stale_spent_dry_run.stderr
    assert (out_dir / "cost_estimate.json").read_bytes() == saved_estimate_bytes
    assert (out_dir / "call_plan.jsonl").read_bytes() == saved_plan_bytes

    dry_run_again = subprocess.run(
        [*command, "--dry-run", "--input-usd-per-mtok", "0"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert dry_run_again.returncode != 0
    assert "input pricing override differs" in dry_run_again.stderr


def test_v1_5_generation_pilot_budget_is_config_frozen_and_reproducible(
    tmp_path: Path,
) -> None:
    env = _subprocess_env()
    env.pop("OPENAI_API_KEY", None)
    # A clean checkout intentionally has no ignored development seed corpus.
    # Keep this budget-contract test hermetic by supplying the minimum unique
    # synthetic cohort required for 52 users plus one held-out seed.
    seed_path = tmp_path / "generation_seeds.jsonl"
    _write_unique_generation_seeds(seed_path)
    script = (
        PROJECT_ROOT
        / "scripts"
        / "v1_5"
        / "20a_run_generation_compatibility_pilot_v1_5.py"
    )
    estimates = []
    for suffix in ("first", "second"):
        out_dir = tmp_path / suffix
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--dry-run",
                "--seed-dialogues",
                str(seed_path),
                "--out-dir",
                str(out_dir),
            ],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        estimates.append(read_json(out_dir / "cost_estimate.json"))
    assert estimates[0]["budget_limits"] == {
        "max_api_calls": GENERATION_PILOT_MAX_ATTEMPTS,
        "max_estimated_usd": 0.018,
        "max_input_tokens_per_call": 4000,
    }
    assert (
        estimates[0]["cost_estimate_sha256"]
        == estimates[1]["cost_estimate_sha256"]
    )

    rejected = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--seed-dialogues",
            str(seed_path),
            "--out-dir",
            str(tmp_path / "override"),
            "--max-estimated-usd",
            "2.0",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "override differs from frozen generation-pilot budget" in rejected.stderr


def test_generation_pilot_recovers_post_success_crash_without_second_http(
    tmp_path: Path, monkeypatch
) -> None:
    script = (
        PROJECT_ROOT
        / "scripts"
        / "20a_run_pm_v2_generation_compatibility_pilot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pm_v2_generation_pilot_recovery", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    seed_path = tmp_path / "seeds.jsonl"
    _write_unique_generation_seeds(seed_path)
    out_dir = tmp_path / "pilot"
    common = [
        str(script),
        "--seed-dialogues",
        str(seed_path),
        "--out-dir",
        str(out_dir),
        "--max-api-calls",
        str(GENERATION_PILOT_MAX_ATTEMPTS),
        "--max-estimated-usd",
        "1",
        "--max-input-tokens-per-call",
        "12000",
    ]
    monkeypatch.setattr(sys, "argv", [*common, "--dry-run"])
    module.main()

    experiment_config_path = PROJECT_ROOT / "configs" / "experiment.yaml"
    pm_v2_config_path = PROJECT_ROOT / "configs" / "pm_v2.yaml"
    experiment_config = load_config(experiment_config_path)
    pm_v2_config = load_config(pm_v2_config_path)
    generation_cfg = pm_v2_config["data_generation"]
    endpoint = endpoint_from_config(
        experiment_config, str(generation_cfg["generator_endpoint"])
    )
    full_user_count = sum(
        int(generation_cfg[key])
        for key in ("train_users", "calibration_users", "internal_test_users")
    )
    contract = build_generation_compatibility_contract(
        project_root=PROJECT_ROOT,
        experiment_config_path=experiment_config_path,
        pm_v2_config_path=pm_v2_config_path,
        seed_dialogues_path=seed_path,
        endpoint=endpoint,
        base_generation_seed=int(generation_cfg["base_seed"]),
        full_user_count=full_user_count,
        input_token_safety_factor=1.5,
        fail_on_reported_input_overrun=True,
        input_usd_per_mtok=0.15,
        output_usd_per_mtok=0.60,
    )
    surfaces = _surface_only_pilot_inputs(contract)

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def chat(self, messages, **kwargs):
            prompt = messages[1]["content"]
            case_field = next(
                field
                for field, _ in GENERATION_CASE_FIELDS
                if f"CASE SLOT (never mention this label): {field}" in prompt
            )
            self.calls.append(case_field)
            surface = surfaces[case_field]
            payload = surface.model_dump(mode="json")
            return (
                CallResult(
                    text=canonical_json(payload),
                    raw_response={
                        "choices": [
                            {"message": {"content": canonical_json(payload)}}
                        ]
                    },
                    usage={
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                    latency_ms=1.0,
                    request_hash=f"fake-{case_field}",
                ),
                surface,
            )

        def close(self) -> None:
            return None

    fake = FakeClient()
    monkeypatch.setattr(module, "make_client", lambda endpoint: fake)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    accepted_hash = read_json(out_dir / "cost_estimate.json")[
        "cost_estimate_sha256"
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *common,
            "--run",
            "--accept-cost-estimate-sha256",
            accepted_hash,
        ],
    )
    module.main()
    assert fake.calls == [field for field, _ in GENERATION_CASE_FIELDS]
    assert read_json(out_dir / "summary.json")["status"] == "PASS"
    assert (out_dir / "artifact_attestation.json").is_file()
    ledger_rows = list(
        iter_jsonl(out_dir / "physical_attempt_ledger.jsonl")
    )
    assert len(ledger_rows) == 2 * GENERATION_PILOT_MINIMUM_CALLS
    assert [row["event"] for row in ledger_rows] == [
        event
        for _ in GENERATION_CASE_FIELDS
        for event in ("STARTED", "SUCCEEDED")
    ]

    def forbidden_client(endpoint):
        raise AssertionError("completed pilot recovery must not create a client")

    monkeypatch.setattr(module, "make_client", forbidden_client)
    module.main()


def test_generation_pilot_uses_one_bounded_repair_and_never_repeats_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = (
        PROJECT_ROOT
        / "scripts"
        / "20a_run_pm_v2_generation_compatibility_pilot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pm_v2_generation_pilot_retry", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    seed_path = tmp_path / "seeds.jsonl"
    _write_unique_generation_seeds(seed_path)
    out_dir = tmp_path / "pilot"
    common = [
        str(script),
        "--seed-dialogues",
        str(seed_path),
        "--out-dir",
        str(out_dir),
        "--max-api-calls",
        str(GENERATION_PILOT_MAX_ATTEMPTS),
        "--max-estimated-usd",
        "1",
        "--max-input-tokens-per-call",
        "12000",
    ]
    monkeypatch.setattr(sys, "argv", [*common, "--dry-run"])
    module.main()
    accepted_hash = read_json(out_dir / "cost_estimate.json")[
        "cost_estimate_sha256"
    ]
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    experiment = load_config(PROJECT_ROOT / "configs" / "experiment.yaml")
    pm_config = load_config(PROJECT_ROOT / "configs" / "pm_v2.yaml")
    generation = pm_config["data_generation"]
    endpoint = endpoint_from_config(
        experiment, generation["generator_endpoint"]
    )
    contract = build_generation_compatibility_contract(
        project_root=PROJECT_ROOT,
        experiment_config_path=PROJECT_ROOT / "configs" / "experiment.yaml",
        pm_v2_config_path=PROJECT_ROOT / "configs" / "pm_v2.yaml",
        seed_dialogues_path=seed_path,
        endpoint=endpoint,
        base_generation_seed=int(generation["base_seed"]),
        full_user_count=sum(
            int(generation[key])
            for key in ("train_users", "calibration_users", "internal_test_users")
        ),
        input_token_safety_factor=1.5,
        fail_on_reported_input_overrun=True,
        input_usd_per_mtok=0.15,
        output_usd_per_mtok=0.60,
    )
    surfaces = _surface_only_pilot_inputs(contract)
    invalid_context = surfaces["context_only"].model_copy(
        update={
            "current_user_text": "I feel unsettled today.",
            "session_summary": "The user feels unsettled today.",
        }
    )

    class RepairingFakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []

        def chat(self, messages, **kwargs):
            prompt = messages[1]["content"]
            case_field = next(
                field
                for field, _ in GENERATION_CASE_FIELDS
                if f"CASE SLOT (never mention this label): {field}" in prompt
            )
            repair = "one pre-authorized repair attempt" in prompt
            self.calls.append((case_field, repair))
            surface = (
                invalid_context
                if case_field == "context_only" and not repair
                else surfaces[case_field]
            )
            payload = surface.model_dump(mode="json")
            response = {
                "choices": [
                    {"message": {"content": canonical_json(payload)}}
                ]
            }
            return (
                CallResult(
                    text=canonical_json(payload),
                    raw_response=response,
                    usage={
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                    latency_ms=1.0,
                    request_hash=f"fake-{case_field}-{'repair' if repair else 'initial'}",
                ),
                surface,
            )

        def close(self) -> None:
            return None

    fake = RepairingFakeClient()
    monkeypatch.setattr(module, "make_client", lambda endpoint: fake)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *common,
            "--run",
            "--accept-cost-estimate-sha256",
            accepted_hash,
        ],
    )
    module.main()
    assert fake.calls[:2] == [
        ("context_only", False),
        ("context_only", True),
    ]
    assert len(fake.calls) == GENERATION_PILOT_MINIMUM_CALLS + 1
    failed = read_json(out_dir / "failed_context_only_initial.json")
    assert failed["lint"]["status"] == "FAIL"
    assert failed["lint"]["errors"][0]["check"] == "current_family_anchor"
    ledger_rows = list(iter_jsonl(out_dir / "physical_attempt_ledger.jsonl"))
    assert [row["event"] for row in ledger_rows[:4]] == [
        "STARTED", "FAILED", "STARTED", "SUCCEEDED"
    ]
    assert ledger_rows[1]["usage"]["total_tokens"] == 150
    summary = read_json(out_dir / "summary.json")
    assert summary["status"] == "PASS"
    assert summary["physical_attempts"] == GENERATION_PILOT_MINIMUM_CALLS + 1
    assert summary["repair_cases"] == ["context_only"]
    assert summary["repair_case_count"] == 1

    def forbidden_client(endpoint):
        raise AssertionError("accepted casewise traces must never be repeated")

    monkeypatch.setattr(module, "make_client", forbidden_client)
    module.main()


def test_generation_contract_ignores_downstream_judges_but_binds_generator(
    tmp_path: Path,
) -> None:
    seed_path = tmp_path / "seeds.jsonl"
    _write_unique_generation_seeds(seed_path)
    source_experiment = PROJECT_ROOT / "configs" / "experiment.yaml"
    source_pm = PROJECT_ROOT / "configs" / "pm_v1_5.yaml"
    experiment_path = tmp_path / "experiment.yaml"
    pm_path = tmp_path / "pm.yaml"
    experiment_text = source_experiment.read_text(encoding="utf-8")
    pm_text = source_pm.read_text(encoding="utf-8")
    experiment_path.write_text(experiment_text, encoding="utf-8")
    pm_path.write_text(pm_text, encoding="utf-8")

    def build() -> dict:
        experiment = load_config(experiment_path)
        pm_config = load_config(pm_path)
        generation = pm_config["data_generation"]
        endpoint = endpoint_from_config(
            experiment, str(generation["generator_endpoint"])
        )
        return build_generation_compatibility_contract(
            project_root=PROJECT_ROOT,
            experiment_config_path=experiment_path,
            pm_v2_config_path=pm_path,
            seed_dialogues_path=seed_path,
            endpoint=endpoint,
            base_generation_seed=int(generation["base_seed"]),
            full_user_count=sum(
                int(generation[key])
                for key in ("train_users", "calibration_users", "internal_test_users")
            ),
            input_token_safety_factor=1.5,
            fail_on_reported_input_overrun=True,
            input_usd_per_mtok=0.15,
            output_usd_per_mtok=0.60,
        )

    baseline = build()
    experiment_path.write_text(
        experiment_text.replace('model: "gpt-4o"', 'model: "gpt-4o-review-only"'),
        encoding="utf-8",
    )
    pm_path.write_text(
        pm_text.replace(
            "pm-v1.5-automated-semantic-review-v4-native-gemini-"
            "deterministic27-plus-paid9",
            "pm-v1.5-automated-semantic-review-v4-native-gemini-"
            "deterministic27-plus-paid9-doc-only-change",
        ),
        encoding="utf-8",
    )
    assert build() == baseline

    experiment_path.write_text(
        experiment_text.replace(
            'model: "gpt-4o-mini"', 'model: "gpt-4o-mini-generator-change"', 1
        ),
        encoding="utf-8",
    )
    pm_path.write_text(pm_text, encoding="utf-8")
    assert build()["contract_sha256"] != baseline["contract_sha256"]

    experiment_path.write_text(experiment_text, encoding="utf-8")
    pm_path.write_text(
        pm_text.replace("max_estimated_usd: 0.018", "max_estimated_usd: 0.017"),
        encoding="utf-8",
    )
    assert build()["contract_sha256"] != baseline["contract_sha256"]


def test_generation_compatibility_attestation_is_exact_and_tamper_evident(
    tmp_path: Path,
) -> None:
    seed_path = tmp_path / "seeds.jsonl"
    _write_unique_generation_seeds(seed_path)
    experiment_config_path = PROJECT_ROOT / "configs" / "experiment.yaml"
    pm_v2_config_path = PROJECT_ROOT / "configs" / "pm_v2.yaml"
    experiment_config = load_config(experiment_config_path)
    pm_v2_config = load_config(pm_v2_config_path)
    generation_cfg = pm_v2_config["data_generation"]
    endpoint = endpoint_from_config(
        experiment_config, str(generation_cfg["generator_endpoint"])
    )
    full_user_count = sum(
        int(generation_cfg[key])
        for key in ("train_users", "calibration_users", "internal_test_users")
    )
    contract = build_generation_compatibility_contract(
        project_root=PROJECT_ROOT,
        experiment_config_path=experiment_config_path,
        pm_v2_config_path=pm_v2_config_path,
        seed_dialogues_path=seed_path,
        endpoint=endpoint,
        base_generation_seed=int(generation_cfg["base_seed"]),
        full_user_count=full_user_count,
        input_token_safety_factor=1.5,
        fail_on_reported_input_overrun=True,
        input_usd_per_mtok=0.15,
        output_usd_per_mtok=0.60,
    )
    _, call_plan, _ = build_generation_compatibility_plan(
        contract, endpoint=endpoint
    )
    assert contract["held_out_seed_index"] >= full_user_count
    assert contract["generator_pricing_usd_per_mtok"] == {
        "input": 0.15,
        "output": 0.60,
    }
    assert contract["generation_config_projection"]["protocol"] == (
        "pm-v2-generation-config-projection-v1"
    )
    assert "experiment_config_sha256" not in contract
    assert "pm_v2_config_sha256" not in contract

    pilot_dir = tmp_path / "pilot_artifacts"
    pilot_dir.mkdir()
    attestation_path = pilot_dir / "artifact_attestation.json"
    with pytest.raises(RuntimeError, match="requires a PASS"):
        require_generation_compatibility_attestation(
            attestation_path, expected_contract=contract
        )

    bundle = _surface_only_pilot_bundle(contract)
    bundle_report = validate_generation_pilot_bundle(bundle, contract)
    assert bundle_report["status"] == "PASS"
    bundle_path = pilot_dir / "pilot_bundle.json"
    write_json(bundle_path, bundle.model_dump(mode="json"))

    ledger_path = pilot_dir / "physical_attempt_ledger.jsonl"
    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=GENERATION_PILOT_STAGE,
        expected_calls={
            str(row["physical_call_key"]): 1 for row in call_plan
        },
        maximum_total_attempts=len(call_plan),
    )
    initial_rows = [
        row for row in call_plan if row["attempt_kind"] == "initial"
    ]
    provider_drafts = bundle.provenance["provider_surface_drafts"]
    provider_responses = bundle.provenance["provider_surface_responses"]
    for row in initial_rows:
        case_field = str(row["case_field"])
        reservation = ledger.reserve(
            str(row["physical_call_key"]),
            record_ids={
                "user_id": contract["pilot_user_id"],
                "case_field": case_field,
                "attempt_kind": "initial",
                "generation_seed": row["generation_seed"],
            },
            prompt_sha256=str(row["prompt_sha256"]),
        )
        ledger.finish(
            reservation,
            succeeded=True,
            request_hash=f"test-request-{case_field}",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            error=None,
            result={
                "surface": provider_drafts[case_field],
                "provider_response": provider_responses[case_field],
                "attempt_kind": "initial",
            },
        )
    summary_path = pilot_dir / "summary.json"
    summary = {
        "status": "PASS",
        "compatibility_contract_sha256": contract["contract_sha256"],
        "accepted_surface_calls": initial_rows,
        "accepted_case_count": len(initial_rows),
        "repair_cases": [],
        "repair_case_count": 0,
        "bundle_validation": bundle_report,
        "physical_attempts": len(initial_rows),
    }
    write_json(summary_path, summary)
    run_manifest_path = pilot_dir / "run_manifest.json"
    cost_estimate_path = pilot_dir / "cost_estimate.json"
    call_plan_path = pilot_dir / "call_plan.jsonl"
    write_json(run_manifest_path, {"stage": GENERATION_PILOT_STAGE})
    cost_payload = {
        "stage": GENERATION_PILOT_STAGE,
        "call_plan_sha256": sha256_text(canonical_json(call_plan)),
    }
    accepted_cost_hash = sha256_text(canonical_json(cost_payload))
    write_json(
        cost_estimate_path,
        {
            **cost_payload,
            "cost_estimate_sha256": accepted_cost_hash,
            "budget_gate": {"status": "PASS"},
        },
    )
    write_jsonl(call_plan_path, call_plan)
    create_artifact_attestation(
        attestation_path,
        stage=GENERATION_PILOT_STAGE,
        inputs={
            "experiment_config": experiment_config_path,
            "pm_v2_config": pm_v2_config_path,
            "seed_dialogues": seed_path,
            "run_manifest": run_manifest_path,
            "cost_estimate": cost_estimate_path,
            "call_plan": call_plan_path,
        },
        outputs={
            "pilot_bundle": (bundle_path, False),
            "physical_attempt_ledger": (ledger_path, True),
            "summary": (summary_path, False),
        },
        parameters={
            "compatibility_contract": contract,
            "compatibility_contract_sha256": contract["contract_sha256"],
            "accepted_cost_estimate_sha256": accepted_cost_hash,
            "provider_trace_mode": "surface_only_casewise",
        },
        expected={
            "minimum_physical_attempts": GENERATION_PILOT_MINIMUM_CALLS,
            "maximum_physical_attempts": GENERATION_PILOT_MAX_ATTEMPTS,
            "actual_physical_attempts": len(initial_rows),
            "regimes": 9,
            "maximum_repairs_per_case": 1,
        },
    )
    verification = require_generation_compatibility_attestation(
        attestation_path, expected_contract=contract
    )
    assert verification["status"] == "PASS"

    review_dir = tmp_path / "generation_semantic_review"
    prepared = prepare_generation_pilot_semantic_review(
        pilot_attestation_path=attestation_path,
        out_dir=review_dir,
    )
    packet_path = Path(prepared["packet"])
    with packet_path.open("r", encoding="utf-8", newline="") as handle:
        packet_rows = list(csv.DictReader(handle))
    completed_paths = []
    for annotator_id in ("reviewer_a", "reviewer_b"):
        completed_path = review_dir / f"completed_{annotator_id}.csv"
        completed_rows = []
        for source_row in packet_rows:
            row = (
                {"item_id": source_row["item_id"]}
                if annotator_id == "reviewer_a"
                else dict(source_row)
            )
            row["annotator_id"] = annotator_id
            row["notes"] = ""
            for field in GENERATION_REVIEW_RATING_FIELDS:
                row[field] = "1"
            completed_rows.append(row)
        with completed_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = (
                GENERATION_SIMPLE_REVIEW_FIELDS
                if annotator_id == "reviewer_a"
                else GENERATION_REVIEW_PACKET_FIELDS
            )
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames
            )
            writer.writeheader()
            writer.writerows(completed_rows)
        completed_paths.append(completed_path)
    review_report_path = review_dir / "generation_pilot_semantic_review_report.json"
    review_attestation_path = review_dir / "semantic_attestation.json"
    review_report = analyze_generation_pilot_semantic_review(
        completed_paths=completed_paths,
        pilot_attestation_path=attestation_path,
        packet_path=packet_path,
        manual_path=Path(prepared["manual"]),
        plan_path=Path(prepared["plan"]),
        report_path=review_report_path,
        attestation_path=review_attestation_path,
    )
    assert review_report["status"] == "PASS"
    review_verification = require_generation_pilot_semantic_review(
        review_attestation_path,
        expected_pilot_attestation_path=attestation_path,
        expected_contract=contract,
    )
    assert review_verification["annotation_count"] == 18

    with completed_paths[1].open("r", encoding="utf-8", newline="") as handle:
        rejected_rows = list(csv.DictReader(handle))
    rejected_rows[0][GENERATION_REVIEW_RATING_FIELDS[0]] = "0"
    with completed_paths[1].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=GENERATION_REVIEW_PACKET_FIELDS
        )
        writer.writeheader()
        writer.writerows(rejected_rows)
    rejected_report = analyze_generation_pilot_semantic_review(
        completed_paths=completed_paths,
        pilot_attestation_path=attestation_path,
        packet_path=packet_path,
        manual_path=Path(prepared["manual"]),
        plan_path=Path(prepared["plan"]),
        report_path=review_report_path,
        attestation_path=review_attestation_path,
        overwrite=True,
    )
    assert rejected_report["status"] == "FAIL"
    with pytest.raises(RuntimeError, match="not PASS"):
        require_generation_pilot_semantic_review(
            review_attestation_path,
            expected_pilot_attestation_path=attestation_path,
            expected_contract=contract,
        )

    wrong_contract = json.loads(json.dumps(contract))
    wrong_contract["generator_pricing_usd_per_mtok"]["input"] = 0.16
    with pytest.raises(RuntimeError, match="lineage differs"):
        require_generation_compatibility_attestation(
            attestation_path, expected_contract=wrong_contract
        )

    write_json(summary_path, {**summary, "status": "FAIL"})
    with pytest.raises(RuntimeError, match="outputs hash mismatch"):
        require_generation_compatibility_attestation(
            attestation_path, expected_contract=contract
        )

    invalid_bundle = _bundle().model_copy(
        update={
            "user_id": contract["pilot_user_id"],
            "provenance": {
                "generation_compatibility_contract_sha256": contract[
                    "contract_sha256"
                ]
            },
        }
    )
    invalid_report = validate_generation_pilot_bundle(invalid_bundle, contract)
    assert invalid_report["status"] == "FAIL"
    assert invalid_report["checks"]["exact_nine_regimes"] is False


def test_full_generation_run_requires_pilot_attestation_before_api_key(
    tmp_path: Path,
) -> None:
    seed_path = tmp_path / "seeds.jsonl"
    _write_unique_generation_seeds(seed_path)
    out_dir = tmp_path / "full_generation"
    missing_attestation = tmp_path / "missing_pilot" / "artifact_attestation.json"
    env = _subprocess_env()
    env.pop("NVIDIA_API_KEY", None)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "20_generate_pm_v2_development_data.py"),
        "--seed-dialogues",
        str(seed_path),
        "--out-dir",
        str(out_dir),
        "--generation-pilot-attestation",
        str(missing_attestation),
        "--max-api-calls",
        "936",
        "--max-estimated-usd",
        "1",
        "--max-input-tokens-per-call",
        "12000",
    ]
    subprocess.run(
        [*command, "--dry-run"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    accepted = read_json(out_dir / "generation_cost_estimate.json")[
        "cost_estimate_sha256"
    ]
    blocked = subprocess.run(
        [
            *command,
            "--run",
            "--accept-cost-estimate-sha256",
            accepted,
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert blocked.returncode != 0
    assert "requires a PASS generation compatibility pilot attestation" in (
        blocked.stderr
    )
    assert "NVIDIA_API_KEY" not in blocked.stderr


def test_human_rating_manual_has_shared_one_to_five_anchors() -> None:
    script = PROJECT_ROOT / "scripts" / "27_prepare_pm_v2_human_audit.py"
    spec = importlib.util.spec_from_file_location("pm_v2_human_packet", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.RESPONSE_SCALE_ANCHORS) == {1, 2, 3, 4, 5}
    assert set(module.RISK_SCALE_ANCHORS) == {0, 1, 2, 3}
    assert set(module.RESPONSE_DIMENSION_DEFINITIONS) == set(module.RESPONSE_FIELDS)
    assert set(module.RESPONSE_DIMENSION_BOUNDARIES) == set(module.RESPONSE_FIELDS)
    assert set(module.RISK_DIMENSION_DEFINITIONS) == set(module.RISK_FIELDS)
    manual = module.rating_manual_markdown(2)
    assert module.MANUAL_VERSION in manual
    assert "Score every dimension independently" in manual
    assert "Do not enter, infer, average, or report an `overall`" in manual
    assert "Generic empathy can score high" in manual
    assert "An appropriate decision not to mention memory can score 5" in manual
    assert "score grounding independently from temporal_consistency" in manual
    assert "at least 2 independent annotators" in manual
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--seed" not in help_result.stdout


def test_human_sample_plan_is_yaml_seeded_and_hash_bound(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs/pm_v2.yaml").read_text(encoding="utf-8")
    )
    config["human_label_audit"]["items"] = 1
    config["human_label_audit"]["sample_seed"] = 9876
    config_path = tmp_path / "pm_v2.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    case = _case(case_id="human_sample_plan")
    state = case_to_state(
        user_id="human_sample_user",
        case=case,
        split=PMV2Split.TRAIN,
        strategy_catalog_count=12429,
        strategy_estimated_tokens=260,
    )
    states_path = tmp_path / "states.jsonl"
    states_path.write_text(
        json.dumps(state.model_dump(mode="json")) + "\n", encoding="utf-8"
    )
    evaluator_path = tmp_path / "evaluator_contexts.jsonl"
    evaluator_path.write_text(
        json.dumps(case_to_evaluator_context(state, case)) + "\n",
        encoding="utf-8",
    )
    outcomes_path = tmp_path / "outcomes.jsonl"
    outcome = {
        "card_id": state.card_id,
        "state_id": state.state_id,
        "user_id": state.user_id,
        "action_id": "M0+R0",
        "response": "That uncertainty sounds difficult; we can take this one step at a time.",
        "selected_memory_ids": [],
        "selected_strategy_ids": [],
        "memory_view": [],
        "strategy_view": [],
        "cost": {
            "pm_input_tokens_est": 100,
            "retrieval_calls": 0,
            "reranker_calls": 0,
            "memory_tokens": 0,
            "strategy_tokens": 0,
            "base_prompt_tokens": 100,
            "total_input_tokens": 100,
            "output_tokens": 15,
            "latency_ms": 1.0,
        },
        "model_name": "fixture-generator",
        "prompt_hash": "fixture-prompt",
        "request_hash": "fixture-request",
        "provenance": {},
    }
    outcomes_path.write_text(json.dumps(outcome) + "\n", encoding="utf-8")

    quality_cfg = config["quality_composite"]
    composite = CompositeSpec(
        version=quality_cfg["version"], weights=quality_cfg["weights"]
    )
    response = ResponseDimensions(
        emotional_support=4,
        personalization=3,
        memory_appropriateness=5,
        factual_grounding=5,
        temporal_consistency=5,
        non_intrusiveness=5,
    )
    risk = RiskDimensions(
        selected_context_misuse=0,
        unnecessary_exposure=0,
        stale_or_conflicting_use=0,
        unsupported_personal_claim=0,
        memory_omission=0,
        strategy_overuse=0,
        strategy_omission=0,
    )
    dimension_mad = {
        **{f"response.{name}": 0.0 for name in ResponseDimensions.model_fields},
        **{f"risk.{name}": 0.0 for name in RiskDimensions.model_fields},
    }
    label = ActionLabel(
        state_id=state.state_id,
        card_id=state.card_id,
        user_id=state.user_id,
        semantic_family=state.semantic_family,
        action_id="M0+R0",
        response=response,
        risk=risk,
        observed_input_tokens=100,
        retrieval_calls=0,
        judge_families=["fixture-a", "fixture-b"],
        judge_count=2,
        max_dimension_mad=0.0,
        dimension_mad=dimension_mad,
        label_reliable=True,
        composite_spec_version=composite.version,
        composite_weights_sha256=sha256_text(canonical_json(composite.weights)),
    )
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(
        json.dumps(label.model_dump(mode="json")) + "\n", encoding="utf-8"
    )
    manifest_path = tmp_path / "judge_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "prompt_contract_hash": prompt_contract_hash(),
                "pm_v2_config_sha256": sha256_file(config_path),
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "human"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/27_prepare_pm_v2_human_audit.py"),
        "--pm-v2-config",
        str(config_path),
        "--states",
        str(states_path),
        "--evaluator-contexts",
        str(evaluator_path),
        "--outcomes",
        str(outcomes_path),
        "--labels",
        str(labels_path),
        "--judge-manifest",
        str(manifest_path),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=True,
    )
    key = json.loads(
        (out_dir / "human_rating_key.json").read_text(encoding="utf-8")
    )
    plan = json.loads(
        (out_dir / "human_sample_plan.json").read_text(encoding="utf-8")
    )
    assert plan["sample_seed"] == 9876
    assert plan["seed_source"] == "pm_v2.yaml:human_label_audit.sample_seed"
    assert key["sample_plan"]["sha256"] == sha256_text(canonical_json(plan))
    assert key["sample_plan"]["file_sha256"] == sha256_file(
        out_dir / "human_sample_plan.json"
    )
    assert key["llm_rubric_contract"]["prompt_contract_sha256"] == (
        prompt_contract_hash()
    )
    assert key["llm_rubric_contract"]["requests_overall_field"] is False
    packet_header = (out_dir / "human_rating_packet.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert "overall" not in packet_header
