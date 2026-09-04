from __future__ import annotations

import numpy as np
import pytest
import metacom_pm.pm_v1_5_rule_router as rule_router_module

from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.contracts import DialogueTurn, MemorySource, StrategyCard
from metacom_pm.io import sha256_file, write_json
from metacom_pm.pm_v1_5_step0 import (
    STRATEGY_LABEL_TO_FAMILY_ID,
    build_step0_observation,
    build_strategy_family_catalog,
    semantic_strategy_readiness,
)
from metacom_pm.pm_v1_5_rule_router import (
    TransparentRuleConfig,
    TransparentRuleRouter,
    compare_development_external_score_diagnostics,
    transparent_rule_candidates,
    transparent_rule_grid_diagnostics,
    transparent_rule_score_diagnostics,
    tune_transparent_rule_router,
)
from metacom_pm.pm_v1_5_shortcut_audit import (
    SHORTCUT_AUDIT_PROTOCOL,
    SHORTCUT_AUDIT_STAGE,
    _group_cv_probe_scores,
    audit_step0_shortcuts,
    require_step0_shortcut_audit_pass,
)
from metacom_pm.pm_v2_contracts import (
    STRATEGY_FAMILY_IDS,
    ObservableSourceSummary,
    PMV2Split,
    PMV2State,
)
from metacom_pm.pm_v2_features import PMV2FeatureBuilder
from metacom_pm.pm_v2_model import (
    SelectionConfig,
    TRANSPARENT_RULE_SELECTION_REASON,
    decision_fallback_kind,
)
from metacom_pm.pm_v2_data import EvaluatorContextIndex


def _cards() -> list[StrategyCard]:
    return [
        StrategyCard(
            strategy_id=f"strat_{index:012x}",
            strategy_label=label,
            retrieval_text=f"Distinct retrieval guidance for {family} support.",
            guidance_text=f"Guidance for {family}.",
            example_response=f"Example for {family}.",
            source_dialogue_id=f"esconv_{index:04d}",
            source_turn_index=index,
        )
        for index, (label, family) in enumerate(
            STRATEGY_LABEL_TO_FAMILY_ID.items(), 1
        )
    ]


def _inventory(embedding: list[float]) -> dict[MemorySource, ObservableSourceSummary]:
    return {
        MemorySource.MP: ObservableSourceSummary(
            available=True,
            count=2,
            min_age_sessions=1,
            median_age_sessions=2.0,
            max_age_sessions=3,
            estimated_tokens=100,
            query_similarity_mean=0.25,
            representation_valid=True,
            catalog_embedding=embedding,
        ),
        MemorySource.MS: ObservableSourceSummary(
            available=False,
            count=0,
            estimated_tokens=0,
        ),
        MemorySource.ME: ObservableSourceSummary(
            available=False,
            count=0,
            estimated_tokens=0,
        ),
    }


def _state(embedding: list[float]) -> PMV2State:
    inventory = _inventory(embedding)
    query = "How can I handle this? I would appreciate advice."
    step0 = build_step0_observation(
        query_text=query,
        inventory=inventory,
        strategy_catalog_count=8,
        strategy_estimated_tokens=180,
        strategy_family_catalog=build_strategy_family_catalog(_cards()),
    )
    return PMV2State(
        state_id="state_step0",
        card_id="card_step0",
        user_id="user_step0",
        split=PMV2Split.TRAIN,
        semantic_family="step0_family",
        surface_form_id="step0_surface",
        current_user_text=query,
        current_session_history=[
            DialogueTurn(role="user", content="I am unsure what to do next.")
        ],
        current_session_summary="The user is deciding how to respond.",
        session_index=4,
        inventory=inventory,
        strategy_catalog_count=8,
        strategy_estimated_tokens=180,
        step0_observation=step0,
        allowed_actions=["M0+R0", "M0+RS", "MP+R0", "MP+RS"],
    )


def test_strategy_step0_is_deterministic_and_exposes_only_scalars() -> None:
    catalog_left = build_strategy_family_catalog(_cards())
    catalog_right = build_strategy_family_catalog(_cards())
    assert catalog_left.digest == catalog_right.digest
    assert set(catalog_left.vectors) == set(STRATEGY_FAMILY_IDS)

    state = _state([0.1, 0.2, 0.3])
    payload = state.step0_observation.model_dump(mode="json")
    assert payload["observation_stage"] == "pre_item_retrieval"
    assert set(payload["strategy"]["family_similarities"]) == set(
        STRATEGY_FAMILY_IDS
    )
    serialized = str(payload).casefold()
    assert "sha256" not in serialized
    assert "vector" not in serialized
    assert "encoder" not in serialized
    assert "card_id" not in serialized


def test_embedding_fingerprint_statistics_cannot_change_pm_features() -> None:
    ordinary = _state([0.1, 0.2, 0.3])
    adversarial = _state([1000.0, -1000.0, 7.0, 99.0])
    builder = PMV2FeatureBuilder(use_precomputed_embeddings=False)
    assert np.allclose(builder._metadata_raw(ordinary), builder._metadata_raw(adversarial))
    assert "catalog_embedding" not in ordinary.model_dump(mode="json")["inventory"]["MP"]


def test_no_step0_ablation_masks_all_step0_scalars() -> None:
    ordinary = _state([0.1, 0.2, 0.3])
    memory = dict(ordinary.step0_observation.memory_sources)
    memory[MemorySource.MP] = memory[MemorySource.MP].model_copy(
        update={"query_to_source_similarity": 0.99}
    )
    changed_step0 = ordinary.step0_observation.model_copy(
        update={"memory_sources": memory}
    )
    changed = ordinary.model_copy(update={"step0_observation": changed_step0})

    full = PMV2FeatureBuilder(
        use_precomputed_embeddings=False, step0_signal_mode="full"
    )
    ablation = PMV2FeatureBuilder(
        use_precomputed_embeddings=False, step0_signal_mode="none"
    )
    assert not np.allclose(full._metadata_raw(ordinary), full._metadata_raw(changed))
    assert np.allclose(ablation._metadata_raw(ordinary), ablation._metadata_raw(changed))
    assert ablation.estimate_action_cost(ordinary, "MP+RS") == ablation.estimate_action_cost(
        changed, "MP+RS"
    )


def test_strategy_family_mapping_and_readiness_fail_closed() -> None:
    bad = _cards()
    bad[0] = bad[0].model_copy(update={"strategy_label": "Unreviewed New Label"})
    with pytest.raises(ValueError, match="unmapped Strategy labels"):
        build_strategy_family_catalog(bad)
    catalog = build_strategy_family_catalog(_cards())
    readiness = semantic_strategy_readiness(
        "Please just listen, no advice.", catalog, None
    )
    assert set(readiness) == {
        "listen_only",
        "explore_first",
        "light_suggestion",
        "structured_plan",
        "ambiguous",
    }
    assert readiness["listen_only"] == max(readiness.values())


def test_shortcut_audit_covers_all_468_states_without_outcome_labels() -> None:
    template = _state([0.1, 0.2, 0.3])
    inventory = dict(template.inventory)
    memory_observations = dict(template.step0_observation.memory_sources)
    for source in (MemorySource.MS, MemorySource.ME):
        inventory[source] = inventory[MemorySource.MP].model_copy(deep=True)
        memory_observations[source] = memory_observations[
            MemorySource.MP
        ].model_copy(deep=True)
    template = template.model_copy(
        deep=True,
        update={
            "inventory": inventory,
            "step0_observation": template.step0_observation.model_copy(
                deep=True, update={"memory_sources": memory_observations}
            ),
            "allowed_actions": [
                "M0+R0",
                "M0+RS",
                "MP+R0",
                "MP+RS",
                "MS+R0",
                "MS+RS",
                "ME+R0",
                "ME+RS",
                "MPMS+R0",
                "MPMS+RS",
                "MPE+R0",
                "MPE+RS",
                "MSE+R0",
                "MSE+RS",
                "MPMSME+R0",
                "MPMSME+RS",
            ],
        },
    )
    regimes = ("strategy_helpful", "strategy_harmful", "ambiguous")
    needed = (["MP"], ["MS"], ["ME"], [])
    states = []
    contexts = {}
    for index in range(468):
        state_id = f"state_audit_{index:04d}"
        state = template.model_copy(
            deep=True,
            update={
                "state_id": state_id,
                "card_id": f"card_audit_{index:04d}",
                "user_id": f"user_{index // 9:03d}",
                "semantic_family": f"family_{index % 18:02d}",
                "surface_form_id": f"surface_{index:04d}",
                "provenance": {
                    "evaluator_context_id": f"eval_{index:04d}"
                },
            },
        )
        states.append(state)
        contexts[state_id] = {
            "state_id": state_id,
            "card_id": state.card_id,
            "evaluator_context_id": f"eval_{index:04d}",
                "regime": regimes[index % len(regimes)],
                "strategy_resource_target": (
                    "use"
                    if regimes[index % len(regimes)] == "strategy_helpful"
                    else "skip"
                    if regimes[index % len(regimes)] == "strategy_harmful"
                    else "ambiguous"
                ),
                "advice_readiness_target": "ambiguous",
                "needed_memory_sources": needed[index % len(needed)],
                "memory_annotations": [],
            }
    index = EvaluatorContextIndex(
        by_state=contexts,
        by_card={row["card_id"]: row for row in contexts.values()},
        source_sha256="a" * 64,
        map_sha256="b" * 64,
    )
    report = audit_step0_shortcuts(
        predictive_states=states[:216],
        structural_states=states,
        evaluator_contexts=index,
        expected_predictive_states=216,
        expected_structural_states=468,
        maximum_single_threshold_balanced_accuracy=0.90,
        maximum_multivariate_probe_balanced_accuracy=0.90,
        centroid_noise_std=0.05,
        shuffle_seed=6113,
        group_cv_folds=6,
        probe_logistic_c=1.0,
        probe_logistic_max_iter=1000,
        probe_tree_max_depth=3,
        probe_tree_min_samples_leaf=9,
    )
    assert report["status"] == "PASS"
    assert report["n_structural_states"] == 468
    assert report["n_predictive_states"] == 216
    assert report["outcome_labels_read"] is False
    assert report["checks"]["all_split_structural_state_count"]
    assert report["checks"]["multivariate_probe_not_near_oracle"]
    assert report["internal_resource_oracle_read"] is False


def test_grouped_multivariate_probe_detects_combined_xor_shortcut() -> None:
    rows = []
    targets = []
    groups = []
    for user_index in range(24):
        for left, right in ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)):
            rows.append([left, right])
            targets.append(bool(left) ^ bool(right))
            groups.append(f"user_{user_index:02d}")
    report = _group_cv_probe_scores(
        np.asarray(rows, dtype=float),
        np.asarray(targets, dtype=bool),
        groups,
        folds=6,
        logistic_c=1.0,
        logistic_max_iter=1000,
        tree_max_depth=3,
        tree_min_samples_leaf=2,
        random_seed=6113,
    )
    assert report["scores"]["shallow_tree"] == 1.0
    assert report["maximum_balanced_accuracy"] == 1.0


def test_shortcut_pre_sweep_attestation_is_content_bound(tmp_path) -> None:
    states = tmp_path / "states.jsonl"
    evaluator = tmp_path / "evaluator.jsonl"
    config = tmp_path / "pm_v1_5.yaml"
    states.write_text('{"state_id":"s1"}\n', encoding="utf-8")
    evaluator.write_text('{"state_id":"s1"}\n', encoding="utf-8")
    config.write_text(
        "version: pm-v1.5\n"
        "shortcut_audit:\n"
        f"  protocol: {SHORTCUT_AUDIT_PROTOCOL}\n"
        "  expected_predictive_train_states: 216\n"
        "  expected_structural_all_split_states: 468\n"
        "  maximum_single_threshold_balanced_accuracy: 0.90\n"
        "  maximum_multivariate_probe_balanced_accuracy: 0.90\n"
        "  fail_on_near_oracle_threshold: true\n"
        "  fail_on_near_oracle_multivariate_probe: true\n",
        encoding="utf-8",
    )
    data_attestation = tmp_path / "data_attestation.json"
    create_artifact_attestation(
        data_attestation,
        stage="pm_v1_5_development_data",
        inputs={"pm_v1_5_config": config},
        outputs={
            "states": (states, True),
            "evaluator_contexts": (evaluator, True),
        },
        parameters={"track": "pm-v1.5"},
    )
    report_path = tmp_path / "shortcut.json"
    input_hashes = {
        "states": sha256_file(states),
        "evaluator_contexts": sha256_file(evaluator),
        "pm_v1_5_config": sha256_file(config),
        "development_data_attestation": sha256_file(data_attestation),
    }
    write_json(
        report_path,
        {
            "status": "PASS",
            "protocol": SHORTCUT_AUDIT_PROTOCOL,
            "n_predictive_states": 216,
            "n_structural_states": 468,
            "maximum_single_threshold_balanced_accuracy": 0.90,
            "maximum_multivariate_probe_balanced_accuracy": 0.90,
            "input_hashes": input_hashes,
            "checks": {
                "single_threshold_not_near_oracle": True,
                "multivariate_probe_not_near_oracle": True,
            },
        },
    )
    attestation = tmp_path / "shortcut_attestation.json"
    create_artifact_attestation(
        attestation,
        stage=SHORTCUT_AUDIT_STAGE,
        inputs={
            "states": states,
            "evaluator_contexts": evaluator,
            "pm_v1_5_config": config,
            "development_data_attestation": data_attestation,
        },
        outputs={"shortcut_audit_report": (report_path, False)},
        parameters={"outcome_labels_read": False},
    )
    assert require_step0_shortcut_audit_pass(
        report_path,
        attestation,
        expected_states_path=states,
        expected_evaluator_contexts_path=evaluator,
        expected_pm_config_path=config,
        expected_generation_attestation_path=data_attestation,
    )["status"] == "PASS"
    states.write_text('{"state_id":"changed"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="attestation|bind current"):
        require_step0_shortcut_audit_pass(
            report_path,
            attestation,
            expected_states_path=states,
            expected_evaluator_contexts_path=evaluator,
            expected_pm_config_path=config,
            expected_generation_attestation_path=data_attestation,
        )


def test_transparent_rule_router_uses_only_frozen_step0_scalars() -> None:
    config = TransparentRuleConfig(
        source_similarity_weight=1.0,
        source_age_penalty=0.0,
        source_cost_penalty=0.0,
        source_minimum_score=0.20,
        maximum_memory_sources=1,
        strategy_family_similarity_weight=0.0,
        strategy_readiness_alignment_weight=1.0,
        question_bonus=0.0,
        strategy_cost_penalty=0.0,
        strategy_minimum_score=-1.0,
    )
    router = TransparentRuleRouter.create(config, SelectionConfig())
    state = _state([0.1, 0.2, 0.3])

    decision = router.choose(state)

    assert decision.chosen_action == "MP+RS"
    assert decision.config_hash == config.digest()
    assert set(decision.predictions) == set(state.allowed_actions)
    assert decision.decision_reason == TRANSPARENT_RULE_SELECTION_REASON
    assert decision_fallback_kind(decision) is None


def test_transparent_rule_tuning_reports_train_role_not_calibration(
    monkeypatch,
) -> None:
    metrics = {
        "mean_quality": 0.8,
        "mean_risk": 0.1,
        "mean_realized_utility": 0.7,
        "mean_observed_input_tokens": 10.0,
        "action_distribution": {"MP+RS": 1},
    }
    monkeypatch.setattr(
        rule_router_module, "evaluate_policy", lambda router, states, labels: metrics
    )
    config = TransparentRuleConfig(
        source_similarity_weight=1.0,
        source_age_penalty=0.0,
        source_cost_penalty=0.0,
        source_minimum_score=0.20,
        maximum_memory_sources=1,
        strategy_family_similarity_weight=0.0,
        strategy_readiness_alignment_weight=1.0,
        question_bonus=0.0,
        strategy_cost_penalty=0.0,
        strategy_minimum_score=-1.0,
    )
    _, report = tune_transparent_rule_router(
        states=[_state([0.1, 0.2, 0.3])],
        labels=[],
        selection_config=SelectionConfig(),
        candidates=[config],
        minimum_quality=0.0,
        maximum_risk=1.0,
        selection_data_role="train",
    )
    assert report["selection_split"] == "train"
    assert report["selection_data_role"] == "train_only"
    assert report["score_diagnostics"]["outcome_labels_used"] is False


def test_transparent_rule_grid_is_finite_and_fail_closed() -> None:
    grid = {
        "source_similarity_weights": [1.0],
        "source_age_penalties": [0.0],
        "source_cost_penalties": [0.0],
        "source_minimum_scores": [0.1, 0.2],
        "maximum_memory_sources": [1],
        "strategy_family_similarity_weights": [1.0],
        "strategy_readiness_alignment_weights": [0.2],
        "question_bonuses": [0.0],
        "strategy_cost_penalties": [0.0],
        "strategy_minimum_scores": [0.1, 0.2],
    }
    assert len(transparent_rule_candidates(grid)) == 4
    with pytest.raises(ValueError, match="grid keys"):
        transparent_rule_candidates({**grid, "unexpected": [1.0]})


def test_rule_grid_diagnostic_hashes_state_mappings_not_only_action_totals() -> None:
    common = dict(
        source_similarity_weight=1.0,
        source_age_penalty=0.0,
        source_cost_penalty=0.0,
        maximum_memory_sources=1,
        strategy_family_similarity_weight=0.0,
        strategy_readiness_alignment_weight=0.0,
        question_bonus=0.0,
        strategy_cost_penalty=0.0,
        strategy_minimum_score=1.0,
    )
    report = transparent_rule_grid_diagnostics(
        [_state([0.1, 0.2, 0.3])],
        [
            TransparentRuleConfig(source_minimum_score=0.20, **common),
            TransparentRuleConfig(source_minimum_score=0.30, **common),
        ],
        SelectionConfig(),
        minimum_unique_policy_mappings=2,
        minimum_maximum_pairwise_disagreement_rate=0.5,
    )
    assert report["status"] == "PASS"
    assert report["unique_policy_mapping_count"] == 2
    assert report["maximum_pairwise_action_disagreement_rate"] == 1.0
    assert len({row["action_by_state_sha256"] for row in report["candidates"]}) == 2


def test_development_external_score_comparison_is_report_only() -> None:
    diagnostics = transparent_rule_score_diagnostics(
        [_state([0.1, 0.2, 0.3])]
    )
    report = compare_development_external_score_diagnostics(
        {"calibration": diagnostics}, diagnostics
    )
    assert report["status"] == "REPORT_ONLY"
    assert report["outcome_labels_used"] is False
    assert report["external_threshold_selection_or_retuning_authorized"] is False
    assert all(
        row["quantile_l1_shift"] == 0.0
        for row in report["metrics"].values()
        if row["status"] == "AVAILABLE"
    )
    assert report["unavailable_distributions"] == [
        "source_similarity.ME",
        "source_similarity.MS",
    ]
