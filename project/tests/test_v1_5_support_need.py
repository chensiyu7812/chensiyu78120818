from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from metacom_pm.contracts import DialogueTurn, MemorySource
from metacom_pm.io import canonical_json, sha256_text
from metacom_pm.pm_v1_5_semantic import (
    FrozenSemanticEncoderSpec,
    SemanticEncoderBinding,
)
from metacom_pm.pm_v2_contracts import (
    ObservableSourceSummary,
    PMV2Split,
    PMV2State,
)
from metacom_pm.v1_5_support_need import (
    DIALOGUE_PHASE_IDS,
    GOAL_IDS,
    NONCLINICAL_URGENCY_IDS,
    SUPPORT_MODE_IDS,
    ExplicitSupportBoundaries,
    MemorySourceOpportunityObservation,
    StrategyFamilyOpportunityObservation,
    SupportNeedObservation,
    SupportNeedPartialLabel,
    SupportNeedUncertainty,
    cross_fit_categorical_soft_head,
    effective_sample_size,
    explicit_support_boundaries,
    group_effective_sample_size,
    group_normalized_weights,
    prepare_support_need_views,
    run_support_need_learning_canary,
    run_train_only_support_need_pilot,
)


class _CanaryEncoder:
    def __init__(self):
        self.spec = FrozenSemanticEncoderSpec(
            model_id="local/canary",
            revision="a" * 40,
            snapshot_tree_sha256="b" * 64,
            max_length=128,
            output_dimension=16,
        )
        self.binding = SemanticEncoderBinding(
            spec_sha256=self.spec.digest(),
            snapshot_tree_sha256=self.spec.snapshot_tree_sha256,
            snapshot_file_count=1,
            implementation="transformers-auto-model-cls-float32",
        )

    def encode(self, texts):
        rows = []
        for text in texts:
            seed = int(sha256_text(str(text))[:16], 16)
            rng = np.random.default_rng(seed)
            vector = rng.normal(size=self.spec.output_dimension)
            vector /= np.linalg.norm(vector)
            rows.append(vector)
        return np.asarray(rows)


def _state() -> PMV2State:
    inventory = {
        source: ObservableSourceSummary(available=False, count=0)
        for source in MemorySource
    }
    return PMV2State(
        state_id="state_1",
        card_id="card_1",
        user_id="user_1",
        split=PMV2Split.TRAIN,
        semantic_family="test",
        surface_form_id="surface",
        current_user_text="I only want someone to listen, one question at a time.",
        current_session_history=[
            DialogueTurn(role="user", content="Everything feels tangled."),
            DialogueTurn(role="assistant", content="What feels heaviest right now?"),
        ],
        current_session_summary="The user feels overwhelmed.",
        session_index=1,
        inventory=inventory,
        allowed_actions=["M0+R0", "M0+RS"],
    )


def _distribution(keys, winner):
    return {key: 1.0 if key == winner else 0.0 for key in keys}


def test_explicit_support_boundaries_are_observable_not_action_labels():
    boundaries = explicit_support_boundaries(
        "I don't want advice. Please just listen and ask one question at a time."
    )
    assert boundaries.advice_rejected
    assert boundaries.listen_first_requested
    assert boundaries.question_or_task_burden_limit
    assert not boundaries.advice_requested


def test_prepare_support_need_views_is_four_views_plus_delta_and_outcome_blind():
    prepared = prepare_support_need_views(_CanaryEncoder(), _state())
    assert prepared.matrix.shape == (5, 16)
    assert prepared.view_names[-1] == "current_vs_context_delta"
    assert np.isclose(np.linalg.norm(prepared.matrix[-1]), 1.0)
    assert prepared.audit["item_level_retrieval_performed"] is False
    assert prepared.audit["internal_test_outcomes_opened"] is False
    assert prepared.audit["external_outcomes_opened"] is False
    assert prepared.explicit_boundaries.listen_first_requested


def test_support_need_contract_separates_modes_from_uncertainty():
    state = _state()
    prepared = prepare_support_need_views(_CanaryEncoder(), state)
    zero_memory = {
        source: MemorySourceOpportunityObservation(
            available=False,
            eligible_count=0,
            semantic_fit=0.0,
            recency_compatibility=0.0,
            expected_cost=0.0,
            opportunity_probability=0.0,
            uncertainty=1.0,
        )
        for source in MemorySource
    }
    zero_strategy = {
        family: StrategyFamilyOpportunityObservation(
            eligible_card_count=0,
            semantic_fit=0.0,
            mode_phase_goal_compatibility=0.0,
            directive_burden_fit=0.0,
            opportunity_probability=0.0,
            uncertainty=1.0,
        )
        for family in (
            "question",
            "other",
            "suggestion",
            "affirmation_reassurance",
            "self_disclosure",
            "reflection",
            "information",
            "restatement",
        )
    }
    observation = SupportNeedObservation(
        state_id=state.state_id,
        user_id=state.user_id,
        explicit_boundaries=prepared.explicit_boundaries,
        support_mode_distribution=_distribution(SUPPORT_MODE_IDS, "listen"),
        goal_probabilities={goal: 0.5 for goal in GOAL_IDS},
        dialogue_phase_distribution=_distribution(
            DIALOGUE_PHASE_IDS, "comforting"
        ),
        nonclinical_urgency_distribution=_distribution(
            NONCLINICAL_URGENCY_IDS, "elevated"
        ),
        memory_source_opportunity=zero_memory,
        strategy_family_opportunity=zero_strategy,
        uncertainty=SupportNeedUncertainty(
            predictive_entropy=0.2,
            group_bootstrap_dispersion=0.1,
            semantic_ood=False,
            metadata_ood=False,
            abstain=False,
            abstain_reasons=[],
        ),
        visible_view_sha256=prepared.audit["visible_view_sha256"],
        feature_contract_sha256=prepared.audit["feature_contract_sha256"],
        provenance={
            "internal_test_outcomes_opened": False,
            "external_outcomes_opened": False,
        },
    )
    assert "ambiguous" not in observation.support_mode_distribution
    with pytest.raises(ValueError, match="support mode"):
        SupportNeedObservation.model_validate(
            {
                **observation.model_dump(mode="json"),
                "support_mode_distribution": {
                    **observation.support_mode_distribution,
                    "ambiguous": 0.0,
                },
            }
        )


def test_group_normalization_prevents_duplicate_rows_inflating_ess():
    original = group_normalized_weights(["a", "b", "c"])
    duplicated = group_normalized_weights(["a", "a", "b", "b", "c", "c"])
    assert effective_sample_size(original) == pytest.approx(3.0)
    # Row-level ESS would be six here; the learning contract must aggregate
    # repeated measurements back to their three independent groups.
    assert effective_sample_size(duplicated) == pytest.approx(6.0)
    assert group_effective_sample_size(
        ["a", "a", "b", "b", "c", "c"], duplicated
    ) == pytest.approx(3.0)
    low_reliability = group_normalized_weights(
        ["a", "b"], base_weights=[0.2, 0.4]
    )
    assert low_reliability.tolist() == pytest.approx([0.2, 0.4])


def test_cross_fit_rejects_group_leakage_by_construction_and_caps_dimension():
    features = []
    targets = []
    groups = []
    for class_index, class_id in enumerate(("a", "b")):
        for index in range(15):
            row = np.zeros(20)
            row[class_index] = 4.0
            row[2 + class_index] = float(index) / 100.0
            features.append(row.tolist())
            targets.append(
                {"a": 1.0 if class_id == "a" else 0.0,
                 "b": 1.0 if class_id == "b" else 0.0}
            )
            groups.append(f"{class_id}_{index}")
    report = cross_fit_categorical_soft_head(
        features=features,
        targets=targets,
        groups=groups,
        class_ids=("a", "b"),
        folds=5,
    )
    assert report["status"] == "COMPLETE_TRAIN_ONLY_OUT_OF_FOLD"
    assert report["accuracy"] >= 0.9
    assert all(
        row["effective_dimension"] <= row["training_groups"] // 5
        for row in report["folds"]
    )
    assert all(
        row["calibration_source"] == "inner-group-oof-only"
        for row in report["folds"]
    )


def test_learning_canary_recovers_signal_and_rejects_permutation():
    first = run_support_need_learning_canary()
    second = run_support_need_learning_canary()
    assert first == second
    assert first["status"] == "PASS"
    assert all(first["checks"].values())
    assert first["report_sha256"] == sha256_text(
        canonical_json(
            {key: value for key, value in first.items() if key != "report_sha256"}
        )
    )


def test_train_only_pilot_uses_soft_partial_labels_and_marks_missing_heads():
    encoder = _CanaryEncoder()
    states = []
    prepared = []
    labels = []
    for class_index, mode in enumerate(SUPPORT_MODE_IDS):
        for row_index in range(3):
            state = _state().model_copy(
                update={
                    "state_id": f"state_{mode}_{row_index}",
                    "card_id": f"card_{mode}_{row_index}",
                    "user_id": f"user_{mode}_{row_index}",
                    "current_user_text": (
                        f"{mode.replace('_', ' ')} signal {class_index} "
                        f"example {row_index}"
                    ),
                }
            )
            states.append(state)
            prepared.append(prepare_support_need_views(encoder, state))
            labels.append(
                SupportNeedPartialLabel(
                    state_id=state.state_id,
                    user_id=state.user_id,
                    group_id=state.user_id,
                    label_source_id=f"structure_{state.state_id}",
                    label_family="minimal_counterfactual",
                    label_role="support_need",
                    source_reliability=0.9,
                    order_stability=1.0,
                    abstain=False,
                    support_mode_distribution=_distribution(
                        SUPPORT_MODE_IDS, mode
                    ),
                    provenance={
                        "automatic_gold_label": False,
                        "internal_test_outcomes_opened": False,
                        "external_outcomes_opened": False,
                    },
                )
            )
    report = run_train_only_support_need_pilot(
        states=states,
        prepared_views=prepared,
        labels=labels,
    )
    assert report["status"] == "COMPLETE_TRAIN_ONLY_DIAGNOSTIC"
    assert "support_mode" in report["complete_heads"]
    assert "dialogue_phase" in report["unsupported_heads"]
    assert report["heads"]["dialogue_phase"]["status"].startswith(
        "UNSUPPORTED"
    )
    assert report["automatic_gold_labels_created"] is False
    assert report["formal_fit_authorized"] is False


def test_partial_label_rejects_abstain_with_hidden_target():
    with pytest.raises(ValueError, match="abstaining"):
        SupportNeedPartialLabel(
            state_id="s",
            user_id="u",
            group_id="u",
            label_source_id="judge",
            label_family="llm_weak",
            label_role="support_need",
            source_reliability=0.5,
            order_stability=0.5,
            abstain=True,
            support_mode_distribution=_distribution(
                SUPPORT_MODE_IDS, "listen"
            ),
            provenance={
                "automatic_gold_label": False,
                "internal_test_outcomes_opened": False,
                "external_outcomes_opened": False,
            },
        )
