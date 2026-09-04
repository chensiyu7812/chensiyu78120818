from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from metacom_pm.io import canonical_json, sha256_text
from metacom_pm.v1_5_support_need_bakeoff import (
    FACTOR_DEFINITIONS,
    FrozenSupportNeedEmbeddingSpec,
    FrozenSupportNeedNLISpec,
    embedding_candidate_feature_views,
    factor_definition_sha256,
    nli_candidate_feature_bundle,
    run_factorized_support_need_bakeoff,
    transparent_feature_views,
)


class _CanaryEmbedding:
    def __init__(self):
        self.spec = SimpleNamespace(
            candidate_id="qwen3_embedding_0_6b_support_need",
            output_dimension=16,
        )

    def encode(self, texts):
        rows = []
        for text in texts:
            seed = int(sha256_text(str(text))[:16], 16)
            rng = np.random.default_rng(seed)
            row = rng.normal(size=16)
            rows.append(row / np.linalg.norm(row))
        return np.asarray(rows, dtype=float)


class _CanaryNLI:
    def score(self, premises, hypotheses):
        rows = []
        for premise, hypothesis in zip(premises, hypotheses, strict=True):
            seed = int(sha256_text(f"{premise}\n{hypothesis}")[:16], 16)
            rng = np.random.default_rng(seed)
            logits = rng.normal(size=3)
            exponent = np.exp(logits - logits.max())
            rows.append(exponent / exponent.sum())
        return np.asarray(rows, dtype=float)


def _fixture_rows():
    modes = (
        ["listen"] * 4
        + ["comfort_reassure"] * 4
        + ["explore"] * 4
        + ["light_guidance"] * 4
        + ["structured_planning"] * 4
    )
    goals = {
        "listen": ["be_heard"],
        "comfort_reassure": ["be_heard", "stabilize"],
        "explore": ["make_sense"],
        "light_guidance": ["stabilize", "act"],
        "structured_planning": ["decide", "act"],
    }
    phases = {
        "listen": "comforting",
        "comfort_reassure": "comforting",
        "explore": "exploration",
        "light_guidance": "action",
        "structured_planning": "action",
    }
    packet_rows = []
    anchor_rows = []
    for index, mode in enumerate(modes):
        blind_item_id = f"factor_need_{index:02d}"
        packet_rows.append(
            {
                "blind_item_id": blind_item_id,
                "visible_state": {
                    "current_user_text": (
                        f"{mode.replace('_', ' ')} visible signal {index}"
                    ),
                    "recent_dialogue": [
                        {
                            "role": "user",
                            "content": f"Earlier user context {index}.",
                        },
                        {
                            "role": "assistant",
                            "content": f"Earlier assistant context {index}.",
                        },
                    ],
                    "session_summary": f"Visible summary {index}.",
                },
            }
        )
        low_burden = True if index < 14 else False if index == 14 else None
        anchor_rows.append(
            {
                "blind_item_id": blind_item_id,
                "human_recommended_low_interaction_burden": low_burden,
                "raw_annotation": {
                    "blind_item_id": blind_item_id,
                    "support_mode": mode,
                    "goals": goals[mode],
                    "dialogue_phase": phases[mode],
                    "nonclinical_urgency": (
                        "routine"
                        if index % 3 == 0
                        else "elevated"
                        if index % 3 == 1
                        else "acute"
                    ),
                    "abstain": False,
                    "confidence": 4,
                },
            }
        )
    return packet_rows, anchor_rows


def test_candidate_specs_are_pinned_and_capacity_bounded():
    embedding = FrozenSupportNeedEmbeddingSpec.model_validate(
        {
            "candidate_id": "candidate",
            "model_id": "org/model",
            "revision": "a" * 40,
            "snapshot_tree_sha256": "b" * 64,
            "pooling": "last_token",
            "padding_side": "left",
            "instruction": "Represent support need.",
            "max_length": 512,
            "hidden_dimension": 1024,
            "output_dimension": 256,
        }
    )
    assert embedding.local_files_only is True
    assert embedding.trust_remote_code is False
    with pytest.raises(ValueError, match="exceeds hidden"):
        FrozenSupportNeedEmbeddingSpec.model_validate(
            {
                **embedding.model_dump(mode="json"),
                "output_dimension": 2048,
            }
        )
    nli = FrozenSupportNeedNLISpec.model_validate(
        {
            "candidate_id": "nli",
            "model_id": "org/nli",
            "revision": "c" * 40,
            "snapshot_tree_sha256": "d" * 64,
            "max_length": 512,
        }
    )
    assert list(nli.label_ids) == [
        "contradiction",
        "entailment",
        "neutral",
    ]
    with pytest.raises(ValueError, match="order"):
        FrozenSupportNeedNLISpec.model_validate(
            {
                **nli.model_dump(mode="json"),
                "label_ids": ["entailment", "contradiction", "neutral"],
            }
        )


def test_factor_contract_is_explicit_and_stable():
    assert [row["factor_id"] for row in FACTOR_DEFINITIONS] == [
        "need_to_be_heard",
        "need_for_emotional_containment",
        "need_for_exploration",
        "advance_readiness",
        "focused_question_readiness",
        "advice_readiness",
        "planning_readiness",
        "low_interaction_burden",
    ]
    assert factor_definition_sha256() == sha256_text(
        canonical_json(list(FACTOR_DEFINITIONS))
    )


def test_factorized_bakeoff_is_deterministic_and_keeps_missing_burden_missing():
    packet_rows, anchor_rows = _fixture_rows()
    transparent = transparent_feature_views(packet_rows)
    embedding = embedding_candidate_feature_views(
        _CanaryEmbedding(), packet_rows
    )
    nli = nli_candidate_feature_bundle(_CanaryNLI(), packet_rows)
    feature_views = {
        **transparent,
        (
            "embedding.qwen3_embedding_0_6b_support_need.multiview"
        ): embedding[
            "embedding.qwen3_embedding_0_6b_support_need.multiview"
        ],
        "nli.current_plus_full": nli["feature_views"][
            "nli.current_plus_full"
        ],
    }
    first = run_factorized_support_need_bakeoff(
        packet_rows=packet_rows,
        normalized_anchor_rows=anchor_rows,
        feature_views=feature_views,
        nli_bundle=nli,
        candidate_runtimes=[],
        hybrid_embedding_candidate_id=(
            "qwen3_embedding_0_6b_support_need"
        ),
    )
    second = run_factorized_support_need_bakeoff(
        packet_rows=packet_rows,
        normalized_anchor_rows=anchor_rows,
        feature_views=feature_views,
        nli_bundle=nli,
        candidate_runtimes=[],
        hybrid_embedding_candidate_id=(
            "qwen3_embedding_0_6b_support_need"
        ),
    )
    assert first == second
    report_core = dict(first)
    report_sha256 = report_core.pop("report_sha256")
    assert report_sha256 == sha256_text(canonical_json(report_core))
    assert first["status"] == "COMPLETE_DIAGNOSTIC_NOT_FORMAL_FIT"
    assert first["heads"]["need_to_be_heard"]["class_counts"] == {
        "negative": 12,
        "positive": 8,
    }
    assert (
        "best_noncollapsed_view_by_soft_log_loss"
        in first["heads"]["need_to_be_heard"]
    )
    burden = first["heads"]["low_interaction_burden"]
    assert burden["labeled_rows"] == 15
    assert burden["class_counts"] == {"negative": 1, "positive": 14}
    assert burden["status"].startswith("UNSUPPORTED")
    assert first["formal_fit_authorized"] is False
    assert first["representation_promotion_authorized"] is False
    assert first["automatic_gold_labels_created"] is False
    assert first["target_supporter_responses_used"] is False
    assert first["internal_test_outcomes_opened"] is False
    assert first["external_outcomes_opened"] is False


def test_factorized_bakeoff_requires_exact_anchor_coverage():
    packet_rows, anchor_rows = _fixture_rows()
    transparent = transparent_feature_views(packet_rows)
    nli = nli_candidate_feature_bundle(_CanaryNLI(), packet_rows)
    feature_views = {
        **transparent,
        (
            "embedding.qwen3_embedding_0_6b_support_need.multiview"
        ): np.ones((len(packet_rows), 8)),
        "nli.current_plus_full": nli["feature_views"][
            "nli.current_plus_full"
        ],
    }
    with pytest.raises(RuntimeError, match="exactly cover"):
        run_factorized_support_need_bakeoff(
            packet_rows=packet_rows,
            normalized_anchor_rows=anchor_rows[:-1],
            feature_views=feature_views,
            nli_bundle=nli,
            candidate_runtimes=[],
            hybrid_embedding_candidate_id=(
                "qwen3_embedding_0_6b_support_need"
            ),
        )


def test_expanded_fit_stage_is_reported_without_opening_confirmation():
    packet_rows, anchor_rows = _fixture_rows()
    transparent = transparent_feature_views(packet_rows)
    embedding = embedding_candidate_feature_views(
        _CanaryEmbedding(), packet_rows
    )
    nli = nli_candidate_feature_bundle(_CanaryNLI(), packet_rows)
    feature_views = {
        **transparent,
        (
            "embedding.qwen3_embedding_0_6b_support_need.multiview"
        ): embedding[
            "embedding.qwen3_embedding_0_6b_support_need.multiview"
        ],
        "nli.current_plus_full": nli["feature_views"][
            "nli.current_plus_full"
        ],
    }
    report = run_factorized_support_need_bakeoff(
        packet_rows=packet_rows,
        normalized_anchor_rows=anchor_rows,
        feature_views=feature_views,
        nli_bundle=nli,
        candidate_runtimes=[],
        hybrid_embedding_candidate_id=(
            "qwen3_embedding_0_6b_support_need"
        ),
        expansion_human_annotations_opened=True,
    )
    assert report["expansion_human_annotations_opened"] is True
    assert "confirmation subset remains sealed" in report["interpretation"]
    assert report["formal_fit_authorized"] is False
