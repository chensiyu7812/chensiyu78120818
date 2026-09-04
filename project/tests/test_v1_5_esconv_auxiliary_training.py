"""ESConv auxiliary train/calibration/internal-test training support.

Uses only the 52 dialogues already used to seed the 52 synthetic development
users, bank-disjoint from the Strategy Bank's source dialogues by
construction, so PM training/calibration gets real exposure to "no
cross-session memory, choose only between M0+R0 and M0+RS" states without
ever touching the held-out 169-dialogue ESConv test split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from metacom_pm.contracts import StrategyCard
from metacom_pm.esconv_v1_5 import (
    ESCONV_V1_5_AUXILIARY_SEMANTIC_FAMILY,
    build_esconv_v1_5_auxiliary_training_artifacts,
    esconv_v1_5_auxiliary_seed_split_assignment,
)
from metacom_pm.io import iter_jsonl, sha256_text, write_jsonl
from metacom_pm.pm_v1_5_semantic import FrozenSemanticEncoderSpec, SemanticEncoderBinding
from metacom_pm.pm_v1_5_step0 import STRATEGY_LABEL_TO_FAMILY_ID
from metacom_pm.pm_v2_contracts import PMV2Split


class _Tokenizer:
    truncation_side = "right"

    def __init__(self) -> None:
        self._to_id: dict[str, int] = {}

    def __call__(self, text, *, add_special_tokens=False, **kwargs):
        del kwargs
        ids = []
        for token in str(text).split():
            if token not in self._to_id:
                self._to_id[token] = 1000 + len(self._to_id)
            ids.append(self._to_id[token])
        if add_special_tokens:
            ids = [101, *ids, 102]
        return {"input_ids": ids}

    def decode(self, ids, **kwargs):
        del kwargs
        inverse = {v: k for k, v in self._to_id.items()}
        return " ".join(inverse[value] for value in ids if value >= 1000)


class _Encoder:
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

    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()

    def encode(self, texts):
        from metacom_pm.pm_v1_5_semantic import FrozenTransformerSemanticEncoder

        rows = []
        for text in texts:
            vector = np.zeros(self.spec.output_dimension, dtype=float)
            for token in str(text).casefold().split():
                vector[int(sha256_text(token)[:8], 16) % len(vector)] += 1.0
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            rows.append(vector)
        return np.vstack(rows)

    def assemble_visible_dialogue_state(self, **kwargs):
        from metacom_pm.pm_v1_5_semantic import FrozenTransformerSemanticEncoder

        return FrozenTransformerSemanticEncoder.assemble_visible_dialogue_state(
            self, **kwargs
        )

    def tokenization_telemetry(self, texts, *, view_names=None):
        from metacom_pm.pm_v1_5_semantic import FrozenTransformerSemanticEncoder

        return FrozenTransformerSemanticEncoder.tokenization_telemetry(
            self, texts, view_names=view_names
        )


def _dialogue(situation: str, n_exchanges: int) -> dict:
    dialog = []
    for i in range(n_exchanges):
        dialog.append({"speaker": "seeker", "content": f"seeker turn {i}."})
        dialog.append(
            {
                "speaker": "supporter",
                "content": f"supporter turn {i}.",
                "annotation": {"strategy": "Question"},
            }
        )
    return {"situation": situation, "dialog": dialog}


def _write_fixture(tmp_path: Path, *, n_seed_dialogues: int) -> dict[str, Path]:
    esconv_path = tmp_path / "esconv.json"
    split_path = tmp_path / "split.jsonl"
    bank_path = tmp_path / "bank.jsonl"
    seeds_path = tmp_path / "seeds.jsonl"

    # index 0..n_seed_dialogues-1: the seed pool (enough exchanges for >=2
    # history turns); index n_seed_dialogues: a bank-source dialogue (must
    # never be pulled into the auxiliary states); index n_seed_dialogues+1: a
    # test-split dialogue (must never be pulled in either).
    dialogues = [_dialogue(f"seed-{i}", 3) for i in range(n_seed_dialogues)]
    dialogues.append(_dialogue("bank-source", 3))
    dialogues.append(_dialogue("test-only", 3))
    esconv_path.write_text(json.dumps(dialogues), encoding="utf-8")

    split_rows = [
        {
            "dialogue_id": f"esconv_{i:04d}",
            "index": i,
            "split": "train",
            "excluded_for_evoemo_overlap": False,
        }
        for i in range(n_seed_dialogues)
    ]
    split_rows.append(
        {
            "dialogue_id": f"esconv_{n_seed_dialogues:04d}",
            "index": n_seed_dialogues,
            "split": "train",
            "excluded_for_evoemo_overlap": False,
        }
    )
    split_rows.append(
        {
            "dialogue_id": f"esconv_{n_seed_dialogues + 1:04d}",
            "index": n_seed_dialogues + 1,
            "split": "test",
            "excluded_for_evoemo_overlap": False,
        }
    )
    write_jsonl(split_path, split_rows)

    write_jsonl(
        bank_path,
        [
            StrategyCard(
                strategy_id=f"strat_{index:016x}",
                strategy_label=label,
                retrieval_text=f"Visible {family} support context.",
                guidance_text=f"Use {family} when appropriate.",
                example_response=f"A {family} response.",
                source_dialogue_id=f"esconv_{n_seed_dialogues:04d}",
                source_turn_index=index,
            ).model_dump(mode="json")
            for index, (label, family) in enumerate(
                STRATEGY_LABEL_TO_FAMILY_ID.items(), 1
            )
        ],
    )
    write_jsonl(
        seeds_path,
        [
            {
                "dialogue_id": f"esconv_{i:04d}",
                "selection_ordinal": i,
                "excluded_for_evoemo_overlap": False,
                "source_split": "train",
            }
            for i in range(n_seed_dialogues)
        ],
    )
    return {
        "esconv": esconv_path,
        "split": split_path,
        "bank": bank_path,
        "seeds": seeds_path,
    }


def test_seed_split_assignment_matches_24_12_16_ordinal_order(tmp_path: Path) -> None:
    seeds_path = tmp_path / "seeds.jsonl"
    write_jsonl(
        seeds_path,
        [
            {
                "dialogue_id": f"esconv_{i:04d}",
                "selection_ordinal": i,
                "excluded_for_evoemo_overlap": False,
                "source_split": "train",
            }
            for i in range(52)
        ],
    )
    assignment = esconv_v1_5_auxiliary_seed_split_assignment(seeds_path)
    assert len(assignment) == 52
    train_ids = {k for k, (split, _) in assignment.items() if split is PMV2Split.TRAIN}
    calib_ids = {
        k for k, (split, _) in assignment.items() if split is PMV2Split.CALIBRATION
    }
    test_ids = {
        k for k, (split, _) in assignment.items() if split is PMV2Split.INTERNAL_TEST
    }
    assert train_ids == {f"esconv_{i:04d}" for i in range(24)}
    assert calib_ids == {f"esconv_{i:04d}" for i in range(24, 36)}
    assert test_ids == {f"esconv_{i:04d}" for i in range(36, 52)}
    assert assignment["esconv_0000"] == (PMV2Split.TRAIN, 0)
    assert assignment["esconv_0024"] == (PMV2Split.CALIBRATION, 0)
    assert assignment["esconv_0036"] == (PMV2Split.INTERNAL_TEST, 0)


def test_seed_split_assignment_rejects_wrong_total(tmp_path: Path) -> None:
    seeds_path = tmp_path / "seeds.jsonl"
    write_jsonl(
        seeds_path,
        [
            {
                "dialogue_id": "esconv_0000",
                "selection_ordinal": 0,
                "excluded_for_evoemo_overlap": False,
                "source_split": "train",
            }
        ],
    )
    with pytest.raises(RuntimeError, match="expected 52"):
        esconv_v1_5_auxiliary_seed_split_assignment(seeds_path)


def test_seed_split_assignment_rejects_evoemo_excluded_source(tmp_path: Path) -> None:
    seeds_path = tmp_path / "seeds.jsonl"
    rows = [
        {
            "dialogue_id": f"esconv_{i:04d}",
            "selection_ordinal": i,
            "excluded_for_evoemo_overlap": i == 0,
            "source_split": "train",
        }
        for i in range(52)
    ]
    write_jsonl(seeds_path, rows)
    with pytest.raises(RuntimeError, match="excluded_for_evoemo_overlap"):
        esconv_v1_5_auxiliary_seed_split_assignment(seeds_path)


def test_auxiliary_builder_only_uses_the_seed_dialogues_and_isolates_gold_fields(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path, n_seed_dialogues=4)
    # Reduce split sizes for this tiny fixture via a monkeypatched order is not
    # available; instead verify behavior directly against the real 24/12/16
    # contract would need 52 dialogues.  Here we only exercise 4 seeds by
    # calling the lower-level per-turn logic through the real function with a
    # temporarily patched split order.
    import metacom_pm.esconv_v1_5 as esconv_v1_5

    original_order = esconv_v1_5.ESCONV_V1_5_AUXILIARY_SPLIT_ORDER
    try:
        esconv_v1_5.ESCONV_V1_5_AUXILIARY_SPLIT_ORDER = (
            (PMV2Split.TRAIN, 2),
            (PMV2Split.CALIBRATION, 1),
            (PMV2Split.INTERNAL_TEST, 1),
        )
        report = build_esconv_v1_5_auxiliary_training_artifacts(
            esconv_path=fixture["esconv"],
            split_manifest_path=fixture["split"],
            strategy_bank_path=fixture["bank"],
            selected_seed_sources_path=fixture["seeds"],
            out_dir=tmp_path / "out",
            semantic_encoder=_Encoder(),
            strategy_estimated_tokens=180,
            require_frozen_split_counts=False,
        )
    finally:
        esconv_v1_5.ESCONV_V1_5_AUXILIARY_SPLIT_ORDER = original_order

    assert report["status"] == "COMPLETE"
    assert report["semantic_family"] == ESCONV_V1_5_AUXILIARY_SEMANTIC_FAMILY
    assert report["bank_disjoint"] is True
    assert report["strategy_bank_source_dialogue_overlap_count"] == 0
    assert report["split_reports"]["train"]["dialogue_count"] == 2
    assert report["split_reports"]["calibration"]["dialogue_count"] == 1
    assert report["split_reports"]["internal_test"]["dialogue_count"] == 1
    assert report["sample_selection_is_outcome_free_deterministic_not_random"] is True

    # The bank-source and test-only dialogues (esconv_0004, esconv_0005) must
    # never appear in any auxiliary split's states.
    for split_name in ("train", "calibration", "internal_test"):
        for row in iter_jsonl(tmp_path / "out" / split_name / "pm_v2_states.jsonl"):
            assert row["user_id"] not in {"esconv_0004", "esconv_0005"}
            assert "gold_response" not in row
            assert "gold_strategy" not in row
            assert row["allowed_actions"] == ["M0+R0", "M0+RS"]
            assert row["split"] == split_name
        for row in iter_jsonl(tmp_path / "out" / split_name / "runtime_states.jsonl"):
            assert "gold_response" not in row
            assert "gold_strategy" not in row
            assert row["current_session_summary"] == ""
            assert (
                row["provenance"]["dialogue_level_situation_exposed_to_pm"]
                is False
            )
            assert row["semantic_family"] == ESCONV_V1_5_AUXILIARY_SEMANTIC_FAMILY
            assert all(not source["available"] for source in row["inventory"].values())
        audit_rows = list(iter_jsonl(tmp_path / "out" / split_name / "audit_only.jsonl"))
        for row in audit_rows:
            assert "gold_response" in row
            assert row["evaluator_only"] is True
    assert report["dialogue_level_situation_exposed_to_pm"] is False


def test_auxiliary_builder_rejects_bank_overlapping_seed(tmp_path: Path) -> None:
    """If a seed dialogue is also a Strategy Bank source, the shared
    audit_esconv_v1_5_split disjointness check must fail closed -- the
    auxiliary builder must never silently proceed with a bank-overlapping
    seed."""

    fixture = _write_fixture(tmp_path, n_seed_dialogues=2)
    # Make one strategy card's source dialogue equal to one of the seed
    # dialogues (instead of the dedicated bank-source dialogue).
    bank_path = fixture["bank"]
    rows = list(iter_jsonl(bank_path))
    rows[0]["source_dialogue_id"] = "esconv_0000"
    write_jsonl(bank_path, rows)

    import metacom_pm.esconv_v1_5 as esconv_v1_5

    original_order = esconv_v1_5.ESCONV_V1_5_AUXILIARY_SPLIT_ORDER
    try:
        esconv_v1_5.ESCONV_V1_5_AUXILIARY_SPLIT_ORDER = (
            (PMV2Split.TRAIN, 1),
            (PMV2Split.CALIBRATION, 1),
            (PMV2Split.INTERNAL_TEST, 0),
        )
        with pytest.raises(RuntimeError, match="data-isolation contract failed"):
            build_esconv_v1_5_auxiliary_training_artifacts(
                esconv_path=fixture["esconv"],
                split_manifest_path=fixture["split"],
                strategy_bank_path=bank_path,
                selected_seed_sources_path=fixture["seeds"],
                out_dir=tmp_path / "out",
                semantic_encoder=_Encoder(),
                strategy_estimated_tokens=180,
                require_frozen_split_counts=False,
            )
    finally:
        esconv_v1_5.ESCONV_V1_5_AUXILIARY_SPLIT_ORDER = original_order
