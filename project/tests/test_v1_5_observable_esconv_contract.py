from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from metacom_pm.contracts import StrategyCard
from metacom_pm.esconv_v1_5 import (
    audit_esconv_v1_5_split,
    build_esconv_v1_5_test_artifacts,
)
from metacom_pm.io import sha256_text, write_jsonl
from metacom_pm.pm_v1_5_semantic import (
    FrozenSemanticEncoderSpec,
    FrozenTransformerSemanticEncoder,
    SemanticEncoderBinding,
)
from metacom_pm.pm_v1_5_step0 import STRATEGY_LABEL_TO_FAMILY_ID
from metacom_pm.pm_v2_data import (
    GENERATION_CASE_FIELDS,
    OBSERVABLE_HISTORY_TURN_TARGETS,
    GeneratedDialogueExchangeDraft,
    GeneratedSurfaceOnlyCaseDraft,
    compiler_surface_from_provider,
    observable_state_design,
)
from metacom_pm.strategy_bank import esconv_turn_states


class _Tokenizer:
    truncation_side = "right"

    def __init__(self) -> None:
        self._to_id: dict[str, int] = {}
        self._to_token: dict[int, str] = {}

    def __call__(self, text, *, add_special_tokens=False, **kwargs):
        del kwargs
        ids = []
        for token in str(text).split():
            if token not in self._to_id:
                token_id = 1000 + len(self._to_id)
                self._to_id[token] = token_id
                self._to_token[token_id] = token
            ids.append(self._to_id[token])
        if add_special_tokens:
            ids = [101, *ids, 102]
        return {"input_ids": ids}

    def decode(self, ids, **kwargs):
        del kwargs
        return " ".join(self._to_token[value] for value in ids if value >= 1000)


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
        rows = []
        for text in texts:
            vector = np.zeros(self.spec.output_dimension, dtype=float)
            for token in str(text).casefold().split():
                vector[int(sha256_text(token)[:8], 16) % len(vector)] += 1.0
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            rows.append(vector)
        return np.vstack(rows)

    def assemble_visible_dialogue_state(self, **kwargs):
        return FrozenTransformerSemanticEncoder.assemble_visible_dialogue_state(
            self, **kwargs
        )

    def tokenization_telemetry(self, texts, *, view_names=None):
        return FrozenTransformerSemanticEncoder.tokenization_telemetry(
            self, texts, view_names=view_names
        )


def test_observable_state_cells_are_counterbalanced_without_outcomes() -> None:
    split_ranges = {
        "train": range(1, 25),
        "calibration": range(25, 37),
        "internal_test": range(37, 53),
    }
    for ordinals in split_ranges.values():
        for case_field, _ in GENERATION_CASE_FIELDS:
            rows = [
                observable_state_design(
                    user_id=f"pm_v1_5_u{ordinal:03d}",
                    case_field=case_field,
                )
                for ordinal in ordinals
            ]
            history = Counter(row["history_turn_target"] for row in rows)
            summary = Counter(row["summary_present"] for row in rows)
            assert set(history) == set(OBSERVABLE_HISTORY_TURN_TARGETS)
            assert max(history.values()) - min(history.values()) <= 1
            assert abs(summary[True] - summary[False]) <= 1
            for target in OBSERVABLE_HISTORY_TURN_TARGETS:
                within_cell = Counter(
                    row["summary_present"]
                    for row in rows
                    if row["history_turn_target"] == target
                )
                assert abs(within_cell[True] - within_cell[False]) <= 1
            assert all(
                row["assignment_uses_outcome_or_target_action"] is False
                for row in rows
            )


def test_compiler_owns_summary_absence_but_provider_must_supply_summary() -> None:
    user_id = next(
        f"pm_v1_5_u{ordinal:03d}"
        for ordinal in range(1, 53)
        if not observable_state_design(
            user_id=f"pm_v1_5_u{ordinal:03d}", case_field="context_only"
        )["summary_present"]
    )
    design = observable_state_design(user_id=user_id, case_field="context_only")
    exchanges = [
        GeneratedDialogueExchangeDraft(
            user_text=f"Earlier user turn {index} about moving.",
            assistant_text=f"Earlier assistant turn {index}.",
        )
        for index in range(int(design["history_turn_target"]) // 2)
    ]
    provider = GeneratedSurfaceOnlyCaseDraft(
        current_user_text="Moving has left me lonely today.",
        dialogue_exchanges_before_current=exchanges,
        session_summary="The current session is about moving and loneliness.",
        authorized_user_context="Only the current session may be used.",
    )
    compiled = compiler_surface_from_provider(
        provider, observable_design=design
    )
    assert provider.session_summary
    assert compiled.session_summary == ""
    assert len(compiled.dialogue_before_current) == design["history_turn_target"]
    assert compiled.dialogue_before_current[-1].role == "assistant"


def test_esconv_current_user_is_not_duplicated_in_history(tmp_path: Path) -> None:
    esconv_path = tmp_path / "esconv.json"
    split_path = tmp_path / "split.jsonl"
    esconv_path.write_text(
        json.dumps(
            [
                {
                    "situation": "A difficult move.",
                    "dialog": [
                        {"speaker": "seeker", "content": "First user."},
                        {
                            "speaker": "supporter",
                            "content": "First response.",
                            "annotation": {"strategy": "Reflection of feelings"},
                        },
                        {"speaker": "seeker", "content": "Current user."},
                        {
                            "speaker": "supporter",
                            "content": "Gold response.",
                            "annotation": {"strategy": "Question"},
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    write_jsonl(
        split_path,
        [
            {
                "dialogue_id": "esconv_0000",
                "index": 0,
                "split": "test",
                "excluded_for_evoemo_overlap": False,
            }
        ],
    )
    rows = esconv_turn_states(esconv_path, split_path, "test")
    target = next(row for row in rows if row["turn_index"] == 3)
    assert target["current_user_text"] == "Current user."
    assert [row["content"] for row in target["history"]] == [
        "First user.",
        "First response.",
    ]
    assert all(
        row["content"] != target["current_user_text"]
        for row in target["history"]
    )


def test_esconv_split_audit_enforces_seed_bank_test_isolation(
    tmp_path: Path,
) -> None:
    esconv_path = tmp_path / "esconv.json"
    split_path = tmp_path / "split.jsonl"
    bank_path = tmp_path / "bank.jsonl"
    seeds_path = tmp_path / "seeds.jsonl"
    esconv_path.write_text(
        json.dumps(
            [
                {"situation": "seed", "dialog": []},
                {"situation": "bank", "dialog": []},
                {"situation": "test", "dialog": []},
            ]
        ),
        encoding="utf-8",
    )
    write_jsonl(
        split_path,
        [
            {
                "dialogue_id": "esconv_0000",
                "index": 0,
                "split": "train",
                "excluded_for_evoemo_overlap": False,
            },
            {
                "dialogue_id": "esconv_0001",
                "index": 1,
                "split": "train",
                "excluded_for_evoemo_overlap": False,
            },
            {
                "dialogue_id": "esconv_0002",
                "index": 2,
                "split": "test",
                "excluded_for_evoemo_overlap": False,
            },
        ],
    )
    write_jsonl(
        bank_path,
        [
            StrategyCard(
                strategy_id="strat_0123456789abcdef",
                strategy_label="Question",
                retrieval_text="Visible dialogue context.",
                guidance_text="Ask a gentle question.",
                example_response="Would you like to say more?",
                source_dialogue_id="esconv_0001",
                source_turn_index=1,
            ).model_dump(mode="json")
        ],
    )
    write_jsonl(seeds_path, [{"dialogue_id": "esconv_0000"}])
    report = audit_esconv_v1_5_split(
        esconv_path=esconv_path,
        split_manifest_path=split_path,
        strategy_bank_path=bank_path,
        selected_seed_sources_path=seeds_path,
        require_frozen_counts=False,
    )
    assert report["status"] == "PASS"
    assert report["checks"][
        "development_seed_and_strategy_instances_are_disjoint"
    ]
    assert report["checks"]["strategy_bank_has_no_validation_or_test_sources"]


def test_esconv_v1_5_builder_uses_supported_history_and_same_feature_contract(
    tmp_path: Path,
) -> None:
    esconv_path = tmp_path / "esconv.json"
    split_path = tmp_path / "split.jsonl"
    bank_path = tmp_path / "bank.jsonl"
    seeds_path = tmp_path / "seeds.jsonl"
    esconv_path.write_text(
        json.dumps(
            [
                {"situation": "seed", "dialog": []},
                {"situation": "bank", "dialog": []},
                {
                    "situation": "The user feels isolated after moving.",
                    "dialog": [
                        {"speaker": "seeker", "content": "I recently moved."},
                        {
                            "speaker": "supporter",
                            "content": "That sounds like a big transition.",
                            "annotation": {"strategy": "Reflection of feelings"},
                        },
                        {
                            "speaker": "seeker",
                            "content": "I still feel lonely here.",
                        },
                        {
                            "speaker": "supporter",
                            "content": "Would it help to talk about the hardest part?",
                            "annotation": {"strategy": "Question"},
                        },
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    write_jsonl(
        split_path,
        [
            {
                "dialogue_id": "esconv_0000",
                "index": 0,
                "split": "train",
                "excluded_for_evoemo_overlap": False,
            },
            {
                "dialogue_id": "esconv_0001",
                "index": 1,
                "split": "train",
                "excluded_for_evoemo_overlap": False,
            },
            {
                "dialogue_id": "esconv_0002",
                "index": 2,
                "split": "test",
                "excluded_for_evoemo_overlap": False,
            },
        ],
    )
    write_jsonl(
        bank_path,
        [
            StrategyCard(
                strategy_id=f"strat_{index:016x}",
                strategy_label=label,
                retrieval_text=f"Visible {family} support context.",
                guidance_text=f"Use {family} when appropriate.",
                example_response=f"A {family} response.",
                source_dialogue_id="esconv_0001",
                source_turn_index=index,
            ).model_dump(mode="json")
            for index, (label, family) in enumerate(
                STRATEGY_LABEL_TO_FAMILY_ID.items(), 1
            )
        ],
    )
    write_jsonl(seeds_path, [{"dialogue_id": "esconv_0000"}])
    development_support = {
        "protocol": "pm-v1.5-development-external-observable-state-support-v1",
        "status": "PASS",
        "history_turn_targets": [2, 4, 6, 8],
        "summary_treatments": ["present", "absent"],
    }
    report = build_esconv_v1_5_test_artifacts(
        esconv_path=esconv_path,
        split_manifest_path=split_path,
        strategy_bank_path=bank_path,
        selected_seed_sources_path=seeds_path,
        out_dir=tmp_path / "out",
        semantic_encoder=_Encoder(),
        strategy_estimated_tokens=180,
        development_observable_state_support=development_support,
        require_frozen_split_counts=False,
    )
    assert report["status"] == "COMPLETE"
    assert report["raw_supporter_turns"] == 2
    assert report["excluded_history_below_two_count"] == 1
    assert report["test_turns"] == 1
    assert report["test_dialogues"] == 1
    assert report["history_turn_counts"] == {2: 1}
    assert report["observable_state_support"]["status"] == "PASS"
    runtime = next(
        json.loads(line)
        for line in (tmp_path / "out" / "runtime_states.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    )
    assert runtime["current_user_text"] == "I still feel lonely here."
    assert runtime["current_session_summary"] == ""
    assert (
        runtime["provenance"]["dialogue_level_situation_exposed_to_pm"]
        is False
    )
    assert [row["content"] for row in runtime["current_session_history"]] == [
        "I recently moved.",
        "That sounds like a big transition.",
    ]
    assert runtime["allowed_actions"] == ["M0+R0", "M0+RS"]
    assert report["summary_present_count"] == 0
    assert report["dialogue_level_situation_exposed_to_pm"] is False
