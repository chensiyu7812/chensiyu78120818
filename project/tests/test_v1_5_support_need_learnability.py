from __future__ import annotations

import numpy as np

from metacom_pm.io import canonical_json, sha256_text
from metacom_pm.pm_v1_5_semantic import (
    FrozenSemanticEncoderSpec,
    SemanticEncoderBinding,
)
from metacom_pm.v1_5_support_need_learnability import (
    run_human_anchor_learnability_diagnostic,
)


class _CanaryEncoder:
    def __init__(self):
        self.spec = FrozenSemanticEncoderSpec(
            model_id="local/canary",
            revision="a" * 40,
            snapshot_tree_sha256="b" * 64,
            output_dimension=16,
            max_length=128,
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
            seed = int(sha256_text(str(text))[:8], 16)
            rng = np.random.default_rng(seed)
            row = rng.normal(size=16)
            rows.append(row / np.linalg.norm(row))
        return np.asarray(rows)


def _fixture_rows():
    modes = (
        ["listen"] * 4
        + ["explore"] * 4
        + ["comfort_reassure"] * 4
        + ["light_guidance"] * 4
        + ["structured_planning"] * 4
    )
    packet_rows = []
    anchor_rows = []
    for index, mode in enumerate(modes):
        blind_item_id = f"need_{index:02d}"
        visible_state = {
            "current_user_text": (
                f"{mode} signal example {index} "
                + " ".join(["detail"] * (index % 5))
                + ("?" if index % 2 else "")
            ),
            "recent_dialogue": [
                {
                    "role": "user" if turn % 2 == 0 else "assistant",
                    "content": f"fixture turn {turn}",
                }
                for turn in range(index % 4)
            ],
            "session_summary": "fixture " + "summary " * (1 + index % 3),
        }
        packet_rows.append(
            {
                "blind_item_id": blind_item_id,
                "visible_state": visible_state,
            }
        )
        raw = {
            "blind_item_id": blind_item_id,
            "support_mode": mode,
            "goals": ["act" if mode.endswith("guidance") else "be_heard"],
            "dialogue_phase": (
                "action"
                if mode in {"light_guidance", "structured_planning"}
                else "comforting"
            ),
            "nonclinical_urgency": (
                "elevated" if index % 2 else "routine"
            ),
            "advice_rejected": None,
            "advice_requested": None,
            "one_small_step_requested": None,
            "listen_first_requested": None,
            "question_or_task_burden_limit": None,
            "abstain": False,
            "confidence": 4,
            "notes": "",
        }
        anchor_rows.append(
            {
                "blind_item_id": blind_item_id,
                "raw_annotation": raw,
            }
        )
    return packet_rows, anchor_rows


def test_anchor_learnability_is_deterministic_and_outcome_blind():
    packet_rows, anchor_rows = _fixture_rows()
    first = run_human_anchor_learnability_diagnostic(
        encoder=_CanaryEncoder(),
        packet_rows=packet_rows,
        normalized_anchor_rows=anchor_rows,
    )
    second = run_human_anchor_learnability_diagnostic(
        encoder=_CanaryEncoder(),
        packet_rows=packet_rows,
        normalized_anchor_rows=anchor_rows,
    )
    assert canonical_json(first) == canonical_json(second)
    assert first["non_abstaining_anchor_count"] == 20
    assert first["esconv_metadata_used_as_features"] is False
    assert first["target_supporter_responses_used"] is False
    assert first["target_strategy_annotations_used"] is False
    assert first["internal_test_outcomes_opened"] is False
    assert first["external_outcomes_opened"] is False


def test_anchor_learnability_requires_exact_packet_coverage():
    packet_rows, anchor_rows = _fixture_rows()
    try:
        run_human_anchor_learnability_diagnostic(
            encoder=_CanaryEncoder(),
            packet_rows=packet_rows,
            normalized_anchor_rows=anchor_rows[:-1],
        )
    except RuntimeError as error:
        assert "exactly cover" in str(error)
    else:
        raise AssertionError("missing normalized anchor must fail closed")
