from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/v1_5/24av_prepare_rs_effect_quality_review_v1_5.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "rs_effect_quality_review", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blinding_is_deterministic_and_not_one_sided() -> None:
    module = _load()
    pair_ids = [f"pair_{index}" for index in range(32)]
    first = [module._a_arm(pair_id) for pair_id in pair_ids]
    second = [module._a_arm(pair_id) for pair_id in pair_ids]
    assert first == second
    counts = Counter(first)
    assert set(counts) == {"R0", "RS"}
    assert min(counts.values()) >= 8


def test_public_packet_hides_treatment_and_judge() -> None:
    packet = (
        ROOT
        / "outputs/pm_v1_5_rs_effect_human_quality_blind_v1_candidate"
        / "human_blind_packet.jsonl"
    )
    if not packet.exists():
        return
    for line in packet.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert set(row) == {
            "protocol",
            "blind_item_id",
            "visible_dialogue",
            "response_a",
            "response_b",
        }
        serialized = json.dumps(row)
        for hidden in (
            "response_a_arm",
            "selected_card",
            "retrieval_score",
            "judge",
        ):
            assert hidden not in serialized


def test_review_has_one_decision_and_no_structured_risk_matrix() -> None:
    template = (
        ROOT
        / "outputs/pm_v1_5_rs_effect_human_quality_blind_v1_candidate"
        / "human_annotation_template.jsonl"
    )
    if not template.exists():
        return
    for line in template.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert set(row) == {
            "protocol",
            "blind_item_id",
            "quality_preference",
            "decisive_criterion",
            "quality_notes",
            "annotator_id",
        }
