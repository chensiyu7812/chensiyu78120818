from __future__ import annotations

import json
from pathlib import Path

from metacom_pm.io import write_jsonl
from metacom_pm.io import write_json
from metacom_pm.v1_5_strategy_bank import (
    build_strategy_bank_v2_candidate,
    validate_strategy_bank_v2_human_annotations,
)


FAMILIES = (
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Affirmation and Reassurance",
    "Providing Suggestions",
)


def test_strategy_bank_v2_excludes_packet_and_never_exposes_raw_examples(
    tmp_path: Path,
):
    esconv = []
    split_rows = []
    raw_rows = []
    for index in range(170):
        dialogue_id = f"esconv_{index:04d}"
        esconv.append(
            {
                "problem_type": f"problem_{index % 4}",
                "emotion_type": f"emotion_{index % 3}",
                "experience_type": "Current Experience",
                "situation": f"situation {index}",
                "survey_score": {
                    "seeker": {
                        "empathy": "4",
                        "relevance": "5",
                        "initial_emotion_intensity": "4",
                        "final_emotion_intensity": "2",
                    },
                    "supporter": {"relevance": "4"},
                },
                "dialog": [
                    {
                        "speaker": "seeker",
                        "content": f"visible user concern {index}",
                    },
                    {
                        "speaker": "supporter",
                        "content": f"unique private raw response {index}",
                    },
                ],
            }
        )
        split_rows.append(
            {
                "index": index,
                "dialogue_id": dialogue_id,
                "split": "train",
                "excluded_for_evoemo_overlap": False,
            }
        )
        family = FAMILIES[index % len(FAMILIES)]
        raw_rows.append(
            {
                "strategy_id": f"strategy_{index}",
                "strategy_label": family,
                "retrieval_text": f"context {index}",
                "guidance_text": "old guidance",
                "example_response": f"unique private raw response {index}",
                "source_dialogue_id": dialogue_id,
                "source_turn_index": 1,
            }
        )
    esconv_path = tmp_path / "esconv.json"
    esconv_path.write_text(json.dumps(esconv), encoding="utf-8")
    split_path = tmp_path / "split.jsonl"
    write_jsonl(split_path, split_rows)
    raw_path = tmp_path / "raw.jsonl"
    write_jsonl(raw_path, raw_rows)
    lineage_path = tmp_path / "packet_lineage.jsonl"
    write_jsonl(
        lineage_path,
        [{"dialogue_id": "esconv_0000"}, {"dialogue_id": "esconv_0001"}],
    )

    result = build_strategy_bank_v2_candidate(
        raw_strategy_bank_path=raw_path,
        esconv_path=esconv_path,
        split_manifest_path=split_path,
        support_need_lineage_path=lineage_path,
    )
    assert len(result["cards"]) == 5
    assert result["report"]["candidate_packet_source_overlap_count"] == 0
    assert result["report"]["packet_source_excluded_raw_card_count"] == 2
    assert (
        result["report"]["no_prior_seeker_turn_excluded_raw_card_count"] == 0
    )
    assert result["report"]["candidate_eligible_for_formal_rs"] is False
    rendered = json.dumps(result["cards"])
    assert "unique private raw response" not in rendered
    assert "unique private raw response" not in json.dumps(
        result["human_review_rows"]
    )
    assert len(result["human_review_rows"]) == 5
    assert len(result["human_annotation_template"]) == 5
    assert all(
        row["source_dialogue_id"] not in {"esconv_0000", "esconv_0001"}
        for row in result["lineage_rows"]
    )

    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    write_json(candidate_dir / "build_report.json", result["report"])
    write_jsonl(
        candidate_dir / "human_review_packet.jsonl",
        result["human_review_rows"],
    )
    annotations = [
        {
            "card_id": row["card_id"],
            "support_move_clear": True,
            "when_to_use_valid": True,
            "when_not_to_use_valid": True,
            "mode_phase_goal_fit_valid": True,
            "burden_and_risk_flags_valid": True,
            "safe_general_technique": True,
            "approve_for_train_only_pilot": True,
            "confidence": 4,
            "required_corrections": "",
            "notes": "",
        }
        for row in result["human_review_rows"]
    ]
    annotations_path = tmp_path / "annotations.jsonl"
    write_jsonl(annotations_path, annotations)
    review = validate_strategy_bank_v2_human_annotations(
        candidate_dir=candidate_dir,
        annotations_path=annotations_path,
    )
    assert review["status"] == "HUMAN_REVIEW_PASS_PENDING_LLM_WEAK_AUDIT"
    assert review["approved_count"] == 5
    assert review["formal_rs_promoted"] is False


def test_strategy_bank_v2_excludes_supporter_turns_without_prior_seeker(
    tmp_path: Path,
):
    esconv = []
    split_rows = []
    raw_rows = []
    for index in range(170):
        dialogue_id = f"esconv_{index:04d}"
        family = FAMILIES[index % len(FAMILIES)]
        esconv.append(
            {
                "problem_type": "problem",
                "emotion_type": "emotion",
                "experience_type": "Current Experience",
                "situation": "situation",
                "survey_score": {},
                "dialog": [
                    {
                        "speaker": "supporter",
                        "content": f"unsupported opening {index}",
                    },
                    {
                        "speaker": "seeker",
                        "content": f"visible user concern {index}",
                    },
                    {
                        "speaker": "supporter",
                        "content": f"eligible response {index}",
                    },
                ],
            }
        )
        split_rows.append(
            {
                "index": index,
                "dialogue_id": dialogue_id,
                "split": "train",
                "excluded_for_evoemo_overlap": False,
            }
        )
        raw_rows.extend(
            [
                {
                    "strategy_id": f"opening_{index}",
                    "strategy_label": family,
                    "retrieval_text": f"opening context {index}",
                    "guidance_text": "old guidance",
                    "example_response": f"unsupported opening {index}",
                    "source_dialogue_id": dialogue_id,
                    "source_turn_index": 0,
                },
                {
                    "strategy_id": f"eligible_{index}",
                    "strategy_label": family,
                    "retrieval_text": f"eligible context {index}",
                    "guidance_text": "old guidance",
                    "example_response": f"eligible response {index}",
                    "source_dialogue_id": dialogue_id,
                    "source_turn_index": 2,
                },
            ]
        )
    esconv_path = tmp_path / "esconv.json"
    esconv_path.write_text(json.dumps(esconv), encoding="utf-8")
    split_path = tmp_path / "split.jsonl"
    write_jsonl(split_path, split_rows)
    raw_path = tmp_path / "raw.jsonl"
    write_jsonl(raw_path, raw_rows)
    lineage_path = tmp_path / "packet_lineage.jsonl"
    write_jsonl(lineage_path, [{"dialogue_id": "not_in_fixture"}])

    result = build_strategy_bank_v2_candidate(
        raw_strategy_bank_path=raw_path,
        esconv_path=esconv_path,
        split_manifest_path=split_path,
        support_need_lineage_path=lineage_path,
    )
    assert (
        result["report"]["no_prior_seeker_turn_excluded_raw_card_count"]
        == 170
    )
    assert result["report"]["candidate_lineage_row_count"] == 170
    assert all(
        row["source_turn_index"] == 2 for row in result["lineage_rows"]
    )
