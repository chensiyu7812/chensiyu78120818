from __future__ import annotations

import json
from pathlib import Path

from metacom_pm.io import (
    canonical_json,
    sha256_file,
    sha256_text,
    write_jsonl,
)
from metacom_pm.v1_5_support_need_packet import (
    FORBIDDEN_BLIND_KEYS,
    build_support_need_expansion_packet,
    build_support_need_expansion_fit_packet,
    build_support_need_annotation_packet,
    normalize_support_need_human_anchors,
    validate_support_need_expansion_annotations,
    validate_support_need_human_annotations,
)


def _dialogue(index: int):
    return {
        "problem_type": f"problem_{index % 4}",
        "emotion_type": f"emotion_{index % 3}",
        "experience_type": "Current Experience",
        "situation": f"Situation {index}",
        "survey_score": {"do_not_use": index},
        "dialog": [
            {"speaker": "seeker", "content": f"hello {index}"},
            {
                "speaker": "supporter",
                "content": f"target response early {index}",
                "annotation": {"strategy": "Question"},
            },
            {"speaker": "seeker", "content": f"middle concern {index}"},
            {
                "speaker": "supporter",
                "content": f"target response middle {index}",
                "annotation": {"strategy": "Reflection"},
            },
            {"speaker": "seeker", "content": f"late concern {index}"},
            {
                "speaker": "supporter",
                "content": f"target response late {index}",
                "annotation": {"strategy": "Suggestion"},
            },
        ],
    }


def test_support_need_packet_is_deterministic_blind_and_dialogue_disjoint(
    tmp_path: Path,
):
    esconv = [_dialogue(index) for index in range(30)]
    esconv_path = tmp_path / "esconv.json"
    esconv_path.write_text(json.dumps(esconv), encoding="utf-8")
    split_path = tmp_path / "split.jsonl"
    write_jsonl(
        split_path,
        [
            {
                "index": index,
                "dialogue_id": f"esconv_{index:04d}",
                "split": "train",
                "excluded_for_evoemo_overlap": False,
            }
            for index in range(30)
        ],
    )
    seeds_path = tmp_path / "seeds.jsonl"
    write_jsonl(seeds_path, [{"dialogue_id": "esconv_0000"}])
    bank_path = tmp_path / "strategy_bank.jsonl"
    write_jsonl(
        bank_path,
        [
            {
                "strategy_id": f"strategy_{index}",
                "source_dialogue_id": f"esconv_{index:04d}",
            }
            for index in range(1, 30)
        ],
    )

    first = build_support_need_annotation_packet(
        esconv_path=esconv_path,
        split_manifest_path=split_path,
        existing_seed_sources_path=seeds_path,
        strategy_bank_path=bank_path,
        packet_size=15,
        human_anchor_size=6,
    )
    second = build_support_need_annotation_packet(
        esconv_path=esconv_path,
        split_manifest_path=split_path,
        existing_seed_sources_path=seeds_path,
        strategy_bank_path=bank_path,
        packet_size=15,
        human_anchor_size=6,
    )
    assert first == second
    assert first["contract"]["unique_dialogues"] == 15
    assert (
        first["contract"]["selection"]["human_anchor_selection"]
        == "round-robin-emotion-by-position"
    )
    assert first["contract"]["target_supporter_responses_used"] is False
    assert first["contract"]["target_strategy_annotations_used"] is False
    assert first["contract"]["survey_outcomes_used"] is False
    assert len({row["dialogue_id"] for row in first["lineage_rows"]}) == 15
    assert all(
        row["must_be_excluded_from_strategy_bank_v2"]
        for row in first["lineage_rows"]
    )
    assert (
        first["contract"]["bank_v2_requirement"][
            "current_raw_bank_source_dialogue_overlap_count"
        ]
        == 15
    )
    assert (
        first["contract"]["bank_v2_requirement"][
            "current_raw_bank_card_overlap_count"
        ]
        == 15
    )
    blind_by_id = {
        row["blind_item_id"]: canonical_json(row)
        for row in first["blind_rows"]
    }
    for lineage in first["lineage_rows"]:
        selected_response = esconv[lineage["dialogue_index"]]["dialog"][
            lineage["turn_index"]
        ]["content"]
        assert selected_response not in blind_by_id[lineage["blind_item_id"]]
    assert all(
        not (set(row) & FORBIDDEN_BLIND_KEYS) for row in first["blind_rows"]
    )
    assert all(
        set(row) == {"blind_item_id", "visible_state"}
        and set(row["visible_state"])
        == {"current_user_text", "recent_dialogue", "session_summary"}
        for row in first["blind_rows"]
    )
    contract = dict(first["contract"])
    expected = contract.pop("contract_sha256")
    assert expected == sha256_text(canonical_json(contract))


def test_human_annotation_binding_requires_exact_coherent_coverage(tmp_path: Path):
    esconv = [_dialogue(index) for index in range(20)]
    esconv_path = tmp_path / "esconv.json"
    esconv_path.write_text(json.dumps(esconv), encoding="utf-8")
    split_path = tmp_path / "split.jsonl"
    write_jsonl(
        split_path,
        [
            {
                "index": index,
                "dialogue_id": f"esconv_{index:04d}",
                "split": "train",
                "excluded_for_evoemo_overlap": False,
            }
            for index in range(20)
        ],
    )
    seeds_path = tmp_path / "seeds.jsonl"
    write_jsonl(seeds_path, [{"dialogue_id": "esconv_0000"}])
    bank_path = tmp_path / "strategy_bank.jsonl"
    write_jsonl(
        bank_path,
        [
            {"source_dialogue_id": f"esconv_{index:04d}"}
            for index in range(1, 20)
        ],
    )
    packet = build_support_need_annotation_packet(
        esconv_path=esconv_path,
        split_manifest_path=split_path,
        existing_seed_sources_path=seeds_path,
        strategy_bank_path=bank_path,
        packet_size=15,
        human_anchor_size=6,
    )
    packet_dir = tmp_path / "packet"
    packet_dir.mkdir()
    (packet_dir / "qualification_contract.json").write_text(
        json.dumps(packet["contract"]), encoding="utf-8"
    )
    write_jsonl(
        packet_dir / "human_blind_packet.jsonl",
        packet["human_anchor_rows"],
    )
    annotations = [
        {
            "blind_item_id": row["blind_item_id"],
            "support_mode": "explore",
            "goals": ["make_sense"],
            "dialogue_phase": "exploration",
            "nonclinical_urgency": "routine",
            "advice_rejected": None,
            "advice_requested": None,
            "one_small_step_requested": None,
            "listen_first_requested": None,
            "question_or_task_burden_limit": True,
            "abstain": False,
            "confidence": 3,
            "notes": "",
        }
        for row in packet["human_anchor_rows"]
    ]
    annotations_path = tmp_path / "annotations.jsonl"
    write_jsonl(annotations_path, annotations)
    report = validate_support_need_human_annotations(
        packet_dir=packet_dir,
        annotations_path=annotations_path,
    )
    assert report["status"] == "COMPLETE_HUMAN_ANCHORS_NOT_AUTOMATIC_GOLD"
    assert report["annotation_count"] == 6
    assert report["automatic_gold_label"] is False

    write_jsonl(annotations_path, annotations[:-1])
    try:
        validate_support_need_human_annotations(
            packet_dir=packet_dir,
            annotations_path=annotations_path,
        )
    except RuntimeError as error:
        assert "coverage is not exact" in str(error)
    else:
        raise AssertionError("incomplete human annotations must fail closed")


def test_human_anchor_normalization_separates_explicit_and_recommended_burden(
    tmp_path: Path,
):
    visible_state = {
        "current_user_text": "I am overwhelmed and do not know what to do.",
        "recent_dialogue": [],
        "session_summary": "",
    }
    blind_item_id = "need_fixture"
    anchor_rows = [
        {"blind_item_id": blind_item_id, "visible_state": visible_state}
    ]
    contract_core = {
        "protocol": "pm-v1.5-support-need-human-packet-v1",
        "human_anchor_packet_sha256": sha256_text(
            canonical_json(anchor_rows)
        ),
    }
    contract = {
        **contract_core,
        "contract_sha256": sha256_text(canonical_json(contract_core)),
    }
    packet_dir = tmp_path / "packet"
    packet_dir.mkdir()
    (packet_dir / "qualification_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    write_jsonl(packet_dir / "human_blind_packet.jsonl", anchor_rows)
    annotations_path = tmp_path / "annotations.jsonl"
    write_jsonl(
        annotations_path,
        [
            {
                "blind_item_id": blind_item_id,
                "support_mode": "comfort_reassure",
                "goals": ["stabilize"],
                "dialogue_phase": "comforting",
                "nonclinical_urgency": "elevated",
                "advice_rejected": None,
                "advice_requested": None,
                "one_small_step_requested": None,
                "listen_first_requested": None,
                "question_or_task_burden_limit": True,
                "abstain": False,
                "confidence": 4,
                "notes": "Keep the next response low burden.",
            }
        ],
    )

    result = normalize_support_need_human_anchors(
        packet_dir=packet_dir,
        annotations_path=annotations_path,
    )
    row = result["normalized_rows"][0]
    assert row["human_recommended_low_interaction_burden"] is True
    assert (
        row["deterministic_current_turn_explicit_boundaries"][
            "question_or_task_burden_limit"
        ]
        is False
    )
    assert (
        row["contextual_human_boundary_judgments"][
            "question_or_task_burden_limit"
        ]
        is True
    )
    assert row["raw_annotation"]["question_or_task_burden_limit"] is True
    assert result["report"]["legacy_field_interpretation"][
        "reannotation_required"
    ] is False
    assert result["report"]["automatic_gold_label"] is False


def test_support_need_expansion_is_fresh_and_quote_validated(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    all_rows = [
        {
            "blind_item_id": f"need_{index:02d}",
            "visible_state": {
                "current_user_text": f"Please listen to concern {index}.",
                "recent_dialogue": [],
                "session_summary": "",
            },
        }
        for index in range(30)
    ]
    write_jsonl(source / "support_need_packet.jsonl", all_rows)
    write_jsonl(source / "human_blind_packet.jsonl", all_rows[:6])
    write_jsonl(
        source / "private_lineage.jsonl",
        [
            {
                "blind_item_id": row["blind_item_id"],
                "dialogue_id": f"dialogue_{index}",
                "position_stratum": ("early", "middle", "late")[index % 3],
            }
            for index, row in enumerate(all_rows)
        ],
    )
    source_contract_core = {"protocol": "fixture"}
    (source / "qualification_contract.json").write_text(
        json.dumps(
            {
                **source_contract_core,
                "contract_sha256": sha256_text(
                    canonical_json(source_contract_core)
                ),
            }
        ),
        encoding="utf-8",
    )
    first = build_support_need_expansion_packet(
        source_packet_dir=source,
        packet_size=12,
        fit_anchor_count=8,
    )
    second = build_support_need_expansion_packet(
        source_packet_dir=source,
        packet_size=12,
        fit_anchor_count=8,
    )
    assert first == second
    assert not (
        {row["blind_item_id"] for row in first["packet_rows"]}
        & {row["blind_item_id"] for row in all_rows[:6]}
    )
    assert sum(
        row["anchor_role"] == "untouched_confirmation"
        for row in first["private_rows"]
    ) == 4
    packet = tmp_path / "expansion"
    packet.mkdir()
    (packet / "qualification_contract.json").write_text(
        json.dumps(first["contract"]), encoding="utf-8"
    )
    write_jsonl(packet / "human_blind_packet.jsonl", first["packet_rows"])
    annotations = [
        {
            "blind_item_id": row["blind_item_id"],
            "support_mode": "listen",
            "goals": ["be_heard"],
            "dialogue_phase": "comforting",
            "nonclinical_urgency": "routine",
            "recommended_response_burden": "minimal_presence",
            "active_explicit_boundary_evidence": [
                {
                    "boundary_type": "listen_first_requested",
                    "exact_user_quote": "Please listen",
                }
            ],
            "abstain": False,
            "confidence": 4,
            "notes": "",
        }
        for row in first["packet_rows"]
    ]
    annotation_path = tmp_path / "annotations.jsonl"
    write_jsonl(annotation_path, annotations)
    report = validate_support_need_expansion_annotations(
        packet_dir=packet,
        annotations_path=annotation_path,
    )
    assert report["annotation_count"] == 12
    annotations[0]["active_explicit_boundary_evidence"][0][
        "exact_user_quote"
    ] = "not in user text"
    write_jsonl(annotation_path, annotations)
    try:
        validate_support_need_expansion_annotations(
            packet_dir=packet,
            annotations_path=annotation_path,
        )
    except RuntimeError as error:
        assert "not an exact visible user substring" in str(error)
    else:
        raise AssertionError("invented boundary quote must fail closed")


def test_expansion_fit_packet_seals_preregistered_confirmation_rows(
    tmp_path: Path,
):
    source = tmp_path / "source"
    source.mkdir()
    all_rows = [
        {
            "blind_item_id": f"need_{index:02d}",
            "visible_state": {
                "current_user_text": f"Visible concern {index}.",
                "recent_dialogue": [],
                "session_summary": "",
            },
        }
        for index in range(24)
    ]
    write_jsonl(source / "support_need_packet.jsonl", all_rows)
    write_jsonl(source / "human_blind_packet.jsonl", all_rows[:6])
    write_jsonl(
        source / "private_lineage.jsonl",
        [
            {
                "blind_item_id": row["blind_item_id"],
                "dialogue_id": f"dialogue_{index}",
            }
            for index, row in enumerate(all_rows)
        ],
    )
    source_contract_core = {"protocol": "fixture"}
    (source / "qualification_contract.json").write_text(
        json.dumps(
            {
                **source_contract_core,
                "contract_sha256": sha256_text(
                    canonical_json(source_contract_core)
                ),
            }
        ),
        encoding="utf-8",
    )
    expansion = build_support_need_expansion_packet(
        source_packet_dir=source,
        packet_size=12,
        fit_anchor_count=8,
    )
    expansion_dir = tmp_path / "expansion"
    expansion_dir.mkdir()
    (expansion_dir / "qualification_contract.json").write_text(
        json.dumps(expansion["contract"]), encoding="utf-8"
    )
    write_jsonl(
        expansion_dir / "human_blind_packet.jsonl",
        expansion["packet_rows"],
    )
    write_jsonl(
        expansion_dir / "human_annotation_template.jsonl",
        expansion["template_rows"],
    )
    write_jsonl(
        expansion_dir / "private_lineage.jsonl",
        expansion["private_rows"],
    )
    candidate_report_core = {
        "status": "COMPLETE_DIAGNOSTIC_NOT_FORMAL_FIT",
        "formal_fit_authorized": False,
        "representation_promotion_authorized": False,
        "expansion_human_annotations_opened": False,
        "internal_test_outcomes_opened": False,
        "external_outcomes_opened": False,
    }
    candidate_report = {
        **candidate_report_core,
        "report_sha256": sha256_text(
            canonical_json(candidate_report_core)
        ),
    }
    candidate_report_path = tmp_path / "candidate_report.json"
    reproduction_report_path = tmp_path / "reproduction_report.json"
    candidate_report_text = json.dumps(candidate_report)
    candidate_report_path.write_text(
        candidate_report_text, encoding="utf-8"
    )
    reproduction_report_path.write_text(
        candidate_report_text, encoding="utf-8"
    )
    bakeoff_binding = tmp_path / "bakeoff.json"
    bakeoff_binding.write_text(
        json.dumps(
            {
                "status": (
                    "COMPLETE_REPRODUCIBLE_QWEN_NOT_PROMOTED_"
                    "AXIS_SPECIFIC_SIGNAL_MORE_HUMAN_ANCHORS_REQUIRED"
                ),
                "next_action": {
                    "decision": (
                        "COLLECT_ONLY_PREPARED_EXPANSION_FIT_ANCHORS"
                    ),
                    "fit_anchor_count": 8,
                    "untouched_confirmation_count": 4,
                },
                "candidate_report": {
                    "path": str(candidate_report_path),
                    "file_sha256": sha256_file(candidate_report_path),
                    "report_sha256": candidate_report[
                        "report_sha256"
                    ],
                },
                "reproduction_report": {
                    "path": str(reproduction_report_path),
                    "file_sha256": sha256_file(
                        reproduction_report_path
                    ),
                    "byte_identical_to_candidate": True,
                    "field_diff_count": 0,
                },
                "formal_fit_authorized": False,
                "representation_promotion_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    first = build_support_need_expansion_fit_packet(
        expansion_packet_dir=expansion_dir,
        bakeoff_binding_path=bakeoff_binding,
    )
    second = build_support_need_expansion_fit_packet(
        expansion_packet_dir=expansion_dir,
        bakeoff_binding_path=bakeoff_binding,
    )
    assert first == second
    assert len(first["packet_rows"]) == 8
    assert first["contract"]["untouched_confirmation_count"] == 4
    assert first["contract"]["confirmation_rows_exposed"] is False
    assert first["contract"]["blind_packet_exposes_anchor_role"] is False
    assert all(
        "anchor_role" not in row
        and not (set(row) & FORBIDDEN_BLIND_KEYS)
        for row in first["packet_rows"]
    )
    fit_ids = {row["blind_item_id"] for row in first["packet_rows"]}
    confirmation_ids = {
        row["blind_item_id"]
        for row in expansion["private_rows"]
        if row["anchor_role"] == "untouched_confirmation"
    }
    assert not (fit_ids & confirmation_ids)

    reproduction_report_path.write_text("{}", encoding="utf-8")
    try:
        build_support_need_expansion_fit_packet(
            expansion_packet_dir=expansion_dir,
            bakeoff_binding_path=bakeoff_binding,
        )
    except RuntimeError as error:
        assert "reports are missing or drifted" in str(error)
    else:
        raise AssertionError("drifted bakeoff reproduction must fail closed")
