from __future__ import annotations

import json
from pathlib import Path

import pytest

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_jsonl,
)
from metacom_pm.v1_5_support_need_adjudication import (
    ADJUDICATION_CONFIG_PROTOCOL,
    build_support_need_fit_adjudication,
    combine_support_need_fit_anchors,
)


def _annotation(
    blind_item_id: str,
    *,
    mode: str,
    goals: list[str],
    phase: str,
    evidence: list[dict] | None = None,
) -> dict:
    return {
        "blind_item_id": blind_item_id,
        "support_mode": mode,
        "goals": goals,
        "dialogue_phase": phase,
        "nonclinical_urgency": "routine",
        "recommended_response_burden": (
            "minimal_presence" if mode == "listen" else "one_focus"
        ),
        "active_explicit_boundary_evidence": evidence or [],
        "abstain": False,
        "confidence": 4,
        "notes": "fixture",
    }


def _fit_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    packet_rows = [
        {
            "blind_item_id": "need_a",
            "visible_state": {
                "current_user_text": "Can you give me one idea?",
                "recent_dialogue": [],
                "session_summary": "",
            },
        },
        {
            "blind_item_id": "need_b",
            "visible_state": {
                "current_user_text": "Thanks for listening.",
                "recent_dialogue": [],
                "session_summary": "",
            },
        },
    ]
    contract_core = {
        "protocol": "fixture",
        "selected_packet_sha256": sha256_text(
            canonical_json(packet_rows)
        ),
        "packet_size": 2,
        "fit_anchor_count": 2,
        "untouched_confirmation_count": 1,
        "confirmation_rows_exposed": False,
    }
    packet_dir = tmp_path / "fit_packet"
    packet_dir.mkdir()
    (packet_dir / "qualification_contract.json").write_text(
        json.dumps(
            {
                **contract_core,
                "contract_sha256": sha256_text(
                    canonical_json(contract_core)
                ),
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(packet_dir / "human_blind_packet.jsonl", packet_rows)
    advice = [
        {
            "boundary_type": "advice_requested",
            "exact_user_quote": "Can you give me one idea?",
        }
    ]
    first_rows = [
        _annotation(
            "need_a",
            mode="explore",
            goals=["make_sense"],
            phase="exploration",
            evidence=advice,
        ),
        _annotation(
            "need_b",
            mode="listen",
            goals=["be_heard"],
            phase="comforting",
        ),
    ]
    second_rows = [
        _annotation(
            "need_a",
            mode="light_guidance",
            goals=["act"],
            phase="action",
            evidence=advice,
        ),
        _annotation(
            "need_b",
            mode="listen",
            goals=["be_heard"],
            phase="comforting",
        ),
    ]
    first_path = tmp_path / "rater_a.jsonl"
    second_path = tmp_path / "rater_b.jsonl"
    write_jsonl(first_path, first_rows)
    write_jsonl(second_path, second_rows)
    decisions = {
        "protocol": ADJUDICATION_CONFIG_PROTOCOL,
        "packet_contract_sha256": sha256_text(
            canonical_json(contract_core)
        ),
        "expected_source_file_sha256": {
            "rater_a": sha256_file(first_path),
            "rater_b": sha256_file(second_path),
        },
        "adjudications": [
            {
                "blind_item_id": "need_a",
                "adjudication_rationale": "The request supports guidance.",
                "final_annotation": second_rows[0],
            },
            {
                "blind_item_id": "need_b",
                "adjudication_rationale": "Both reviews agree.",
                "final_annotation": second_rows[1],
            },
        ],
    }
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(json.dumps(decisions), encoding="utf-8")
    return packet_dir, first_path, second_path, decisions_path


def test_fit_adjudication_preserves_one_group_per_item(tmp_path: Path):
    packet_dir, first_path, second_path, decisions_path = _fit_fixture(
        tmp_path
    )
    result = build_support_need_fit_adjudication(
        packet_dir=packet_dir,
        rater_a_path=first_path,
        rater_b_path=second_path,
        decisions_path=decisions_path,
    )
    assert result["report"]["source_review_row_count"] == 4
    assert result["report"]["independent_dialogue_groups"] == 2
    assert result["report"]["source_review_rows_do_not_increase_ess"] is True
    assert result["report"]["confirmation_rows_exposed"] is False
    trace = result["trace_rows"][0]
    assert trace["pre_adjudication_distributions"]["support_mode"] == {
        "explore": 0.5,
        "light_guidance": 0.5,
    }
    assert result["normalized_rows"][0][
        "human_recommended_low_interaction_burden"
    ] is True


def test_fit_adjudication_cannot_invent_evidence(tmp_path: Path):
    packet_dir, first_path, second_path, decisions_path = _fit_fixture(
        tmp_path
    )
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    decisions["adjudications"][0]["final_annotation"][
        "active_explicit_boundary_evidence"
    ] = [
        {
            "boundary_type": "one_small_step_requested",
            "exact_user_quote": "Can you give me one idea?",
        }
    ]
    decisions_path.write_text(json.dumps(decisions), encoding="utf-8")
    with pytest.raises(RuntimeError, match="cannot invent"):
        build_support_need_fit_adjudication(
            packet_dir=packet_dir,
            rater_a_path=first_path,
            rater_b_path=second_path,
            decisions_path=decisions_path,
        )


def test_combined_fit_anchors_do_not_duplicate_groups(tmp_path: Path):
    packet_dir, first_path, second_path, decisions_path = _fit_fixture(
        tmp_path
    )
    fit = build_support_need_fit_adjudication(
        packet_dir=packet_dir,
        rater_a_path=first_path,
        rater_b_path=second_path,
        decisions_path=decisions_path,
    )
    initial_packet_rows = [
        {
            "blind_item_id": "need_initial",
            "visible_state": {
                "current_user_text": "I need support.",
                "recent_dialogue": [],
                "session_summary": "",
            },
        }
    ]
    initial_contract_core = {
        "human_anchor_packet_sha256": sha256_text(
            canonical_json(initial_packet_rows)
        )
    }
    initial_packet_dir = tmp_path / "initial_packet"
    initial_packet_dir.mkdir()
    (initial_packet_dir / "qualification_contract.json").write_text(
        json.dumps(
            {
                **initial_contract_core,
                "contract_sha256": sha256_text(
                    canonical_json(initial_contract_core)
                ),
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        initial_packet_dir / "human_blind_packet.jsonl",
        initial_packet_rows,
    )
    initial_normalized_rows = [
        {
            "blind_item_id": "need_initial",
            "raw_annotation": {
                "abstain": False,
                "confidence": 4,
            },
        }
    ]
    initial_normalized_path = tmp_path / "initial_normalized.jsonl"
    write_jsonl(initial_normalized_path, initial_normalized_rows)
    binding_core = {
        "normalized_rows_sha256": sha256_text(
            canonical_json(initial_normalized_rows)
        )
    }
    binding_path = tmp_path / "initial_binding.json"
    binding_path.write_text(
        json.dumps(
            {
                **binding_core,
                "normalization_binding_sha256": sha256_text(
                    canonical_json(binding_core)
                ),
            }
        ),
        encoding="utf-8",
    )
    fit_packet_rows = [
        dict(row)
        for row in iter_jsonl(packet_dir / "human_blind_packet.jsonl")
    ]
    result = combine_support_need_fit_anchors(
        initial_packet_dir=initial_packet_dir,
        initial_normalized_path=initial_normalized_path,
        initial_normalization_binding_path=binding_path,
        fit_packet_rows=fit_packet_rows,
        fit_normalized_rows=fit["normalized_rows"],
        adjudication_report=fit["report"],
    )
    assert result["report"]["combined_packet_rows"] == 3
    assert result["report"]["independent_dialogue_groups"] == 3
    assert result["report"]["confirmation_rows_exposed"] is False


def test_support_need_fit_adjudication_binding_hash_is_valid():
    root = Path(__file__).resolve().parents[1]
    binding = read_json(
        root
        / "data/pm_v1_5_contracts/support_need_fit_adjudication_v1.json"
    )
    digest = binding.pop("binding_sha256")
    assert digest == sha256_text(canonical_json(binding))
