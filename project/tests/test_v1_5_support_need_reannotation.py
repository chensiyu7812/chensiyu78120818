from __future__ import annotations

import json
from pathlib import Path

from metacom_pm.io import (
    canonical_json,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.v1_5_support_need_packet import (
    validate_support_need_expansion_annotations,
)
from metacom_pm.v1_5_support_need_reannotation import (
    ALLOWED_VISIBLE_KEYS,
    build_historical_support_need_reannotation_packet,
)


def _judge_rows(prefix: str, start: int) -> list[dict]:
    return [
        {
            "blind_item_id": f"{prefix}_{index}",
            "authorized_user_context": "must be stripped",
            "candidate_a": {
                "response": "must be stripped",
                "selected_context": {"memory": ["secret"], "strategy": []},
            },
            "candidate_b": {
                "response": "must be stripped",
                "selected_context": {"memory": [], "strategy": ["secret"]},
            },
            "visible_state": {
                "current_user_text": f"visible concern {start + index}",
                "recent_dialogue": [
                    {"role": "user", "content": f"history {start + index}"},
                    {"role": "assistant", "content": "visible reply"},
                ],
                "current_session_summary": f"summary {start + index}",
            },
        }
        for index in range(12)
    ]


def _binding(root: Path, low: Path, role: Path, current: Path) -> Path:
    report_core = {
        "protocol": "pm-v1.5-historical-human-annotation-asset-audit-v1",
        "status": "COMPLETE_USE_RESTRICTED_INVENTORY",
        "historical_judge_state_reannotation_opportunity": {
            "status": "ELIGIBLE_AS_NEW_BLINDED_ACTIVE_LEARNING_POOL_ONLY",
            "source_packet_file_sha256": {
                "low_budget": sha256_file(low),
                "role_decomposed": sha256_file(role),
                "current_support_need": sha256_file(current),
            },
        },
    }
    report = {
        **report_core,
        "report_sha256": sha256_text(canonical_json(report_core)),
    }
    out = root / "audit"
    out.mkdir()
    report_path = out / "report.json"
    write_json(report_path, report)
    binding_core = {
        "protocol": (
            "pm-v1.5-historical-human-annotation-asset-audit-binding-v1"
        ),
        "status": "COMPLETE_REPRODUCIBLE_USE_RESTRICTED_INVENTORY",
        "output_directory": "audit",
        "candidate_repro_byte_identical": True,
        "report_file_sha256": sha256_file(report_path),
        "report_sha256": report["report_sha256"],
        "historical_judge_state_reannotation_candidate_count": 24,
        "historical_judge_state_current_support_need_exact_overlap": 0,
        "historical_judge_labels_reused_as_need_targets": False,
    }
    binding = {
        **binding_core,
        "binding_sha256": sha256_text(canonical_json(binding_core)),
    }
    path = root / "audit_binding.json"
    write_json(path, binding)
    return path


def test_historical_reannotation_strips_treatment_and_reuses_v2_validator(
    tmp_path: Path,
):
    low = tmp_path / "low.jsonl"
    role = tmp_path / "role.jsonl"
    current = tmp_path / "current.jsonl"
    write_jsonl(low, _judge_rows("low", 0))
    write_jsonl(role, _judge_rows("role", 100))
    write_jsonl(
        current,
        [
            {
                "blind_item_id": "current",
                "visible_state": {
                    "current_user_text": "different current state",
                    "recent_dialogue": [],
                    "session_summary": "",
                },
            }
        ],
    )
    binding = _binding(tmp_path, low, role, current)
    result = build_historical_support_need_reannotation_packet(
        project_root=tmp_path,
        human_asset_audit_binding_path=binding,
        low_budget_packet_path=low,
        role_decomposed_packet_path=role,
        current_support_need_packet_path=current,
    )
    assert len(result["packet_rows"]) == 24
    assert len(result["private_rows"]) == 24
    assert result["contract"]["old_human_labels_reused"] is False
    assert result["contract"]["prevalence_estimation_allowed"] is False
    assert all(
        set(row) == {"blind_item_id", "visible_state"}
        and set(row["visible_state"]) == ALLOWED_VISIBLE_KEYS
        for row in result["packet_rows"]
    )
    rendered = json.dumps(result["packet_rows"])
    assert "authorized_user_context" not in rendered
    assert "candidate_a" not in rendered
    assert "must be stripped" not in rendered
    assert "secret" not in rendered

    packet_dir = tmp_path / "packet"
    packet_dir.mkdir()
    write_json(
        packet_dir / "qualification_contract.json", result["contract"]
    )
    write_jsonl(
        packet_dir / "human_blind_packet.jsonl", result["packet_rows"]
    )
    annotations = [
        {
            "blind_item_id": row["blind_item_id"],
            "support_mode": "explore",
            "goals": ["make_sense"],
            "dialogue_phase": "exploration",
            "nonclinical_urgency": "routine",
            "recommended_response_burden": "one_focus",
            "active_explicit_boundary_evidence": [],
            "abstain": False,
            "confidence": 3,
            "notes": "",
        }
        for row in result["packet_rows"]
    ]
    annotation_path = tmp_path / "annotations.jsonl"
    write_jsonl(annotation_path, annotations)
    validation = validate_support_need_expansion_annotations(
        packet_dir=packet_dir,
        annotations_path=annotation_path,
    )
    assert validation["annotation_count"] == 24
    assert validation["automatic_gold_label"] is False


def test_historical_reannotation_is_deterministic(tmp_path: Path):
    low = tmp_path / "low.jsonl"
    role = tmp_path / "role.jsonl"
    current = tmp_path / "current.jsonl"
    write_jsonl(low, _judge_rows("low", 0))
    write_jsonl(role, _judge_rows("role", 100))
    write_jsonl(
        current,
        [
            {
                "blind_item_id": "current",
                "visible_state": {
                    "current_user_text": "different",
                    "recent_dialogue": [],
                    "session_summary": "",
                },
            }
        ],
    )
    binding = _binding(tmp_path, low, role, current)
    kwargs = {
        "project_root": tmp_path,
        "human_asset_audit_binding_path": binding,
        "low_budget_packet_path": low,
        "role_decomposed_packet_path": role,
        "current_support_need_packet_path": current,
    }
    assert build_historical_support_need_reannotation_packet(
        **kwargs
    ) == build_historical_support_need_reannotation_packet(**kwargs)

