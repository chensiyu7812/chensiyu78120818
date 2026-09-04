from __future__ import annotations

import json
from pathlib import Path

import pytest

from metacom_pm.v1_5_judge_qualification_analysis import (
    aggregate_qualification,
    require_tracked_human_anchor,
    validate_human_annotations,
)
from metacom_pm.io import (
    canonical_json,
    read_json,
    sha256_file,
    sha256_text,
)


ROOT = Path(__file__).resolve().parents[1]


def test_tracked_human_anchor_is_complete_and_content_addressed(
    tmp_path: Path,
):
    pairs = [
        {
            "pair_id": "pair_1",
            "candidate_a": {"response": "A"},
            "candidate_b": {"response": "B"},
        }
    ]
    packet = [
        {
            "blind_item_id": "blind_1",
            "candidate_a": {"response": "A"},
            "candidate_b": {"response": "B"},
        }
    ]
    annotations = [
        {
            "blind_item_id": "blind_1",
            "overall_preference": "A",
            "support_quality_preference": "A",
            "evidence_handling_preference": "A",
            "safety_preference": "tie",
            "confidence": 4,
            "notes": "independent anchor",
        }
    ]
    annotation_path = tmp_path / "annotations.jsonl"
    annotation_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in annotations
        ),
        encoding="utf-8",
    )
    rendered_packet = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in packet
    )
    unsigned_binding = {
        "protocol": "pm-v1.5-low-budget-judge-human-anchor-v1",
        "status": "FROZEN_BEFORE_MODEL_RESULTS",
        "annotation_role": (
            "independent_researcher_anchor_not_automatic_gold"
        ),
        "annotations": {
            "expected_rows": 1,
            "path": "annotations.jsonl",
            "sha256": sha256_file(annotation_path),
        },
        "human_blind_packet": {
            "expected_rows": 1,
            "canonical_content_sha256": sha256_text(
                canonical_json(packet)
            ),
            "source_file_sha256": sha256_text(rendered_packet),
        },
        "decision_boundary": {
            "may_be_used_as_training_labels": False,
            "may_auto_promote_bulk_labeler": False,
            "majority_vote_is_gold": False,
            "requires_researcher_signoff": True,
        },
    }
    binding_path = tmp_path / "binding.json"
    binding_path.write_text(
        json.dumps(
            {
                **unsigned_binding,
                "binding_sha256": sha256_text(
                    canonical_json(unsigned_binding)
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    record = require_tracked_human_anchor(
        root=tmp_path,
        binding_path=binding_path,
        pairs=pairs,
        human_packet=packet,
    )
    assert record["annotations"] == 1
    assert record["automatic_gold"] is False
    assert record["may_auto_promote_bulk_labeler"] is False

    drifted_packet = json.loads(json.dumps(packet))
    drifted_packet[0]["candidate_a"]["response"] += " drift"
    with pytest.raises(RuntimeError, match="packet content drifted"):
        require_tracked_human_anchor(
            root=tmp_path,
            binding_path=binding_path,
            pairs=pairs,
            human_packet=drifted_packet,
        )

    binding = read_json(binding_path)
    binding["annotations"]["sha256"] = "0" * 64
    unsigned = dict(binding)
    unsigned.pop("binding_sha256")
    binding["binding_sha256"] = sha256_text(canonical_json(unsigned))
    bad_annotation_binding = tmp_path / "bad_annotation_binding.json"
    bad_annotation_binding.write_text(
        json.dumps(binding, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="annotations hash mismatch"):
        require_tracked_human_anchor(
            root=tmp_path,
            binding_path=bad_annotation_binding,
            pairs=pairs,
            human_packet=packet,
        )

    binding = read_json(binding_path)
    binding["status"] = "EDITED_AFTER_RESULTS"
    bad_binding = tmp_path / "bad_binding.json"
    bad_binding.write_text(
        json.dumps(binding, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="binding hash mismatch"):
        require_tracked_human_anchor(
            root=tmp_path,
            binding_path=bad_binding,
            pairs=pairs,
            human_packet=packet,
        )


def test_repository_human_anchor_binds_the_tracked_annotations():
    binding_path = (
        ROOT
        / "data/pm_v1_5_contracts/low_budget_judge_human_anchor_v1.json"
    )
    binding = read_json(binding_path)
    unsigned = dict(binding)
    binding_hash = unsigned.pop("binding_sha256")
    assert binding_hash == sha256_text(canonical_json(unsigned))
    annotation_path = ROOT / binding["annotations"]["path"]
    assert sha256_file(annotation_path) == binding["annotations"]["sha256"]
    assert binding["annotation_role"].endswith("not_automatic_gold")
    assert binding["decision_boundary"]["majority_vote_is_gold"] is False


def _pairwise_parsed(preference: str) -> dict:
    return {
        "overall_preference": preference,
        "support_quality_preference": preference,
        "evidence_handling_preference": preference,
        "safety_preference": preference,
        "overall_reason": "reason",
        "support_quality_reason": "reason",
        "evidence_handling_reason": "reason",
        "safety_reason": "reason",
    }


def _complete_results():
    rows = []
    pair_ids = [f"p{index}" for index in range(12)]
    for condition, ids in (
        ("qwen_nonthinking_pairwise", pair_ids),
        ("qwen_thinking_pairwise", pair_ids),
        ("gpt_high_quality_pairwise_anchor", pair_ids[:8]),
    ):
        for pair_id in ids:
            for order in (0, 1):
                rows.append(
                    {
                        "condition": condition,
                        "record_ids": {
                            "pair_id": pair_id,
                            "state_id": f"s{pair_id}",
                            "order_variant": order,
                        },
                        "parsed": _pairwise_parsed("A" if order == 0 else "B"),
                    }
                )
    combined = {
        "emotional_support": 4.0,
        "personalization": 4.0,
        "memory_appropriateness": 4.0,
        "factual_grounding": 4.0,
        "temporal_consistency": 4.0,
        "non_intrusiveness": 4.0,
        "selected_context_misuse": 0.0,
        "unnecessary_exposure": 0.0,
        "stale_or_conflicting_use": 0.0,
        "unsupported_personal_claim": 0.0,
        "memory_omission": 0.0,
        "strategy_overuse": 0.0,
        "strategy_omission": 0.0,
        "quality_rationale": "quality",
        "risk_rationale": "risk",
    }
    quality = {
        key: value
        for key, value in combined.items()
        if key
        in {
            "emotional_support",
            "personalization",
            "memory_appropriateness",
            "factual_grounding",
            "temporal_consistency",
            "non_intrusiveness",
        }
    }
    quality["rationale"] = "quality"
    risk = {
        key: value
        for key, value in combined.items()
        if key
        in {
            "selected_context_misuse",
            "unnecessary_exposure",
            "stale_or_conflicting_use",
            "unsupported_personal_claim",
            "memory_omission",
            "strategy_overuse",
            "strategy_omission",
        }
    }
    risk["rationale"] = "risk"
    for index in range(9):
        record_ids = {
            "equivalence_id": f"e{index}",
            "state_id": f"s{index}",
            "regime": f"r{index}",
        }
        rows.extend(
            [
                {
                    "condition": "qwen_nonthinking_combined",
                    "record_ids": record_ids,
                    "parsed": combined,
                },
                {
                    "condition": "qwen_nonthinking_split_quality",
                    "record_ids": record_ids,
                    "parsed": quality,
                },
                {
                    "condition": "qwen_nonthinking_split_risk",
                    "record_ids": record_ids,
                    "parsed": risk,
                },
            ]
        )
    assert len(rows) == 91
    return rows, set(pair_ids[:8])


def _contract() -> dict:
    return json.loads(
        (
            ROOT
            / "data/pm_v1_5_contracts/"
            "low_budget_judge_qualification_v2.json"
        ).read_text(encoding="utf-8")
    )


def test_automatic_qualification_pass_still_requires_human_signoff():
    rows, anchors = _complete_results()
    human = [
        {
            "pair_id": f"p{index}",
            "canonical_overall_preference": "A",
            "canonical_support_quality_preference": "A",
            "canonical_evidence_handling_preference": "A",
            "canonical_safety_preference": "A",
        }
        for index in range(12)
    ]
    report = aggregate_qualification(
        result_rows=rows,
        contract=_contract(),
        gpt_anchor_pair_ids=anchors,
        human_normalized_rows=human,
    )
    assert report["status"] == (
        "AUTOMATIC_METRICS_PASS_REQUIRES_HUMAN_SIGNOFF"
    )
    assert all(report["automatic_checks"].values())
    assert report["bulk_labeling_authorized"] is False
    assert (
        report["combined_vs_split_equivalence"]["mean_absolute_error"]
        == 0.0
    )


def test_incomplete_result_matrix_fails_closed():
    rows, anchors = _complete_results()
    with pytest.raises(RuntimeError, match="incomplete"):
        aggregate_qualification(
            result_rows=rows[:-1],
            contract=_contract(),
            gpt_anchor_pair_ids=anchors,
            human_normalized_rows=None,
        )


def test_human_annotations_require_exact_blind_coverage():
    pairs = [
        {
            "pair_id": "p1",
            "candidate_a": {"response": "a"},
            "candidate_b": {"response": "b"},
        }
    ]
    packet = [
        {
            "blind_item_id": "h1",
            "candidate_a": {"response": "b"},
            "candidate_b": {"response": "a"},
        }
    ]
    annotation = {
        "blind_item_id": "h1",
        "overall_preference": "A",
        "support_quality_preference": "B",
        "evidence_handling_preference": "tie",
        "safety_preference": "insufficient",
        "confidence": 4,
        "notes": "",
    }
    normalized, report = validate_human_annotations(
        pairs=pairs,
        human_packet=packet,
        annotation_rows=[annotation],
    )
    assert report["status"] == "COMPLETE"
    assert normalized[0]["order_variant"] == 1
    assert normalized[0]["canonical_overall_preference"] == "B"
    assert normalized[0]["canonical_support_quality_preference"] == "A"
    with pytest.raises(RuntimeError, match="exactly once"):
        validate_human_annotations(
            pairs=pairs,
            human_packet=packet,
            annotation_rows=[],
        )


def test_human_mapping_rejects_repeated_candidate_content():
    pairs = [
        {
            "pair_id": "p1",
            "candidate_a": {"response": "a"},
            "candidate_b": {"response": "b"},
        },
        {
            "pair_id": "p2",
            "candidate_a": {"response": "a"},
            "candidate_b": {"response": "b"},
        },
    ]
    packet = [
        {
            "blind_item_id": "h1",
            "candidate_a": {"response": "a"},
            "candidate_b": {"response": "b"},
        },
        {
            "blind_item_id": "h2",
            "candidate_a": {"response": "b"},
            "candidate_b": {"response": "a"},
        },
    ]
    with pytest.raises(RuntimeError, match="repeated candidate content"):
        validate_human_annotations(
            pairs=pairs,
            human_packet=packet,
            annotation_rows=[],
        )


def test_aggregate_rejects_incomplete_human_pair_coverage():
    rows, anchors = _complete_results()
    human = [
        {
            "pair_id": "p0",
            "canonical_overall_preference": "A",
            "canonical_support_quality_preference": "A",
            "canonical_evidence_handling_preference": "A",
            "canonical_safety_preference": "A",
        }
    ]
    with pytest.raises(RuntimeError, match="exactly once"):
        aggregate_qualification(
            result_rows=rows,
            contract=_contract(),
            gpt_anchor_pair_ids=anchors,
            human_normalized_rows=human,
        )
