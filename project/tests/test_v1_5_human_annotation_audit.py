from __future__ import annotations

from pathlib import Path

from metacom_pm.io import canonical_json, read_json, sha256_text
from metacom_pm.v1_5_human_annotation_audit import (
    audit_historical_human_annotations,
)


ROOT = Path(__file__).resolve().parents[1]


def test_historical_human_annotation_audit_separates_data_roles():
    report = audit_historical_human_annotations(project_root=ROOT)
    assets = {
        row["asset_id"]: row for row in report["logical_assets"]
    }
    assert report["summary"]["completed_support_need_human_rows"] == 40
    assert (
        report["support_need_fit_decision"][
            "independent_dialogue_groups"
        ]
        == 39
    )
    assert (
        report["support_need_fit_decision"][
            "raw_rater_rows_add_independent_groups"
        ]
        is False
    )
    assert assets["support_need_human_anchor_v1"][
        "support_need_fit_eligible"
    ]
    assert assets["support_need_factorized_fit_adjudication_v1"][
        "support_need_fit_eligible"
    ]
    assert not assets["low_budget_judge_human_anchor_v1"][
        "support_need_fit_eligible"
    ]
    assert not assets["role_decomposed_judge_human_anchor_v1"][
        "support_need_fit_eligible"
    ]
    assert assets["strategy_bank_v2_human_template"]["status"] == (
        "TEMPLATE_UNANNOTATED"
    )
    assert assets["support_need_confirmation_subset_v1"]["status"] == (
        "SEALED_UNANNOTATED_CONFIRMATION"
    )
    opportunity = report[
        "historical_judge_state_reannotation_opportunity"
    ]
    assert opportunity["candidate_dialogue_states"] == 24
    assert opportunity["low_budget_role_exact_overlap"] == 0
    assert (
        opportunity["judge_state_current_support_need_exact_overlap"] == 0
    )
    assert opportunity["old_human_labels_reused_as_need_targets"] is False
    report_core = dict(report)
    digest = report_core.pop("report_sha256")
    assert digest == sha256_text(canonical_json(report_core))


def test_human_annotation_asset_audit_binding_hash_is_valid():
    binding = read_json(
        ROOT
        / "data/pm_v1_5_contracts/human_annotation_asset_audit_v1.json"
    )
    digest = binding.pop("binding_sha256")
    assert digest == sha256_text(canonical_json(binding))
