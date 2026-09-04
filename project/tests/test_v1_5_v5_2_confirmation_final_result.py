from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "outputs/pm_v1_5_v5_2_confirmation_final_labels_v1"
ANALYSIS = ROOT / "outputs/pm_v1_5_v5_2_confirmation_analysis_v1"
SENSITIVITY = ROOT / "outputs/pm_v1_5_v5_2_confirmation_analysis_primary_sensitivity_v1"
REPORT = ROOT / "outputs/pm_v1_5_v5_2_confirmation_final_report_v1"


def read(name: Path) -> dict:
    return json.loads(name.read_text())


def jsonl(name: Path) -> list[dict]:
    return [json.loads(line) for line in name.read_text().splitlines() if line.strip()]


def test_final_label_resolution_is_narrow_and_traceable() -> None:
    manifest = read(LABELS / "label_resolution_manifest.json")
    assert manifest["status"] == "FINAL_LABELS_READY_FOR_FROZEN_POLICY_REPLAY"
    assert manifest["manual_representatives"] == {"quality": 255, "risk": 423}
    assert manifest["third_adjudication_items"] == {"quality": 15, "risk": 13}
    assert manifest["construct_validity_repairs"]["new_human_review_requested"] == 0
    assert manifest["response_policy_feature_threshold_or_executor_changed"] is False
    assert manifest["final_distribution"]["quality"] == {"A": 116, "B": 133, "tie": 6}
    assert manifest["final_distribution"]["risk"] == {"no": 365, "yes": 58}
    assert manifest["final_distribution"]["risk_categories"]["fabricated_recall"] == 4
    lineage = jsonl(LABELS / "adjudication_lineage.jsonl")
    assert len(lineage) == 28
    assert sum(row["disposition"] == "narrow_quality_construct_repair" for row in lineage) == 3
    assert sum(
        row["disposition"] == "narrow_risk_category_repair_binary_yes_unchanged"
        for row in lineage
    ) == 2


def test_frozen_system_gate_fails_only_the_recorded_two_requirements() -> None:
    result = read(ANALYSIS / "confirmation_analysis.json")
    assert result["status"] == "FAIL_SINGLE_USE_V5_2_SYSTEM_GATE"
    assert result["overall_pass"] is False
    failed = {name for name, row in result["gate_results"].items() if not row["pass"]}
    assert failed == {"quality_vs_fixed_high_lower_ci", "critical_grounding_count"}
    learned = result["policy_summary"]["learned_pm"]
    fixed = result["policy_summary"]["component_fixed_high"]
    assert learned["selected_on_fraction"] == 0.5078125
    assert learned["critical_grounding_event_count"] == 4
    assert learned["material_risk_rate"] < fixed["material_risk_rate"]
    comparison = result["learned_comparisons"]["component_fixed_high"]
    assert comparison["quality_mean_difference"] == 0.03125
    assert comparison["quality_cluster_bootstrap_95_ci"] == [-0.109375, 0.1796875]
    assert comparison["risk_cluster_bootstrap_95_ci"] == [-0.1171875, -0.015625]


def test_primary_only_sensitivity_reproduces_the_same_failure() -> None:
    result = read(SENSITIVITY / "confirmation_analysis.json")
    failed = {name for name, row in result["gate_results"].items() if not row["pass"]}
    assert result["status"] == "FAIL_SINGLE_USE_V5_2_SYSTEM_GATE"
    assert failed == {"quality_vs_fixed_high_lower_ci", "critical_grounding_count"}
    assert result["learned_comparisons"]["component_fixed_high"]["quality_mean_difference"] == 0.0546875
    assert result["learned_comparisons"]["component_fixed_high"]["quality_cluster_bootstrap_95_ci"][0] == -0.0859375
    assert result["policy_summary"]["learned_pm"]["critical_grounding_event_count"] == 4


def test_portable_report_contains_the_frozen_result_and_semantic_fallback() -> None:
    artifact = read(REPORT / "artifact.json")
    html = (REPORT / "report.html").read_text()
    assert artifact["surface"] == "report"
    assert len(artifact["manifest"]["charts"]) == 3
    assert len(artifact["manifest"]["tables"]) == 6
    assert "完整系统门未通过" in artifact["manifest"]["title"]
    assert "FAIL_SINGLE_USE_V5_2_SYSTEM_GATE" in html
    assert "critical fabricated recall" in html
    assert "semantic" in html.lower()
