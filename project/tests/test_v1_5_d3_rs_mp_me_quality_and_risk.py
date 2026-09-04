from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "outputs/pm_v1_5_d3_rs_mp_me_quality_aggregation_v1"
RISK = ROOT / "outputs/pm_v1_5_d3_rs_mp_me_on_win_risk_review_v1_candidate"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_d3_quality_aggregation_excludes_original_ms_without_cherry_picking() -> None:
    report = _json(AGG / "aggregation_report.json")
    annotations = _jsonl(AGG / "validated_all_160_quality_annotations.jsonl")
    pairs = _jsonl(AGG / "pair_quality_measurements_pre_risk.jsonl")
    states = _jsonl(AGG / "state_soft_targets_pre_risk.jsonl")
    fidelity = _jsonl(AGG / "mp_me_fidelity_rows_for_evidence_audit.jsonl")
    assert report["status"] == "PASS_35_COMPONENT_ON_WINS_PENDING_MINIMAL_RISK"
    assert all(report["checks"].values())
    assert len(annotations) == 160
    assert len(pairs) == 120
    assert len(states) == 96
    assert len(fidelity) == 4
    assert Counter(row["component"] for row in pairs) == Counter({"RS": 40, "MP": 40, "ME": 40})
    assert Counter(row["component"] for row in states) == Counter({"RS": 32, "MP": 32, "ME": 32})
    assert Counter(row["quality_verdict"] for row in pairs) == Counter(
        {"control": 52, "tie": 33, "treatment": 35}
    )
    assert all(row["state_training_weight"] == 1.0 for row in states)


def test_d3_on_win_risk_packet_is_bounded_and_lineage_hidden() -> None:
    manifest = _json(RISK / "manifest.json")
    public = _jsonl(RISK / "human_risk_packet.jsonl")
    private = _jsonl(RISK / "private_risk_key.jsonl")
    blank = _jsonl(RISK / "blank_risk_annotations.jsonl")
    assert manifest["status"] == "READY_FOR_ONE_BOUNDED_35_ITEM_ON_WIN_MINIMAL_RISK_REVIEW"
    assert all(manifest["checks"].values())
    assert len(public) == len(private) == len(blank) == 35
    assert Counter(row["component"] for row in private) == Counter({"ME": 15, "RS": 13, "MP": 7})
    assert {row["review_item_id"] for row in public} == {row["review_item_id"] for row in private}
    hidden = {"component", "pair_id", "state_id", "user_id", "source_blind_item_id"}
    assert all(not (set(row) & hidden) for row in public)
    assert all(row["quality_label"] == "ON_QUALITY_WIN_PENDING_MATERIAL_RISK" for row in private)
    assert manifest["not_an_overall_component_risk_rate"] is True
    assert manifest["not_a_control_vs_treatment_risk_difference"] is True
