from pathlib import Path

from metacom_pm.io import iter_jsonl, read_json
from metacom_pm.v1_5_v3_observation_static_gate import audit_pre_human_static_gate


ROOT = Path(__file__).resolve().parents[1]


def test_realized_orthogonal_data_passes_full_pre_human_static_gate() -> None:
    blueprint = [
        dict(row)
        for row in iter_jsonl(
            ROOT / "data/pm_v1_5_v3_observation_orthogonal_v1/private/construction_blueprint.jsonl"
        )
    ]
    candidates = [
        dict(row)
        for row in iter_jsonl(
            ROOT / "outputs/pm_v1_5_v3_observation_orthogonal_exact_rank1_v1/candidate_rows_private.jsonl"
        )
    ]
    materialization = read_json(
        ROOT / "outputs/pm_v1_5_v3_observation_orthogonal_exact_rank1_v1/materialization_report.json"
    )
    report = audit_pre_human_static_gate(
        blueprint_rows=blueprint,
        candidate_rows=candidates,
        materialization_report=materialization,
    )
    assert report["status"] == "PASS", report["failures"]
    assert report["review_packet_allowed"] is True
