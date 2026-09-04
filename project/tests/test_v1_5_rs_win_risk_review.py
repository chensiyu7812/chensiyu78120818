from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v1_5/24ax_prepare_rs_win_risk_review_v1_5.py"


def _load():
    spec = importlib.util.spec_from_file_location("rs_win_risk_review", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_risk_review_has_four_applicable_atomic_categories() -> None:
    module = _load()
    assert set(module.RISK_CATEGORIES) == {
        "explicit_boundary_or_permission_violation",
        "unsupported_or_overstated_claim",
        "excessive_directiveness_or_burden",
        "domain_or_high_stakes_overreach",
    }


def test_public_risk_packet_has_only_candidate_response() -> None:
    packet = (
        ROOT
        / "outputs/pm_v1_5_rs_win_minimal_risk_review_v1_candidate"
        / "human_risk_packet.jsonl"
    )
    if not packet.exists():
        return
    for line in packet.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert set(row) == {
            "protocol",
            "review_item_id",
            "visible_dialogue",
            "candidate_response",
        }
        serialized = json.dumps(row)
        for hidden in (
            "R0",
            "selected_card",
            "strategy_family",
            "retrieval_score",
            "judge",
        ):
            assert hidden not in serialized
