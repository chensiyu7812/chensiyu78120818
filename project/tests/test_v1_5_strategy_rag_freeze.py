from __future__ import annotations

import hashlib
import json
from pathlib import Path

from metacom_pm.contracts import StrategyCard


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT / "outputs/pm_v1_5_strategy_rag_frozen_v1/freeze_report.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_qualified_strategy_rag_freeze_binds_runtime_and_bank() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "FROZEN_AND_QUALIFIED_FOR_G3_CLEAN_TREATMENT"
    assert report["card_count"] == 6
    assert report["top_k"] == 1
    assert report["strategy_min_score"] == 0.05
    assert report["bge_enabled"] is False
    assert report["formal_response_quality_benefit_proven"] is False

    manifest_path = ROOT / report["manifest_relative_path"]
    config_path = ROOT / report["config_relative_path"]
    bank_path = ROOT / report["bank_relative_path"]
    assert _sha(manifest_path) == report["manifest_sha256"]
    assert _sha(config_path) == report["config_sha256"]
    assert _sha(bank_path) == report["bank_sha256"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["retrieval"]["top_k"] == 1
    assert manifest["retrieval"]["ranker"] == "term_frequency_cosine"
    assert manifest["prompt_compiler"]["source_example_text_injected"] is False
    cards = [
        StrategyCard.model_validate(json.loads(line))
        for line in bank_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(cards) == 6
