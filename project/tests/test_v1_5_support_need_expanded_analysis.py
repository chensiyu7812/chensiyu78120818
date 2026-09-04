from __future__ import annotations

import json
from pathlib import Path

import pytest

from metacom_pm.io import canonical_json, read_json, sha256_text
from metacom_pm.v1_5_support_need_expanded_analysis import (
    _validate_report,
)


def test_expanded_analysis_rejects_invalid_report_hash(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"status": "bad", "report_sha256": "0" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="content hash"):
        _validate_report(
            json.loads(path.read_text(encoding="utf-8")),
            label="bad",
        )


def test_report_hash_fixture_is_valid():
    core = {"status": "ok"}
    report = {
        **core,
        "report_sha256": sha256_text(canonical_json(core)),
    }
    _validate_report(report, label="fixture")


def test_expanded_bakeoff_binding_hash_is_valid():
    root = Path(__file__).resolve().parents[1]
    binding = read_json(
        root
        / "data/pm_v1_5_contracts/"
        "support_need_factorized_bakeoff_expanded_v1.json"
    )
    digest = binding.pop("binding_sha256")
    assert digest == sha256_text(canonical_json(binding))
