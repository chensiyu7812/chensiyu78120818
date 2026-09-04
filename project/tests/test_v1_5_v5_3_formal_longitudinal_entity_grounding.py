from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
USERS = ROOT / "data/pm_v1_5_v5_3_formal_longitudinal_catalog_intake_v1/canonical_users"
CONTRACT = ROOT / "data/pm_v1_5_contracts/v5_3_complete_training_data_generation_v1.json"
VALIDATOR = ROOT / "scripts/v1_5/82l_validate_formal_longitudinal_user_v1_5.py"


def _module():
    spec = importlib.util.spec_from_file_location("formal_longitudinal_validator_test", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tracked_formal_intake_has_grounded_relationship_timelines() -> None:
    validator = _module()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    paths = sorted(USERS.glob("*.json"))
    assert len(paths) == 11

    for path in paths:
        user = json.loads(path.read_text(encoding="utf-8"))
        result = validator._validate_user(user, contract)
        assert result["hard_issue_count"] == 0, (user["user_id"], result["issues"])
        assert result["me_compiler"]["typed_exact_span_core_valid"] == 8
        for relationship in user["relationships"]:
            assert set(relationship) == {
                "entity_id",
                "name",
                "relationship",
                "valid_from_session",
                "valid_until_session",
            }


def test_repaired_formal_intake_batch_audit_is_frozen_pass() -> None:
    report = json.loads(
        (USERS.parent / "batch_audit_report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "PASS"
    assert report["source_fidelity"]["protected_surface_drift_count"] == 0
    assert report["cross_user_overlap"]["cross_user_exact_groups"] == 0
    assert report["cross_user_overlap"]["cross_user_normalized_8gram_groups"] == 0
    assert report["external_overlap"]["exact_collision_count"] == 0
    assert report["external_overlap"]["normalized_ngram_collision_count"] == 0
