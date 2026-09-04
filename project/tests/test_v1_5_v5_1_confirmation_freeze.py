from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import pytest

from metacom_pm.io import iter_jsonl, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def requires_artifact(path: Path):
    return pytest.mark.skipif(
        not path.is_file(),
        reason=f"artifact replay input is not included in a clean checkout: {path}",
    )


def _rows(path: Path) -> list[dict]:
    return [dict(row) for row in iter_jsonl(path)]


@requires_artifact(
    ROOT
    / "outputs/pm_v1_5_v5_1_confirmation_surface_v1/data_integrity_report.json"
)
def test_confirmation_surface_is_outcome_blind_and_disjoint() -> None:
    out = ROOT / "outputs/pm_v1_5_v5_1_confirmation_surface_v1"
    report = json.loads((out / "data_integrity_report.json").read_text())
    features = _rows(out / "confirmation_feature_rows_private.jsonl")
    assert report["status"] == "PASS_READY_FOR_FROZEN_CONFIRMATION_CALL_PLAN"
    assert all(report["checks"].values())
    assert report["api_calls"] == 0
    assert report["human_or_generated_outcomes_read"] == 0
    assert report["sealed_or_external_read"] is False
    assert len(features) == 128
    assert len({row["state_id"] for row in features}) == 128
    assert len({row["decision_surface_sha256"] for row in features}) == 128
    assert Counter(row["component"] for row in features) == Counter(
        {"MP": 32, "MS": 32, "ME": 32, "RS": 32}
    )
    groups = Counter(row["semantic_group_id_private_analysis_only"] for row in features)
    assert len(groups) == 64
    assert set(groups.values()) == {2}


@requires_artifact(
    ROOT / "outputs/pm_v1_5_v5_1_confirmation_plan_v1/freeze_manifest.json"
)
def test_confirmation_call_and_policy_plan_is_complete_and_frozen() -> None:
    out = ROOT / "outputs/pm_v1_5_v5_1_confirmation_plan_v1"
    manifest = json.loads((out / "freeze_manifest.json").read_text())
    calls = _rows(out / "call_plan_private.jsonl")
    bindings = _rows(out / "state_policy_bindings_private.jsonl")
    assert manifest["status"] == "READY_FOR_SINGLE_V5_1_CONFIRMATION_EXECUTION"
    assert manifest["planned_calls"] == 512
    assert manifest["states"] == 128
    assert manifest["counterfactual_groups"] == 64
    assert all(manifest["checks"].values())
    assert manifest["api_calls"] == 0
    assert manifest["human_or_generated_outcomes_read"] == 0
    assert manifest["sealed_or_external_read"] is False
    assert sha256_file(out / "call_plan_private.jsonl") == manifest["call_plan_sha256"]
    assert (
        sha256_file(out / "state_policy_bindings_private.jsonl")
        == manifest["state_policy_bindings_sha256"]
    )

    assert len(calls) == 512
    assert len({row["call_id"] for row in calls}) == 512
    paired: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in calls:
        paired[(row["state_id"], row["seed_hex"])].append(row)
    assert len(paired) == 256
    assert all(Counter(item["arm"] for item in pair) == Counter({"ON": 1, "OFF": 1}) for pair in paired.values())
    assert all(len({item["seed"] for item in pair}) == 1 for pair in paired.values())
    assert all(
        "No external resource was requested" in row["messages"][0]["content"]
        for row in calls
        if row["arm"] == "OFF"
    )
    assert all(
        "Resource 1" in row["messages"][0]["content"]
        for row in calls
        if row["arm"] == "ON"
    )

    assert len(bindings) == 128
    learned_on = sum(row["policy_decisions"]["learned_pm"] for row in bindings)
    assert learned_on == 66
    assert 0.10 <= learned_on / len(bindings) <= 0.90
    assert all(
        set(row["policy_decisions"])
        == {"always_off", "component_fixed_high", "transparent_rule", "learned_pm"}
        for row in bindings
    )
    assert all(row["confirmation_response_or_label_read"] is False for row in bindings)
