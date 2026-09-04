from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def requires_artifact(path: Path):
    return pytest.mark.skipif(
        not path.is_file(),
        reason=f"artifact replay input is not included in a clean checkout: {path}",
    )


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_v5_2_confirmation_identity_is_content_disjoint_and_single_use() -> None:
    report = _json(
        ROOT
        / "outputs/pm_v1_5_v5_2_confirmation_freeze_v1/freeze_and_data_quality_report.json"
    )
    assert report["status"] == "PASS_V5_2_CONTENT_DISJOINT_CONFIRMATION_IDENTITY_FROZEN"
    assert all(report["checks"].values())
    assert report["api_calls"] == 0
    assert report["human_labels_read"] == 0
    assert report["external_outcomes_read"] == 0
    assert all(
        row["all_states_exposed"]
        for row in report["historical_exposure_audit"].values()
    )


@requires_artifact(
    ROOT
    / "data/pm_v1_5_v5_2_confirmation_v1/private/candidate_rows_private.jsonl"
)
def test_v5_2_confirmation_candidate_and_group_shape() -> None:
    data_dir = ROOT / "data/pm_v1_5_v5_2_confirmation_v1/private"
    candidates = _jsonl(data_dir / "candidate_rows_private.jsonl")
    blueprint = _jsonl(data_dir / "construction_blueprint.jsonl")
    assert len(candidates) == 128
    assert Counter(row["target_component_private_not_model_input"] for row in candidates) == Counter(
        {"MP": 32, "MS": 32, "ME": 32, "RS": 32}
    )
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in blueprint:
        groups[row["counterfactual_group_id"]].append(row)
    assert len(groups) == 64
    assert all(len(rows) == 2 for rows in groups.values())
    assert all(
        {row["private_benefit_enrichment"] for row in rows}
        == {"HIGH", "LOW_OR_NEUTRAL"}
        for rows in groups.values()
    )
    assert all(
        row["exact_rank1_candidate"]["candidate_present"]
        and row["exact_rank1_candidate"]["selected_rank"] == 1
        and not row["response_or_outcome_read"]
        for row in candidates
    )


@requires_artifact(
    ROOT / "outputs/pm_v1_5_v5_2_confirmation_plan_v1/freeze_manifest.json"
)
def test_v5_2_confirmation_plan_is_paired_and_policy_non_degenerate() -> None:
    plan_dir = ROOT / "outputs/pm_v1_5_v5_2_confirmation_plan_v1"
    manifest = _json(plan_dir / "freeze_manifest.json")
    calls = _jsonl(plan_dir / "call_plan_private.jsonl")
    bindings = _jsonl(plan_dir / "state_policy_bindings_private.jsonl")
    assert manifest["status"] == "READY_FOR_SINGLE_V5_2_CONTENT_DISJOINT_CONFIRMATION_EXECUTION"
    assert len(calls) == 512
    assert len(bindings) == 128
    pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in calls:
        pairs[(row["state_id"], row["seed_hex"])].add(row["arm"])
        assert row["selection_or_prompt_uses_confirmation_human_external_outcome"] is False
    assert len(pairs) == 256
    assert all(arms == {"ON", "OFF"} for arms in pairs.values())
    learned_on = sum(row["policy_decisions"]["learned_pm"] for row in bindings)
    assert learned_on == 65
    assert manifest["learned_requested_on_fraction"] == 65 / 128
    assert 0.10 <= learned_on / 128 <= 0.90
