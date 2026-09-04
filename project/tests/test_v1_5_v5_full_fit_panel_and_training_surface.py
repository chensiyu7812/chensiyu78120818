from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from metacom_pm.v1_5_final_candidate_contract import FINAL_FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "outputs/pm_v1_5_v5_single_full_fit_outcome_review_v1"
FEATURES = ROOT / "outputs/pm_v1_5_v5_fit_training_surface_v1"


def requires_artifact(path: Path):
    return pytest.mark.skipif(
        not path.is_file(),
        reason=f"artifact replay input is not included in a clean checkout: {path}",
    )


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _load_aggregator():
    path = ROOT / "scripts/v1_5/26a_aggregate_and_train_v5_fit_v1_5.py"
    spec = importlib.util.spec_from_file_location("v5_fit_aggregator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@requires_artifact(PANEL / "manifest.json")
def test_v5_full_fit_panel_is_blind_balanced_and_group_aware() -> None:
    manifest = json.loads((PANEL / "manifest.json").read_text())
    quality = _rows(PANEL / "primary_quality_packet.jsonl")
    risk = _rows(PANEL / "primary_grounding_risk_packet.jsonl")
    overlap_quality = _rows(PANEL / "independent_overlap_quality_packet.jsonl")
    overlap_risk = _rows(
        PANEL / "independent_overlap_grounding_risk_packet.jsonl"
    )
    key = _rows(PANEL / "private_blind_key.jsonl")
    assert (len(quality), len(risk), len(key)) == (512, 512, 512)
    assert (len(overlap_quality), len(overlap_risk)) == (104, 104)
    assert manifest["independent_counterfactual_groups"] == 128
    assert manifest["independent_overlap_semantic_groups"] == 52
    assert {tuple(sorted(row)) for row in quality} == {
        tuple(
            sorted(
                (
                    "protocol",
                    "blind_item_id",
                    "visible_conversation",
                    "response_a",
                    "response_b",
                )
            )
        )
    }
    assert {tuple(sorted(row)) for row in risk} == {
        tuple(
            sorted(
                (
                    "protocol",
                    "risk_item_id",
                    "visible_conversation",
                    "authorized_evidence_and_instruction",
                    "candidate_response",
                )
            )
        )
    }
    overlap_keys = [row for row in key if row["in_independent_overlap"]]
    assert len({row["state_id"] for row in overlap_keys}) == 52
    assert len({row["semantic_group_id"] for row in overlap_keys}) == 52
    for component in ("MP", "MS", "ME", "RS"):
        subset = [row for row in key if row["component"] == component]
        assert sum(row["a_arm"] == "ON" for row in subset) == 64
        assert sum(row["a_arm"] == "OFF" for row in subset) == 64
        assert len({row["state_id"] for row in subset}) == 64
        assert len({row["semantic_group_id"] for row in subset}) == 32
        assert len(
            {
                row["semantic_group_id"]
                for row in subset
                if row["in_independent_overlap"]
            }
        ) == 13
    for name, expected in manifest["outputs"].items():
        actual = hashlib.sha256((PANEL / name).read_bytes()).hexdigest()
        assert actual == expected


@requires_artifact(FEATURES / "freeze_report.json")
def test_v5_fit_feature_rows_are_pre_action_and_frozen() -> None:
    report = json.loads((FEATURES / "freeze_report.json").read_text())
    rows = _rows(FEATURES / "fit_feature_rows_private.jsonl")
    assert report["status"] == "FROZEN_READY_FOR_SINGLE_HUMAN_OUTCOME_PANEL"
    assert len(rows) == 256
    assert len({row["semantic_group_id_private_cv_only"] for row in rows}) == 128
    for component, names in FINAL_FEATURE_NAMES.items():
        subset = [row for row in rows if row["component"] == component]
        assert len(subset) == 64
        assert all(set(row["model_features"]) == set(names) for row in subset)
        assert all(row["human_fit_outcome_labels_read"] == 0 for row in subset)
        assert all(
            row["generated_response_or_execution_outcome_read"] is False
            for row in subset
        )


def test_v5_state_label_uses_itt_quality_risk_and_cost_rule() -> None:
    module = _load_aggregator()
    key = [
        {
            "state_id": "s1",
            "component": "ME",
            "semantic_group_id": "g1",
            "semantic_family": "f1",
            "blind_item_id": "q1",
            "risk_item_id": "r1",
            "a_arm": "ON",
            "b_arm": "OFF",
            "on_fallback_used": False,
        },
        {
            "state_id": "s1",
            "component": "ME",
            "semantic_group_id": "g1",
            "semantic_family": "f1",
            "blind_item_id": "q2",
            "risk_item_id": "r2",
            "a_arm": "OFF",
            "b_arm": "ON",
            "on_fallback_used": True,
        },
    ]
    quality = {
        "q1": {"quality_preference": "A"},
        "q2": {"quality_preference": "tie"},
    }
    risk = {
        "r1": {
            "any_material_risk": "no",
            "resource_functionally_contributed": "yes",
        },
        "r2": {
            "any_material_risk": "no",
            "resource_functionally_contributed": "no",
        },
    }
    label = module._derive_state_labels(
        private_key=key, quality=quality, risk=risk
    )[0]
    assert label["hard_worth_opening"] == 1
    assert label["seed_on_fallback_used"] == [False, True]
    risk["r2"]["any_material_risk"] = "yes"
    vetoed = module._derive_state_labels(
        private_key=key, quality=quality, risk=risk
    )[0]
    assert vetoed["hard_worth_opening"] == 0


@requires_artifact(PANEL / "private_blind_key.jsonl")
def test_v5_full_aggregation_entry_runs_without_post_outcome_repair(
    tmp_path: Path,
) -> None:
    key = _rows(PANEL / "private_blind_key.jsonl")
    key_by_quality = {row["blind_item_id"]: row for row in key}
    key_by_risk = {row["risk_item_id"]: row for row in key}
    positive_state = {
        row["state_id"]: int(hashlib.sha256(row["state_id"].encode()).hexdigest(), 16)
        % 2
        == 0
        for row in key
    }

    def quality_rows(packet: Path, annotator: str) -> list[dict]:
        result = []
        for item in _rows(packet):
            private = key_by_quality[item["blind_item_id"]]
            if positive_state[private["state_id"]]:
                label = "A" if private["a_arm"] == "ON" else "B"
                criterion = "immediate_helpfulness"
                notes = "Synthetic test annotation; not a research label."
            else:
                label = "tie"
                criterion = "materially_equivalent"
                notes = ""
            result.append(
                {
                    "protocol": "pm-v1.5-v5-fit-outcome-quality-blind-v1",
                    "blind_item_id": item["blind_item_id"],
                    "quality_preference": label,
                    "decisive_criterion": criterion,
                    "quality_notes": notes,
                    "annotator_id": annotator,
                }
            )
        return result

    def risk_rows(packet: Path, annotator: str) -> list[dict]:
        return [
            {
                "protocol": "pm-v1.5-v5-fit-outcome-grounding-risk-v1",
                "risk_item_id": item["risk_item_id"],
                "any_material_risk": "no",
                "selected_categories": [],
                "evidence_by_category": {},
                "resource_functionally_contributed": "yes",
                "functional_evidence_excerpt": "Synthetic test evidence.",
                "review_notes": "",
                "annotator_id": annotator,
            }
            for item in _rows(packet)
        ]

    files = {
        "primary_quality.jsonl": quality_rows(
            PANEL / "primary_quality_packet.jsonl", "primary_test"
        ),
        "primary_risk.jsonl": risk_rows(
            PANEL / "primary_grounding_risk_packet.jsonl", "primary_risk_test"
        ),
        "overlap_quality.jsonl": quality_rows(
            PANEL / "independent_overlap_quality_packet.jsonl", "overlap_test"
        ),
        "overlap_risk.jsonl": risk_rows(
            PANEL / "independent_overlap_grounding_risk_packet.jsonl",
            "overlap_risk_test",
        ),
    }
    for name, rows in files.items():
        (tmp_path / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    out = tmp_path / "result"
    command = [
        str(ROOT.parent / ".venv-pm-v1-5/bin/python"),
        str(ROOT / "scripts/v1_5/26a_aggregate_and_train_v5_fit_v1_5.py"),
        "--primary-quality",
        str(tmp_path / "primary_quality.jsonl"),
        "--primary-risk",
        str(tmp_path / "primary_risk.jsonl"),
        "--overlap-quality",
        str(tmp_path / "overlap_quality.jsonl"),
        "--overlap-risk",
        str(tmp_path / "overlap_risk.jsonl"),
        "--out-dir",
        str(out),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={"PYTHONNOUSERSITE": "1", "PYTHONPATH": "src"},
        check=True,
        capture_output=True,
        text=True,
    )
    assert "pm-v1.5-v5-fit-outcome-aggregation-and-training-v1" in completed.stdout
    report = json.loads((out / "fit_report.json").read_text())
    assert set(report["heads"]) == {"MP", "MS", "ME", "RS"}
    assert report["post_label_method_change_allowed"] is False
