from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from metacom_pm.config import load_config
from metacom_pm.io import read_json, write_json
from metacom_pm.pm_v2_contracts import ResponseDimensions, RiskDimensions
import metacom_pm.pm_v2_pilot_human_spot_check as spot


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_spot_check_is_exactly_18_blinded_no_overall_items() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pm_v2.yaml")
    settings = spot.pilot_human_spot_check_settings(config)
    assert settings["train_states_per_regime"] == 1
    assert settings["actions_per_regime"] == 2
    assert settings["minimum_annotators"] >= 2
    assert len(settings["diagnostic_actions_by_regime"]) == 9
    assert all(
        len(actions) == 2
        for actions in settings["diagnostic_actions_by_regime"].values()
    )
    assert "overall" not in spot.PACKET_FIELDS
    assert "support_rating" not in spot.PACKET_FIELDS
    assert len(spot.RESPONSE_FIELDS) == 6
    assert len(spot.RISK_FIELDS) == 7
    manual = spot.pilot_human_spot_check_manual(settings)
    assert (
        "Use the displayed selected context only when scoring the seven risk fields"
        in manual
    )


def test_selection_uses_one_train_state_and_two_frozen_actions_per_regime() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pm_v2.yaml")
    settings = spot.pilot_human_spot_check_settings(config)
    by_regime = {}
    outcomes = {}
    for regime, actions in settings["diagnostic_actions_by_regime"].items():
        by_regime[regime] = [
            {
                "state_id": f"state_{regime}_b",
                "card_id": f"card_{regime}_b",
                "split": "train",
            },
            {
                "state_id": f"state_{regime}_a",
                "card_id": f"card_{regime}_a",
                "split": "train",
            },
        ]
        for row in by_regime[regime]:
            for action in actions:
                outcomes[(row["state_id"], action)] = SimpleNamespace(
                    request_hash=f"request-{row['state_id']}-{action}"
                )
    bundle = {
        "settings": settings,
        "by_regime": by_regime,
        "outcome_map": outcomes,
    }
    first = spot.select_pilot_human_spot_check_items(source_bundle=bundle)
    second = spot.select_pilot_human_spot_check_items(source_bundle=bundle)
    assert first == second
    assert len(first) == 18
    assert {row["split"] for row in first} == {"train"}
    for regime, actions in settings["diagnostic_actions_by_regime"].items():
        rows = [row for row in first if row["regime"] == regime]
        assert len({row["state_id"] for row in rows}) == 1
        assert {row["action_id"] for row in rows} == set(actions)


def _write_packet(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(spot.PACKET_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _analysis_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    biased_second_family: bool,
) -> dict:
    config = load_config(PROJECT_ROOT / "configs" / "pm_v2.yaml")
    settings = spot.pilot_human_spot_check_settings(config)
    paths = {
        name: tmp_path / filename
        for name, filename in {
            "config": "pm_v2.yaml",
            "states": "states.jsonl",
            "contexts": "contexts.jsonl",
            "pilot_plan": "pilot_plan.json",
            "outcomes": "outcomes.jsonl",
            "labels": "labels.jsonl",
            "raw": "raw.jsonl",
            "summary": "summary.json",
            "judge_attestation": "judge_attestation.json",
            "manual": "manual.md",
            "human_plan": "human_plan.json",
        }.items()
    }
    for path in paths.values():
        path.write_text("{}\n", encoding="utf-8")

    selections = []
    packet_rows = []
    raw_map = {}
    label_map = {}
    response_three = {field: 3.0 for field in spot.RESPONSE_FIELDS}
    response_middle = {field: 3.5 for field in spot.RESPONSE_FIELDS}
    risk_one = {field: 1.0 for field in spot.RISK_FIELDS}
    for index in range(18):
        item_id = f"phs_{index:02d}"
        state_id = f"state_{index:02d}"
        action_id = "M0+R0" if index % 2 == 0 else "M0+RS"
        selections.append(
            {
                "item_id": item_id,
                "state_id": state_id,
                "action_id": action_id,
            }
        )
        packet_rows.append(
            {
                "item_id": item_id,
                "current_user_text": f"turn {index}",
                "recent_dialogue_json": "[]",
                "current_session_summary": "summary",
                "authorized_user_context": "authorized",
                "selected_context_json": "[]",
                "candidate_response": f"response {index}",
                **{field: "" for field in (*spot.RESPONSE_FIELDS, *spot.RISK_FIELDS)},
                "annotator_id": "",
                "notes": "",
            }
        )
        raw_map[(state_id, action_id, "family_a")] = {
            "response": dict(response_three),
            "risk": dict(risk_one),
        }
        raw_map[(state_id, action_id, "family_b")] = {
            "response": {
                field: (4.0 if biased_second_family else 3.0)
                for field in spot.RESPONSE_FIELDS
            },
            "risk": dict(risk_one),
        }
        label_map[(state_id, action_id)] = SimpleNamespace(
            response=ResponseDimensions(
                **(response_middle if biased_second_family else response_three)
            ),
            risk=RiskDimensions(**risk_one),
        )
    packet_path = tmp_path / "packet.csv"
    _write_packet(packet_path, packet_rows)
    completed = []
    for annotator in ("annotator_a", "annotator_b"):
        rows = []
        for source in packet_rows:
            rows.append(
                {
                    **source,
                    **{field: "3" for field in spot.RESPONSE_FIELDS},
                    **{field: "1" for field in spot.RISK_FIELDS},
                    "annotator_id": annotator,
                }
            )
        path = tmp_path / f"{annotator}.csv"
        _write_packet(path, rows)
        completed.append(path)

    plan = {
        "plan_sha256": "h" * 64,
        "selected_items": selections,
    }
    packet_by_item = {row["item_id"]: row for row in packet_rows}
    evaluator_index = SimpleNamespace(map_sha256="e" * 64)
    source_bundle = {
        "settings": settings,
        "plan": {"pilot_plan_sha256": "p" * 64},
        "judge_families": ["family_a", "family_b"],
        "raw_map": raw_map,
        "label_map": label_map,
    }
    monkeypatch.setattr(
        spot, "_validate_pilot_source_bundle", lambda **kwargs: source_bundle
    )
    monkeypatch.setattr(
        spot,
        "_validate_human_plan",
        lambda **kwargs: (plan, packet_by_item),
    )
    return {
        "config": config,
        "settings": settings,
        "paths": paths,
        "packet": packet_path,
        "completed": completed,
        "plan": plan,
        "packet_by_item": packet_by_item,
        "source_bundle": source_bundle,
        "evaluator_index": evaluator_index,
    }


def _analyze(fixture: dict, tmp_path: Path) -> tuple[dict, Path, Path]:
    report_path = tmp_path / "report.json"
    attestation_path = tmp_path / "artifact_attestation.json"
    paths = fixture["paths"]
    report = spot.analyze_pilot_human_spot_check(
        completed_paths=fixture["completed"],
        config=fixture["config"],
        config_path=paths["config"],
        states_path=paths["states"],
        evaluator_contexts_path=paths["contexts"],
        evaluator_contexts=fixture["evaluator_index"],
        pilot_plan_path=paths["pilot_plan"],
        outcomes_path=paths["outcomes"],
        labels_path=paths["labels"],
        raw_results_path=paths["raw"],
        judge_compatibility_summary_path=paths["summary"],
        judge_compatibility_attestation_path=paths["judge_attestation"],
        packet_path=fixture["packet"],
        manual_path=paths["manual"],
        plan_path=paths["human_plan"],
        report_path=report_path,
        attestation_path=attestation_path,
    )
    return report, report_path, attestation_path


def test_each_raw_judge_family_is_gated_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _analysis_fixture(
        tmp_path, monkeypatch, biased_second_family=True
    )
    report, _, _ = _analyze(fixture, tmp_path)
    assert report["status"] == "FAIL"
    checks = report["gate"]["checks"][
        "llm_vs_human_by_source_and_dimension"
    ]
    assert checks["family_a"]["response.emotional_support.mae"] is True
    assert checks["cross_family_median"][
        "response.emotional_support.mae"
    ] is True
    assert checks["family_b"]["response.emotional_support.mae"] is False


def test_pass_attestation_helper_is_tamper_evident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _analysis_fixture(
        tmp_path, monkeypatch, biased_second_family=False
    )
    report, report_path, attestation_path = _analyze(fixture, tmp_path)
    assert report["status"] == "PASS"
    monkeypatch.setattr(spot, "load_states", lambda path: [])
    monkeypatch.setattr(
        spot,
        "load_evaluator_context_index",
        lambda *args, **kwargs: fixture["evaluator_index"],
    )
    paths = fixture["paths"]
    result = spot.require_pilot_human_spot_check_pass(
        report_path=report_path,
        attestation_path=attestation_path,
        config=fixture["config"],
        config_path=paths["config"],
        states_path=paths["states"],
        evaluator_contexts_path=paths["contexts"],
        pilot_plan_path=paths["pilot_plan"],
        outcomes_path=paths["outcomes"],
        labels_path=paths["labels"],
        raw_results_path=paths["raw"],
        judge_compatibility_summary_path=paths["summary"],
        judge_compatibility_attestation_path=paths["judge_attestation"],
    )
    assert result["status"] == "PASS"
    changed = read_json(report_path)
    changed["status"] = "FAIL"
    write_json(report_path, changed)
    with pytest.raises(RuntimeError, match="outputs hash mismatch"):
        spot.require_pilot_human_spot_check_pass(
            report_path=report_path,
            attestation_path=attestation_path,
            config=fixture["config"],
            config_path=paths["config"],
            states_path=paths["states"],
            evaluator_contexts_path=paths["contexts"],
            pilot_plan_path=paths["pilot_plan"],
            outcomes_path=paths["outcomes"],
            labels_path=paths["labels"],
            raw_results_path=paths["raw"],
            judge_compatibility_summary_path=paths["summary"],
            judge_compatibility_attestation_path=paths["judge_attestation"],
        )


def test_human_scores_reject_float_strings() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        spot._integer_score("3.0", item_id="item", field="emotional_support")
