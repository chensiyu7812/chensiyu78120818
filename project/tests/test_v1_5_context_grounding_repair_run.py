from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from metacom_pm.api import Endpoint
from metacom_pm.io import iter_jsonl, read_json, sha256_file
from metacom_pm.pm_v2_data import load_bundles
from metacom_pm.v1_5_context_grounding_data_repair import (
    FieldOnlyRepairOutput,
    VISIBLE_SURFACE_REPAIR_TURN_INDICES,
    VisibleSurfaceRepairOutput,
)
from metacom_pm.v1_5_context_grounding_repair import (
    DEFAULT_CLASSIFICATION_PATH,
    load_context_grounding_defect_classification,
)
from metacom_pm.v1_5_context_grounding_repair_run import (
    REPAIR_PILOT_STATE_IDS,
    build_repair_run_plan,
    materialize_full_repair_overlays,
    select_repair_records,
    validate_repair_result,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLES_PATH = (
    ROOT
    / "data"
    / "pm_v1_5_formal_v8_18_duplicate_repair_candidate"
    / "pm_v2_bundles.jsonl"
)


def _require_real_bundles() -> None:
    if not BUNDLES_PATH.is_file():
        pytest.skip(
            "requires the private local V8.18 development-data artifact; "
            "the artifact is intentionally excluded from Git"
        )


@pytest.fixture(scope="module")
def records():
    return load_context_grounding_defect_classification(DEFAULT_CLASSIFICATION_PATH)


def _endpoint() -> Endpoint:
    return Endpoint(
        base_url="https://api.openai.com",
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        family="openai_gpt4o",
    )


def _plan(records, scope):
    _require_real_bundles()
    return build_repair_run_plan(
        records=records,
        scope=scope,
        endpoint=_endpoint(),
        original_bundles_path=BUNDLES_PATH,
        classification_path=DEFAULT_CLASSIFICATION_PATH,
        experiment_config_path=ROOT / "configs" / "experiment.yaml",
        pm_v1_5_config_path=ROOT / "configs" / "pm_v1_5.yaml",
        input_token_safety_factor=1.5,
        input_usd_per_mtok=0.15,
        output_usd_per_mtok=0.60,
    )


def test_pilot_selection_is_frozen_six_and_full_is_exact_25(records) -> None:
    pilot = select_repair_records(records, scope="pilot")
    full = select_repair_records(records, scope="full")
    assert [record.state_id for record in pilot] == list(REPAIR_PILOT_STATE_IDS)
    assert len({record.state_id for record in pilot}) == 6
    assert len({record.state_id for record in full}) == 25
    assert {record.state_id for record in pilot} < {record.state_id for record in full}


def test_repair_plan_is_deterministic_and_binds_inputs(records) -> None:
    first, rows, messages = _plan(records, "pilot")
    second, second_rows, second_messages = _plan(records, "pilot")
    assert first == second
    assert rows == second_rows
    assert messages == second_messages
    assert first["minimum_logical_calls"] == 6
    assert first["maximum_physical_api_attempts"] == 18
    assert len({row["physical_call_key"] for row in rows}) == 6
    assert first["cost_estimate_sha256"]
    contract = first["contract"]
    assert contract["original_bundles_path"].startswith("data/")
    assert contract["classification_path"].startswith("data/")
    assert contract["shared_code_manifest_sha256"]
    assert "repair_runner" in contract["shared_code_manifest"]
    summary_row = next(
        row for row in rows if row["state_id"] == "state_1d327ed7b58f3b13298b968c"
    )
    summary_prompt = " ".join(
        message["content"] for message in messages[summary_row["physical_call_key"]]
    )
    assert "NON-EMPTY session_summary" in summary_prompt


def test_full_plan_cannot_be_confused_with_pilot(records) -> None:
    pilot, pilot_rows, _ = _plan(records, "pilot")
    full, full_rows, _ = _plan(records, "full")
    assert len(full_rows) == 25
    assert full["contract_sha256"] != pilot["contract_sha256"]
    assert full["call_plan_sha256"] != pilot["call_plan_sha256"]
    assert full["cost_estimate_sha256"] != pilot["cost_estimate_sha256"]


def test_field_only_validation_discards_model_echo_for_frozen_summary(records) -> None:
    record = next(r for r in records if r.state_id == "state_062b3dd61638973f95199d76")
    result = validate_repair_result(
        record,
        FieldOnlyRepairOutput(
            authorized_user_context="The user reports difficulty sleeping at night.",
            session_summary="A model rewrite that must never be used.",
        ),
    )
    assert result["session_summary"] is None
    assert result["recent_dialogue_patch"] == {}


def test_visible_validation_rejects_wrong_turn_count(records) -> None:
    record = next(r for r in records if r.state_id == "state_d0eee004b44a562c57a0e6c6")
    with pytest.raises(RuntimeError, match="frozen contract requires"):
        validate_repair_result(
            record,
            VisibleSurfaceRepairOutput(
                repaired_turn_contents=["Only one repaired conversational turn."],
                authorized_user_context="The user is grieving their grandfather.",
                session_summary="The user is processing grief after a family loss.",
            ),
        )


def test_full_overlay_materializer_rejects_pilot_results(records) -> None:
    _require_real_bundles()
    bundles = load_bundles(str(BUNDLES_PATH))
    pilot_results = {}
    for record in select_repair_records(records, scope="pilot"):
        pilot_results[record.state_id] = {
            "state_id": record.state_id,
            "repair_mode": record.repair_mode,
            "authorized_user_context": "A substantive repaired user context.",
            "session_summary": None,
            "recent_dialogue_patch": {},
        }
    with pytest.raises(RuntimeError, match="exactly all 25"):
        materialize_full_repair_overlays(
            records=records,
            bundles=bundles,
            classification_sha256=sha256_file(DEFAULT_CLASSIFICATION_PATH),
            validated_results=pilot_results,
        )


def test_full_overlay_materializer_emits_exact_25_hash_bound_rows(records) -> None:
    _require_real_bundles()
    bundles = load_bundles(str(BUNDLES_PATH))
    validated = {}
    for record in select_repair_records(records, scope="full"):
        patch = {
            index: f"Substantive repaired dialogue turn at index {index}."
            for index in VISIBLE_SURFACE_REPAIR_TURN_INDICES.get(record.state_id, ())
        }
        validated[record.state_id] = {
            "state_id": record.state_id,
            "repair_mode": record.repair_mode,
            "authorized_user_context": "A substantive repaired user context.",
            "session_summary": (
                "A substantive repaired session summary."
                if record.repair_mode == "VISIBLE_SURFACE_REPAIR"
                or record.state_id
                in {
                    "state_1d327ed7b58f3b13298b968c",
                    "state_df5f52854a6eaae0e0a229bb",
                }
                else None
            ),
            "recent_dialogue_patch": patch,
        }
    overlays = materialize_full_repair_overlays(
        records=records,
        bundles=bundles,
        classification_sha256=sha256_file(DEFAULT_CLASSIFICATION_PATH),
        validated_results=validated,
    )
    assert len(overlays) == 25
    assert {overlay.state_id for overlay in overlays} == set(validated)
    assert {
        overlay.classification_sha256 for overlay in overlays
    } == {sha256_file(DEFAULT_CLASSIFICATION_PATH)}


def test_cli_pilot_dry_run_is_zero_api_and_writes_exact_plan(tmp_path) -> None:
    _require_real_bundles()
    out_dir = tmp_path / "pilot"
    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "v1_5" / "20e_run_context_grounding_repair_v1_5.py"),
            "--dry-run",
            "--scope",
            "pilot",
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    estimate = read_json(out_dir / "cost_estimate.json")
    rows = list(iter_jsonl(out_dir / "call_plan.jsonl"))
    assert estimate["budget_gate"]["status"] == "PASS"
    assert estimate["minimum_logical_calls"] == len(rows) == 6
    assert not (out_dir / "physical_attempt_ledger.jsonl").exists()
    assert not (out_dir / "repair_overlays.jsonl").exists()

    # Even with the exact dry-run hash, an absent central per-stage approval
    # must stop execution before an API client/ledger is created.
    attempted = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "v1_5" / "20e_run_context_grounding_repair_v1_5.py"),
            "--run",
            "--scope",
            "pilot",
            "--out-dir",
            str(out_dir),
            "--accept-cost-estimate-sha256",
            str(estimate["cost_estimate_sha256"]),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert attempted.returncode != 0
    assert "paid-run approval" in attempted.stderr
    assert not (out_dir / "physical_attempt_ledger.jsonl").exists()
