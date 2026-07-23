from __future__ import annotations

from pathlib import Path

import pytest

from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.io import canonical_json, sha256_file, sha256_text, write_json, write_jsonl
from metacom_pm.v1_5_actual_corpus_qualification import (
    ACTUAL_CORPUS_QUALIFICATION_STAGE,
    ACTUAL_CORPUS_QUALIFIED_STATUS,
    FROZEN_CONTROL_MISS_ITEM_ID,
    FROZEN_UNRECOVERABLE_CALL,
    FROZEN_UNRECOVERABLE_CONTRACT_ERRORS,
    build_actual_corpus_posthoc_qualification,
    require_actual_corpus_posthoc_qualification,
)
from metacom_pm.v1_5_actual_corpus_review import (
    ACTUAL_CITATION_POLICY,
    ACTUAL_CORPUS_CONTROL_PROTOCOL,
    ACTUAL_CORPUS_REVIEW_PROTOCOL,
    ACTUAL_DETERMINISTIC_FIELDS,
    ACTUAL_PANEL_POLICY,
    ACTUAL_SEMANTIC_FIELDS,
)


def _fixture(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    for name in (
        "experiment_config",
        "states",
        "evaluator_contexts",
        "memory_backend",
        "runtime_states",
        "bundles",
        "data_report",
        "strategy_bank",
        "pm_config",
    ):
        path = tmp_path / f"{name}.jsonl"
        path.write_text('{"name":"' + name + '"}\n', encoding="utf-8")
        files[name] = path

    classification = tmp_path / "classification.jsonl"
    write_jsonl(
        classification,
        [
            {
                "state_id": f"state_{index:03d}",
                "classification": (
                    "DATA_DEFECT" if index < 25 else "INSTRUMENT_AMBIGUITY"
                ),
            }
            for index in range(91)
        ],
    )
    overlays = tmp_path / "overlays.jsonl"
    write_jsonl(overlays, [{"state_id": f"state_{index:03d}"} for index in range(25)])
    files.update(classification=classification, overlays=overlays)

    output_hashes = {
        "pm_v2_states.jsonl": sha256_file(files["states"]),
        "evaluator_contexts.jsonl": sha256_file(files["evaluator_contexts"]),
        "memory_backend.jsonl": sha256_file(files["memory_backend"]),
        "runtime_states.jsonl": sha256_file(files["runtime_states"]),
        "pm_v2_bundles.jsonl": sha256_file(files["bundles"]),
        "pm_v2_data_report.json": sha256_file(files["data_report"]),
    }
    recompile = tmp_path / "recompile.json"
    write_json(
        recompile,
        {
            "protocol": "pm-v1.5-context-grounding-repair-recompile-verification-v1",
            "status": "PASS",
            "total_states": 468,
            "repaired_state_count": 25,
            "untouched_state_count": 443,
            "unexpectedly_changed_untouched_states": {},
            "unexpectedly_unchanged_repaired_states": [],
            "classification_sha256": sha256_file(classification),
            "repair_overlays_sha256": sha256_file(overlays),
            "output_file_sha256s": output_hashes,
        },
    )
    files["recompile"] = recompile

    data_attestation = tmp_path / "data_attestation.json"
    create_artifact_attestation(
        data_attestation,
        stage="pm_v1_5_development_data",
        inputs={
            "classification": classification,
            "repair_overlays": overlays,
            "strategy_bank": files["strategy_bank"],
            "pm_v1_5_config": files["pm_config"],
        },
        outputs={
            "states": (files["states"], True),
            "evaluator_contexts": (files["evaluator_contexts"], True),
            "backend": (files["memory_backend"], True),
            "runtime_states": (files["runtime_states"], True),
            "bundles": (files["bundles"], True),
            "data_report": (files["data_report"], False),
        },
        parameters={"repaired_state_count": 25},
    )
    files["data_attestation"] = data_attestation

    call_plan = tmp_path / "call_plan.jsonl"
    rows = [{"physical_call_key": f"call_{index:04d}"} for index in range(1880)]
    write_jsonl(call_plan, rows)
    cost = tmp_path / "cost.json"
    write_json(
        cost,
        {
            "protocol": ACTUAL_CORPUS_REVIEW_PROTOCOL,
            "stage": "pm_v1_5_actual_corpus_semantic_review",
            "review_scope": "actual_468",
            "n_logical_calls": 1880,
            "call_plan_sha256": sha256_text(canonical_json(rows)),
            "budget_gate": {"status": "PASS"},
        },
    )
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text('{"event":"SUCCEEDED"}\n', encoding="utf-8")
    files.update(call_plan=call_plan, cost=cost, ledger=ledger)

    recovery = tmp_path / "recovery.json"
    write_json(
        recovery,
        {
            "protocol": "pm-v1.5-actual-468-offline-length-bound-recovery-v1",
            "status": "PARTIALLY_RECOVERED",
            "unrecoverable_calls": [{"call_key": "frozen", **FROZEN_UNRECOVERABLE_CALL}],
        },
    )
    aggregate = {
        "protocol": ACTUAL_CORPUS_REVIEW_PROTOCOL,
        "status": "FAIL",
        "n_real_cases": 468,
        "n_real_semantic_packets": 936,
        "n_controls": 4,
        "control_protocol": ACTUAL_CORPUS_CONTROL_PROTOCOL,
        "panel_policy": ACTUAL_PANEL_POLICY,
        "citation_policy": ACTUAL_CITATION_POLICY,
        "semantic_fields": list(ACTUAL_SEMANTIC_FIELDS),
        "real_case_rejections": [],
        "real_case_panel_passes": ["x"] * 619,
        "real_case_disagreements": ["x"] * 316,
        "control_misses": [
            {
                "item_id": FROZEN_CONTROL_MISS_ITEM_ID,
                "field": "context_grounding_match",
                "verdicts": {"a": "supported", "b": "supported"},
            }
        ],
        "control_contract_errors": sorted(FROZEN_UNRECOVERABLE_CONTRACT_ERRORS),
        "deterministic_gate": {
            "status": "PASS",
            "n_states": 468,
            "n_failures": 0,
            "failures": [],
            "field_pass_counts": {field: 468 for field in ACTUAL_DETERMINISTIC_FIELDS},
        },
        "citation_audit": {
            "policy": ACTUAL_CITATION_POLICY,
            "n_observations": 1880,
            "n_valid": 1493,
            "integrity_rate": 1493 / 1880,
        },
    }
    recovered = tmp_path / "recovered.json"
    write_json(
        recovered,
        {
            "protocol": "pm-v1.5-actual-468-offline-length-bound-recovery-v1",
            "status": "RECOVERED_GATE_DECISION_POST_RUN_AMENDMENT",
            "recovery_report_sha256": sha256_text(canonical_json(
                __import__("json").load(recovery.open(encoding="utf-8"))
            )),
            "logical_calls_planned": 1880,
            "logical_calls_recovered": 11,
            "logical_calls_unrecoverable": 1,
            "aggregated_gate": aggregate,
        },
    )
    incomplete = tmp_path / "incomplete.json"
    write_json(
        incomplete,
        {
            "protocol": ACTUAL_CORPUS_REVIEW_PROTOCOL,
            "status": "INCOMPLETE_NO_GATE_DECISION",
            "logical_calls_planned": 1880,
        },
    )
    files.update(recovery=recovery, recovered=recovered, incomplete=incomplete)
    return files


def _build(files: dict[str, Path]) -> dict:
    return build_actual_corpus_posthoc_qualification(
        recovered_gate_report_path=files["recovered"],
        recovery_report_path=files["recovery"],
        incomplete_gate_report_path=files["incomplete"],
        physical_attempt_ledger_path=files["ledger"],
        cost_estimate_path=files["cost"],
        call_plan_path=files["call_plan"],
        recompile_verification_path=files["recompile"],
        development_data_attestation_path=files["data_attestation"],
        classification_path=files["classification"],
        repair_overlays_path=files["overlays"],
        expected_states_path=files["states"],
        expected_evaluator_contexts_path=files["evaluator_contexts"],
        expected_backend_path=files["memory_backend"],
        expected_runtime_states_path=files["runtime_states"],
        expected_bundles_path=files["bundles"],
        expected_data_report_path=files["data_report"],
        expected_strategy_bank_path=files["strategy_bank"],
        expected_pm_config_path=files["pm_config"],
    )


def test_posthoc_qualification_requires_exact_frozen_residuals(tmp_path: Path) -> None:
    files = _fixture(tmp_path)
    report = _build(files)
    assert report["status"] == ACTUAL_CORPUS_QUALIFIED_STATUS
    assert report["original_gate_status"] == "FAIL"
    assert report["original_gate_pass_claimed"] is False

    value = __import__("json").load(files["recovered"].open(encoding="utf-8"))
    value["aggregated_gate"]["real_case_rejections"] = [{"state_id": "bad"}]
    write_json(files["recovered"], value)
    with pytest.raises(RuntimeError, match="unanimous real-case rejection"):
        _build(files)


def test_posthoc_qualification_rejects_control_or_repair_drift(tmp_path: Path) -> None:
    files = _fixture(tmp_path)
    value = __import__("json").load(files["recovered"].open(encoding="utf-8"))
    value["aggregated_gate"]["control_misses"][0]["item_id"] = "different"
    write_json(files["recovered"], value)
    with pytest.raises(RuntimeError, match="control residual"):
        _build(files)

    files = _fixture(tmp_path / "repair")
    value = __import__("json").load(files["recompile"].open(encoding="utf-8"))
    value["repaired_state_count"] = 24
    write_json(files["recompile"], value)
    with pytest.raises(RuntimeError, match="25-state repair"):
        _build(files)


def test_qualification_attestation_binds_current_corpus(tmp_path: Path) -> None:
    files = _fixture(tmp_path)
    report = _build(files)
    report_path = tmp_path / "qualification.json"
    write_json(report_path, report)
    attestation = tmp_path / "qualification_attestation.json"
    create_artifact_attestation(
        attestation,
        stage=ACTUAL_CORPUS_QUALIFICATION_STAGE,
        inputs={
            "experiment_config": files["experiment_config"],
            "states": files["states"],
            "evaluator_contexts": files["evaluator_contexts"],
            "memory_backend": files["memory_backend"],
            "strategy_bank": files["strategy_bank"],
            "pm_v1_5_config": files["pm_config"],
        },
        outputs={"qualification_report": (report_path, False)},
        parameters={"post_hoc": True},
    )
    result = require_actual_corpus_posthoc_qualification(
        report_path,
        attestation,
        expected_experiment_config_path=files["experiment_config"],
        expected_states_path=files["states"],
        expected_evaluator_contexts_path=files["evaluator_contexts"],
        expected_backend_path=files["memory_backend"],
        expected_strategy_bank_path=files["strategy_bank"],
        expected_pm_config_path=files["pm_config"],
    )
    assert result["status"] == ACTUAL_CORPUS_QUALIFIED_STATUS
    files["states"].write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="attestation|bind current"):
        require_actual_corpus_posthoc_qualification(
            report_path,
            attestation,
            expected_experiment_config_path=files["experiment_config"],
            expected_states_path=files["states"],
            expected_evaluator_contexts_path=files["evaluator_contexts"],
            expected_backend_path=files["memory_backend"],
            expected_strategy_bank_path=files["strategy_bank"],
            expected_pm_config_path=files["pm_config"],
        )
