from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from metacom_pm.api import CallResult, RetryableProviderError
from metacom_pm.io import canonical_json, read_json, write_json
from metacom_pm.pm_v2_generation_review_v8 import (
    RATING_FIELDS,
    VALIDATION_CASES_PER_REGIME,
    generate_v8_review_cases,
)
from metacom_pm.v1_5_automated_semantic_review import (
    V1_5_REVIEW_STRATEGY_CARD_IDS,
    build_control_manifest,
    build_positive_controls,
)
from metacom_pm.v1_5_semantic_review_diagnostic import (
    ATOMIC_DEFINITION_AND_EXHAUSTIVE_EVIDENCE,
    DETERMINISTIC_DIAGNOSTIC_FIELDS,
    DIAGNOSTIC_FIELDS,
    DIAGNOSTIC_POLARITIES,
    DIAGNOSTIC_RUN_PROTOCOL,
    DIAGNOSTIC_RUN_STAGE,
    DIAGNOSTIC_STATUS,
    FIELD_REVIEW_SPECIFICATIONS,
    MAX_EVIDENCE_QUOTE_CHARS,
    MAX_REASON_CHARS,
    RecoveredSingleFieldDiagnosticOutput,
    SEMANTIC_DIAGNOSTIC_FIELDS,
    SingleFieldDiagnosticOutput,
    aggregate_v4_single_field_diagnostic,
    assess_single_field_diagnostic_output,
    build_v4_root_cause_diagnostic_packet,
    evaluate_deterministic_diagnostic_item,
    recover_length_bound_failure,
    validate_single_field_diagnostic_output,
    validate_v4_diagnostic_source_artifacts,
)

from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "v1_5" / "20c_prepare_v4_single_field_diagnostic_v1_5.py"
RUNNER = ROOT / "scripts" / "v1_5" / "20d_run_v4_single_field_diagnostic_v1_5.py"


@pytest.fixture(scope="module")
def v4_calibration_material() -> tuple[list, list[dict], dict]:
    cases = generate_v8_review_cases(
        strategy_bank_path=ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl",
        cases_per_regime=VALIDATION_CASES_PER_REGIME,
        seed=20260716,
        strategy_card_ids=V1_5_REVIEW_STRATEGY_CARD_IDS,
    )
    controls = build_positive_controls(
        cases,
        seed=20260716,
        required_fields=RATING_FIELDS,
        controls_per_field=2,
    )
    packet = build_v4_root_cause_diagnostic_packet(cases, controls)
    return cases, controls, packet


def test_field_truth_table_covers_rubric_and_keeps_derived_labels_out_of_judge() -> None:
    assert list(FIELD_REVIEW_SPECIFICATIONS) == list(RATING_FIELDS)
    assert FIELD_REVIEW_SPECIFICATIONS["regime_match"].evaluation_mode == (
        "derived_from_validated_primitives"
    )
    assert tuple(DETERMINISTIC_DIAGNOSTIC_FIELDS) == (
        "source_type_match",
        "dialogue_temporal_order_match",
        "memory_age_design_match",
        "surface_naturalness_match",
    )
    assert set(SEMANTIC_DIAGNOSTIC_FIELDS).isdisjoint(DETERMINISTIC_DIAGNOSTIC_FIELDS)
    assert set(SEMANTIC_DIAGNOSTIC_FIELDS) | set(DETERMINISTIC_DIAGNOSTIC_FIELDS) == set(
        DIAGNOSTIC_FIELDS
    )
    assert "candidate_regime" in FIELD_REVIEW_SPECIFICATIONS[
        "regime_match"
    ].prohibited_evidence


def test_packet_is_balanced_calibration_only_and_freezes_narrow_claim(
    v4_calibration_material,
) -> None:
    _cases, _controls, packet = v4_calibration_material
    assert packet["status"] == DIAGNOSTIC_STATUS
    assert packet["formal_gate"] is False
    assert packet["api_calls_made"] == 0
    assert packet["uses_existing_v4_calibration_data"] is True
    assert packet["creates_new_heldout"] is False
    assert packet["creates_paid_run_identity"] is False
    assert packet["changes_v4_gate_result"] is False
    assert "not real-user" in packet["claim_boundary"].lower()
    assert packet["semantic_protocol"] == ATOMIC_DEFINITION_AND_EXHAUSTIVE_EVIDENCE
    assert packet["diagnostic_polarities"] == list(DIAGNOSTIC_POLARITIES)
    assert packet["prepared_prompt_count"] == 4
    assert packet["deterministic_check_count"] == 8
    assert len(packet["items"]) == 12
    assert {
        (item["field"], item["polarity"]) for item in packet["items"]
    } == {
        (field, polarity)
        for field in DIAGNOSTIC_FIELDS
        for polarity in DIAGNOSTIC_POLARITIES
    }
    serialized = json.dumps(packet, ensure_ascii=False)
    for forbidden in (
        "cost_estimate_sha256",
        "paid_execution_authorized",
        "stage_approvals",
        "api_key_env",
    ):
        assert forbidden not in serialized


def test_semantic_prompts_are_atomic_exhaustive_and_do_not_leak_answers(
    v4_calibration_material,
) -> None:
    _cases, _controls, packet = v4_calibration_material
    semantic_items = [
        item for item in packet["items"] if item["evaluation_mode"] == "semantic_atomic_claim"
    ]
    assert len(semantic_items) == 4
    for item in semantic_items:
        payload = json.loads(item["messages"][1]["content"])
        assert payload["field"] == item["field"]
        assert payload["candidate_claim"] == item["candidate_claim"]
        assert payload["exhaustive_evidence"] == item["allowed_evidence"]
        assert payload["valid_evidence_keys"] == list(item["allowed_evidence"])
        assert payload["field_definition"]
        assert payload["decision_rule"]
        assert payload["minimum_distinct_evidence_citations"] == item[
            "minimum_evidence_citations"
        ]
        for forbidden in (
            "expected_verdict",
            "source_control_id",
            "source_case_id",
            "corrupted_field",
            "override",
            "coverage_rationale",
            "polarity",
            "regime",
            "utility",
        ):
            assert forbidden not in payload


def test_balanced_pairs_are_genuine_positive_and_negative_examples(
    v4_calibration_material,
) -> None:
    _cases, _controls, packet = v4_calibration_material
    by_key = {(item["field"], item["polarity"]): item for item in packet["items"]}

    source_pos = by_key[("source_type_match", "positive")]
    source_neg = by_key[("source_type_match", "negative")]
    assert source_pos["candidate_claim"]["proposed_source"] == "MP"
    assert source_neg["candidate_claim"]["proposed_source"] == "MS"
    assert source_pos["allowed_evidence"] == {"registered_source": "MP"}
    assert source_neg["allowed_evidence"] == {"registered_source": "MP"}
    assert "messages" not in source_pos
    assert "messages" not in source_neg

    context_pos = by_key[("context_grounding_match", "positive")]
    context_neg = by_key[("context_grounding_match", "negative")]
    assert context_pos["candidate_claim"]["session_summary"] != context_neg[
        "candidate_claim"
    ]["session_summary"]
    assert context_pos["minimum_evidence_citations"] == 2
    assert all(
        "claim" not in row
        for row in context_pos["allowed_evidence"]["claim_provenance"]
    )

    readiness_pos = by_key[("advice_readiness_match", "positive")]
    readiness_neg = by_key[("advice_readiness_match", "negative")]
    assert readiness_pos["candidate_claim"]["proposed_advice_readiness"] == "explore_first"
    assert readiness_neg["candidate_claim"]["proposed_advice_readiness"] == "structured_plan"

    surface_pos = by_key[("surface_naturalness_match", "positive")]
    surface_neg = by_key[("surface_naturalness_match", "negative")]
    assert "generated user" not in surface_pos["allowed_evidence"][
        "current_user_text"
    ].lower()
    assert "generated user" in surface_neg["allowed_evidence"][
        "current_user_text"
    ].lower()


def test_deterministic_truths_accept_positive_and_reject_negative_without_llm(
    v4_calibration_material,
) -> None:
    _cases, _controls, packet = v4_calibration_material
    items = [
        item for item in packet["items"] if item["evaluation_mode"] == "deterministic_code"
    ]
    assert len(items) == 8
    for item in items:
        assert "messages" not in item
        observed = evaluate_deterministic_diagnostic_item(item)
        assert observed == item["deterministic_assessment"]
        assert observed["correct"] is True
        assert observed["code_result"] is (item["polarity"] == "positive")


def test_offline_packet_builder_verifies_observed_v4_hashes(
    tmp_path: Path,
    v4_calibration_material,
) -> None:
    _cases, controls, _packet = v4_calibration_material
    observed = tmp_path / "controls.json"
    write_json(observed, build_control_manifest(controls))
    out = tmp_path / "diagnostic_packet.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--observed-controls-manifest",
            str(observed),
            "--out",
            str(out),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    packet = read_json(out)
    assert packet["api_calls_made"] == 0
    assert packet["observed_v4_controls_manifest"]["verified"] is True
    assert packet["observed_v4_controls_manifest"]["observed_control_count"] == 24
    assert packet["prepared_prompt_count"] == 4
    assert '"api_calls_made": 0' in completed.stdout
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in (
        "make_client",
        "require_paid_run_release",
        'add_argument("--run"',
        "accept-cost-estimate",
    ):
        assert forbidden not in source


def _quote(value) -> str:
    return canonical_json(value)


def test_semantic_output_supports_multi_section_citations_and_exact_scoring(
    v4_calibration_material,
) -> None:
    _cases, _controls, packet = v4_calibration_material
    item = next(
        row
        for row in packet["items"]
        if row["field"] == "context_grounding_match" and row["polarity"] == "positive"
    )
    keys = ["history", "current_user_text"]
    output = SingleFieldDiagnosticOutput(
        verdict="supported",
        evidence_keys=keys,
        evidence_quotes=[_quote(item["allowed_evidence"][key]) for key in keys],
        reason="The current request and preceding event jointly entail the summary.",
    )
    assessment = validate_single_field_diagnostic_output(item=item, output=output)
    assert assessment["correct"] is True
    assert assessment["evidence_citation_count"] == 2

    wrong = output.model_copy(update={"verdict": "not_supported"})
    assert validate_single_field_diagnostic_output(item=item, output=wrong)[
        "false_negative"
    ] is True


def test_semantic_output_rejects_unknown_missing_and_duplicate_citations(
    v4_calibration_material,
) -> None:
    _cases, _controls, packet = v4_calibration_material
    item = next(
        row
        for row in packet["items"]
        if row["field"] == "context_grounding_match" and row["polarity"] == "negative"
    )
    with pytest.raises(RuntimeError, match="too few"):
        validate_single_field_diagnostic_output(
            item=item,
            output=SingleFieldDiagnosticOutput(
                verdict="not_supported",
                evidence_keys=["current_user_text"],
                evidence_quotes=[item["allowed_evidence"]["current_user_text"]],
                reason="One citation is insufficient for this relational claim.",
            ),
        )
    with pytest.raises(RuntimeError, match="distinct"):
        validate_single_field_diagnostic_output(
            item=item,
            output=SingleFieldDiagnosticOutput(
                verdict="not_supported",
                evidence_keys=["current_user_text", "current_user_text"],
                evidence_quotes=[
                    item["allowed_evidence"]["current_user_text"],
                    item["allowed_evidence"]["current_user_text"],
                ],
                reason="Duplicate citations are not independent evidence sections.",
            ),
        )
    with pytest.raises(RuntimeError, match="unknown"):
        validate_single_field_diagnostic_output(
            item=item,
            output=SingleFieldDiagnosticOutput(
                verdict="not_supported",
                evidence_keys=["expected_verdict", "current_user_text"],
                evidence_quotes=["not supported", item["allowed_evidence"]["current_user_text"]],
                reason="Leaked label.",
            ),
        )


def _gemini_response(finish_reason: str) -> dict:
    return {"candidates": [{"finishReason": finish_reason}]}


def _openai_compatible_response(finish_reason: str) -> dict:
    return {"choices": [{"finish_reason": finish_reason}]}


def test_recovered_schema_accepts_a_reason_and_quote_over_the_live_length_caps() -> None:
    over_long_reason = "x " * (MAX_REASON_CHARS)  # well over MAX_REASON_CHARS
    over_long_quote = "y " * (MAX_EVIDENCE_QUOTE_CHARS)  # well over MAX_EVIDENCE_QUOTE_CHARS
    recovered = RecoveredSingleFieldDiagnosticOutput(
        verdict="supported",
        evidence_keys=["current_user_text"],
        evidence_quotes=[over_long_quote],
        reason=over_long_reason,
    )
    assert len(recovered.reason) > MAX_REASON_CHARS
    assert len(recovered.evidence_quotes[0]) > MAX_EVIDENCE_QUOTE_CHARS
    with pytest.raises(ValidationError):
        SingleFieldDiagnosticOutput(
            verdict="supported",
            evidence_keys=["current_user_text"],
            evidence_quotes=[over_long_quote],
            reason=over_long_reason,
        )


@pytest.mark.parametrize(
    "bad_kwargs,match",
    [
        ({"verdict": "maybe"}, None),
        ({"reason": ""}, None),
        ({"evidence_keys": []}, None),
        ({"evidence_keys": ["a", "b", "c", "d", "e"]}, None),
        ({"evidence_quotes": ["a", "b", "c", "d", "e"]}, None),
        ({"extra_field": "not allowed"}, None),
    ],
)
def test_recovered_schema_still_rejects_every_non_length_defect(bad_kwargs, match) -> None:
    base = dict(
        verdict="supported",
        evidence_keys=["current_user_text"],
        evidence_quotes=["a real quote"],
        reason="a real reason",
    )
    base.update(bad_kwargs)
    with pytest.raises(ValidationError):
        RecoveredSingleFieldDiagnosticOutput(**base)


def test_recover_length_bound_failure_refuses_a_truncated_gemini_response() -> None:
    with pytest.raises(ValueError, match="not a normal completion"):
        recover_length_bound_failure(
            raw_provider_response=_gemini_response("MAX_TOKENS"),
            parsed_payload={
                "verdict": "supported",
                "evidence_keys": ["current_user_text"],
                "evidence_quotes": ["a quote"],
                "reason": "x" * (MAX_REASON_CHARS + 10),
            },
        )


def test_recover_length_bound_failure_refuses_a_truncated_openai_compatible_response() -> None:
    with pytest.raises(ValueError, match="not a normal completion"):
        recover_length_bound_failure(
            raw_provider_response=_openai_compatible_response("length"),
            parsed_payload={
                "verdict": "supported",
                "evidence_keys": ["current_user_text"],
                "evidence_quotes": ["a quote"],
                "reason": "x" * (MAX_REASON_CHARS + 10),
            },
        )


def test_recover_length_bound_failure_recovers_a_complete_over_length_response(
    v4_calibration_material,
) -> None:
    _cases, _controls, packet = v4_calibration_material
    item = next(
        row
        for row in packet["items"]
        if row["field"] == "context_grounding_match" and row["polarity"] == "positive"
    )
    keys = ["history", "current_user_text"]
    over_long_reason = (
        "The current request and preceding event jointly entail the summary. " * 20
    )
    assert len(over_long_reason) > MAX_REASON_CHARS
    recovered = recover_length_bound_failure(
        raw_provider_response=_gemini_response("STOP"),
        parsed_payload={
            "verdict": "supported",
            "evidence_keys": keys,
            "evidence_quotes": [_quote(item["allowed_evidence"][key]) for key in keys],
            "reason": over_long_reason,
        },
    )
    assessment = assess_single_field_diagnostic_output(item=item, output=recovered)
    assert assessment["citation_valid"] is True
    assert assessment["correct"] is True


def test_recover_length_bound_failure_reports_citation_defects_honestly_not_as_valid(
    v4_calibration_material,
) -> None:
    _cases, _controls, packet = v4_calibration_material
    item = next(
        row
        for row in packet["items"]
        if row["field"] == "context_grounding_match" and row["polarity"] == "positive"
    )
    over_long_reason = "This reason is deliberately over the live length cap. " * 20
    assert len(over_long_reason) > MAX_REASON_CHARS
    recovered = recover_length_bound_failure(
        raw_provider_response=_openai_compatible_response("stop"),
        parsed_payload={
            "verdict": "supported",
            "evidence_keys": ["current_user_text"],
            "evidence_quotes": ["a quote that does not appear anywhere in the evidence"],
            "reason": over_long_reason,
        },
    )
    assessment = assess_single_field_diagnostic_output(item=item, output=recovered)
    assert assessment["citation_valid"] is False
    assert "quote_absent_from_cited_evidence:current_user_text" in assessment["citation_errors"]
    assert assessment["correct"] is False


def _failed_v4_source_artifacts(packet: dict, controls: list[dict]):
    manifest = build_control_manifest(controls)
    endpoint_names = [
        "training_judge_gemini_flash_lite",
        "training_judge_deepseek_flash",
    ]
    negative_items = [item for item in packet["items"] if item["polarity"] == "negative"]
    judgments = {
        item["source_control_id"]: {
            endpoint: {
                "ratings": {field: 1 for field in RATING_FIELDS},
                "notes": "",
            }
            for endpoint in endpoint_names
        }
        for item in negative_items
    }
    gate = {
        "status": "FAIL",
        "n_controls": 24,
        "control_misses": [{"item_id": row["item_id"]} for row in manifest],
        "control_catches": [],
        "judge_families": endpoint_names,
    }
    return manifest, judgments, gate, endpoint_names


def test_source_binding_and_balanced_aggregate_do_not_create_a_formal_gate(
    v4_calibration_material,
) -> None:
    _cases, controls, packet = v4_calibration_material
    manifest, judgments, gate, endpoints = _failed_v4_source_artifacts(packet, controls)
    validation = validate_v4_diagnostic_source_artifacts(
        packet=packet,
        observed_controls=manifest,
        observed_control_judgments=judgments,
        observed_gate_report=gate,
        endpoint_names=endpoints,
    )
    assert validation["source_binding_verified"] is True
    assert len(validation["p0_negative_control_rows"]) == 12
    assert len(validation["deterministic_rows"]) == 8
    assert all(row["correct"] for row in validation["deterministic_rows"])

    result_rows = [
        {
            "field": field,
            "polarity": polarity,
            "endpoint_name": endpoint,
            "verdict": "supported" if polarity == "positive" else "not_supported",
            "verdict_correct": True,
            "citation_valid": True,
            "correct": True,
        }
        for field in SEMANTIC_DIAGNOSTIC_FIELDS
        for polarity in DIAGNOSTIC_POLARITIES
        for endpoint in endpoints
    ]
    report = aggregate_v4_single_field_diagnostic(
        deterministic_rows=validation["deterministic_rows"],
        result_rows=result_rows,
        endpoint_names=endpoints,
    )
    assert report["formal_gate"] is False
    assert report["authorizes_v5"] is False
    assert report["authorizes_training"] is False
    assert report["measurement_instrument_ready"] is True
    assert report["semantic_support_classifier_ready"] is True
    assert report["citation_audit_ready"] is True
    assert report["citation_integrity_is_outcome_gate"] is False
    assert report["n_result_rows"] == 8
    assert all(row["balanced_accuracy"] == 1.0 for row in report["field_strata"])


def _write_runner_inputs(tmp_path: Path, material):
    _cases, controls, packet = material
    manifest, judgments, gate, _endpoints = _failed_v4_source_artifacts(packet, controls)
    paths = {
        "packet": tmp_path / "packet.json",
        "controls": tmp_path / "controls.json",
        "judgments": tmp_path / "judgments.json",
        "gate": tmp_path / "gate.json",
    }
    write_json(paths["packet"], packet)
    write_json(paths["controls"], manifest)
    write_json(paths["judgments"], judgments)
    write_json(paths["gate"], gate)
    return paths


def _runner_args(paths: dict[str, Path], out_dir: Path) -> list[str]:
    return [
        str(RUNNER),
        "--diagnostic-packet",
        str(paths["packet"]),
        "--observed-controls",
        str(paths["controls"]),
        "--observed-control-judgments",
        str(paths["judgments"]),
        "--observed-gate-report",
        str(paths["gate"]),
        "--out-dir",
        str(out_dir),
    ]


def test_diagnostic_runner_dry_run_is_bound_budgeted_and_api_free(
    tmp_path: Path,
    v4_calibration_material,
) -> None:
    paths = _write_runner_inputs(tmp_path, v4_calibration_material)
    out_dir = tmp_path / "run"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--dry-run", *_runner_args(paths, out_dir)[1:]],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    estimate = read_json(out_dir / "cost_estimate.json")
    plan = [
        json.loads(line)
        for line in (out_dir / "call_plan.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert estimate["protocol"] == DIAGNOSTIC_RUN_PROTOCOL
    assert estimate["stage"] == DIAGNOSTIC_RUN_STAGE
    assert estimate["formal_gate"] is False
    assert estimate["authorizes_v5"] is False
    assert estimate["n_logical_calls"] == 8
    assert estimate["maximum_physical_api_attempts"] == 80
    assert estimate["retry_contract"][
        "bounded_provider_output_retry_classes"
    ] == ["missing_field", "provider_output_format"]
    assert estimate["retry_contract"][
        "provider_output_maximum_failures"
    ] == 2
    assert estimate["retry_contract"][
        "provider_output_failures_are_independent_of_transport_attempts"
    ] is True
    assert estimate["retry_contract"]["backoff_seconds"] == [
        10.0,
        30.0,
        60.0,
        120.0,
        300.0,
        300.0,
        600.0,
        600.0,
        900.0,
    ]
    assert estimate["budget_gate"]["status"] == "PASS"
    assert len(plan) == 8
    assert {row["polarity"] for row in plan} == set(DIAGNOSTIC_POLARITIES)
    assert "DRY_RUN_COMPLETE_NOT_A_FORMAL_GATE" in completed.stdout
    assert not (out_dir / "physical_attempt_ledger.jsonl").exists()


def test_diagnostic_runner_records_bad_citation_and_completes_full_mock_matrix(
    tmp_path: Path,
    v4_calibration_material,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_runner_inputs(tmp_path, v4_calibration_material)
    out_dir = tmp_path / "mock_run"
    spec = importlib.util.spec_from_file_location("v4_diag_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    common_args = _runner_args(paths, out_dir)
    monkeypatch.setattr(sys, "argv", [common_args[0], "--dry-run", *common_args[1:]])
    module.main()
    accepted = read_json(out_dir / "cost_estimate.json")["cost_estimate_sha256"]

    class FakeClient:
        def __init__(self, *, gemini: bool) -> None:
            self.gemini = gemini

        def chat(self, messages, **_kwargs):
            payload = json.loads(messages[1]["content"])
            field = payload["field"]
            claim = payload["candidate_claim"]
            evidence = payload["exhaustive_evidence"]
            if field == "context_grounding_match":
                supported = "recently moved" not in claim["session_summary"].lower()
            elif field == "advice_readiness_match":
                supported = claim["proposed_advice_readiness"] == "explore_first"
            else:  # pragma: no cover - frozen semantic cohort
                raise AssertionError(field)
            minimum = int(payload["minimum_distinct_evidence_citations"])
            keys = list(evidence)[:minimum]
            quotes = [
                canonical_json(evidence[key])[:MAX_EVIDENCE_QUOTE_CHARS]
                for key in keys
            ]
            # A citation pointer defect is recorded separately and must not
            # redefine an otherwise correct binary semantic verdict.
            if not self.gemini and field == "context_grounding_match" and supported:
                keys = ["memory_id"]
                quotes = [
                    canonical_json(evidence["current_user_text"])[
                        :MAX_EVIDENCE_QUOTE_CHARS
                    ]
                ]
            parsed = SingleFieldDiagnosticOutput(
                verdict="supported" if supported else "not_supported",
                evidence_keys=keys,
                evidence_quotes=quotes,
                reason="The cited exhaustive evidence decides the atomic claim.",
            )
            usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
            if self.gemini:
                usage.update(
                    {
                        "gemini_prompt_tokens": 100,
                        "gemini_candidate_tokens": 18,
                        "gemini_thought_tokens": 2,
                        "gemini_tool_use_prompt_tokens": 0,
                        "gemini_cached_content_tokens": 0,
                    }
                )
            return (
                CallResult(
                    text=parsed.model_dump_json(),
                    raw_response={"choices": []},
                    usage=usage,
                    latency_ms=1.0,
                    request_hash="a" * 64,
                    provider_finish_reason="stop",
                    normalized_finish_reason="complete",
                ),
                parsed,
            )

        def close(self):
            return None

    monkeypatch.setenv("GEMINI_API_KEY", "mock")
    monkeypatch.setenv("NVIDIA_API_KEY", "mock")
    monkeypatch.setattr(
        module,
        "make_client",
        lambda endpoint: FakeClient(gemini=endpoint.family == "google_gemini"),
    )
    monkeypatch.setattr(
        module,
        "require_paid_run_release",
        lambda *_args, **_kwargs: {"status": "MOCK_APPROVED"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            common_args[0],
            "--run",
            *common_args[1:],
            "--accept-cost-estimate-sha256",
            accepted,
        ],
    )
    module.main()
    report = read_json(out_dir / "diagnostic_report.json")
    results = read_json(out_dir / "diagnostic_results.json")
    assert report["formal_gate"] is False
    assert report["authorizes_v5"] is False
    assert report["measurement_instrument_ready"] is True
    assert report["semantic_support_classifier_ready"] is True
    assert report["citation_audit_ready"] is False
    assert report["citation_integrity_is_outcome_gate"] is False
    assert report["n_result_rows"] == 8
    assert report["semantic_verdict_accuracy"] == 1.0
    assert report["citation_integrity_rate"] == 7 / 8
    assert report["semantic_joint_validated_accuracy"] == 7 / 8
    assert report["reported_thought_tokens"] == 8
    assert report["reported_visible_candidate_tokens"] == 152
    assert report["reported_output_tokens"] == 160
    assert report["reported_total_tokens"] == 960
    assert len(results) == 8
    invalid = [row for row in results if not row["evidence_attribution_valid"]]
    assert len(invalid) == 1
    assert invalid[0]["endpoint_name"] == "training_judge_deepseek_flash"
    assert invalid[0]["citation_errors"] == [
        "too_few_evidence_sections",
        "unknown_evidence_key:memory_id",
    ]
    assert (out_dir / "source_validation.json").is_file()
    assert (out_dir / "artifact_attestation.json").is_file()


def test_diagnostic_runner_continues_matrix_after_one_provider_exhausts(
    tmp_path: Path,
    v4_calibration_material,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_runner_inputs(tmp_path, v4_calibration_material)
    out_dir = tmp_path / "partial_mock_run"
    spec = importlib.util.spec_from_file_location("v4_diag_runner_partial", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    common_args = _runner_args(paths, out_dir)
    monkeypatch.setattr(sys, "argv", [common_args[0], "--dry-run", *common_args[1:]])
    module.main()
    accepted = read_json(out_dir / "cost_estimate.json")["cost_estimate_sha256"]

    class FakeClient:
        def __init__(self, *, fail_first_deepseek_item: bool) -> None:
            self.fail_first_deepseek_item = fail_first_deepseek_item

        def chat(self, messages, **_kwargs):
            payload = json.loads(messages[1]["content"])
            field = payload["field"]
            claim = payload["candidate_claim"]
            evidence = payload["exhaustive_evidence"]
            if self.fail_first_deepseek_item and (
                field == "context_grounding_match"
                and "recently moved" not in claim["session_summary"].lower()
            ):
                raise RetryableProviderError(
                    "injected provider outage",
                    last_retry_class="http_5xx",
                    last_status_code=503,
                    attempts_tried=1,
                    request_hash="f" * 64,
                    response_diagnostics={"status_code": 503},
                )
            if field == "context_grounding_match":
                supported = "recently moved" not in claim["session_summary"].lower()
            elif field == "advice_readiness_match":
                supported = claim["proposed_advice_readiness"] == "explore_first"
            else:  # pragma: no cover - frozen semantic cohort
                raise AssertionError(field)
            minimum = int(payload["minimum_distinct_evidence_citations"])
            keys = list(evidence)[:minimum]
            parsed = SingleFieldDiagnosticOutput(
                verdict="supported" if supported else "not_supported",
                evidence_keys=keys,
                evidence_quotes=[
                    canonical_json(evidence[key])[:MAX_EVIDENCE_QUOTE_CHARS]
                    for key in keys
                ],
                reason="The cited exhaustive evidence decides the atomic claim.",
            )
            return (
                CallResult(
                    text=parsed.model_dump_json(),
                    raw_response={"choices": []},
                    usage={
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                    },
                    latency_ms=1.0,
                    request_hash="a" * 64,
                    provider_finish_reason="stop",
                    normalized_finish_reason="complete",
                    structured_output_audit={"initially_valid_json": True},
                ),
                parsed,
            )

        def close(self):
            return None

    monkeypatch.setenv("GEMINI_API_KEY", "mock")
    monkeypatch.setenv("NVIDIA_API_KEY", "mock")
    monkeypatch.setattr(
        module,
        "make_client",
        lambda endpoint: FakeClient(
            fail_first_deepseek_item=endpoint.family == "deepseek"
        ),
    )
    monkeypatch.setattr(
        module,
        "require_paid_run_release",
        lambda *_args, **_kwargs: {"status": "MOCK_APPROVED"},
    )
    real_execute = module.execute_with_bounded_retry

    def execute_without_wait(*args, **kwargs):
        kwargs["sleep"] = lambda _seconds: None
        return real_execute(*args, **kwargs)

    monkeypatch.setattr(module, "execute_with_bounded_retry", execute_without_wait)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            common_args[0],
            "--run",
            *common_args[1:],
            "--accept-cost-estimate-sha256",
            accepted,
        ],
    )
    module.main()
    report = read_json(out_dir / "diagnostic_report.json")
    results = read_json(out_dir / "diagnostic_results.json")
    assert report["status"] == (
        "INCONCLUSIVE_PROVIDER_AVAILABILITY_NOT_A_FORMAL_GATE"
    )
    assert report["measurement_instrument_ready"] is False
    assert report["result_matrix_complete"] is False
    assert report["n_result_rows"] == 7
    assert report["n_missing_result_rows"] == 1
    assert len(results) == 7
    assert len(report["unavailable_calls"]) == 1
    assert report["unavailable_calls"][0]["retry_class"] == "http_5xx"
    assert report["unavailable_calls"][0]["attempts"] == 10
    assert report["transport_retry_summary"]["physical_attempts"] == 17


def test_diagnostic_runner_retains_provider_usage_breakdown() -> None:
    spec = importlib.util.spec_from_file_location("v4_diag_runner_audit", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parsed = SingleFieldDiagnosticOutput(
        verdict="not_supported",
        evidence_keys=["current_user_text"],
        evidence_quotes=["example"],
        reason="The field is not established by the evidence.",
    )
    result = CallResult(
        text=parsed.model_dump_json(),
        raw_response={"usageMetadata": {"thoughtsTokenCount": 2}},
        usage={
            "prompt_tokens": 265,
            "completion_tokens": 111,
            "total_tokens": 376,
            "gemini_prompt_tokens": 265,
            "gemini_candidate_tokens": 109,
            "gemini_thought_tokens": 2,
            "gemini_tool_use_prompt_tokens": 0,
            "gemini_cached_content_tokens": 0,
        },
        latency_ms=1.0,
        request_hash="b" * 64,
        provider_finish_reason="STOP",
        normalized_finish_reason="complete",
    )
    payload = module._audited_result_payload(result=result, parsed=parsed)
    assert payload["provider_usage_breakdown"]["gemini_thought_tokens"] == 2
    assert len(payload["provider_response_sha256"]) == 64


def test_real_diagnostic_run_is_blocked_before_api_without_exact_approval(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "blocked"
    env = os.environ.copy()
    env.pop("GEMINI_API_KEY", None)
    env.pop("NVIDIA_API_KEY", None)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--run",
            "--accept-cost-estimate-sha256",
            "f" * 64,
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "paid-run approval is stale" in completed.stderr
    assert not (out_dir / "physical_attempt_ledger.jsonl").exists()
    source = RUNNER.read_text(encoding="utf-8")
    assert "require_paid_run_release" in source
    assert "authorizes_v5" in source
    for forbidden in (
        "require_automated_semantic_review_pass",
        "pm_v2_states.jsonl",
        "22_train_pm_v2",
    ):
        assert forbidden not in source
