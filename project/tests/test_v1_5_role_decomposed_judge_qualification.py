from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from metacom_pm.api import CallResult
from metacom_pm.attempt_ledger import PersistentAttemptLedger
from metacom_pm.v1_5_role_decomposed_judge_qualification import (
    EvidenceRiskFinding,
    RoleDecomposedEvidenceRiskOutput,
    RoleDecomposedQualityOutput,
    aggregate_role_decomposed_qualification,
    build_call_plan,
    build_evidence_risk_messages,
    evidence_excerpt_is_exact,
    finish_paid_postcondition_failure,
    require_human_anchor,
    validate_audit_output_dimensions,
    validate_endpoint_contract,
    validate_schema_contract,
)
from metacom_pm.io import canonical_json, read_json, sha256_file, sha256_text


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/v1_5/21y_prepare_role_decomposed_judge_qualification_v1_5.py"
)
DRY_RUN_SCRIPT = (
    ROOT
    / "scripts/v1_5/"
    "21za_dry_run_role_decomposed_judge_qualification_v1_5.py"
)
RUN_SCRIPT = (
    ROOT
    / "scripts/v1_5/"
    "21zb_run_role_decomposed_judge_qualification_v1_5.py"
)
PACKET_DIR = (
    ROOT
    / "data/pm_v1_5_contracts/role_decomposed_judge_packet_v1"
)
TRACKED_CONTRACT = (
    ROOT
    / "data/pm_v1_5_contracts/"
    "role_decomposed_judge_qualification_v1.json"
)
HUMAN_BINDING = (
    ROOT
    / "data/pm_v1_5_contracts/"
    "role_decomposed_judge_human_anchor_v1.json"
)
ENDPOINT_CONTRACT = (
    ROOT / "configs/pm_v1_5_role_decomposed_judge_qualification_v1.json"
)


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_role_decomposed_schemas_reject_mixed_or_invalid_outputs():
    RoleDecomposedQualityOutput(
        overall_preference="A",
        support_quality_preference="tie",
        overall_reason="A is more directly responsive.",
        support_quality_reason="Both are similarly supportive.",
    )
    with pytest.raises(Exception):
        RoleDecomposedQualityOutput(
            overall_preference="A",
            support_quality_preference="A",
            overall_reason="ok",
            support_quality_reason="ok",
            safety_preference="A",
        )
    with pytest.raises(Exception):
        EvidenceRiskFinding(
            dimension="unsupported_personal_claim",
            verdict="no_violation",
            severity=2,
            response_excerpt="[none]",
            evidence_excerpt="[none]",
            reason="No violation was found.",
        )
    with pytest.raises(Exception):
        EvidenceRiskFinding(
            dimension="strategy_omission",
            verdict="insufficient_evidence",
            severity=1,
            response_excerpt="[omission]",
            evidence_excerpt="[none]",
            reason="The available material cannot establish an omission.",
        )


def test_audit_cross_field_contract_is_explicit_in_both_prompt_messages():
    messages = build_evidence_risk_messages(
        {
            "visible_state": {
                "current_user_text": "I feel stuck.",
                "recent_dialogue": [],
                "current_session_summary": "",
            },
            "authorized_user_context": "",
            "selected_context": {"memory": [], "strategy": []},
            "response": "That sounds difficult.",
            "applicable_dimensions": ["strategy_omission"],
        }
    )
    assert "insufficient_evidence always use severity" in messages[0][
        "content"
    ]
    assert "insufficient_evidence. Set severity=1, 2," in messages[1][
        "content"
    ]
    assert "Never assign positive severity to an abstention." in messages[1][
        "content"
    ]


def test_audit_output_requires_exact_compiler_dimensions():
    output = RoleDecomposedEvidenceRiskOutput(
        findings=[
            EvidenceRiskFinding(
                dimension="unsupported_personal_claim",
                verdict="no_violation",
                severity=0,
                response_excerpt="[none]",
                evidence_excerpt="[none]",
                reason="No unsupported claim appears.",
            )
        ]
    )
    validate_audit_output_dimensions(output, ["unsupported_personal_claim"])
    with pytest.raises(ValueError):
        validate_audit_output_dimensions(
            output, ["unsupported_personal_claim", "memory_omission"]
        )


def test_evidence_excerpt_validation_uses_raw_visible_strings():
    surface = {
        "visible_state": {
            "current_user_text": 'They said "not yet" yesterday.'
        },
        "selected_context": {"memory": ["A separate fact."]},
    }
    assert evidence_excerpt_is_exact('"not yet"', surface)
    assert evidence_excerpt_is_exact("A separate fact", surface)
    assert evidence_excerpt_is_exact("[none]", surface)
    assert not evidence_excerpt_is_exact("invented fact", surface)


def test_zero_api_preparation_is_reproducible_and_blind(tmp_path: Path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    for out in (out_a, out_b):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--out-dir", str(out)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    names = sorted(path.name for path in out_a.iterdir())
    assert names == sorted(path.name for path in out_b.iterdir())
    for name in names:
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()

    summary = json.loads(
        (out_a / "preparation_summary.json").read_text(encoding="utf-8")
    )
    assert summary["pairs"] == 24
    assert summary["quality_ordered_items"] == 48
    assert summary["evidence_risk_items"] == 48
    assert summary["human_blind_items"] == 12
    assert summary["api_calls_made"] == 0
    assert summary["training_labels_created"] is False
    contract = json.loads(
        (out_a / "qualification_contract.json").read_text(encoding="utf-8")
    )
    for name in ("states", "evaluator_contexts", "action_outcomes"):
        value = contract["source_lineage"]["longitudinal"][name]
        assert "file_sha256" not in value
        assert "consumed_bytes_sha256" in value

    internal = _rows(out_a / "selected_pairs_internal.jsonl")
    assert {row["domain"] for row in internal} == {
        "longitudinal",
        "esconv_auxiliary",
    }
    assert len({row["state_id"] for row in internal}) == 24
    human = _rows(out_a / "human_blind_packet.jsonl")
    assert len(human) == 12
    forbidden = {
        "state_id",
        "user_id",
        "domain",
        "regime",
        "action_id",
        "action_a",
        "action_b",
        "model",
        "proxy_expected_winner",
    }
    for row in human:
        assert not (set(row) & forbidden)
        encoded = json.dumps(row, ensure_ascii=False).lower()
        assert "pmv2_train_" not in encoded
        assert "esconv_" not in encoded


def test_preparation_never_opens_audit_only():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "audit_only" not in source
    assert "gold_response" not in source
    assert "gold_strategy" not in source


def test_tracked_human_anchor_is_frozen_and_not_automatic_gold():
    contract = read_json(TRACKED_CONTRACT)
    binding = require_human_anchor(
        root=ROOT,
        binding_path=HUMAN_BINDING,
        qualification_contract=contract,
    )
    assert binding["annotations"] == 12
    assert binding["automatic_gold"] is False
    assert binding["may_auto_promote_bulk_labeler"] is False
    assert binding["requires_researcher_signoff"] is True


def test_human_annotation_drift_is_rejected(tmp_path: Path):
    source_binding = read_json(HUMAN_BINDING)
    scratch_root = tmp_path / "root"
    scratch_root.mkdir()
    relative_paths = {
        "annotations_path": Path("data/annotations.jsonl"),
        "human_blind_packet_path": Path("outputs/packet.jsonl"),
        "human_annotation_template_path": Path("outputs/template.jsonl"),
        "human_anchor_mapping_path": Path("outputs/mapping.jsonl"),
    }
    for key, relative in relative_paths.items():
        target = scratch_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source_binding[key], target)
    binding_payload = {
        **{
            key: value
            for key, value in source_binding.items()
            if key != "binding_sha256"
        },
        **{key: str(value) for key, value in relative_paths.items()},
        "annotations_file_sha256": sha256_file(
            scratch_root / relative_paths["annotations_path"]
        ),
        "human_blind_packet_file_sha256": sha256_file(
            scratch_root / relative_paths["human_blind_packet_path"]
        ),
        "human_annotation_template_file_sha256": sha256_file(
            scratch_root / relative_paths["human_annotation_template_path"]
        ),
        "human_anchor_mapping_file_sha256": sha256_file(
            scratch_root / relative_paths["human_anchor_mapping_path"]
        ),
    }
    binding = {
        **binding_payload,
        "binding_sha256": sha256_text(canonical_json(binding_payload)),
    }
    binding_path = scratch_root / "data/binding.json"
    binding_path.write_text(
        json.dumps(binding, ensure_ascii=False), encoding="utf-8"
    )
    contract = {
        "preparation_contract_sha256": binding[
            "preparation_contract_sha256"
        ],
        "human_anchor_binding_path": "data/binding.json",
        "human_anchor_binding_file_sha256": sha256_file(binding_path),
        "human_anchor_binding_sha256": binding["binding_sha256"],
    }
    require_human_anchor(
        root=scratch_root,
        binding_path=binding_path,
        qualification_contract=contract,
    )
    annotations_path = scratch_root / relative_paths["annotations_path"]
    annotations_path.write_text(
        annotations_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="artifact drifted"):
        require_human_anchor(
            root=scratch_root,
            binding_path=binding_path,
            qualification_contract=contract,
        )


def test_endpoint_or_price_drift_is_rejected():
    endpoint = read_json(ENDPOINT_CONTRACT)
    validate_endpoint_contract(endpoint)
    drifted = json.loads(json.dumps(endpoint))
    drifted["candidates"]["openai_gpt_5_mini"][
        "input_usd_per_million_tokens"
    ] = 0.24
    with pytest.raises(RuntimeError, match="drifted"):
        validate_endpoint_contract(drifted)


def test_tracked_contract_must_bind_the_live_canonical_schemas():
    contract = read_json(TRACKED_CONTRACT)
    validate_schema_contract(contract)
    drifted = json.loads(json.dumps(contract))
    drifted["schemas"]["audit_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="canonical response schema drifted"):
        validate_schema_contract(drifted)


def test_call_plan_binds_exact_audit_dimensions():
    endpoint = read_json(ENDPOINT_CONTRACT)
    plan = build_call_plan(
        quality_rows=_rows(PACKET_DIR / "quality_ordered_items_internal.jsonl"),
        audit_rows=_rows(PACKET_DIR / "evidence_risk_items_internal.jsonl"),
        endpoint_contract=endpoint,
    )
    assert len(plan) == 288
    assert all(
        row["applicable_dimensions"] == []
        for row in plan
        if row["role"] == "quality"
    )
    assert all(
        row["applicable_dimensions"]
        for row in plan
        if row["role"] == "evidence_risk"
    )
    anthropic = [
        row
        for row in plan
        if row["candidate_key"] == "anthropic_claude_haiku_4_5"
    ]
    assert {
        row["request_parameters"]["input_token_safety_factor"]
        for row in anthropic
    } == {2.25}
    assert {
        row["request_parameters"]["provider_schema_projection_protocol"]
        for row in anthropic
    } == {
        "anthropic-strict-tool-schema-projection-v2-strip-numeric-and-array-bounds"
    }
    evidence = next(
        row for row in anthropic if row["role"] == "evidence_risk"
    )
    assert "Never assign positive severity to an abstention." in evidence[
        "messages"
    ][1]["content"]
    assert (
        evidence["request_parameters"]["provider_schema_sha256"]
        != evidence["response_schema_sha256"]
    )


def test_paid_postcondition_failure_is_terminally_ledgered(tmp_path: Path):
    call_key = "a" * 64
    ledger = PersistentAttemptLedger(
        tmp_path / "ledger.jsonl",
        stage="test",
        expected_calls={call_key: 1},
        maximum_total_attempts=1,
    )
    reservation = ledger.reserve(
        call_key,
        record_ids={"item": "x"},
        prompt_sha256="b" * 64,
    )
    parsed = RoleDecomposedQualityOutput(
        overall_preference="A",
        support_quality_preference="A",
        overall_reason="brief",
        support_quality_reason="brief",
    )
    result = CallResult(
        text='{"ok":true}',
        raw_response={"id": "paid"},
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
        latency_ms=1.0,
        request_hash="c" * 64,
        provider_finish_reason="stop",
        normalized_finish_reason="complete",
    )
    finish_paid_postcondition_failure(
        ledger=ledger,
        reservation=reservation,
        result=result,
        parsed=parsed,
        error=RuntimeError("reported prompt usage exceeds bound"),
    )
    terminal = ledger.terminal_row(call_key)
    assert terminal is not None
    assert terminal["event"] == "FAILED"
    assert terminal["usage"]["prompt_tokens"] == 100
    assert terminal["result"]["parsed"]["overall_preference"] == "A"
    assert terminal["metadata"]["retry_class"] == "stage_postcondition_failure"


def _synthetic_complete_results(plan: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for plan_row in plan:
        if plan_row["role"] == "quality":
            preference = (
                "A"
                if int(plan_row["record_ids"]["order_variant"]) == 0
                else "B"
            )
            parsed = {
                "overall_preference": preference,
                "support_quality_preference": preference,
                "overall_reason": "The selected candidate is more responsive.",
                "support_quality_reason": (
                    "The selected candidate is more supportive."
                ),
            }
        else:
            parsed = {
                "findings": [
                    {
                        "dimension": dimension,
                        "verdict": "no_violation",
                        "severity": 0,
                        "response_excerpt": "[none]",
                        "evidence_excerpt": "[none]",
                        "reason": "No violation was found.",
                    }
                    for dimension in plan_row["applicable_dimensions"]
                ]
            }
        rows.append(
            {
                "candidate_key": plan_row["candidate_key"],
                "role": plan_row["role"],
                "record_ids": plan_row["record_ids"],
                "parsed": parsed,
            }
        )
    return rows


def test_preoutcome_aggregation_promotes_roles_separately():
    endpoint = read_json(ENDPOINT_CONTRACT)
    plan = build_call_plan(
        quality_rows=_rows(PACKET_DIR / "quality_ordered_items_internal.jsonl"),
        audit_rows=_rows(PACKET_DIR / "evidence_risk_items_internal.jsonl"),
        endpoint_contract=endpoint,
    )
    results = _synthetic_complete_results(plan)
    report = aggregate_role_decomposed_qualification(
        plan_rows=plan,
        result_rows=results,
        audit_rows=_rows(PACKET_DIR / "evidence_risk_items_internal.jsonl"),
        human_mapping_rows=_rows(
            PACKET_DIR / "human_anchor_mapping_internal.jsonl"
        ),
        human_annotation_rows=_rows(
            ROOT
            / "data/pm_v1_5_contracts/"
            "role_decomposed_judge_human_annotations_v1.jsonl"
        ),
        contract=read_json(TRACKED_CONTRACT),
    )
    assert report["supported_quality_candidates"] == list(
        sorted(report["candidate_reports"])
    )
    assert report["supported_evidence_risk_candidates"] == list(
        sorted(report["candidate_reports"])
    )
    assert report["automatic_gold_created"] is False
    assert report["training_labels_created"] is False

    for row in results:
        if (
            row["candidate_key"] == "openai_gpt_5_mini"
            and row["role"] == "quality"
            and int(row["record_ids"]["order_variant"]) == 1
        ):
            row["parsed"]["overall_preference"] = "A"
            row["parsed"]["support_quality_preference"] = "A"
    failed = aggregate_role_decomposed_qualification(
        plan_rows=plan,
        result_rows=results,
        audit_rows=_rows(PACKET_DIR / "evidence_risk_items_internal.jsonl"),
        human_mapping_rows=_rows(
            PACKET_DIR / "human_anchor_mapping_internal.jsonl"
        ),
        human_annotation_rows=_rows(
            ROOT
            / "data/pm_v1_5_contracts/"
            "role_decomposed_judge_human_annotations_v1.jsonl"
        ),
        contract=read_json(TRACKED_CONTRACT),
    )
    assert "openai_gpt_5_mini" not in failed[
        "supported_quality_candidates"
    ]
    assert "openai_gpt_5_mini" in failed[
        "supported_evidence_risk_candidates"
    ]


def test_zero_api_dry_run_is_reproducible_and_binds_execution(tmp_path: Path):
    out_a = tmp_path / "dry_a"
    out_b = tmp_path / "dry_b"
    for out in (out_a, out_b):
        result = subprocess.run(
            [
                sys.executable,
                str(DRY_RUN_SCRIPT),
                "--dry-run",
                "--out-dir",
                str(out),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    for name in ("call_plan.jsonl", "cost_estimate.json", "summary.json"):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()
    estimate = read_json(out_a / "cost_estimate.json")
    assert estimate["logical_calls"] == 288
    assert estimate["maximum_physical_attempts"] == 576


def test_zero_api_compatibility_pilot_is_two_role_complete_calls(
    tmp_path: Path,
):
    out_a = tmp_path / "pilot_a"
    out_b = tmp_path / "pilot_b"
    for out in (out_a, out_b):
        result = subprocess.run(
            [
                sys.executable,
                str(DRY_RUN_SCRIPT),
                "--dry-run",
                "--compatibility-pilot",
                "--out-dir",
                str(out),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    for name in ("call_plan.jsonl", "cost_estimate.json", "summary.json"):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()
    plan = _rows(out_a / "call_plan.jsonl")
    assert len(plan) == 2
    assert {row["role"] for row in plan} == {"quality", "evidence_risk"}
    assert {row["candidate_key"] for row in plan} == {
        "anthropic_claude_haiku_4_5"
    }
    estimate = read_json(out_a / "cost_estimate.json")
    assert estimate["stage"].endswith("_compatibility_pilot_v2")
    assert estimate["logical_calls"] == 2
    assert estimate["maximum_physical_attempts"] == 4
    assert estimate["api_calls_made"] == 0
    assert estimate["training_labels_created"] is False
    assert (
        estimate["code_manifest"]["execution"]["sha256"]
        == sha256_file(RUN_SCRIPT)
    )


def test_execution_source_preserves_qualification_boundary():
    source = RUN_SCRIPT.read_text(encoding="utf-8")
    assert "training_labels_created" in source
    assert "validate_audit_output_dimensions" in source
    assert "require_paid_run_release" in source
    assert "require_output_directory_not_previously_consumed" in source
    assert "audit_only" not in source
