from __future__ import annotations

from pathlib import Path

import pytest

from metacom_pm.io import canonical_json, read_json, sha256_file, sha256_text, write_json, write_jsonl
from metacom_pm.pm_v2_contracts import ActionLabel, ResponseDimensions, RiskDimensions
from metacom_pm.v1_5_weak_supervision import (
    WEAK_SUPERVISION_PROTOCOL,
    WEAK_SUPERVISION_STATUS,
    _load_judge_results,
    require_weak_supervision_contract,
    require_weak_supervision_split,
    require_weak_supervision_source_hashes,
    write_weak_supervision_bundle,
)


def _contract(path: Path, *, source_hash: str = "a" * 64) -> dict:
    body = {
        "protocol": WEAK_SUPERVISION_PROTOCOL,
        "status": WEAK_SUPERVISION_STATUS,
        "automatic_gold_label_claimed": False,
        "human_anchors_used_as_automatic_gold": False,
        "source_lineage": {"source_sha256": source_hash},
        "judge_aggregation": {
            "judge_families": ["judge_a", "judge_b"],
        },
    }
    value = {
        **body,
        "contract_sha256": sha256_text(canonical_json(body)),
    }
    write_json(path, value)
    return value


def _label(state_id: str) -> ActionLabel:
    dimensions = {
        **{f"response.{name}": 0.5 for name in ResponseDimensions.model_fields},
        **{f"risk.{name}": 0.0 for name in RiskDimensions.model_fields},
    }
    return ActionLabel(
        state_id=state_id,
        card_id=f"card_{state_id}",
        user_id=f"user_{state_id}",
        semantic_family="context_only",
        action_id="M0+R0",
        response=ResponseDimensions(
            emotional_support=4,
            personalization=3,
            memory_appropriateness=4,
            factual_grounding=4,
            temporal_consistency=4,
            non_intrusiveness=5,
        ),
        risk=RiskDimensions(
            selected_context_misuse=0,
            unnecessary_exposure=0,
            stale_or_conflicting_use=0,
            unsupported_personal_claim=0,
            memory_omission=0,
            strategy_overuse=0,
            strategy_omission=0,
        ),
        observed_input_tokens=100,
        retrieval_calls=0,
        judge_families=["judge_a", "judge_b"],
        judge_count=2,
        max_dimension_mad=0.5,
        dimension_mad=dimensions,
        label_reliable=True,
        composite_weights_sha256="b" * 64,
        provenance={
            "supervision_kind": WEAK_SUPERVISION_STATUS,
            "automatic_gold_label": False,
        },
    )


def test_contract_is_content_addressed_and_rejects_gold_promotion(tmp_path: Path):
    path = tmp_path / "contract.json"
    value = _contract(path)
    assert require_weak_supervision_contract(path)["contract_sha256"] == value[
        "contract_sha256"
    ]
    value["automatic_gold_label_claimed"] = True
    write_json(path, value)
    with pytest.raises(RuntimeError, match="self-hash mismatch"):
        require_weak_supervision_contract(path)


def test_weak_contract_allows_only_explicit_non_internal_splits(tmp_path: Path):
    path = tmp_path / "contract.json"
    value = _contract(path)
    value["scope"] = {"esconv_auxiliary": ["train", "calibration"]}
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    value["contract_sha256"] = sha256_text(canonical_json(body))
    write_json(path, value)
    contract = require_weak_supervision_contract(path)
    require_weak_supervision_split(
        contract, domain="esconv_auxiliary", split="calibration"
    )
    with pytest.raises(RuntimeError, match="does not authorize"):
        require_weak_supervision_split(
            contract, domain="esconv_auxiliary", split="internal_test"
        )


def test_source_hash_guard_rejects_drift(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    source.write_text('{"value":1}\n', encoding="utf-8")
    contract_path = tmp_path / "contract.json"
    contract = _contract(contract_path, source_hash=sha256_file(source))
    assert require_weak_supervision_source_hashes(
        contract, {"source": source}
    ) == {"source": sha256_file(source)}
    source.write_text('{"value":2}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        require_weak_supervision_source_hashes(contract, {"source": source})


def test_raw_judge_loader_rejects_incomplete_or_duplicate_rows(tmp_path: Path):
    row = {
        "state_id": "state_a",
        "action_id": "M0+R0",
        "judge_family": "judge_a",
        "judge_model": "model_a",
        "status": "SUCCESS",
        "schema_success": True,
        "response": {
            "emotional_support": 4,
            "personalization": 3,
            "memory_appropriateness": 4,
            "factual_grounding": 4,
            "temporal_consistency": 4,
            "non_intrusiveness": 5,
            "rationale": "supported",
        },
        "risk": {
            "selected_context_misuse": 0,
            "unnecessary_exposure": 0,
            "stale_or_conflicting_use": 0,
            "unsupported_personal_claim": 0,
            "memory_omission": 0,
            "strategy_overuse": 0,
            "strategy_omission": 0,
            "rationale": "no observed issue",
        },
        "response_request_hash": "r",
        "risk_request_hash": "q",
    }
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [row, row])
    with pytest.raises(RuntimeError, match="duplicate raw judge result"):
        _load_judge_results(path)
    write_jsonl(path, [{**row, "schema_success": False}])
    with pytest.raises(RuntimeError, match="schema-success"):
        _load_judge_results(path)


def test_bundle_attestation_is_deterministic_and_keeps_non_gold_boundary(
    tmp_path: Path,
):
    source = tmp_path / "source.jsonl"
    source.write_text('{"value":1}\n', encoding="utf-8")
    contract_path = tmp_path / "contract.json"
    _contract(contract_path, source_hash=sha256_file(source))
    code = tmp_path / "code.py"
    code.write_text("pass\n", encoding="utf-8")
    labels = [_label("state_a")]
    results = []
    for name in ("a", "b"):
        results.append(
            write_weak_supervision_bundle(
                out_dir=tmp_path / name,
                contract_path=contract_path,
                source_paths={"source": source},
                longitudinal_labels=labels,
                auxiliary_labels=labels,
                code_paths={"compiler": code},
            )
        )
    assert results[0]["attestation_sha256"] == results[1]["attestation_sha256"]
    for filename in (
        "longitudinal_train_calibration_weak_labels.jsonl",
        "esconv_auxiliary_train_weak_labels.jsonl",
        "weak_supervision_report.json",
        "weak_supervision_attestation.json",
    ):
        assert (tmp_path / "a" / filename).read_bytes() == (
            tmp_path / "b" / filename
        ).read_bytes()
    report = read_json(tmp_path / "a" / "weak_supervision_report.json")
    assert report["automatic_gold_label_claimed"] is False
    assert report["internal_test_outcomes_opened"] is False
    assert report["human_anchors_used_as_automatic_gold"] is False
    assert report["dual_domain_training_readiness"] == (
        "BLOCKED_AUXILIARY_CALIBRATION_WEAK_LABELS_PENDING"
    )


def test_bundle_with_calibration_becomes_fit_ready_without_internal_outcomes(
    tmp_path: Path,
):
    source = tmp_path / "source.jsonl"
    source.write_text('{"value":1}\n', encoding="utf-8")
    contract_path = tmp_path / "contract.json"
    _contract(contract_path, source_hash=sha256_file(source))
    code = tmp_path / "code.py"
    code.write_text("pass\n", encoding="utf-8")
    labels = [_label("state_a")]
    result = write_weak_supervision_bundle(
        out_dir=tmp_path / "bundle",
        contract_path=contract_path,
        source_paths={"source": source},
        longitudinal_labels=labels,
        auxiliary_labels=labels,
        auxiliary_calibration_labels=[_label("state_b")],
        code_paths={"compiler": code},
    )
    report = read_json(tmp_path / "bundle" / "weak_supervision_report.json")
    assert report["dual_domain_training_readiness"] == (
        "READY_TRAIN_CALIBRATION_WEAK_SUPERVISION"
    )
    assert report["remaining_required_supervision"] is None
    assert report["internal_test_outcomes_opened"] is False
    assert result["paths"]["auxiliary_calibration_labels"].endswith(
        "esconv_auxiliary_calibration_weak_labels.jsonl"
    )
