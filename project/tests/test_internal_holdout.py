from __future__ import annotations

from pathlib import Path

import pytest

from metacom_pm.internal_holdout import (
    begin_internal_test_consumption,
    finish_internal_test_consumption,
    freeze_candidate_manifest,
    require_completed_internal_consumption,
    seal_internal_label_bundle,
)
from metacom_pm.io import write_json


def test_internal_test_is_spent_before_read_and_cannot_restart(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"frozen")
    labels = tmp_path / "internal.jsonl"
    labels.write_text('{"state_id":"s1","private":"outcome"}\n', encoding="utf-8")
    seal = tmp_path / "sealed_internal.json"
    seal_internal_label_bundle(seal, internal_labels_path=labels)
    manifest_path = tmp_path / "candidate.json"
    freeze_candidate_manifest(
        manifest_path,
        run_identity="repair-run-001",
        artifacts={"checkpoint": checkpoint, "sealed_internal_bundle": seal},
        parameters={"primary": "learned_step0", "rule": "transparent_step0"},
    )
    ledger = tmp_path / "consumption.jsonl"

    begin_internal_test_consumption(
        ledger,
        candidate_manifest_path=manifest_path,
        internal_labels_path=labels,
    )
    with pytest.raises(RuntimeError, match="already been consumed"):
        begin_internal_test_consumption(
            ledger,
            candidate_manifest_path=manifest_path,
            internal_labels_path=labels,
        )

    report = tmp_path / "report.json"
    write_json(report, {"status": "NOT_SUPPORTED"})
    finish_internal_test_consumption(ledger, report_path=report)
    verified = require_completed_internal_consumption(
        ledger,
        candidate_manifest_path=manifest_path,
        report_path=report,
    )
    assert verified["status"] == "PASS"


def test_candidate_manifest_is_immutable(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.write_text("one", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    freeze_candidate_manifest(
        manifest,
        run_identity="run-a",
        artifacts={"artifact": artifact},
        parameters={},
    )
    artifact.write_text("two", encoding="utf-8")
    with pytest.raises(RuntimeError, match="different content"):
        freeze_candidate_manifest(
            manifest,
            run_identity="run-a",
            artifacts={"artifact": artifact},
            parameters={},
        )


def test_internal_labels_cannot_change_after_pretraining_seal(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "internal.jsonl"
    labels.write_text('{"state_id":"s1","private":"before"}\n', encoding="utf-8")
    seal = tmp_path / "seal.json"
    seal_internal_label_bundle(seal, internal_labels_path=labels)
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"frozen")
    manifest = tmp_path / "candidate.json"
    freeze_candidate_manifest(
        manifest,
        run_identity="sealed-run",
        artifacts={"checkpoint": checkpoint, "sealed_internal_bundle": seal},
        parameters={},
    )
    labels.write_text('{"state_id":"s1","private":"after"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="sealed internal label bundle"):
        begin_internal_test_consumption(
            tmp_path / "ledger.jsonl",
            candidate_manifest_path=manifest,
            internal_labels_path=labels,
        )


def test_two_internal_domains_use_distinct_seals_ledgers_and_reports(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"frozen")
    artifacts = {"checkpoint": checkpoint}
    labels_by_domain = {}
    for domain in ("longitudinal_synthetic", "esconv_auxiliary"):
        labels = tmp_path / f"{domain}.jsonl"
        labels.write_text(
            f'{{"state_id":"{domain}_state","private":"outcome"}}\n',
            encoding="utf-8",
        )
        seal = tmp_path / f"{domain}_seal.json"
        seal_internal_label_bundle(seal, internal_labels_path=labels)
        artifacts[f"sealed_internal_bundle_{domain}"] = seal
        labels_by_domain[domain] = labels
    manifest = tmp_path / "candidate.json"
    freeze_candidate_manifest(
        manifest,
        run_identity="dual-domain-run",
        artifacts=artifacts,
        parameters={"domains": sorted(labels_by_domain)},
    )
    for domain, labels in labels_by_domain.items():
        ledger = tmp_path / f"{domain}_ledger.jsonl"
        begin_internal_test_consumption(
            ledger,
            candidate_manifest_path=manifest,
            internal_labels_path=labels,
            sealed_artifact_name=f"sealed_internal_bundle_{domain}",
            consumption_domain=domain,
        )
        report = tmp_path / f"{domain}_report.json"
        write_json(report, {"status": "NOT_SUPPORTED", "domain": domain})
        finish_internal_test_consumption(
            ledger, report_path=report, expected_domain=domain
        )
        verified = require_completed_internal_consumption(
            ledger,
            candidate_manifest_path=manifest,
            report_path=report,
            expected_domain=domain,
        )
        assert verified["status"] == "PASS"

    with pytest.raises(RuntimeError, match="consumption domain mismatch"):
        require_completed_internal_consumption(
            tmp_path / "longitudinal_synthetic_ledger.jsonl",
            candidate_manifest_path=manifest,
            report_path=tmp_path / "longitudinal_synthetic_report.json",
            expected_domain="esconv_auxiliary",
        )
