from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from metacom_pm.io import sha256_file
from metacom_pm.strategy_bank_approval import (
    STRATEGY_BANK_APPROVAL_DECISION,
    STRATEGY_BANK_APPROVAL_PROTOCOL,
    require_strategy_bank_human_approval,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[dict[str, Path], dict]:
    paths = {
        "bank": tmp_path / "strategy_cards.jsonl",
        "split": tmp_path / "split.jsonl",
        "bank_audit": tmp_path / "bank_audit.json",
        "audit_summary": tmp_path / "audit_summary.json",
        "candidate": tmp_path / "candidate.json",
        "approval": tmp_path / "human_approval.json",
    }
    paths["bank"].write_text(
        '{"strategy_id":"s1"}\n{"strategy_id":"s2"}\n', encoding="utf-8"
    )
    paths["split"].write_text('{"dialogue_id":"d1","split":"train"}\n', encoding="utf-8")
    _write_json(paths["bank_audit"], {"status": "PASS"})
    _write_json(paths["audit_summary"], {"human_review_required": True})
    candidate_payload = {
        "status": "PROVISIONAL_FROZEN_CANDIDATE_AWAITING_HUMAN_APPROVAL",
        "gate_unlocked": False,
        "bank": {
            "sha256": sha256_file(paths["bank"]),
            "card_count": 2,
        },
    }
    from metacom_pm.io import canonical_json, sha256_text

    _write_json(
        paths["candidate"],
        {
            **candidate_payload,
            "manifest_sha256": sha256_text(canonical_json(candidate_payload)),
        },
    )
    approval = {
        "protocol": STRATEGY_BANK_APPROVAL_PROTOCOL,
        "decision": STRATEGY_BANK_APPROVAL_DECISION,
        "approver_id": "principal_investigator_01",
        "approved_at_utc": "2026-07-16T13:00:00Z",
        "artifact_bindings": {
            "strategy_bank_sha256": sha256_file(paths["bank"]),
            "strategy_bank_card_count": 2,
            "split_manifest_sha256": sha256_file(paths["split"]),
            "bank_audit_sha256": sha256_file(paths["bank_audit"]),
            "strategy_rag_audit_summary_sha256": sha256_file(
                paths["audit_summary"]
            ),
            "provisional_candidate_manifest_sha256": sha256_file(
                paths["candidate"]
            ),
        },
        "declarations": {
            "reviewed_bank_lineage": True,
            "reviewed_leakage_audit": True,
            "reviewed_distribution_shift": True,
            "confirms_legacy_bank_is_not_mixed": True,
            "approves_formal_pm_v2_2_use": True,
        },
        "notes": "Approved after reviewing the frozen audit packet.",
    }
    _write_json(paths["approval"], approval)
    return paths, approval


def _require(paths: dict[str, Path]) -> dict:
    return require_strategy_bank_human_approval(
        paths["approval"],
        strategy_bank_path=paths["bank"],
        split_manifest_path=paths["split"],
        bank_audit_path=paths["bank_audit"],
        strategy_rag_audit_summary_path=paths["audit_summary"],
        provisional_candidate_manifest_path=paths["candidate"],
    )


def test_strategy_bank_approval_requires_real_human_decision_and_exact_hashes(
    tmp_path: Path,
) -> None:
    paths, _ = _fixture(tmp_path)
    result = _require(paths)

    assert result["status"] == "APPROVED"
    assert result["artifact_bindings"]["strategy_bank_card_count"] == 2
    assert len(result["approval_file_sha256"]) == 64

    paths["bank"].write_text(
        paths["bank"].read_text(encoding="utf-8") + '{"strategy_id":"s3"}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="does not bind the current artifacts"):
        _require(paths)


@pytest.mark.parametrize("field", ["decision", "approver_id", "approved_at_utc"])
def test_strategy_bank_approval_rejects_placeholder_or_invalid_human_fields(
    tmp_path: Path, field: str
) -> None:
    paths, approval = _fixture(tmp_path)
    changed = deepcopy(approval)
    changed[field] = "REPLACE_ME"
    _write_json(paths["approval"], changed)

    with pytest.raises(RuntimeError):
        _require(paths)


def test_strategy_bank_approval_rejects_unchecked_declaration(tmp_path: Path) -> None:
    paths, approval = _fixture(tmp_path)
    changed = deepcopy(approval)
    changed["declarations"]["reviewed_leakage_audit"] = False
    _write_json(paths["approval"], changed)

    with pytest.raises(RuntimeError, match="every frozen.*declaration must be true"):
        _require(paths)
