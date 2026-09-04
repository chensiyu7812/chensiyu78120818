from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .io import canonical_json, iter_jsonl, read_json, sha256_file, sha256_text


STRATEGY_BANK_APPROVAL_PROTOCOL = "pm-v2.2-strategy-bank-human-approval-v1"
STRATEGY_BANK_APPROVAL_DECISION = "APPROVE_FOR_PM_V2_2_FORMAL_STUDY"

_TOP_LEVEL_KEYS = {
    "protocol",
    "decision",
    "approver_id",
    "approved_at_utc",
    "artifact_bindings",
    "declarations",
    "notes",
}
_BINDING_KEYS = {
    "strategy_bank_sha256",
    "strategy_bank_card_count",
    "split_manifest_sha256",
    "bank_audit_sha256",
    "strategy_rag_audit_summary_sha256",
    "provisional_candidate_manifest_sha256",
}
_DECLARATION_KEYS = {
    "reviewed_bank_lineage",
    "reviewed_leakage_audit",
    "reviewed_distribution_shift",
    "confirms_legacy_bank_is_not_mixed",
    "approves_formal_pm_v2_2_use",
}
_PLACEHOLDER_APPROVER_IDS = {
    "",
    "TODO",
    "TBD",
    "REPLACE_ME",
    "APPROVER_ID",
}


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, label: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise RuntimeError(
            f"{label} keys do not match the frozen approval schema: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def _require_utc_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("strategy-bank approval lacks approved_at_utc")
    raw = value.strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError("strategy-bank approved_at_utc is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeError("strategy-bank approved_at_utc must include a UTC offset")
    if parsed.utcoffset().total_seconds() != 0:
        raise RuntimeError("strategy-bank approved_at_utc must be UTC")
    return raw


def require_strategy_bank_human_approval(
    approval_path: str | Path,
    *,
    strategy_bank_path: str | Path,
    split_manifest_path: str | Path,
    bank_audit_path: str | Path,
    strategy_rag_audit_summary_path: str | Path,
    provisional_candidate_manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify an explicit human decision bound to the exact Strategy Bank.

    Automated lineage and leakage audits are evidence for this decision, not a
    substitute for it.  This gate intentionally cannot create or infer human
    approval.
    """

    approval_path = Path(approval_path).resolve()
    if not approval_path.is_file():
        raise FileNotFoundError(
            "formal PM-v2.2 work requires a completed private Strategy Bank "
            f"approval file: {approval_path}"
        )
    raw = read_json(approval_path)
    if not isinstance(raw, Mapping):
        raise RuntimeError("strategy-bank approval must be a JSON object")
    _require_exact_keys(raw, _TOP_LEVEL_KEYS, label="strategy-bank approval")
    if raw["protocol"] != STRATEGY_BANK_APPROVAL_PROTOCOL:
        raise RuntimeError("strategy-bank approval protocol is stale")
    if raw["decision"] != STRATEGY_BANK_APPROVAL_DECISION:
        raise RuntimeError("Strategy Bank has not been approved for formal PM-v2.2 use")
    approver_id = str(raw["approver_id"]).strip()
    if approver_id.upper() in _PLACEHOLDER_APPROVER_IDS:
        raise RuntimeError("strategy-bank approval has a placeholder approver_id")
    approved_at_utc = _require_utc_timestamp(raw["approved_at_utc"])
    if not isinstance(raw["notes"], str):
        raise RuntimeError("strategy-bank approval notes must be a string")

    bindings = raw["artifact_bindings"]
    declarations = raw["declarations"]
    if not isinstance(bindings, Mapping) or not isinstance(declarations, Mapping):
        raise RuntimeError("strategy-bank approval bindings/declarations must be objects")
    _require_exact_keys(bindings, _BINDING_KEYS, label="artifact_bindings")
    _require_exact_keys(declarations, _DECLARATION_KEYS, label="declarations")
    if any(value is not True for value in declarations.values()):
        raise RuntimeError("every frozen Strategy Bank approval declaration must be true")

    strategy_bank_path = Path(strategy_bank_path).resolve()
    split_manifest_path = Path(split_manifest_path).resolve()
    bank_audit_path = Path(bank_audit_path).resolve()
    strategy_rag_audit_summary_path = Path(
        strategy_rag_audit_summary_path
    ).resolve()
    provisional_candidate_manifest_path = Path(
        provisional_candidate_manifest_path
    ).resolve()
    required_paths = (
        strategy_bank_path,
        split_manifest_path,
        bank_audit_path,
        strategy_rag_audit_summary_path,
        provisional_candidate_manifest_path,
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    card_count = sum(1 for _ in iter_jsonl(strategy_bank_path))
    expected_bindings = {
        "strategy_bank_sha256": sha256_file(strategy_bank_path),
        "strategy_bank_card_count": card_count,
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "bank_audit_sha256": sha256_file(bank_audit_path),
        "strategy_rag_audit_summary_sha256": sha256_file(
            strategy_rag_audit_summary_path
        ),
        "provisional_candidate_manifest_sha256": sha256_file(
            provisional_candidate_manifest_path
        ),
    }
    if dict(bindings) != expected_bindings:
        mismatches = {
            key: {"approved": bindings.get(key), "current": value}
            for key, value in expected_bindings.items()
            if bindings.get(key) != value
        }
        raise RuntimeError(
            "strategy-bank approval does not bind the current artifacts: "
            f"{mismatches}"
        )

    candidate = read_json(provisional_candidate_manifest_path)
    candidate_payload = {
        key: value for key, value in candidate.items() if key != "manifest_sha256"
    }
    if candidate.get("manifest_sha256") != sha256_text(
        canonical_json(candidate_payload)
    ):
        raise RuntimeError("provisional Strategy Bank candidate manifest is corrupt")
    candidate_bank = candidate.get("bank") or {}
    if (
        candidate_bank.get("sha256") != expected_bindings["strategy_bank_sha256"]
        or int(candidate_bank.get("card_count", -1)) != card_count
    ):
        raise RuntimeError("provisional candidate describes a different Strategy Bank")

    return {
        "status": "APPROVED",
        "protocol": STRATEGY_BANK_APPROVAL_PROTOCOL,
        "decision": STRATEGY_BANK_APPROVAL_DECISION,
        "approver_id": approver_id,
        "approved_at_utc": approved_at_utc,
        "approval_file_sha256": sha256_file(approval_path),
        "artifact_bindings": expected_bindings,
        "declarations_sha256": sha256_text(canonical_json(dict(declarations))),
    }
