"""Build a clean SupportNeed reannotation packet from old judge states.

Only the visible dialogue is reused. Candidate responses, selected/authorized
context, action information, and every old preference/risk label are omitted.
The resulting states are an active-learning pool, not a prevalence sample.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    stable_hex,
)
from .text import normalize_space


PROTOCOL = "pm-v1.5-support-need-historical-state-reannotation-packet-v1"
ALLOWED_VISIBLE_KEYS = frozenset(
    {"current_user_text", "recent_dialogue", "session_summary"}
)


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    declared = str(value.get(field) or "")
    body = {key: item for key, item in value.items() if key != field}
    actual = sha256_text(canonical_json(body))
    if declared != actual:
        raise RuntimeError(f"invalid {field}")
    return declared


def _visible_state(raw: Mapping[str, Any]) -> dict[str, Any]:
    current = normalize_space(raw.get("current_user_text") or "")
    if not current:
        raise RuntimeError("historical judge state has no current user text")
    history = []
    for turn in raw.get("recent_dialogue") or []:
        role = str(turn.get("role") or "")
        content = normalize_space(turn.get("content") or "")
        if role not in {"user", "assistant"} or not content:
            raise RuntimeError("historical judge state has malformed dialogue")
        history.append({"role": role, "content": content})
    return {
        "current_user_text": current,
        "recent_dialogue": history,
        "session_summary": normalize_space(
            raw.get("session_summary")
            or raw.get("current_session_summary")
            or ""
        ),
    }


def _source_rows(path: Path, source_task: str) -> list[dict[str, Any]]:
    rows = []
    source_sha = sha256_file(path)
    for raw in iter_jsonl(path):
        source_id = str(raw.get("blind_item_id") or "")
        if not source_id or not isinstance(raw.get("visible_state"), Mapping):
            raise RuntimeError("historical judge packet is malformed")
        visible = _visible_state(raw["visible_state"])
        blind_item_id = "need_reann_" + stable_hex(
            PROTOCOL,
            source_task,
            source_sha,
            source_id,
            canonical_json(visible),
            n=20,
        )
        rows.append(
            {
                "blind": {
                    "blind_item_id": blind_item_id,
                    "visible_state": visible,
                },
                "private": {
                    "blind_item_id": blind_item_id,
                    "source_task": source_task,
                    "source_blind_item_id": source_id,
                    "source_packet_file_sha256": source_sha,
                    "source_candidate_responses_copied": False,
                    "source_selected_context_copied": False,
                    "source_authorized_context_copied": False,
                    "source_human_annotations_copied": False,
                    "source_preferences_or_risk_labels_copied": False,
                    "sampling_role": (
                        "response_difference_enriched_active_learning_"
                        "not_prevalence"
                    ),
                },
            }
        )
    return rows


def build_historical_support_need_reannotation_packet(
    *,
    project_root: str | Path,
    human_asset_audit_binding_path: str | Path,
    low_budget_packet_path: str | Path,
    role_decomposed_packet_path: str | Path,
    current_support_need_packet_path: str | Path,
) -> dict[str, Any]:
    """Strip post-treatment fields and produce 24 newly blinded states."""

    root = Path(project_root).resolve()
    audit_binding_path = Path(human_asset_audit_binding_path).resolve()
    audit_binding = read_json(audit_binding_path)
    if (
        audit_binding.get("protocol")
        != "pm-v1.5-historical-human-annotation-asset-audit-binding-v1"
        or audit_binding.get("candidate_repro_byte_identical") is not True
        or audit_binding.get(
            "historical_judge_state_reannotation_candidate_count"
        )
        != 24
        or audit_binding.get(
            "historical_judge_state_current_support_need_exact_overlap"
        )
        != 0
        or audit_binding.get(
            "historical_judge_labels_reused_as_need_targets"
        )
        is not False
    ):
        raise RuntimeError("historical human-asset audit does not authorize reuse")
    _self_hash(audit_binding, "binding_sha256")
    audit_report_path = (
        root / str(audit_binding["output_directory"]) / "report.json"
    )
    if (
        not audit_report_path.is_file()
        or sha256_file(audit_report_path)
        != audit_binding["report_file_sha256"]
    ):
        raise RuntimeError("historical human-asset audit report drifted")
    audit_report = read_json(audit_report_path)
    _self_hash(audit_report, "report_sha256")
    opportunity = dict(
        audit_report["historical_judge_state_reannotation_opportunity"]
    )
    if opportunity.get("status") != (
        "ELIGIBLE_AS_NEW_BLINDED_ACTIVE_LEARNING_POOL_ONLY"
    ):
        raise RuntimeError("historical judge states are not reusable")

    low_path = Path(low_budget_packet_path).resolve()
    role_path = Path(role_decomposed_packet_path).resolve()
    current_path = Path(current_support_need_packet_path).resolve()
    expected_hashes = opportunity["source_packet_file_sha256"]
    for name, path in (
        ("low_budget", low_path),
        ("role_decomposed", role_path),
        ("current_support_need", current_path),
    ):
        if not path.is_file() or sha256_file(path) != expected_hashes[name]:
            raise RuntimeError(f"historical reannotation source drifted: {name}")

    selected = [
        *_source_rows(low_path, "low_budget_judge"),
        *_source_rows(role_path, "role_decomposed_judge"),
    ]
    if len(selected) != 24:
        raise RuntimeError("historical SupportNeed pool must contain 24 states")
    blind_ids = [str(row["blind"]["blind_item_id"]) for row in selected]
    if len(blind_ids) != len(set(blind_ids)):
        raise RuntimeError("historical SupportNeed blind IDs are not unique")

    current_visible = {
        canonical_json(_visible_state(row["visible_state"]))
        for row in iter_jsonl(current_path)
    }
    selected_visible = [
        canonical_json(row["blind"]["visible_state"]) for row in selected
    ]
    if (
        len(set(selected_visible)) != 24
        or set(selected_visible) & current_visible
    ):
        raise RuntimeError(
            "historical SupportNeed pool overlaps current fit or itself"
        )

    selected.sort(
        key=lambda row: stable_hex(
            PROTOCOL,
            "blind-order",
            row["blind"]["blind_item_id"],
            n=32,
        )
    )
    packet_rows = [dict(row["blind"]) for row in selected]
    private_rows = [dict(row["private"]) for row in selected]
    template_rows = [
        {
            "blind_item_id": row["blind_item_id"],
            "support_mode": None,
            "goals": [],
            "dialogue_phase": None,
            "nonclinical_urgency": None,
            "recommended_response_burden": None,
            "active_explicit_boundary_evidence": [],
            "abstain": None,
            "confidence": None,
            "notes": "",
        }
        for row in packet_rows
    ]
    if any(
        set(row["visible_state"]) != ALLOWED_VISIBLE_KEYS
        for row in packet_rows
    ):
        raise RuntimeError("historical reannotation blind surface drifted")

    contract_core = {
        "protocol": PROTOCOL,
        "status": (
            "PREPARED_OUTCOME_BLIND_ACTIVE_LEARNING_POOL_"
            "AWAITING_NEW_HUMAN_ANNOTATION"
        ),
        "human_asset_audit_binding_file_sha256": sha256_file(
            audit_binding_path
        ),
        "human_asset_audit_binding_sha256": audit_binding["binding_sha256"],
        "human_asset_audit_report_sha256": audit_report["report_sha256"],
        "source_packet_file_sha256": {
            "low_budget": sha256_file(low_path),
            "role_decomposed": sha256_file(role_path),
            "current_support_need": sha256_file(current_path),
        },
        "selected_packet_sha256": sha256_text(canonical_json(packet_rows)),
        "template_sha256": sha256_text(canonical_json(template_rows)),
        "private_lineage_sha256": sha256_text(
            canonical_json(private_rows)
        ),
        "packet_size": len(packet_rows),
        "independent_dialogue_states": len(packet_rows),
        "exact_overlap_with_current_support_need": 0,
        "sampling_role": (
            "response_difference_enriched_active_learning_not_prevalence"
        ),
        "source_candidates_exposed": False,
        "source_selected_context_exposed": False,
        "source_authorized_context_exposed": False,
        "old_human_labels_reused": False,
        "old_preferences_or_risk_labels_reused": False,
        "annotation_semantics": {
            "explicit_boundary": (
                "requires_exact_quote_from_visible_user_text"
            ),
            "recommended_response_burden": (
                "contextual_human_partial_label_not_explicit_user_fact"
            ),
            "support_mode": "human_partial_label_not_pm_action_gold",
        },
        "automatic_gold_label": False,
        "pm_action_gold": False,
        "prevalence_estimation_allowed": False,
        "formal_fit_authorized": False,
        "confirmation_rows_opened": False,
        "internal_test_outcomes_opened": False,
        "external_outcomes_opened": False,
        "api_calls_made": 0,
    }
    return {
        "packet_rows": packet_rows,
        "template_rows": template_rows,
        "private_rows": private_rows,
        "contract": {
            **contract_core,
            "contract_sha256": sha256_text(canonical_json(contract_core)),
        },
    }

