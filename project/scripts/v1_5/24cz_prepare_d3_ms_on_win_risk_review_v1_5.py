#!/usr/bin/env python3
"""Prepare the one-item minimal risk review for the surviving D3 MS win."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from metacom_pm.contracts import MemoryBackendRecord, RuntimeState
from metacom_pm.io import (
    iter_jsonl,
    sha256_file,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-d3-ms-on-win-minimal-risk-review-v1"
AGGREGATION_PROTOCOL = "pm-v1.5-d3-ms-grounded-quality-aggregation-v1"
STATUS = "READY_FOR_ONE_MS_ON_WIN_MINIMAL_RISK_REVIEW"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _risk_module() -> Any:
    path = ROOT / "scripts/v1_5/24bz_prepare_transport_repaired_on_win_risk_review_v1_5.py"
    spec = importlib.util.spec_from_file_location("_d3ms_risk_html", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load minimal-risk review instrument")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(
    *,
    aggregation_dir: Path,
    blind_dir: Path,
    blueprint_dir: Path,
    execution_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    aggregation = json.loads(
        (aggregation_dir / "aggregation_report.json").read_text(encoding="utf-8")
    )
    if (
        aggregation.get("protocol") != AGGREGATION_PROTOCOL
        or aggregation.get("status") != "PASS_ONE_MS_ON_WIN_PENDING_MINIMAL_RISK"
        or not all(aggregation.get("checks", {}).values())
        or aggregation.get("quality_on_wins_pending_risk") != 1
    ):
        raise RuntimeError("MS quality aggregation is stale or unqualified")
    measurements = _rows(
        aggregation_dir / "pair_quality_measurements_pre_risk.jsonl"
    )
    winners = [row for row in measurements if row["quality_verdict"] == "treatment"]
    if len(winners) != 1:
        raise RuntimeError(f"expected one MS quality winner, found {len(winners)}")
    winner = winners[0]
    source_public = {
        str(row["blind_item_id"]): row
        for row in _rows(blind_dir / "human_blind_packet.jsonl")
    }
    packet = source_public[str(winner["blind_item_id"])]
    states = {
        state.state_id: state
        for state in (
            RuntimeState.model_validate(row)
            for row in iter_jsonl(blueprint_dir / "runtime_states.jsonl")
        )
    }
    backends = {
        record.card_id: {item.memory_id: item for item in record.items}
        for record in (
            MemoryBackendRecord.model_validate(row)
            for row in iter_jsonl(blueprint_dir / "memory_backend.jsonl")
        )
    }
    treatment_outcomes = [
        row
        for row in _rows(execution_dir / "generation_outcomes.jsonl")
        if row["pair_id"] == winner["pair_id"] and row["arm"] == "treatment"
    ]
    if len(treatment_outcomes) != 1:
        raise RuntimeError("winning treatment response lineage is incomplete")
    treatment = treatment_outcomes[0]
    state = states[str(winner["state_id"])]
    catalog = backends[state.card_id]
    selected_ids = [str(value) for value in treatment["selected_memory_ids"]]
    evidence = []
    private_evidence = []
    for index, memory_id in enumerate(selected_ids, 1):
        item = catalog[memory_id]
        relative_age = int(state.session_index) - int(item.created_session)
        if relative_age <= 0:
            raise RuntimeError("risk packet contains non-prior memory")
        evidence_id = f"evidence_{index:02d}"
        evidence.append(
            {
                "evidence_id": evidence_id,
                "relative_age_sessions": relative_age,
                "content": item.text,
            }
        )
        private_evidence.append(
            {
                "evidence_id": evidence_id,
                "memory_id": memory_id,
                "source": item.source.value,
                "created_session": int(item.created_session),
                "relative_age_sessions": relative_age,
                "content_sha256": sha256_text(item.text),
            }
        )
    candidate_response = str(treatment["response"])
    if candidate_response not in {packet["response_a"], packet["response_b"]}:
        raise RuntimeError("winning response is absent from the blind packet")
    review_id = "d3ms_risk_" + stable_hex(PROTOCOL, winner["pair_id"], n=20)
    item = {
        "protocol": PROTOCOL,
        "review_item_id": review_id,
        "current_session_summary": packet.get("current_session_summary") or "",
        "recent_dialogue": packet["recent_dialogue"],
        "current_user_text": packet["current_user_text"],
        "authorized_auxiliary_evidence": evidence,
        "candidate_response": candidate_response,
    }
    risk_module = _risk_module()
    private = {
        "protocol": PROTOCOL,
        "review_item_id": review_id,
        "pair_id": winner["pair_id"],
        "contrast_slot_id": winner["contrast_slot_id"],
        "state_id": winner["state_id"],
        "user_id": winner["user_id"],
        "component": "MS",
        "candidate_arm": "treatment",
        "candidate_response_sha256": sha256_text(candidate_response),
        "authorized_evidence": private_evidence,
        "risk_applicability": risk_module._risk_applicability(
            selected_memory_count=len(selected_ids),
            current_user_text=str(packet["current_user_text"]),
        ),
        "quality_label": "ON_QUALITY_WIN_PENDING_MATERIAL_RISK",
    }
    checks = {
        "exactly_one_quality_winner": len(winners) == 1,
        "winner_is_ms_treatment": winner["component"] == "MS"
        and treatment["arm"] == "treatment",
        "candidate_response_lineage_matches": sha256_text(candidate_response)
        == private["candidate_response_sha256"],
        "authorized_evidence_nonempty": bool(evidence),
        "authorized_evidence_strictly_prior": all(
            row["relative_age_sessions"] > 0 for row in evidence
        ),
        "no_quality_rejudgment_requested": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"MS risk packet checks failed: {checks}")
    manifest = {
        "protocol": PROTOCOL,
        "status": STATUS,
        "items": 1,
        "risk_construct": "interaction-and-grounding risk proxy",
        "not_clinical_safety": True,
        "quality_not_rescored": True,
        "risk_categories": risk_module.RISK_CATEGORIES,
        "checks": checks,
        "final_training_authorized": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    public_path = out_dir / "human_risk_packet.jsonl"
    private_path = out_dir / "private_risk_key.jsonl"
    blank_path = out_dir / "blank_risk_annotations.jsonl"
    html_path = out_dir / "human_risk_review.html"
    write_jsonl(public_path, [item])
    write_jsonl(private_path, [private])
    write_jsonl(
        blank_path,
        [
            {
                "protocol": PROTOCOL,
                "review_item_id": review_id,
                "any_material_risk": None,
                "selected_categories": [],
                "evidence_by_category": {},
                "risk_notes": "",
                "annotator_id": "",
            }
        ],
    )
    html_text = risk_module._render_html(manifest, [item])
    html_text = html_text.replace(
        "pm_v15_transport_on_win_minimal_risk_v1",
        "pm_v15_d3_ms_on_win_minimal_risk_v1",
    ).replace(
        "transport_component_on_win_risk_annotations.jsonl",
        "d3_ms_on_win_risk_annotations.jsonl",
    )
    html_path.write_text(html_text, encoding="utf-8")
    manifest["outputs"] = {
        path.name: sha256_file(path)
        for path in (public_path, private_path, blank_path, html_path)
    }
    manifest["aggregation_report_sha256"] = sha256_file(
        aggregation_dir / "aggregation_report.json"
    )
    manifest["generation_outcomes_sha256"] = sha256_file(
        execution_dir / "generation_outcomes.jsonl"
    )
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregation-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_d3_ms_grounded_quality_aggregation_v1",
    )
    parser.add_argument(
        "--blind-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_d3_ms_replacement_blind_v1",
    )
    parser.add_argument(
        "--blueprint-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_d3_ms_replacement_step0_v1",
    )
    parser.add_argument(
        "--execution-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_d3_ms_replacement_generation_v1_execution",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_d3_ms_on_win_risk_review_v1_candidate",
    )
    args = parser.parse_args()
    manifest = build(
        aggregation_dir=args.aggregation_dir,
        blind_dir=args.blind_dir,
        blueprint_dir=args.blueprint_dir,
        execution_dir=args.execution_dir,
        out_dir=args.out_dir,
    )
    print(
        json.dumps(
            {key: manifest[key] for key in ("protocol", "status", "items")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
