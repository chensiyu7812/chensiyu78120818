#!/usr/bin/env python3
"""Audit V3 structured-gate coverage without labels, outcomes, or external data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl
from metacom_pm.v1_5_observation_contract import (
    StructuredCandidateMetadata,
    structured_hard_gate_observation_v3,
)


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v3-structured-gate-coverage-audit-v1"


def main() -> None:
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V3 requires {FORMAL_PYTHON}; got {sys.executable}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_v3_p2_exact_rank1_v4/candidate_rows_private.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_structured_gate_coverage_v1",
    )
    args = parser.parse_args()
    rows = [dict(row) for row in iter_jsonl(args.candidates)]
    if len(rows) != 640:
        raise RuntimeError("structured coverage audit requires all 640 V3 states")

    outcomes: defaultdict[str, Counter[str]] = defaultdict(Counter)
    gates: defaultdict[str, defaultdict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        component = str(row["target_component_private_not_model_input"])
        exact = dict(row["exact_rank1_candidate"])
        raw = dict(row["structured_candidate_metadata"])
        observation = structured_hard_gate_observation_v3(
            state_id=str(row["state_id"]),
            component=component,
            current_user_text=str(row["current_user_text"]),
            visible_dialogue=list(row["visible_dialogue"]),
            candidate_text=str(exact.get("candidate_text") or ""),
            candidate_subtype=str(exact["compiler_subtype_hint"]),
            metadata=StructuredCandidateMetadata(
                candidate_present=bool(raw["candidate_present"]),
                state_owner_id=str(raw["state_owner_id"]),
                candidate_owner_id=str(raw["candidate_owner_id"]),
                candidate_active=bool(raw["candidate_active"]),
                candidate_superseded=bool(raw["candidate_superseded"]),
                candidate_content_sha256=str(raw["candidate_content_sha256"]),
            ),
        )
        outcome = "eligible" if observation.runtime_eligible else "off"
        outcomes[component][outcome] += 1
        outcomes[component][
            "unknown_off" if observation.unknown_forced_off else "fully_known"
        ] += 1
        for gate, decision in observation.gate_decisions.items():
            gates[component][gate][decision] += 1
        audit_rows.append(
            {
                "protocol": PROTOCOL,
                "state_id": str(row["state_id"]),
                "component": component,
                "gate_decisions": dict(observation.gate_decisions),
                "runtime_eligible": observation.runtime_eligible,
                "unknown_forced_off": observation.unknown_forced_off,
                "human_label_read": False,
                "construction_intent_read": False,
                "response_or_outcome_read": False,
                "external_lockbox_read": False,
            }
        )

    zero_coverage_components = [
        component
        for component in ("MP", "MS", "ME", "RS")
        if outcomes[component]["eligible"] == 0
    ]
    status = (
        "FAIL_DEGENERATE_ZERO_COMPONENT_COVERAGE"
        if zero_coverage_components
        else "PASS_NONDEGENERATE_COVERAGE"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.out_dir / "audit_rows.jsonl"
    write_jsonl(audit_path, audit_rows)
    report = {
        "protocol": PROTOCOL,
        "status": status,
        "candidate_rows": len(rows),
        "candidate_rows_sha256": sha256_file(args.candidates),
        "audit_rows_sha256": sha256_file(audit_path),
        "per_component_outcomes": {
            component: dict(outcomes[component])
            for component in ("MP", "MS", "ME", "RS")
        },
        "per_component_gate_decisions": {
            component: {
                gate: dict(counts)
                for gate, counts in gates[component].items()
            }
            for component in ("MP", "MS", "ME", "RS")
        },
        "zero_coverage_components": zero_coverage_components,
        "human_labels_read": 0,
        "construction_intent_read": False,
        "responses_generated": 0,
        "external_lockbox_read": False,
        "scientific_interpretation": (
            "Unknown-to-OFF is not a viable pre-Step1 policy on the complete V3 "
            "development decision surface. Only objective hard denials may bypass "
            "Step1; unresolved goal or increment must remain observable uncertainty "
            "for component-effect learning rather than become a fabricated negative."
        ),
    }
    write_json(args.out_dir / "coverage_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
