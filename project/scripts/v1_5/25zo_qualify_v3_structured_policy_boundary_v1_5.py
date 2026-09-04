#!/usr/bin/env python3
"""Qualify objective pre-Step1 denials while preserving semantic uncertainty."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl
from metacom_pm.v1_5_observation_contract import (
    StructuredCandidateMetadata,
    structured_policy_observation_v4,
)


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v3-structured-policy-boundary-qualification-v1"


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
        "--contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/v3_observation_structured_hard_gate_amendment_v2.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_structured_policy_boundary_v1",
    )
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if sha256_file(args.candidates) != contract["trigger"]["candidate_rows_sha256"]:
        raise RuntimeError("candidate rows changed after structured-boundary freeze")
    rows = [dict(row) for row in iter_jsonl(args.candidates)]
    if len(rows) != 640:
        raise RuntimeError("qualification requires the complete 640-state surface")

    by_track: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_component_track: defaultdict[str, Counter[str]] = defaultdict(Counter)
    semantic: defaultdict[str, defaultdict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    audit_rows: list[dict[str, Any]] = []
    provenance_complete = 0
    for row in rows:
        component = str(row["target_component_private_not_model_input"])
        track = str(row["track_private_not_model_input"])
        exact = dict(row["exact_rank1_candidate"])
        raw = dict(row["structured_candidate_metadata"])
        complete = all(
            raw.get(key) is not None
            for key in (
                "candidate_present",
                "state_owner_id",
                "candidate_owner_id",
                "candidate_active",
                "candidate_superseded",
                "candidate_content_sha256",
            )
        )
        provenance_complete += int(complete)
        observation = structured_policy_observation_v4(
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
        decision = "admit_to_step1" if observation.step1_candidate_admissible else "hard_off"
        by_track[track][decision] += 1
        by_component_track[f"{track}:{component}"][decision] += 1
        for name, value in observation.semantic_feature_decisions.items():
            semantic[component][name][value] += 1
        audit_rows.append(
            {
                "protocol": PROTOCOL,
                "state_id": str(row["state_id"]),
                "component": component,
                "track": track,
                "hard_gate_decisions": dict(observation.hard_gate_decisions),
                "semantic_feature_decisions": dict(
                    observation.semantic_feature_decisions
                ),
                "step1_candidate_admissible": observation.step1_candidate_admissible,
                "hard_denial_reasons": list(observation.hard_denial_reasons),
                "semantic_unknown_is_gold": False,
                "human_label_read": False,
                "response_or_outcome_read": False,
                "external_lockbox_read": False,
            }
        )

    checks = {
        "640_complete_provenance_rows": provenance_complete == 640,
        "all_512_effect_rows_reach_step1": by_track["COMPONENT_EFFECT"]["admit_to_step1"] == 512,
        "no_effect_row_hard_off": by_track["COMPONENT_EFFECT"]["hard_off"] == 0,
        "each_component_has_128_effect_rows": all(
            by_component_track[f"COMPONENT_EFFECT:{component}"]["admit_to_step1"] == 128
            for component in ("MP", "MS", "ME", "RS")
        ),
        "eligibility_audit_contains_hard_off_controls": by_track["ELIGIBILITY_AUDIT"]["hard_off"] > 0,
        "no_human_or_outcome_or_external_read": True,
    }
    status = "PASS_H_STEP2_PREPARATION_AUTHORIZED" if all(checks.values()) else "FAIL"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.out_dir / "policy_observation_rows.jsonl"
    write_jsonl(audit_path, audit_rows)
    report = {
        "protocol": PROTOCOL,
        "status": status,
        "checks": checks,
        "candidate_rows_sha256": sha256_file(args.candidates),
        "contract_sha256": sha256_file(args.contract),
        "policy_observation_rows_sha256": sha256_file(audit_path),
        "provenance_complete_rows": provenance_complete,
        "track_decisions": {key: dict(value) for key, value in by_track.items()},
        "component_track_decisions": {
            key: dict(value) for key, value in by_component_track.items()
        },
        "semantic_feature_shape": {
            component: {
                name: dict(values) for name, values in semantic[component].items()
            }
            for component in ("MP", "MS", "ME", "RS")
        },
        "human_labels_read": 0,
        "responses_generated": 0,
        "external_lockbox_read": False,
        "step1_labels_created": 0,
        "scientific_interpretation": (
            "Objective provenance and explicit boundary denials are qualified as "
            "pre-Step1 hard gates. Semantic unknowns remain inputs for effect learning; "
            "this result does not prove Step2 execution or Step1 learning."
        ),
        "next_step": "PREPARE_ONE_VERSIONED_H_STEP2_EXECUTION_QUALIFICATION_PACKET",
    }
    write_json(args.out_dir / "qualification_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
