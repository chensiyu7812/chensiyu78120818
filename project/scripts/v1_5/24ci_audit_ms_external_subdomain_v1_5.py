#!/usr/bin/env python3
"""Audit the V1.5 MS construct and internal-superdomain/EvoEmo-subdomain claim.

This audit is deliberately outcome blind.  It reads frozen runtime/catalog
artifacts and response-arm lineage, but never reads generated responses,
quality labels, risk labels, PM predictions, or external outcomes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from metacom_pm.contracts import MemorySource
from metacom_pm.io import iter_jsonl, write_json
from metacom_pm.pm_v2_data import load_bundles
from metacom_pm.v1_5_memory_transport import (
    compile_bounded_memory,
    synthetic_basic_info,
    synthetic_prior_session,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-ms-construct-and-external-subdomain-audit-v1"

BUNDLES = (
    ROOT
    / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate"
    / "pm_v2_bundles.jsonl"
)
RUNTIME_STATES = (
    ROOT
    / "data/pm_v1_5_evo_style_synthetic_memory_v1_candidate"
    / "runtime_states.jsonl"
)
ARM_REGISTRY = (
    ROOT
    / "outputs/pm_v1_5_transport_repaired_memory_generation_v1"
    / "response_arm_registry.jsonl"
)
EVOEMO = ROOT / "data/external/evo_emo.json"
MIMIC_AUDIT = (
    ROOT
    / "outputs/pm_v1_5_evo_mimic_equivalence_audit_v1"
    / "equivalence_audit.json"
)
PM_CONFIG = ROOT / "configs/pm_v1_5.yaml"
OUTPUT = (
    ROOT
    / "outputs/pm_v1_5_ms_external_subdomain_audit_v1"
    / "ms_external_subdomain_audit.json"
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluation_turn_indices(config_text: str) -> list[int]:
    marker = "turn_indices:"
    for line in config_text.splitlines():
        if marker not in line:
            continue
        suffix = line.split(marker, 1)[1].strip()
        if suffix.startswith("[") and suffix.endswith("]"):
            return [int(value.strip()) for value in suffix[1:-1].split(",")]
    raise RuntimeError("external evaluation turn_indices not found")


def _internal_summary_origin() -> tuple[dict[tuple[str, str], str], dict[str, Any]]:
    origin: dict[tuple[str, str], str] = {}
    supplied = 0
    fallback = 0
    sessions = 0
    cases = 0
    users = load_bundles(BUNDLES)
    for bundle in users:
        ordered = sorted(bundle.cases, key=lambda case: case.session_index)
        cases += len(ordered)
        session_rows = [synthetic_prior_session(case) for case in ordered]
        items, _ = compile_bounded_memory(
            {
                "id": bundle.user_id,
                "basic_info": synthetic_basic_info(bundle),
                "dialog_history": session_rows,
            }
        )
        for item in items:
            if item.source is not MemorySource.MS:
                continue
            case = ordered[int(item.created_session) - 1]
            kind = (
                "supplied_session_summary"
                if case.session_summary.strip()
                else "fallback_current_user_text"
            )
            origin[(bundle.user_id, item.memory_id)] = kind
            supplied += int(kind == "supplied_session_summary")
            fallback += int(kind == "fallback_current_user_text")
            sessions += 1
    return origin, {
        "users": len(users),
        "current_states": cases,
        "compiled_session_summaries": sessions,
        "supplied_session_summaries": supplied,
        "fallback_current_user_text": fallback,
        "fallback_rate": fallback / sessions if sessions else None,
    }


def _old_ms_pair_audit(
    origin: dict[tuple[str, str], str],
) -> dict[str, Any]:
    states = {
        str(row["state_id"]): row for row in iter_jsonl(RUNTIME_STATES)
    }
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in iter_jsonl(ARM_REGISTRY):
        if row["component"] == "MS":
            pairs[str(row["contrast_slot_id"])][str(row["arm"])] = row

    counts: Counter[str] = Counter()
    for contrast_slot_id, pair in pairs.items():
        if set(pair) != {"control", "treatment"}:
            raise RuntimeError(f"incomplete MS pair: {contrast_slot_id}")
        control = pair["control"]
        treatment = pair["treatment"]
        if control["state_id"] != treatment["state_id"]:
            raise RuntimeError("MS contrast changes state")
        added = set(treatment["selected_memory_ids"]) - set(
            control["selected_memory_ids"]
        )
        kinds = [
            origin.get((str(treatment["user_id"]), str(memory_id)))
            for memory_id in sorted(added)
        ]
        if not kinds or any(kind is None for kind in kinds):
            raise RuntimeError("MS treatment item origin cannot be resolved")
        state = states[str(treatment["state_id"])]
        has_fallback = "fallback_current_user_text" in kinds
        current_summary_present = bool(
            str(state.get("current_session_summary") or "").strip()
        )
        counts["pairs"] += 1
        counts["selected_ms_items"] += len(kinds)
        counts["pairs_with_fallback_item"] += int(has_fallback)
        counts["pairs_with_only_supplied_summaries"] += int(not has_fallback)
        counts["pairs_with_current_session_summary"] += int(
            current_summary_present
        )
        counts["pairs_with_empty_current_session_summary"] += int(
            not current_summary_present
        )
        counts["pairs_clean_under_narrowed_v1_5_contract"] += int(
            not has_fallback and not current_summary_present
        )
    return dict(counts)


def _external_profile() -> dict[str, Any]:
    users = _load_json(EVOEMO)
    sessions = [
        session
        for user in users
        for session in (user.get("dialog_history") or [])
    ]
    nonempty = sum(
        bool(str(session.get("summary") or "").strip()) for session in sessions
    )
    return {
        "users": len(users),
        "sessions": len(sessions),
        "nonempty_supplied_session_summaries": nonempty,
        "supplied_summary_rate": nonempty / len(sessions) if sessions else None,
        "current_session_summary_runtime_policy": "always_empty",
    }


def build_audit() -> dict[str, Any]:
    origin, internal = _internal_summary_origin()
    old_pairs = _old_ms_pair_audit(origin)
    external = _external_profile()
    mimic = _load_json(MIMIC_AUDIT)
    turn_indices = _evaluation_turn_indices(PM_CONFIG.read_text(encoding="utf-8"))
    visible_history = []
    for turn_index in turn_indices:
        prior_messages = 2 * int(turn_index) - 2
        retained = min(8, prior_messages)
        visible_history.append(
            {
                "evaluation_seeker_turn": int(turn_index),
                "prior_messages_before_current": prior_messages,
                "retained_by_current_runtime": retained,
                "omitted_by_current_runtime": prior_messages - retained,
                "current_session_summary_present": False,
            }
        )

    descriptor_support = dict(
        mimic["candidate_support_gate"]["observed_support_by_source"]
    )
    exact_checks = list(mimic["exact_mechanism_checks"])
    exact_pass = all(row["status"] == "PASS" for row in exact_checks)

    return {
        "protocol": PROTOCOL,
        "status": "D1B_REQUIRED_BEFORE_D2_OR_RETRAINING",
        "api_calls_made": 0,
        "human_decisions_made": 0,
        "outcomes_read": {
            "generated_responses": False,
            "quality_labels": False,
            "risk_labels": False,
            "pm_predictions": False,
            "external_outcomes": False,
        },
        "historical_ontology": {
            "current_session_context": (
                "Visible current-session dialogue and optional token-budget "
                "compaction; fixed across actions and not an MP/MS/ME resource."
            ),
            "MS_SESSION": (
                "A bounded summary of one strictly completed prior session."
            ),
            "MS_PATTERN": (
                "A cross-session pattern, recurring concern, or state evolution "
                "supported by at least two strictly prior sessions."
            ),
            "v1_5_scope": "MS_SESSION only",
            "v2_reserved": "MS_PATTERN",
            "action_decision": (
                "Keep one MS bit and 16 total actions. If MS_PATTERN is later "
                "implemented, treat it as a typed candidate under the same bit "
                "unless simultaneous independent injection is required."
            ),
        },
        "implementation_findings": {
            "current_ms_runtime": (
                "One MemoryItem per completed prior session. Supplied summary "
                "is used when available; otherwise current code copies the last "
                "seeker message/current_user_text into MS."
            ),
            "cross_session_pattern_compiler_implemented": False,
            "internal": internal,
            "external": external,
            "external_visible_current_session": visible_history,
            "old_ms_pair_lineage": old_pairs,
        },
        "severity_decision": {
            "severity": "HIGH",
            "confidence": "HIGH",
            "why": [
                "59 of 64 old MS contrasts inject at least one fallback utterance mislabeled as a summary.",
                "Only four old MS contrasts combine supplied-only MS with an empty current-session summary.",
                "Internal current-session summary presence and EvoEmo always-empty policy are not transport-equivalent.",
                "At external evaluation turn 8, six earlier messages are omitted and no current-session summary compensates.",
                "The implemented resource cannot support a claim about learned cross-session patterns."
            ],
        },
        "external_subdomain_contract": {
            "relationship": (
                "EvoEmo content is disjoint external data, while its resource "
                "ontology and pre-outcome candidate feature support must be a "
                "covered subdomain of the broader internal controlled environment."
            ),
            "must_match_exactly": [
                "MemoryItem schema and resource ontology",
                "causal prior-history rule",
                "compiler semantics",
                "query builder",
                "retriever/filter/Top-k",
                "candidate descriptor",
                "PM decision timing",
                "prompt compiler and generator treatment"
            ],
            "must_be_content_disjoint": [
                "user identity",
                "dialogue and memory text",
                "external response-quality and risk outcomes"
            ],
            "current_exact_mechanism_pass": exact_pass,
            "current_descriptor_support": descriptor_support,
            "descriptor_gate_minimum": 0.80,
            "descriptor_target": 0.90,
            "semantic_subtype_gate": "NOT_YET_PASSED",
            "reporting": [
                "Report all external states unweighted.",
                "Also report in-support and OOD strata.",
                "OOD candidates deterministically abstain/off; OOD rows are never deleted from the all-state result.",
                "EvoEmo is development-informed external because its outcome-free schema and descriptor distribution have already been inspected."
            ],
        },
        "required_repairs_before_D2": [
            "Freeze V1.5 MS to qualified MS_SESSION and remove MS_PATTERN language from the V1.5 claim.",
            "Forbid last-user-message fallback from qualifying as MS.",
            "Make the current-session visible-state policy identical internally and externally.",
            "For the bounded ten-turn EvoEmo experiment, retain the complete current-session dialogue and use no current-session summary.",
            "Downgrade all 64 old MS pairs to diagnosis; do not select the four surviving pairs after inspecting this audit.",
            "Create eight fresh MS_SESSION states with supplied, causal, auditable summaries before D2."
        ],
        "revised_D2_budget": {
            "RS_MP_ME_existing_states": 24,
            "MS_fresh_states": 8,
            "pairs_per_existing_state": 2,
            "pairs_per_fresh_ms_state": 3,
            "new_response_api_calls": 144,
            "primary_reviewer_pair_decisions": 72,
            "second_reviewer_fixed_overlap_decisions": 32,
            "total_human_decisions": 104,
            "difference_from_prior_D2": {
                "response_api_calls": 16,
                "human_decisions": 8
            },
        },
        "claim_boundary": {
            "allowed": (
                "The internal controlled environment covers the resource types "
                "and decision conditions tested by EvoEmo while adding internal-only "
                "tests for preference memory, misuse, subtype competition, and all "
                "16 actions."
            ),
            "forbidden": [
                "EvoEmo content is a subset of internal training content.",
                "The current V1.5 MS implementation learns cross-session patterns.",
                "Descriptor min/max support proves semantic transport.",
                "Internal-only preference results are externally validated by EvoEmo."
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = build_audit()
    write_json(args.output, report)
    print(
        {
            "protocol": report["protocol"],
            "status": report["status"],
            "old_ms_pairs": report["implementation_findings"][
                "old_ms_pair_lineage"
            ]["pairs"],
            "fallback_affected_pairs": report["implementation_findings"][
                "old_ms_pair_lineage"
            ]["pairs_with_fallback_item"],
            "new_D2_calls": report["revised_D2_budget"][
                "new_response_api_calls"
            ],
            "output": str(args.output),
        }
    )


if __name__ == "__main__":
    main()
