#!/usr/bin/env python3
"""Profile W8/P2R against the complete formal-data contract (zero API)."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import read_json, write_json


ROOT = Path(__file__).resolve().parents[2]
CATALOG_DIR = ROOT / "outputs/pm_v1_5_v5_3_p2r_longitudinal_catalog_v1"
BLUEPRINT_DIR = ROOT / "outputs/pm_v1_5_v5_3_p2r_learning_blueprint_v1"
AUDIT_DIR = ROOT / "outputs/pm_v1_5_v5_3_p2r_learning_blueprint_audit_v1"
CONTRACT_PATH = (
    ROOT / "data/pm_v1_5_contracts/v5_3_complete_training_data_generation_v1.json"
)
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_w8_complete_contract_gap_audit_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def main() -> None:
    contract = read_json(CONTRACT_PATH)
    catalog = _read_jsonl(CATALOG_DIR / "catalog.jsonl")
    users = read_json(CATALOG_DIR / "users.json")
    catalog_coverage = read_json(CATALOG_DIR / "coverage_report.json")
    blueprint = read_json(BLUEPRINT_DIR / "report.json")
    static_audit = read_json(AUDIT_DIR / "report.json")

    by_subtype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in catalog:
        by_subtype[str(row["subtype"])].append(row)
    subtype_profile = {}
    for subtype, rows in sorted(by_subtype.items()):
        unique = len({_normalized(str(row["text"])) for row in rows})
        subtype_profile[subtype] = {
            "rows": len(rows),
            "unique_texts": unique,
            "unique_text_rate": unique / len(rows),
            "exact_duplicate_rows": len(rows) - unique,
        }

    target_items = contract["catalog_targets"]
    current_counts = Counter(str(row["subtype"]) for row in catalog)
    catalog_gaps = {
        "formal_longitudinal_users": contract["formal_longitudinal_users"] - len(users),
        "sessions": target_items["sessions"]
        - sum(int(row["n_sessions"]) for row in users),
        "base_and_update_mp_profile_items": (
            target_items["base_mp_profile_items"] + target_items["profile_update_items"]
            - current_counts["MP_PROFILE"]
        ),
        "mp_preference_items": target_items["mp_preference_items"]
        - current_counts["MP_PREFERENCE"],
        "ms_session_items": target_items["ms_session_items"]
        - current_counts["MS_SESSION"],
        "me_reusable_outcome_items": target_items["me_reusable_outcome_items"]
        - current_counts["ME_REUSABLE_OUTCOME"],
        "me_unresolved_event_items": target_items["me_unresolved_event_items"]
        - current_counts["ME_UNRESOLVED_EVENT"],
        "me_context_event_items": target_items["me_context_event_items"]
        - current_counts["ME_CONTEXT_EVENT"],
    }
    target_states = contract["current_state_targets"]
    component_counts = blueprint["component_counts"]
    state_gaps = {
        "MP": target_states["mp_states_total"] - int(component_counts["MP"]),
        "MS": target_states["ms_states_total"] - int(component_counts["MS"]),
        "ME": target_states["me_states_total"] - int(component_counts["ME"]),
        "RS": target_states["rs_states_total"] - int(component_counts["RS"]),
        "interaction": target_states["formal_interaction_states_total"]
        - int(blueprint["interaction_count"]),
    }

    report = {
        "protocol": "pm-v1.5-v5.3-w8-complete-contract-gap-audit-v1",
        "status": "W8_VALID_DEVELOPMENT_CATALOG_FORMAL_TRAINING_DATA_INCOMPLETE",
        "grain": {
            "catalog": "typed longitudinal resource item",
            "blueprint": "component current state",
            "formal_split": "user cluster",
        },
        "w8": {
            "users": len(users),
            "sessions": sum(int(row["n_sessions"]) for row in users),
            "catalog_items": len(catalog),
            "unique_catalog_texts": len({_normalized(str(row["text"])) for row in catalog}),
            "unique_catalog_text_rate": len(
                {_normalized(str(row["text"])) for row in catalog}
            )
            / len(catalog),
            "subtype_text_profile": subtype_profile,
            "owner_isolation": catalog_coverage["owner_isolation"],
            "external_overlap": catalog_coverage["external_overlap"],
        },
        "p2r_blueprint": {
            "states": blueprint["row_count"],
            "user_clusters": blueprint["unique_user_clusters"],
            "counterfactual_groups": blueprint["unique_counterfactual_groups"],
            "component_counts": component_counts,
            "interactions": blueprint["interaction_count"],
            "static_audit_status": static_audit["status"],
            "static_blockers": static_audit["blockers"],
            "transparent_observability": static_audit["transparent_observability"],
        },
        "formal_contract_gaps": {
            "catalog": catalog_gaps,
            "current_states": state_gaps,
            "paired_effect_labels": contract["formal_paired_effect_subset"][
                "effect_states_total"
            ],
            "formal_user_split_frozen": False,
            "full_chain_step1_trained": False,
        },
        "verdict": {
            "longitudinal_catalog_structure_usable": True,
            "formal_content_diversity_sufficient": False,
            "formal_independent_user_count_sufficient": False,
            "formal_paired_effect_supervision_exists": False,
            "ready_to_claim_pm_learned": False,
        },
        "api_calls": 0,
        "quality_risk_or_external_outcome_read": False,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
