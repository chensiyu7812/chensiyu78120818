#!/usr/bin/env python3
"""Audit MP subtype coverage and freeze a V1.5-compatible repair."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-mp-subtype-coverage-and-action-design-audit-v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _normalized_preference(text: str) -> str:
    value = re.sub(r"^Stable Preference \d+:\s*", "", text)
    value = re.sub(
        r"about .*?, with one gentle question at a time$",
        "about <TOPIC>, with one gentle question at a time",
        value,
    )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory-backend",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_evo_style_synthetic_memory_v1_candidate/"
        "memory_backend.jsonl",
    )
    parser.add_argument(
        "--transport-audit",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_evo_mimic_equivalence_audit_v1/"
        "equivalence_audit.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_mp_subtype_design_audit_v1",
    )
    args = parser.parse_args()

    backend = _rows(args.memory_backend)
    transport = read_json(args.transport_audit)
    mp_items = [
        item
        for row in backend
        for item in row["items"]
        if item["source"] == "MP"
    ]
    texts = [str(item["text"]) for item in mp_items]
    normalized = {_normalized_preference(text) for text in texts}
    profiles = transport["content_profiles"]
    internal_fields = profiles["repaired_internal"]["profile_field_names"]
    external_fields = profiles["evoemo_external"]["profile_field_names"]

    report = {
        "protocol": PROTOCOL,
        "status": (
            "KEEP_16_ACTIONS_USE_TYPED_MP_AND_INTERNAL_COMPLEMENTARY_TESTS"
        ),
        "correction_to_prior_interpretation": (
            "MP was historically defined as stable profile, preference, or "
            "boundary memory. Internal preference rows and EvoEmo profile "
            "facts are different valid subtypes, not evidence that preference "
            "memory should be removed."
        ),
        "observed_coverage": {
            "historical_mp_ontology": (
                "stable profile, preference, or boundary memory"
            ),
            "internal": {
                "subtype": "MP_PREFERENCE",
                "state_item_records": len(mp_items),
                "unique_memory_ids": len(
                    {str(item["memory_id"]) for item in mp_items}
                ),
                "unique_exact_texts": len(set(texts)),
                "normalized_preference_templates": len(normalized),
                "normalized_templates": sorted(normalized),
                "all_acknowledgement_before_suggestions": all(
                    "prefers acknowledgement before suggestions" in text
                    for text in texts
                ),
                "all_one_gentle_question": all(
                    "with one gentle question at a time" in text
                    for text in texts
                ),
                "profile_fields": internal_fields,
            },
            "evoemo": {
                "subtype": "MP_PROFILE",
                "items": int(
                    profiles["evoemo_external"]["item_tokens"]["MP"]["n"]
                ),
                "profile_fields": external_fields,
                "support_preference_fields": [],
            },
            "field_intersection": sorted(
                set(internal_fields) & set(external_fields)
            ),
            "quality_problem": (
                "Subtype support is disjoint and the internal preference "
                "subtype has near-zero behavioral diversity."
            ),
        },
        "design_options": [
            {
                "option": "replace_internal_preferences_with_evoemo_profiles",
                "decision": "reject",
                "reason": (
                    "Deletes a plausible response-quality contribution and "
                    "abandons the project's original MP ontology."
                ),
            },
            {
                "option": "add_a_fifth_MP_profile_or_preference_action_bit",
                "decision": "defer_to_v2",
                "reason": (
                    "Expands 16 actions to 32, adds a fifth effect head and "
                    "new interactions, and materially increases the V1.5 data "
                    "burden without being necessary when runtime injects one "
                    "qualified MP candidate."
                ),
            },
            {
                "option": "typed_MP_catalog_single_MP_action_bit",
                "decision": "adopt_for_v1_5",
                "reason": (
                    "Preserves the 16 legal actions while letting retrieval, "
                    "hard gates, PM features, and evaluation distinguish "
                    "preference from profile candidates."
                ),
            },
        ],
        "typed_mp_contract": {
            "action_bit": "MP",
            "subtypes": {
                "MP_PREFERENCE": (
                    "An explicit, stable preference about support pacing, "
                    "tone, sequencing, permission, format, or use of history."
                ),
                "MP_PROFILE": (
                    "A stable demographic or life fact that can directly and "
                    "non-stereotypically change the current response."
                ),
            },
            "preference_strength": [
                "HARD_DURABLE_CONTROL",
                "SOFT_CONTEXTUAL_STYLE",
            ],
            "runtime": {
                "current_turn_boundary": (
                    "Always obey from visible context; it is not an MP "
                    "retrieval benefit."
                ),
                "hard_durable_control": (
                    "Compile only explicit, active, nonconflicting settings "
                    "into a bounded deterministic preference contract outside "
                    "the optional MP action."
                ),
                "soft_contextual_style_and_profile": (
                    "Enter the typed MP candidate pool; retrieve and filter "
                    "Top-3, select at most one candidate, then let the single "
                    "MP effect head decide injection."
                ),
                "current_turn_overrides_stored": True,
                "conflict_or_revocation": "hard off; do not inject",
                "wrong_user": "hard off; do not inject",
                "default_optional_mp_injection": False,
                "maximum_optional_mp_items_injected": 1,
            },
            "pm_features": [
                "candidate_state_match_score",
                "candidate_grounding_or_nonredundancy_score",
                "candidate_is_preference",
                "background_MS_on",
                "background_ME_on",
                "background_RS_on",
            ],
            "feature_count": 6,
            "capacity_requirement": (
                "At least 48 independent development users permits up to 9 "
                "features under floor(groups/5); six remains compliant."
            ),
        },
        "preference_diversity_blueprint": [
            "listen_or_acknowledge_before_advice",
            "ask_permission_before_suggestions",
            "one_question_or_one_task_at_a_time",
            "brief_concise_vs_more_elaborated_support",
            "reflection_before_further_exploration",
            "options_and_choice_vs_directives",
            "direct_gentle_or_neutral_tone",
            "permission_to_reference_prior_sessions_or_sensitive_topics",
        ],
        "internal_development_requirement": {
            "new_content_disjoint_users": 32,
            "MP_PREFERENCE": {
                "independent_users": 16,
                "on_benefit": 8,
                "nonpositive": 8,
                "preference_families_min": 8,
            },
            "MP_PROFILE": {
                "independent_users": 16,
                "on_benefit": 8,
                "nonpositive": 8,
                "profile_fields": [
                    "age_or_life_stage",
                    "job_or_role",
                    "location_or_living_context",
                    "education_or_training_context",
                    "relationship_or_household_context",
                    "language_or_communication_context",
                ],
            },
            "matched_nonpositive_conditions": [
                "irrelevant but stable",
                "already visible in current context",
                "current request conflicts with stored preference",
                "stale or explicitly revoked",
                "wrong person",
                "sensitive profile without direct present relevance",
                "preference overgeneralized beyond its scope",
                "profile fact would invite stereotyping",
            ],
            "selection_timing": "before any new response outcome",
        },
        "internal_confirmation_requirement": {
            "MP_PREFERENCE": {
                "positive_independent_groups_min": 8,
                "nonpositive_independent_groups_min": 8,
            },
            "MP_PROFILE": {
                "positive_independent_groups_min": 8,
                "nonpositive_independent_groups_min": 8,
            },
            "mixed_catalog_competition_groups_min": 8,
            "mixed_catalog_examples": [
                "preference relevant and profile irrelevant",
                "profile relevant and preference irrelevant",
                "both available but current context already covers one",
                "stored preference contradicted by the current turn",
            ],
        },
        "evaluation_roles": {
            "ESConv": (
                "RS quality and correct memory-off behavior; no claim that "
                "ESConv validates longitudinal preference or profile memory."
            ),
            "EvoEmo": (
                "External MP_PROFILE plus MS/ME transport and retrieval; it "
                "does not validate MP_PREFERENCE when preference fields are "
                "absent."
            ),
            "internal_controlled_environment": (
                "Primary evidence for MP_PREFERENCE, preference misuse, "
                "profile relevance, subtype competition, and all 16 actions."
            ),
            "aggregation_rule": (
                "Report aggregate MP and subtype-conditioned MP separately. "
                "Never let strong internal preference results mask external "
                "profile failure or claim external validation of preference."
            ),
        },
        "risk_measurement": {
            "existing_atomic_parent": (
                "stale_conflicting_wrong-person_memory_use"
            ),
            "MP_subtype_breakdown": [
                "stale_or_revoked_preference_use",
                "preference_scope_overgeneralization",
                "wrong_person_preference_or_profile",
                "sensitive_profile_exposure_or_stereotyping",
                "hard_preference_or_boundary_omission",
            ],
            "training_policy": (
                "These remain deterministic guards and audited events, not a "
                "small-sample learned risk head."
            ),
        },
        "claim_boundary": (
            "The internal controlled environment may establish preference "
            "routing and misuse handling; EvoEmo supplies external evidence "
            "only for the profile subtype. This is complementary evidence, "
            "not a pooled claim that both external datasets test every MP "
            "capability."
        ),
        "lineage": {
            "memory_backend_sha256": sha256_file(args.memory_backend),
            "transport_audit_sha256": sha256_file(args.transport_audit),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "mp_subtype_design_audit.json", report)
    print(
        {
            "status": report["status"],
            "internal_state_item_records": len(mp_items),
            "internal_unique_memory_ids": len(
                {str(item["memory_id"]) for item in mp_items}
            ),
            "normalized_templates": len(normalized),
            "action_count": 16,
            "new_mp_action_bit": False,
            "new_development_users": 32,
        }
    )


if __name__ == "__main__":
    main()
