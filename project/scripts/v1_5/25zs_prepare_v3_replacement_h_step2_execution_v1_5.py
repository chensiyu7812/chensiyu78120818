#!/usr/bin/env python3
"""Build the one eligible-only 32-item replacement V3 H-Step2 plan.

This script is outcome-blind and performs no API calls.  It reuses exact Rank-1
surfaces already frozen by H-Eligibility, removes known unexecutable payloads,
and constructs eight explicitly joint-feasible multi-component states.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
from pathlib import Path
import re
import sys
from typing import Any

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    sha256_file,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)
from metacom_pm.v1_5_generator_alignment_audit import (
    qualify_step2_resource_surface,
    resource_bundle_output_schema_for_components,
)
from metacom_pm.v1_5b_policy_runtime import (
    COMPONENTS as RUNTIME_COMPONENTS,
    ComponentFeasibility,
    project_joint_feasibility,
)


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v3-replacement-h-step2-plan-v1"
COMPONENTS = ("MP", "MS", "ME", "RS")


def _load_legacy_builder() -> Any:
    path = ROOT / "scripts/v1_5/25zp_prepare_v3_h_step2_execution_v1_5.py"
    spec = importlib.util.spec_from_file_location("_v3_h_step2_legacy_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared H-Step2 builder helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY = _load_legacy_builder()


def _require_formal_python() -> None:
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V3 requires {FORMAL_PYTHON}; got {sys.executable}")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _topic(row: dict[str, Any]) -> str:
    match = re.search(
        r"dealing with the (.+?) situation", str(row["current_user_text"]), re.IGNORECASE
    )
    if not match:
        raise RuntimeError(f"cannot recover topic from {row['state_id']}")
    return match.group(1).strip().lower()


def _subtype(row: dict[str, Any]) -> str:
    component = str(row["target_component_private_not_model_input"])
    hint = str(row["exact_rank1_candidate"]["compiler_subtype_hint"])
    return str(LEGACY._resource_subtype(component, hint))


def _eligible_candidates(
    *, reference_path: Path, candidates_path: Path
) -> list[dict[str, Any]]:
    reference = {
        (str(row["state_id"]), str(row["component"])): row
        for row in _rows(reference_path)
        if row["derived_eligibility"] == "eligible"
    }
    candidates = _rows(candidates_path)
    joined: list[dict[str, Any]] = []
    for row in candidates:
        key = (
            str(row["state_id"]),
            str(row["target_component_private_not_model_input"]),
        )
        frozen = reference.get(key)
        if frozen is None:
            continue
        exact = row["exact_rank1_candidate"]
        if (
            str(exact["candidate_id"]) != str(frozen["candidate_id"])
            or str(exact["candidate_text_sha256"])
            != str(frozen["candidate_text_sha256"])
        ):
            raise RuntimeError(f"eligibility/candidate binding changed: {key}")
        qualification = qualify_step2_resource_surface(
            component=key[1],
            resource_subtype=_subtype(row),
            selected_resource=str(exact["candidate_text"]),
            current_user_text=str(row["current_user_text"]),
        )
        joined.append(
            {
                **row,
                "frozen_eligibility": frozen,
                "topic": _topic(row),
                "step2_surface_qualification": qualification,
            }
        )
    if Counter(str(row["target_component_private_not_model_input"]) for row in joined) != Counter(
        {component: 16 for component in COMPONENTS}
    ):
        raise RuntimeError("frozen eligible join is not 16 rows per component")
    return joined


def _pick(
    pool: list[dict[str, Any]],
    *,
    used: set[str],
    component: str,
    topic: str | None = None,
    subtype: str | None = None,
    contains: str | None = None,
) -> dict[str, Any]:
    matches = []
    for row in pool:
        if str(row["state_id"]) in used:
            continue
        if row["target_component_private_not_model_input"] != component:
            continue
        if topic is not None and row["topic"] != topic:
            continue
        if subtype is not None and _subtype(row) != subtype:
            continue
        candidate = str(row["exact_rank1_candidate"]["candidate_text"])
        if contains is not None and contains.lower() not in candidate.lower():
            continue
        matches.append(row)
    matches.sort(key=lambda row: stable_hex(PROTOCOL, "pick", str(row["state_id"]), n=32))
    if not matches:
        raise RuntimeError(
            f"no unused eligible candidate: {component=} {topic=} {subtype=} {contains=}"
        )
    chosen = matches[0]
    used.add(str(chosen["state_id"]))
    return chosen


def _lineage(row: dict[str, Any]) -> dict[str, Any]:
    exact = row["exact_rank1_candidate"]
    return {
        "candidate_id": str(exact["candidate_id"]),
        "candidate_text_sha256": str(exact["candidate_text_sha256"]),
        "source_state_id": str(row["state_id"]),
        "selected_rank": 1,
        "eligibility": "eligible",
        "eligibility_reference_protocol": str(row["frozen_eligibility"]["protocol"]),
        "source": "v3_exact_rank1_materialization_v3",
    }


def _multi_specifications() -> list[dict[str, Any]]:
    return [
        {
            "components": ("MP", "MS"),
            "topic": "project deadline",
            "current": (
                "The project deadline pressure is back. Please remind me what narrow goal "
                "we kept in the earlier conversation, then keep the response concise with "
                "one reflection before at most one focused question."
            ),
            "selectors": {
                "MP": {"topic": "project deadline", "subtype": "MP_PREFERENCE"},
                "MS": {"topic": "project deadline", "contains": "goal was"},
            },
        },
        {
            "components": ("MP", "ME"),
            "topic": "shift sleep",
            "current": (
                "The shift sleep problem is pressing again. I am open to one optional, "
                "realistic step based on something that helped before, and it needs to fit "
                "my actual circumstances."
            ),
            "selectors": {
                "MP": {"topic": "shift sleep", "subtype": "MP_PROFILE"},
                "ME": {"topic": "shift sleep"},
            },
        },
        {
            "components": ("MS", "ME"),
            "topic": "relationship routine",
            "current": (
                "The relationship routine issue is back, and I cannot separate emotional "
                "uncertainty from practical workload. Please use the earlier distinction "
                "tentatively, and I am open to one optional approach that helped before."
            ),
            "selectors": {
                "MS": {"topic": "relationship routine", "contains": "rather than"},
                "ME": {"topic": "relationship routine"},
            },
        },
        {
            "components": ("MP", "MS", "ME"),
            "topic": "city adjustment",
            "current": (
                "The city adjustment issue feels difficult again. Remind me of the specific "
                "workable timing recorded earlier, then offer one concise optional approach "
                "that helped before and still fits my actual circumstances."
            ),
            "selectors": {
                "MP": {"topic": "city adjustment", "subtype": "MP_PROFILE"},
                "MS": {"topic": "city adjustment", "contains": "recorded that"},
                "ME": {"topic": "city adjustment"},
            },
        },
        {
            "components": ("MP", "RS"),
            "topic": "city adjustment",
            "current": (
                "The city adjustment issue is weighing on me. I welcome one reversible "
                "adjustment to the immediate environment, as long as it is realistic for my "
                "actual circumstances and does not become a list."
            ),
            "selectors": {
                "MP": {"topic": "city adjustment", "subtype": "MP_PROFILE"},
                "RS": {"topic": "city adjustment", "contains": "reversible adjustment"},
            },
        },
        {
            "components": ("MS", "RS"),
            "topic": "project deadline",
            "current": (
                "The project deadline pressure is tangled again. Continue the narrow goal "
                "from the earlier conversation and ask exactly one focused question about "
                "the feeling that remains unclear."
            ),
            "selectors": {
                "MS": {"topic": "project deadline", "contains": "goal was"},
                "RS": {"topic": "project deadline", "contains": "focused question"},
            },
        },
        {
            "components": ("ME", "RS"),
            "topic": "relationship routine",
            "current": (
                "In the relationship routine issue, the most pressing concern is fear of "
                "repeating the same conflict. Briefly paraphrase that concern, then offer one "
                "optional approach grounded in something that helped before."
            ),
            "selectors": {
                "ME": {"topic": "relationship routine"},
                "RS": {"topic": "relationship routine", "contains": "Paraphrase"},
            },
        },
        {
            "components": ("MP", "MS", "ME", "RS"),
            "topic": "shift sleep",
            "current": (
                "With the shift sleep issue, I feel torn between protecting my energy and "
                "meeting my responsibilities. Remind me of the workable timing recorded "
                "earlier, tentatively reflect that tension, and offer one optional approach "
                "that helped before while keeping my actual circumstances in view."
            ),
            "selectors": {
                "MP": {"topic": "shift sleep", "subtype": "MP_PROFILE"},
                "MS": {"topic": "shift sleep", "contains": "recorded that"},
                "ME": {"topic": "shift sleep"},
                "RS": {"topic": "family caregiving", "contains": "emotional tension"},
            },
        },
    ]


def _build_multi_calls(
    *, pool: list[dict[str, Any]], used: set[str], generation: SupporterGenerationContract
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for index, spec in enumerate(_multi_specifications(), 1):
        selected: dict[str, dict[str, Any]] = {}
        for component in spec["components"]:
            selected[component] = _pick(
                pool,
                used=used,
                component=component,
                **spec["selectors"][component],
            )
        resources = {
            component: str(row["exact_rank1_candidate"]["candidate_text"])
            for component, row in selected.items()
        }
        subtypes = {component: _subtype(row) for component, row in selected.items()}
        qualifications = {
            component: qualify_step2_resource_surface(
                component=component,
                resource_subtype=subtypes[component],
                selected_resource=resources[component],
                current_user_text=spec["current"],
            )
            for component in spec["components"]
        }
        if not all(row["qualified"] for row in qualifications.values()):
            raise RuntimeError({"multi_index": index, "qualifications": qualifications})

        requested_action = LEGACY._action_id(spec["components"])
        feasibility = {
            component: ComponentFeasibility(
                component=component,
                candidate_present=component in spec["components"],
                hard_gate_pass=component in spec["components"],
                opportunity_score=1.0 if component in spec["components"] else 0.0,
                incremental_tokens=(len(resources[component].split()) if component in resources else 0),
                conflicts_with=frozenset(),
            )
            for component in RUNTIME_COMPONENTS
        }
        projection = project_joint_feasibility(
            requested_action_id=requested_action,
            feasibility=feasibility,
            maximum_incremental_tokens=1000,
        )
        if projection.feasible_action_id != requested_action:
            raise RuntimeError(f"multi state {index} is not jointly feasible")

        state_id = "v3h2r_multi_" + stable_hex(PROTOCOL, index, spec["topic"], n=20)
        call = LEGACY._call(
            call_kind="replacement_multi",
            state_id=state_id,
            user_id="user_" + state_id,
            components=tuple(spec["components"]),
            subtype_by_component=subtypes,
            resources=resources,
            lineage={component: _lineage(row) for component, row in selected.items()},
            visible_dialogue=[
                {
                    "role": "user",
                    "content": f"The {spec['topic']} issue has come up again.",
                },
                {
                    "role": "assistant",
                    "content": "I will not assume the earlier situation is unchanged.",
                },
            ],
            current_user_text=spec["current"],
            generation=generation,
            protocol=PROTOCOL,
        )
        call["step2_surface_qualifications"] = qualifications
        call["joint_feasibility"] = {
            "requested_action_id": projection.requested_action_id,
            "feasible_action_id": projection.feasible_action_id,
            "dropped_components": list(projection.dropped_components),
            "incremental_tokens": projection.incremental_tokens,
        }
        calls.append(call)
    return calls


def _ms_family(row: dict[str, Any]) -> str | None:
    text = str(row["exact_rank1_candidate"]["candidate_text"]).lower()
    if "recorded that" in text:
        return "specific_observation"
    if "rather than" in text:
        return "pressure_distinction"
    if "goal was" in text:
        return "prior_goal"
    return None


def _build_single_calls(
    *, pool: list[dict[str, Any]], used: set[str], generation: SupporterGenerationContract
) -> list[dict[str, Any]]:
    by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        if str(row["state_id"]) in used:
            continue
        if not row["step2_surface_qualification"]["qualified"]:
            continue
        by_component[str(row["target_component_private_not_model_input"])].append(row)
    selected: dict[str, list[dict[str, Any]]] = {}
    for component in COMPONENTS:
        rows = sorted(
            by_component[component],
            key=lambda row: stable_hex(PROTOCOL, "single", component, str(row["state_id"]), n=32),
        )
        if component == "MP":
            chosen = [row for row in rows if _subtype(row) == "MP_PREFERENCE"][:3]
            chosen += [row for row in rows if _subtype(row) == "MP_PROFILE"][:3]
        elif component == "MS":
            families: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                family = _ms_family(row)
                if family:
                    families[family].append(row)
            chosen = sum(
                (families[family][:2] for family in (
                    "specific_observation", "pressure_distinction", "prior_goal"
                )),
                [],
            )
        elif component == "RS":
            chosen = []
            seen_moves: set[str] = set()
            for row in rows:
                move = str(row["exact_rank1_candidate"]["candidate_text"]).splitlines()[0]
                if move not in seen_moves or len(chosen) >= 5:
                    chosen.append(row)
                    seen_moves.add(move)
                if len(chosen) == 6:
                    break
        else:
            chosen = rows[:6]
        if len(chosen) != 6:
            raise RuntimeError(f"replacement singles lack six qualified {component} rows")
        selected[component] = chosen

    calls: list[dict[str, Any]] = []
    for component in COMPONENTS:
        for row in selected[component]:
            used.add(str(row["state_id"]))
            exact = row["exact_rank1_candidate"]
            call = LEGACY._call(
                call_kind="replacement_single",
                state_id=str(row["state_id"]),
                user_id=str(row["user_id_private_not_model_input"]),
                components=(component,),
                subtype_by_component={component: _subtype(row)},
                resources={component: str(exact["candidate_text"])},
                lineage={component: _lineage(row)},
                visible_dialogue=list(row["visible_dialogue"]),
                current_user_text=str(row["current_user_text"]),
                generation=generation,
                protocol=PROTOCOL,
            )
            call["step2_surface_qualifications"] = {
                component: row["step2_surface_qualification"]
            }
            calls.append(call)
    return calls


def build(
    *,
    reference_path: Path,
    candidates_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    _require_formal_python()
    pm_config = load_config(ROOT / "configs/pm_v1_5.yaml")
    experiment = load_config(ROOT / "configs/experiment.yaml")
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, generation.generator_endpoint)
    pool = _eligible_candidates(
        reference_path=reference_path, candidates_path=candidates_path
    )
    used: set[str] = set()
    multi = _build_multi_calls(pool=pool, used=used, generation=generation)
    singles = _build_single_calls(pool=pool, used=used, generation=generation)
    calls = singles + multi
    component_single_counts = Counter(
        call["requested_components"][0]
        for call in singles
        if len(call["requested_components"]) == 1
    )
    checks = {
        "32_unique_calls": len(calls) == 32 and len({call["call_id"] for call in calls}) == 32,
        "24_singles_six_each": len(singles) == 24
        and component_single_counts == Counter({component: 6 for component in COMPONENTS}),
        "8_multi_calls": len(multi) == 8,
        "all_source_rows_frozen_eligible": all(
            lineage["eligibility"] == "eligible"
            for call in calls
            for lineage in call["resource_lineage_private"].values()
        ),
        "all_surfaces_static_qualified": all(
            qualification["qualified"]
            for call in calls
            for qualification in call["step2_surface_qualifications"].values()
        ),
        "all_multi_requested_equals_feasible": all(
            call["joint_feasibility"]["requested_action_id"]
            == call["joint_feasibility"]["feasible_action_id"]
            for call in multi
        ),
        "all_component_lineage_unique": len(
            [
                lineage["source_state_id"]
                for call in calls
                for lineage in call["resource_lineage_private"].values()
            ]
        )
        == len(
            {
                lineage["source_state_id"]
                for call in calls
                for lineage in call["resource_lineage_private"].values()
            }
        ),
        "no_response_or_external_outcome_selection": all(
            not call["selection_or_prompt_uses_external_response_quality_risk_or_judge"]
            for call in calls
        ),
    }
    if not all(checks.values()):
        raise RuntimeError({"replacement_h_step2_static_checks": checks})
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "call_plan_private.jsonl"
    write_jsonl(plan_path, calls)
    primary_input = sum(int(call["input_token_upper_bound"]) for call in calls)
    primary_output = sum(int(call["primary_output_token_cap"]) for call in calls)
    fallback_input = sum(
        int(call["maximum_fallback_input_token_upper_bound"]) for call in calls
    )
    fallback_output = sum(
        int(call["maximum_fallback_output_token_cap"]) for call in calls
    )
    report = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_T1_REPLACEMENT_PAID_GENERATION_REVIEW",
        "generator": endpoint.model,
        "planned_primary_calls": 32,
        "maximum_fallback_calls": 32,
        "primary_input_token_upper_bound": primary_input,
        "primary_output_token_cap_total": primary_output,
        "maximum_fallback_input_token_upper_bound": fallback_input,
        "maximum_fallback_output_token_cap_total": fallback_output,
        "estimated_primary_generation_usd": LEGACY._usd(
            primary_input, primary_output
        ),
        "estimated_generation_usd_with_all_fallbacks": LEGACY._usd(
            primary_input + fallback_input, primary_output + fallback_output
        ),
        "single_component_counts": dict(component_single_counts),
        "multi_action_ids": [call["requested_action_id"] for call in multi],
        "source_eligible_pool_per_component": dict(
            Counter(str(row["target_component_private_not_model_input"]) for row in pool)
        ),
        "checks": checks,
        "reference_sha256": sha256_file(reference_path),
        "candidates_sha256": sha256_file(candidates_path),
        "call_plan_sha256": sha256_file(plan_path),
        "api_calls": 0,
        "responses_generated": 0,
        "human_labels_read_for_selection": 0,
        "external_lockbox_read": False,
        "formal_python_executable": sys.executable,
        "formal_python_version": sys.version.split()[0],
    }
    implementation_paths = {
        "component_binding_and_step2_contract": ROOT
        / "src/metacom_pm/v1_5_generator_alignment_audit.py",
        "replacement_plan_builder": Path(__file__).resolve(),
        "generation_runner": ROOT
        / "scripts/v1_5/24es_run_final_system_generation_v1_5.py",
        "review_packet_builder": ROOT
        / "scripts/v1_5/25zt_prepare_v3_replacement_h_step2_human_review_v1_5.py",
        "review_aggregator": ROOT
        / "scripts/v1_5/25zu_aggregate_v3_replacement_h_step2_review_v1_5.py",
        "joint_feasibility_runtime": ROOT / "src/metacom_pm/v1_5b_policy_runtime.py",
    }
    schema_hashes = {
        action_id: sha256_text(
            canonical_json(
                resource_bundle_output_schema_for_components(components).model_json_schema()
            )
        )
        for action_id, components in sorted(
            {
                str(call["requested_action_id"]): tuple(call["requested_components"])
                for call in calls
            }.items()
        )
    }
    freeze_manifest = {
        "protocol": PROTOCOL,
        "status": "T0_FROZEN_READY_FOR_SINGLE_T1_CONSUMPTION",
        "call_plan_sha256": report["call_plan_sha256"],
        "implementation_sha256": {
            name: sha256_file(path) for name, path in implementation_paths.items()
        },
        "exact_action_schema_sha256": schema_hashes,
        "generator": endpoint.model,
        "formal_python_executable": sys.executable,
        "post_T1_prompt_schema_validator_builder_revision_allowed": False,
    }
    report["t0_freeze_manifest"] = freeze_manifest
    write_json(out_dir / "generation_preflight.json", report)
    write_json(out_dir / "t0_freeze_manifest.json", freeze_manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_h_eligibility_reference_v3/eligibility_reference.jsonl",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_p2_exact_rank1_v3/candidate_rows_private.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_replacement_h_step2_plan_v1_candidate",
    )
    args = parser.parse_args()
    print(build(reference_path=args.reference, candidates_path=args.candidates, out_dir=args.out_dir))


if __name__ == "__main__":
    main()
