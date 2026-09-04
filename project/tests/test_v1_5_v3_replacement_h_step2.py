from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path

from metacom_pm.io import iter_jsonl, read_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]
PLAN_DIR = ROOT / "outputs/pm_v1_5_v3_replacement_h_step2_plan_v1_candidate"
REFERENCE = ROOT / "outputs/pm_v1_5_v3_h_eligibility_reference_v3/eligibility_reference.jsonl"


def _load_aggregator():
    path = ROOT / "scripts/v1_5/25zu_aggregate_v3_replacement_h_step2_review_v1_5.py"
    spec = importlib.util.spec_from_file_location("v3_replacement_h_step2_aggregator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(path: Path) -> list[dict]:
    return [dict(row) for row in iter_jsonl(path)]


def test_replacement_plan_is_exactly_one_32_item_gate() -> None:
    report = read_json(PLAN_DIR / "generation_preflight.json")
    calls = _rows(PLAN_DIR / "call_plan_private.jsonl")
    assert report["status"] == "READY_FOR_T1_REPLACEMENT_PAID_GENERATION_REVIEW"
    assert len(calls) == 32
    assert len({row["call_id"] for row in calls}) == 32
    assert report["call_plan_sha256"] == sha256_file(
        PLAN_DIR / "call_plan_private.jsonl"
    )
    assert report["api_calls"] == 0
    assert report["responses_generated"] == 0
    freeze = read_json(PLAN_DIR / "t0_freeze_manifest.json")
    assert freeze == report["t0_freeze_manifest"]
    assert freeze["status"] == "T0_FROZEN_READY_FOR_SINGLE_T1_CONSUMPTION"
    assert freeze["call_plan_sha256"] == report["call_plan_sha256"]
    assert len(freeze["implementation_sha256"]) == 6
    assert len(freeze["exact_action_schema_sha256"]) == 12


def test_replacement_singles_are_six_per_component_and_frozen_eligible() -> None:
    reference = {
        (str(row["state_id"]), str(row["component"])): row
        for row in _rows(REFERENCE)
    }
    calls = _rows(PLAN_DIR / "call_plan_private.jsonl")
    singles = [row for row in calls if len(row["requested_components"]) == 1]
    assert Counter(row["requested_components"][0] for row in singles) == Counter(
        {"MP": 6, "MS": 6, "ME": 6, "RS": 6}
    )
    for call in calls:
        for component, lineage in call["resource_lineage_private"].items():
            frozen = reference[(lineage["source_state_id"], component)]
            assert frozen["derived_eligibility"] == "eligible"
            assert lineage["candidate_id"] == frozen["candidate_id"]
            assert lineage["candidate_text_sha256"] == frozen["candidate_text_sha256"]


def test_replacement_surfaces_are_non_echo_and_typed_before_generation() -> None:
    calls = _rows(PLAN_DIR / "call_plan_private.jsonl")
    for call in calls:
        for component, qualification in call["step2_surface_qualifications"].items():
            assert qualification["component"] == component
            assert qualification["qualified"], qualification["errors"]
            assert qualification["errors"] == []
        prompt = str(call["messages"])
        assert "resource_support_excerpt" in prompt
        assert "response_evidence_excerpt" in prompt
        assert "Evidence from one component may never justify another component" in prompt


def test_replacement_multi_actions_are_distinct_and_jointly_feasible() -> None:
    calls = _rows(PLAN_DIR / "call_plan_private.jsonl")
    multi = [row for row in calls if len(row["requested_components"]) > 1]
    assert len(multi) == 8
    assert len({row["requested_action_id"] for row in multi}) == 8
    assert all(
        row["joint_feasibility"]["requested_action_id"]
        == row["joint_feasibility"]["feasible_action_id"]
        for row in multi
    )
    assert all(not row["joint_feasibility"]["dropped_components"] for row in multi)


def test_replacement_lineage_does_not_reuse_one_source_surface() -> None:
    calls = _rows(PLAN_DIR / "call_plan_private.jsonl")
    source_states = [
        lineage["source_state_id"]
        for call in calls
        for lineage in call["resource_lineage_private"].values()
    ]
    assert len(source_states) == len(set(source_states))


def _synthetic_review_inputs() -> tuple[list[dict], list[dict], list[dict]]:
    module = _load_aggregator()
    calls = _rows(PLAN_DIR / "call_plan_private.jsonl")
    annotations: list[dict] = []
    packet: list[dict] = []
    private: list[dict] = []
    for index, call in enumerate(calls):
        review_id = f"synthetic_{index:02d}"
        components = list(call["requested_components"])
        packet.append(
            {
                "review_item_id": review_id,
                "requested_action_id": call["requested_action_id"],
                "requested_components": components,
                "application_response": "usable response",
                "fallback_response": None,
            }
        )
        private.append(
            {
                "review_item_id": review_id,
                "call_id": call["call_id"],
                "state_id": call["state_id"],
                "call_kind": "single" if len(components) == 1 else "multi",
                "fallback_executed_private": False,
                "schema_failure_private": False,
            }
        )
        annotations.append(
            {
                "protocol": module.PROTOCOL,
                "review_item_id": review_id,
                "adjudicable": "yes",
                "component_functional": {component: "yes" for component in components},
                "material_misuse": "no",
                "misuse_categories": [],
                "fallback_usable": "not_applicable",
                "literal_response_excerpt": "usable response",
                "review_notes": "synthetic unit-test judgment",
                "annotator_id": "unit_test",
            }
        )
    return annotations, packet, private


def test_replacement_human_gate_passes_only_the_frozen_thresholds() -> None:
    module = _load_aggregator()
    annotations, packet, private = _synthetic_review_inputs()
    report, frozen = module.aggregate(
        annotations=annotations, packet=packet, private=private
    )
    assert report["passed"]
    assert report["status"] == "PASS_FREEZE_V3_REPLACEMENT_STEP2"
    assert len(frozen) == 32
    assert all(values["yes"] == 6 for values in report["functional_by_single_component"].values())
    assert report["multi_all_requested_functional"] == 8


def test_replacement_human_gate_failure_is_terminal_not_a_repair_trigger() -> None:
    module = _load_aggregator()
    annotations, packet, private = _synthetic_review_inputs()
    mp_singles = [
        row
        for row in annotations
        if row["component_functional"] == {"MP": "yes"}
    ]
    for row in mp_singles[:2]:
        row["component_functional"]["MP"] = "no"
    report, _ = module.aggregate(
        annotations=annotations, packet=packet, private=private
    )
    assert not report["passed"]
    assert report["status"] == "TERMINAL_BOUNDED_COMPONENT_PASS_OR_NOT_QUALIFIED"
    assert not report["gates"]["single_component_functional_minimum_5_of_6_each"]
    assert not report["post_result_prompt_or_packet_repair_allowed"]
