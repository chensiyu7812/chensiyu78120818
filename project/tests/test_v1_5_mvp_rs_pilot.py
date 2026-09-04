from __future__ import annotations

from pathlib import Path

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.v1_5_mvp_rs_pilot import (
    build_minimum_rs_clean_pair_plan,
    explicit_boundary_cue,
    validate_minimum_rs_execution_inputs,
)


ROOT = Path(__file__).resolve().parents[1]


def _real_plan() -> dict:
    pm_config = load_config(ROOT / "configs/pm_v1_5.yaml")
    experiment = load_config(ROOT / "configs/experiment.yaml")
    contract = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, contract.generator_endpoint)
    return build_minimum_rs_clean_pair_plan(
        runtime_states_path=str(
            ROOT
            / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate"
            / "runtime_states.jsonl"
        ),
        strategy_cards_path=str(
            ROOT
            / "outputs/pm_v1_5_strategy_bank_v2_review_candidate_v2"
            / "strategy_cards_v2_candidate.jsonl"
        ),
        strategy_lineage_path=str(
            ROOT
            / "outputs/pm_v1_5_strategy_bank_v2_review_candidate_v2"
            / "strategy_bank_v2_lineage.jsonl"
        ),
        selected_seed_sources_path=str(
            ROOT / "data/strategy/pm_v1_5_selected_seed_sources.jsonl"
        ),
        generation_contract=contract,
        generator_identity={
            "base_url": endpoint.base_url,
            "model": endpoint.model,
            "family": endpoint.family,
            "transport": endpoint.transport,
        },
    )


def test_explicit_boundary_cues_are_literal_and_conflict_checked() -> None:
    assert explicit_boundary_cue("Any tips?") == "advice_welcome"
    assert (
        explicit_boundary_cue(
            "For this turn, I mainly need to feel heard before thinking about solutions."
        )
        == "listen_only"
    )
    assert explicit_boundary_cue("I feel overwhelmed today.") is None


def test_real_minimum_rs_plan_is_balanced_grouped_and_zero_api() -> None:
    payload = _real_plan()
    report = payload["report"]
    assert report["status"] == "BLOCKED_ONLY_ON_FIVE_CARD_HUMAN_REVIEW"
    assert report["api_calls_made"] == 0
    assert report["selected_states"] == 49
    assert report["independent_user_groups"] == 24
    assert report["planned_logical_calls"] == 98
    assert report["strategy_development_seed_source_overlap_count"] == 0
    assert report["boundary_cue_counts"] == {
        "advice_welcome": 27,
        "listen_only": 22,
    }
    assert all(report["scientific_checks"].values())


def test_rs_plan_uses_same_stack_and_never_exposes_raw_examples() -> None:
    payload = _real_plan()
    by_pair: dict[str, list[dict]] = {}
    for row in payload["call_plan"]:
        by_pair.setdefault(row["pair_id"], []).append(row)
    assert len(by_pair) == 49
    for rows in by_pair.values():
        assert {row["arm"] for row in rows} == {"R0", "RS"}
        r0 = next(row for row in rows if row["arm"] == "R0")
        rs = next(row for row in rows if row["arm"] == "RS")
        assert r0["generation"] == rs["generation"]
        assert r0["generator_identity"] == rs["generator_identity"]
        assert r0["messages"][0] == rs["messages"][0]
        assert "Current-session summary:\n(none)" in r0["messages"][1]["content"]
        assert "Current-session summary:\n(none)" in rs["messages"][1]["content"]
        assert "Example style" not in rs["messages"][1]["content"]
        if rs["boundary_cue"] == "advice_welcome":
            assert rs["selected_strategy_family"] == "Providing Suggestions"
        else:
            assert rs["selected_strategy_family"] != "Providing Suggestions"


def test_human_review_is_the_only_remaining_local_plan_gate() -> None:
    payload = _real_plan()
    pm_config = load_config(ROOT / "configs/pm_v1_5.yaml")
    experiment = load_config(ROOT / "configs/experiment.yaml")
    contract = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, contract.generator_endpoint)
    promoted = build_minimum_rs_clean_pair_plan(
        runtime_states_path=payload["report"]["runtime_states_path"],
        strategy_cards_path=payload["report"]["strategy_cards_path"],
        strategy_lineage_path=payload["report"]["strategy_lineage_path"],
        selected_seed_sources_path=payload["report"][
            "selected_seed_sources_path"
        ],
        generation_contract=contract,
        generator_identity={
            "base_url": endpoint.base_url,
            "model": endpoint.model,
            "family": endpoint.family,
            "transport": endpoint.transport,
        },
        human_bank_review_passed=True,
    )
    assert promoted["report"]["status"] == "READY_FOR_TRAIN_ONLY_GENERATION"
    preflight = validate_minimum_rs_execution_inputs(
        report=promoted["report"],
        call_rows=promoted["call_plan"],
        generation_contract=contract,
        generator_identity=promoted["report"]["generator_identity"],
        human_review_binding={
            "status": "HUMAN_REVIEW_PASS_PENDING_LLM_WEAK_AUDIT",
            "review_count": 5,
            "approved_count": 5,
        },
    )
    assert preflight["status"] == "READY_FOR_TRAIN_ONLY_GENERATION"
    assert preflight["ready"] is True


def test_execution_preflight_blocks_without_human_binding_before_client_use() -> None:
    payload = _real_plan()
    pm_config = load_config(ROOT / "configs/pm_v1_5.yaml")
    contract = SupporterGenerationContract.from_config(pm_config)
    preflight = validate_minimum_rs_execution_inputs(
        report=payload["report"],
        call_rows=payload["call_plan"],
        generation_contract=contract,
        generator_identity=payload["report"]["generator_identity"],
        human_review_binding=None,
    )
    assert preflight["status"] == "BLOCKED_ONLY_ON_FIVE_CARD_HUMAN_REVIEW"
    assert preflight["api_calls_made"] == 0
    assert preflight["errors"] == []
