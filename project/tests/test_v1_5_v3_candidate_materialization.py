import json
from pathlib import Path

from metacom_pm.v1_5_v3_candidate_materialization import materialize_v3_candidates
from metacom_pm.v1_5_v3_effect_blueprint import build_blueprint
from metacom_pm.v1_5_v3_state_realization import realize_v3_states


ROOT = Path(__file__).resolve().parents[1]


def _cards() -> list[dict]:
    path = ROOT / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_v3_formal_exact_rank1_materialization_passes_without_labels_or_outcomes() -> None:
    blueprint = build_blueprint()
    states = realize_v3_states(blueprint)
    rows, report = materialize_v3_candidates(
        states=states,
        blueprint_rows=blueprint,
        strategy_cards=_cards(),
    )
    assert report["status"] == "PASS", report["failures"]
    assert len(rows) == 640
    assert report["duplicate_complete_step1_decision_surfaces"] == 0
    assert all(row["human_gold"] is None for row in rows)
    assert all(not row["response_or_outcome_read"] for row in rows)
    assert all("structured_candidate_metadata" in row for row in rows)
    for row in rows:
        metadata = row["structured_candidate_metadata"]
        assert metadata["candidate_present"] is True
        assert metadata["state_owner_id"]
        assert metadata["candidate_active"] is True
        assert metadata["candidate_superseded"] is False
        if row["target_component_private_not_model_input"] != "RS":
            assert metadata["candidate_owner_id"] == metadata["state_owner_id"]


def test_every_target_component_surface_is_present_and_exact_rank1() -> None:
    blueprint = build_blueprint()
    rows, _report = materialize_v3_candidates(
        states=realize_v3_states(blueprint),
        blueprint_rows=blueprint,
        strategy_cards=_cards(),
    )
    assert all(row["exact_rank1_candidate"]["candidate_present"] for row in rows)
    assert all(row["exact_rank1_candidate"]["selected_rank"] == 1 for row in rows)
