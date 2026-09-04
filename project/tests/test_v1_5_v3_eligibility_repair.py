import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_dual_review_audit_passes_reliability_but_does_not_freeze_bad_realization() -> None:
    report = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_v3_h_eligibility_dual_review_audit_v1/audit_report.json"
        ).read_text(encoding="utf-8")
    )
    final = report["agreement"]["derived_eligibility"]
    assert report["review_reliability_gate_pass"] is True
    assert final["agreement_count"] == 31
    assert final["n"] == 32
    assert final["cohen_kappa"] == 0.9375
    assert report["current_gold_frozen"] is False
    assert report["current_rows_trainable"] is False
    assert report["construction_intent_mismatch_count"] == 14


def test_repaired_exact_rank1_surfaces_fix_the_observed_semantic_family_swaps() -> None:
    blueprint = {
        row["blueprint_row_id"]: row
        for row in _jsonl(
            ROOT
            / "data/pm_v1_5_v3_effect_blueprint_v1/private/construction_blueprint.jsonl"
        )
    }
    rows = _jsonl(
        ROOT / "outputs/pm_v1_5_v3_p2_exact_rank1_v2/candidate_rows_private.jsonl"
    )
    by_family: dict[str, list[dict]] = {}
    for row in rows:
        family = blueprint[row["state_id"]]["logic_family"]
        if blueprint[row["state_id"]]["track"] == "ELIGIBILITY_AUDIT":
            by_family.setdefault(family, []).append(row)

    assert all(
        "emotional uncertainty rather than practical workload"
        in row["exact_rank1_candidate"]["candidate_text"]
        for row in by_family["MS_ELIGIBLE_MS_DISTINCTION"]
    )
    assert all(
        "current question is about me, not my friend" in row["current_user_text"]
        and "belonged to the user's friend"
        in row["exact_rank1_candidate"]["candidate_text"]
        for row in by_family["MS_WRONG_OWNER_OR_GOAL"]
    )
    assert all(
        "reversible adjustment to the immediate environment or routine"
        in row["exact_rank1_candidate"]["candidate_text"]
        for row in by_family["RS_ELIGIBLE_RS_SUGGESTION"]
    )
    assert all(
        "clarify the feeling I already indicated" in row["current_user_text"]
        for row in by_family["RS_CURRENT_REQUEST_ALREADY_SPECIFIES_MOVE"]
    )


def test_v3_ranker_uses_the_low_burden_rs_profile_for_one_focused_question() -> None:
    blueprint = {
        row["blueprint_row_id"]: row
        for row in _jsonl(
            ROOT
            / "data/pm_v1_5_v3_effect_blueprint_v1/private/construction_blueprint.jsonl"
        )
    }
    rows = _jsonl(
        ROOT / "outputs/pm_v1_5_v3_p2_exact_rank1_v3/candidate_rows_private.jsonl"
    )
    current_request_rows = [
        row
        for row in rows
        if blueprint[row["state_id"]]["logic_family"]
        == "RS_CURRENT_REQUEST_ALREADY_SPECIFIES_MOVE"
    ]
    assert len(current_request_rows) == 4
    for row in current_request_rows:
        text = row["exact_rank1_candidate"]["candidate_text"]
        assert "Prefer this profile when the user requests low burden" in text
        assert "Do not use this profile to add any second support action" in text


def test_repair_delta_contains_only_all_changed_eligibility_surfaces() -> None:
    out = ROOT / "outputs/pm_v1_5_v3_h_eligibility_repair_delta_v2_candidate"
    manifest = json.loads((out / "freeze_manifest.json").read_text(encoding="utf-8"))
    primary = json.loads((out / "primary_packet.json").read_text(encoding="utf-8"))
    independent = json.loads((out / "independent_packet.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "READY_FOR_DUAL_REVIEW_OF_CHANGED_DECISION_SURFACES_ONLY"
    assert manifest["changed_items"] == 24
    assert manifest["per_component"] == {"MS": 12, "RS": 12}
    assert manifest["unchanged_eligibility_rows_eligible_for_exact_surface_carry_forward"] == 104
    assert manifest["python_executable"].endswith("/.venv-pm-v1-5/bin/python")
    assert manifest["python_version"] == "3.13.2"
    assert len(primary["items"]) == len(independent["items"]) == 24
    assert {row["blind_item_id"] for row in primary["items"]} == {
        row["blind_item_id"] for row in independent["items"]
    }
    forbidden = {
        "logic_family",
        "private_coverage_intent_not_gold",
        "old_decision_surface_sha256",
        "new_decision_surface_sha256",
    }
    assert all(not (forbidden & set(row)) for row in primary["items"])
