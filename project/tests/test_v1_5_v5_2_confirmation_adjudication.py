from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pm_v1_5_v5_2_confirmation_adjudication_v1_candidate"


def jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (OUT / name).read_text().splitlines() if line.strip()]


def test_review_inputs_are_complete_and_exactly_joined() -> None:
    report = json.loads((OUT / "adjudication_manifest.json").read_text())
    quality = report["quality_reliability"]
    risk = report["risk_reliability"]
    assert report["review_data_quality"]["missing_or_extra_ids"] == 0
    assert report["review_data_quality"]["uncertain"] == 0
    assert quality["n"] == 52
    assert quality["agreement_count"] == 37
    assert quality["disagreements"] == 15
    assert quality["direct_a_b_reversals"] == 7
    assert risk["n"] == 86
    assert risk["agreement_count"] == 76
    assert risk["yes_no_disagreements"] == 10
    assert risk["category_disagreements_when_both_yes"] == 3
    assert risk["adjudication_union"] == 13


def test_only_disagreements_enter_the_single_adjudication() -> None:
    report = json.loads((OUT / "adjudication_manifest.json").read_text())
    quality = jsonl("quality_disagreement_packet.jsonl")
    risk = jsonl("risk_disagreement_packet.jsonl")
    assert len(quality) == report["adjudication"]["quality_items"] == 15
    assert len(risk) == report["adjudication"]["risk_items"] == 13
    assert len({row["blind_item_id"] for row in quality}) == len(quality)
    assert len({row["risk_item_id"] for row in risk}) == len(risk)


def test_adjudication_pages_do_not_reveal_prior_labels_or_annotators() -> None:
    for name in ("human_quality_adjudication.html", "human_risk_adjudication.html"):
        text = (OUT / name).read_text()
        assert '"panel_role":"adjudication"' in text
        assert "Primary_ChatGPT" not in text
        assert "Claude_independent_overlap_sighted" not in text
        assert "prior_labels" not in text
