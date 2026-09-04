#!/usr/bin/env python3
"""Freeze the G2-qualified six-card Strategy RAG as a G3 method input."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from metacom_pm.contracts import StrategyCard
from metacom_pm.v1_5_strategy_rag_runtime import (
    MINIMUM_SCORE,
    MOVE_IDS,
    PROTOCOL as RUNTIME_PROTOCOL,
    QualifiedStrategyRAG,
    TOP_K,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-qualified-strategy-rag-freeze-v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    source_bank = (
        ROOT
        / "outputs/pm_v1_5_strategy_g1_final_bank_v1"
        / "strategy_cards_g1_runtime.jsonl"
    )
    g1_report_path = (
        ROOT
        / "outputs/pm_v1_5_strategy_g1_final_bank_v1"
        / "g1_dual_human_freeze_report.json"
    )
    control_report_path = (
        ROOT
        / "outputs/pm_v1_5_strategy_g2_controlled_retrieval_v1"
        / "controlled_report.json"
    )
    real_report_path = (
        ROOT
        / "outputs/pm_v1_5_strategy_g2_real_qualification_v1"
        / "real_qualification_report.json"
    )
    real_results_path = (
        ROOT
        / "outputs/pm_v1_5_strategy_g2_real_qualification_v1"
        / "real_query_results.jsonl"
    )
    runtime_path = ROOT / "src/metacom_pm/v1_5_strategy_rag_runtime.py"
    prompt_path = ROOT / "src/metacom_pm/prompts.py"
    prereg_path = (
        ROOT
        / "data/pm_v1_5_contracts/strategy_g2_real_qualification_v1.json"
    )

    g1 = _read_json(g1_report_path)
    controlled = _read_json(control_report_path)
    real = _read_json(real_report_path)
    if g1["status"] != "G1_COMPLETE_SIX_CARD_BANK_FROZEN_FOR_G2_DEVELOPMENT":
        raise RuntimeError("G1 six-card definition Bank is not frozen")
    if controlled["eligibility_expected_set_passes"] != controlled[
        "eligibility_expected_set_denominator"
    ]:
        raise RuntimeError("synthetic eligibility controls did not all pass")
    if real["status"] != "RAG_QUALIFIED_FOR_G3_CLEAN_TREATMENT":
        raise RuntimeError("real G2 qualification did not pass")
    if real["ranker_decision"]["selected"] != "lexical":
        raise RuntimeError("this freeze expects the qualified lexical ranker")

    card_rows = _read_jsonl(source_bank)
    cards = [StrategyCard.model_validate(row) for row in card_rows]
    QualifiedStrategyRAG(cards)
    if {card.strategy_label for card in cards} != MOVE_IDS:
        raise RuntimeError("frozen move inventory drift")
    if any(
        card.example_response
        != (
            "No source response is provided. Compose a new response from "
            "the visible dialogue and this guidance only."
        )
        for card in cards
    ):
        raise RuntimeError("a runtime card exposes or changes a source example")

    bank_path = ROOT / "data/strategy/strategy_cards_v1_5_minimal.jsonl"
    manifest_path = (
        ROOT / "data/strategy/strategy_rag_v1_5_minimal_freeze.json"
    )
    config_path = ROOT / "configs/pm_v1_5_minimal_rag_v1.json"
    out_dir = ROOT / "outputs/pm_v1_5_strategy_rag_frozen_v1"
    _write_jsonl(bank_path, card_rows)

    manifest = {
        "protocol": PROTOCOL,
        "status": "FROZEN_AND_QUALIFIED_FOR_G3_CLEAN_TREATMENT",
        "scope": (
            "Minimal topic-agnostic Strategy Guidance opportunity and Top-1 "
            "retrieval; response benefit is not yet established."
        ),
        "bank": {
            "relative_path": str(bank_path.relative_to(ROOT)),
            "sha256": _sha256(bank_path),
            "card_count": len(cards),
            "move_ids": sorted(MOVE_IDS),
            "raw_source_responses_available": False,
            "formative_source_examples_are_gold": False,
        },
        "retrieval": {
            "runtime_protocol": RUNTIME_PROTOCOL,
            "runtime_relative_path": str(runtime_path.relative_to(ROOT)),
            "runtime_sha256": _sha256(runtime_path),
            "eligibility_before_ranking": True,
            "ranker": "term_frequency_cosine",
            "score_floor": MINIMUM_SCORE,
            "top_k": TOP_K,
            "no_eligible_or_below_floor": "abstain_and_keep_RS_off",
            "bge_promoted": False,
        },
        "prompt_compiler": {
            "relative_path": str(prompt_path.relative_to(ROOT)),
            "sha256": _sha256(prompt_path),
            "strategy_guidance_only": True,
            "source_example_text_injected": False,
            "maximum_strategy_cards": TOP_K,
            "generator_must_compose_new_wording": True,
        },
        "qualification_evidence": {
            "g1_report": str(g1_report_path.relative_to(ROOT)),
            "g1_report_sha256": _sha256(g1_report_path),
            "synthetic_control_report": str(control_report_path.relative_to(ROOT)),
            "synthetic_control_report_sha256": _sha256(control_report_path),
            "real_report": str(real_report_path.relative_to(ROOT)),
            "real_report_sha256": _sha256(real_report_path),
            "real_results": str(real_results_path.relative_to(ROOT)),
            "real_results_sha256": _sha256(real_results_path),
            "preregistration": str(prereg_path.relative_to(ROOT)),
            "preregistration_sha256": _sha256(prereg_path),
        },
        "same_stack_requirement": (
            "This exact bank, eligibility implementation, lexical ranker, "
            "score floor, Top-1 cap, and prompt compiler must be reused in "
            "G3 development, ESConv evaluation, and EvoEmo evaluation."
        ),
        "not_claimed": [
            "Strategy Guidance improves response quality",
            "the next ESConv native strategy is a gold card label",
            "the catalog covers clinical or specialist guidance",
            "BGE similarity estimates component utility",
        ],
        "next_gate": (
            "Build RS R0-vs-RS clean treatment pairs and verify retrieval "
            "realization, generator uptake, quality, risk, and cost."
        ),
    }
    _write_json(manifest_path, manifest)
    config = {
        "protocol": PROTOCOL,
        "status": "ACTIVE_FOR_NEW_PM_V1_5_G3_WORK",
        "strategy_bank": {
            "relative_path": manifest["bank"]["relative_path"],
            "sha256": manifest["bank"]["sha256"],
            "card_count": manifest["bank"]["card_count"],
        },
        "eligibility_runtime": {
            "relative_path": manifest["retrieval"]["runtime_relative_path"],
            "sha256": manifest["retrieval"]["runtime_sha256"],
        },
        "retrieval": {
            "ranker": manifest["retrieval"]["ranker"],
            "strategy_min_score": MINIMUM_SCORE,
            "strategy_top_k": TOP_K,
            "bge_enabled": False,
        },
        "prompt_compiler": {
            "relative_path": manifest["prompt_compiler"]["relative_path"],
            "sha256": manifest["prompt_compiler"]["sha256"],
            "inject_source_examples": False,
        },
        "manifest": {
            "relative_path": str(manifest_path.relative_to(ROOT)),
        },
        "legacy_config_policy": (
            "configs/pm_v1_5.yaml remains historical evidence for the failed "
            "11590-card/top3 path and must not be used for new G3 runs."
        ),
    }
    _write_json(config_path, config)
    freeze_report = {
        "protocol": PROTOCOL,
        "status": manifest["status"],
        "manifest_relative_path": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": _sha256(manifest_path),
        "config_relative_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "bank_relative_path": str(bank_path.relative_to(ROOT)),
        "bank_sha256": _sha256(bank_path),
        "card_count": len(cards),
        "top_k": TOP_K,
        "strategy_min_score": MINIMUM_SCORE,
        "bge_enabled": False,
        "formal_response_quality_benefit_proven": False,
    }
    _write_json(out_dir / "freeze_report.json", freeze_report)
    print(json.dumps(freeze_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
