#!/usr/bin/env python3
"""Diagnostic-only Hybrid (lexical + semantic) retrieval report.

Part 3 of the approved PM-v1.5 hybrid-retrieval plan. This script is
strictly report-only:

- It never touches a real, frozen pipeline artifact under outputs/ owned by
  a formal stage, never calls a paid API, and never changes what any real
  consumer does. HybridMemoryRetriever/HybridStrategyRetriever
  (src/metacom_pm/hybrid_retrieval.py) remain isolated from every real
  consumer -- see tests/test_hybrid_retrieval_isolation.py.
- Score floors are calibrated using ONLY the synthetic development corpus's
  TRAIN split (reported against, never fit from, the CALIBRATION split).
  internal_test/external_test rows and every EvoEmo-derived number below are
  strictly report-only, computed only after the floors/contract above are
  already frozen, and must never feed back into calibration. See
  src/metacom_pm/hybrid_retrieval_diagnostics.py's module docstring for the
  full data-boundary rule this script enforces by construction (it calls
  into that module rather than re-deriving the logic here).
- The two EvoEmo comparisons (fixed top-k, fixed evidence-token budget)
  deliberately use each user's `topic` field as a query stand-in --
  evaluator-only-for-diagnostics-only, structurally separate from the real,
  observable query path (`hybrid_retrieval.observable_query_with_hash`).

Any future decision to adopt Hybrid for real consumers is explicitly
deferred (Part 4 of the plan) and is not made or implied by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

_START = time.monotonic()


def _log(message: str) -> None:
    print(f"[{time.monotonic() - _START:8.1f}s] {message}", file=sys.stderr, flush=True)

from metacom_pm.config import load_config
from metacom_pm.contracts import MemorySource, StrategyCard
from metacom_pm.evoemo import evo_memory_global_catalog_digest, load_evoemo
from metacom_pm.hybrid_retrieval import (
    HYBRID_RETRIEVAL_PROTOCOL,
    RECIPROCAL_RANK_FUSION_K,
    HybridMemoryRetriever,
    HybridStrategyRetriever,
    SourceScoreFloors,
    hybrid_retrieval_contract_hash,
)
from metacom_pm.hybrid_retrieval_diagnostics import (
    calibrate_source_floors,
    compare_fixed_token_budget,
    compare_fixed_top_k,
    evaluate_case_memory_retrieval_quality,
    evaluate_esconv_strategy_retrieval_quality,
    iter_case_calibration_examples,
    split_of_user_id,
)
from metacom_pm.io import iter_jsonl, sha256_file, write_json
from metacom_pm.pm_v1_5_semantic import (
    FrozenTransformerSemanticEncoder,
    semantic_encoder_spec_from_config,
)
from metacom_pm.retrieval import DEFAULT_MEMORY_TOP_K, MemoryRetriever, StrategyRetriever

ROOT = Path(__file__).resolve().parents[2]

# No per-candidate strategy-utility ground truth exists in the synthetic
# corpus (only memory items carry item_utility/stale/conflicts_with_current_
# state labels) -- freeze the strategy floor at zero (no exclusion) rather
# than fabricate a calibrated value from data that does not exist. Recorded
# in the report, not silently assumed.
STRATEGY_FLOOR_CALIBRATION_STATUS = (
    "no_labeled_synthetic_strategy_candidates_available_in_this_corpus; "
    "strategy floors frozen at (0.0, 0.0) -- no exclusion -- rather than "
    "fabricated from nonexistent ground truth"
)
STRATEGY_FLOORS = SourceScoreFloors(lexical_min_score=0.0, semantic_min_score=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evoemo", type=Path, default=ROOT / "data" / "external" / "evo_emo.json"
    )
    parser.add_argument(
        "--development-data-candidate",
        type=Path,
        default=(
            ROOT
            / "data"
            / "pm_v1_5_formal_v8_17_casewise_repair_candidate"
            / "_generated_bundles_work.jsonl"
        ),
        help=(
            "Real, currently-best-available synthetic development bundles, "
            "used only to calibrate TRAIN/CALIBRATION score floors. This is "
            "NOT the frozen formal development-data artifact -- see the "
            "report's development_data_candidate.disclosure field."
        ),
    )
    parser.add_argument(
        "--pm-v2-config", type=Path, default=ROOT / "configs" / "pm_v1_5.yaml"
    )
    parser.add_argument(
        "--esconv", type=Path, default=ROOT / "data" / "external" / "ESConv.json"
    )
    parser.add_argument(
        "--esconv-split-manifest",
        type=Path,
        default=ROOT / "data" / "strategy" / "esconv_split_manifest_v1_5.jsonl",
    )
    parser.add_argument(
        "--strategy-bank",
        type=Path,
        default=ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl",
    )
    parser.add_argument(
        "--memory-eval-split",
        default="calibration",
        choices=["calibration"],
        help=(
            "Split evaluate_case_memory_retrieval_quality reads. Hard-"
            "restricted to 'calibration' -- the only split the approved "
            "plan permits for adoption-relevant evidence. 'train' would "
            "reuse the split floors were fit from; internal_test/"
            "external_test are confirmatory and must never be touched "
            "before the retriever design is frozen and a decision is made."
        ),
    )
    parser.add_argument(
        "--esconv-eval-split",
        default="validation",
        choices=["validation"],
        help=(
            "Split evaluate_esconv_strategy_retrieval_quality reads. Hard-"
            "restricted to 'validation'; 'test' is confirmatory and must "
            "never be used before the retriever design is frozen and a "
            "decision is made."
        ),
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=360,
        help="Fixed evidence-token budget for the second comparison.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_hybrid_retrieval_diagnostic.json",
    )
    args = parser.parse_args()

    _log("loading config and semantic encoder")
    config = load_config(args.pm_v2_config)
    encoder_spec = semantic_encoder_spec_from_config(config)
    encoder = FrozenTransformerSemanticEncoder.load(encoder_spec)
    _log("semantic encoder loaded")

    bundles = list(iter_jsonl(args.development_data_candidate))
    split_counts = Counter(split_of_user_id(bundle["user_id"]) for bundle in bundles)
    calibration_examples = iter_case_calibration_examples(bundles)
    _log(f"calibrating floors from {len(calibration_examples)} examples")
    source_calibration = calibrate_source_floors(calibration_examples, encoder=encoder)
    _log("floor calibration complete")
    memory_floors_by_source = {
        source: SourceScoreFloors(
            lexical_min_score=calib.lexical_min_score,
            semantic_min_score=calib.semantic_min_score,
        )
        for source, calib in source_calibration.items()
    }
    contract_hash = hybrid_retrieval_contract_hash(
        memory_floors_by_source=memory_floors_by_source,
        strategy_floors=STRATEGY_FLOORS,
        rrf_k=RECIPROCAL_RANK_FUSION_K,
        semantic_encoder_spec_sha256=encoder_spec.digest(),
    )

    users = load_evoemo(args.evoemo)
    evo_memory_digest = evo_memory_global_catalog_digest(users)
    # Configured for every source (not just ME) so the same retriever pair
    # serves both the ME-only EvoEmo comparison below (which always passes
    # selected_sources={ME} explicitly) and the all-source
    # evaluate_case_memory_retrieval_quality legitimate-evidence pass.
    memory_min_score = float(config["retrieval"]["memory_min_score"])
    lexical_retriever = MemoryRetriever(
        top_k_by_source=dict(DEFAULT_MEMORY_TOP_K),
        minimum_score_by_source={source: memory_min_score for source in MemorySource},
    )
    hybrid_retriever = HybridMemoryRetriever(
        semantic_encoder=encoder,
        top_k_by_source=dict(DEFAULT_MEMORY_TOP_K),
        floors_by_source=memory_floors_by_source,
        rrf_k=RECIPROCAL_RANK_FUSION_K,
    )
    _log(f"running EvoEmo fixed-top-k comparison over {len(users)} users")
    fixed_top_k_report = compare_fixed_top_k(
        users, lexical_retriever=lexical_retriever, hybrid_retriever=hybrid_retriever
    )
    _log("EvoEmo fixed-top-k comparison complete; running fixed-token-budget comparison")
    fixed_budget_report = compare_fixed_token_budget(
        users, hybrid_retriever=hybrid_retriever, token_budget=int(args.token_budget)
    )
    _log("EvoEmo fixed-token-budget comparison complete")

    _log(f"running case memory retrieval-quality eval (split={args.memory_eval_split})")
    memory_retrieval_quality_report = evaluate_case_memory_retrieval_quality(
        bundles,
        split=args.memory_eval_split,
        lexical_retriever=lexical_retriever,
        hybrid_retriever=hybrid_retriever,
        progress=lambda done, total: (
            _log(f"  memory case {done}/{total}") if done % 10 == 0 or done == total else None
        ),
    )
    _log("case memory retrieval-quality eval complete")

    strategy_cards = [
        StrategyCard.model_validate(row) for row in iter_jsonl(args.strategy_bank)
    ]
    _log(f"loaded {len(strategy_cards)} strategy cards; building retrievers (batched encode)")
    strategy_top_k = int(config["retrieval"]["strategy_top_k"])
    strategy_min_score = float(config["retrieval"]["strategy_min_score"])
    lexical_strategy_retriever = StrategyRetriever(
        strategy_cards, top_k=strategy_top_k, minimum_score=strategy_min_score
    )
    hybrid_strategy_retriever = HybridStrategyRetriever(
        strategy_cards,
        semantic_encoder=encoder,
        top_k=strategy_top_k,
        floors=STRATEGY_FLOORS,
        rrf_k=RECIPROCAL_RANK_FUSION_K,
        embedding_progress=lambda done, total: (
            _log(f"  strategy bank encode batch {done}/{total}")
            if done % 10 == 0 or done == total
            else None
        ),
    )
    _log("strategy bank batched encode complete; running ESConv strategy retrieval-quality eval")
    esconv_strategy_report = evaluate_esconv_strategy_retrieval_quality(
        str(args.esconv),
        str(args.esconv_split_manifest),
        split=args.esconv_eval_split,
        lexical_retriever=lexical_strategy_retriever,
        hybrid_retriever=hybrid_strategy_retriever,
        progress=lambda done, total: (
            _log(f"  esconv turn {done}/{total}") if done % 100 == 0 or done == total else None
        ),
    )
    _log("ESConv strategy retrieval-quality eval complete; writing report")

    report = {
        "protocol": HYBRID_RETRIEVAL_PROTOCOL,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "REPORT_ONLY_NOT_A_FORMAL_ARTIFACT_NO_ADOPTION_DECISION_MADE",
        "inputs": {
            "evoemo_path": str(args.evoemo),
            "evoemo_sha256": sha256_file(args.evoemo),
            "evo_memory_builder_contract_sha256": evo_memory_digest[
                "builder_contract_sha256"
            ],
            "evo_memory_global_catalog_sha256": evo_memory_digest[
                "global_catalog_sha256"
            ],
            "development_data_candidate": {
                "path": str(args.development_data_candidate),
                "sha256": sha256_file(args.development_data_candidate),
                "user_count": len(bundles),
                "split_counts": dict(sorted(split_counts.items())),
                "disclosure": (
                    "Real, generated content produced by "
                    "scripts/v1_5/20_generate_pm_v2_development_data_v1_5.py's "
                    "v8.17 casewise-repair attempt -- NOT the frozen formal "
                    "development-data artifact (as of this report, that stage "
                    "has not been formally closed out / paid-approved for "
                    "every user). Floors calibrated from it should be "
                    "re-derived once the formal artifact exists; the "
                    "calibration method itself (TRAIN-positive minimum, "
                    "CALIBRATION-only reporting) does not change."
                ),
            },
            "esconv_path": str(args.esconv),
            "esconv_sha256": sha256_file(args.esconv),
            "esconv_split_manifest_path": str(args.esconv_split_manifest),
            "esconv_split_manifest_sha256": sha256_file(args.esconv_split_manifest),
            "strategy_bank_path": str(args.strategy_bank),
            "strategy_bank_sha256": sha256_file(args.strategy_bank),
        },
        "semantic_encoder": {
            "spec_sha256": encoder_spec.digest(),
            "binding": encoder.binding.model_dump(mode="json"),
        },
        "floor_calibration_query_construction": (
            "Each calibration example's query is built via "
            "retrieval.context_query(current_user_text, recent_dialogue, "
            "session_summary) -- the same construction real retrieval uses "
            "at deployment time -- not bare current_user_text alone. "
            "Floors calibrated against a shorter, differently-shaped query "
            "would not be calibrated against the query distribution they "
            "are meant to gate."
        ),
        "floor_calibration_by_memory_source": {
            source.value: asdict(calib) for source, calib in source_calibration.items()
        },
        "strategy_floor_calibration_status": STRATEGY_FLOOR_CALIBRATION_STATUS,
        "frozen_contract": {
            "rrf_k": RECIPROCAL_RANK_FUSION_K,
            "memory_floors_by_source": {
                source.value: asdict(floors)
                for source, floors in memory_floors_by_source.items()
            },
            "strategy_floors": asdict(STRATEGY_FLOORS),
            "contract_sha256": contract_hash,
        },
        "legitimate_adoption_evidence": {
            "note": (
                "The only numbers in this report that may inform a Part 4 "
                "adoption decision, per the approved plan's data-boundary "
                "rule: real per-case candidate pools scored via the actual "
                "retrieve() pipeline (fusion, ranking, top-k), never a raw- "
                "score-only floor check. Memory uses the CALIBRATION split "
                "only (never TRAIN, already spent fitting floors above); "
                "Strategy uses the ESConv VALIDATION split only (never "
                "test, which is confirmatory). Both use the same "
                "context_query construction real retrieval uses."
            ),
            "memory_retrieval_quality": {
                "split": args.memory_eval_split,
                "top_k_by_source": {
                    source.value: k for source, k in DEFAULT_MEMORY_TOP_K.items()
                },
                "by_method": {
                    method: {
                        source_value: asdict(summary)
                        for source_value, summary in per_source.items()
                    }
                    for method, per_source in memory_retrieval_quality_report[
                        "by_method_by_source"
                    ].items()
                },
                "negative_source_and_harmful_retrieval": memory_retrieval_quality_report[
                    "negative_source_and_harmful_retrieval"
                ],
            },
            "esconv_strategy_retrieval_quality": {
                "split": args.esconv_eval_split,
                "strategy_top_k": strategy_top_k,
                "by_method": {
                    method: asdict(summary)
                    for method, summary in esconv_strategy_report["by_method"].items()
                },
                "paired_dialogue_cluster_bootstrap": esconv_strategy_report[
                    "paired_dialogue_cluster_bootstrap"
                ],
            },
        },
        "report_only_evoemo_diagnostic_not_confirmatory": {
            "note": (
                "Every number below uses EvoEmo's evaluator-only "
                "subsequent_topics/related_sessions ground truth and topic "
                "text as a query stand-in -- report-only, computed after "
                "the frozen_contract above was already fixed, and never fed "
                "back into floor calibration or the legitimate evidence "
                "above. Scope is ME only: related_sessions ground truth has "
                "no MP/MS analogue."
            ),
            "top_k_by_source": {"ME": DEFAULT_MEMORY_TOP_K[MemorySource.ME]},
            "fixed_top_k_comparison": {
                name: asdict(summary) for name, summary in fixed_top_k_report.items()
            },
            "fixed_token_budget_comparison": {
                "token_budget": int(args.token_budget),
                **{
                    name: asdict(summary)
                    for name, summary in fixed_budget_report.items()
                },
            },
        },
    }
    write_json(args.out, report)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "out": str(args.out),
                "contract_sha256": contract_hash,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
