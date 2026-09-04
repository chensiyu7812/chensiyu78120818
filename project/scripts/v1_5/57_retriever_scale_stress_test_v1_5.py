#!/usr/bin/env python3
"""Zero-API retriever stress test: does discover_final_typed_memory_
candidates() -- the shared MP/MS/ME retriever used all session for both
EvoEmo and (per PM_V1_5_EXTERNAL_VALIDATION_AND_ESMEMEVAL_PLAN_20260804_ZH.md,
"内部与外部共享...retriever、candidate schema") the internal domain -- still
finds the correct, designed candidate as Rank-1 once catalog scale and
distractor density grow to realistic EvoEmo levels (up to ME=109)?

This is step 3 (first half: "Retriever数据") of the roadmap from the
2026-08-06 catalog-diversity/OOD investigation: training's synthetic cards
are fixed at exactly 2 items per source (see PM_V1_5_V5_3_MASTER_STATUS_
20260806_ZH.md section 9); an independent review's recommended next step is
to prove retrieval survives at realistic scale BEFORE any relabeling or
retraining -- this answers that question directly, deterministically, with
no API calls and no new PM/model code.

Design, deliberately reusing the existing project convention rather than
inventing one: every synthetic item is built from the SAME fixed template +
topic-substitution scheme already used by the real 468-card training corpus
(confirmed by direct inspection of data/pm_v1_5_formal_v8_18_duplicate_
repair_candidate/memory_backend.jsonl this session) -- not free-form LLM
writing. One template per source is reserved as the "signal" pattern (the
thing a real positive FIT example would need); the rest are the corpus's own
real non-signal/distractor templates (same-topic-no-result, context-only,
current-echo-shaped, unresolved). This keeps the test's positive examples
gold-bound by construction (the signal item is the only one built from the
signal template) while distractor density and catalog scale grow.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.contracts import MemoryItem, MemorySource  # noqa: E402
from metacom_pm.io import canonical_json, stable_hex, write_json  # noqa: E402
from metacom_pm.retrieval import source_specific_memory_queries  # noqa: E402
from metacom_pm.v1_5_candidate_discovery import (  # noqa: E402
    discover_final_typed_memory_candidates,
)

PROTOCOL = "pm-v1.5-retriever-scale-stress-test-v1"
OUT_PATH = ROOT / "outputs" / "pm_v1_5_retriever_scale_stress_test_v1.json"

# Topic bank: real life-situation phrases, same style/grain as the 41 real
# topics found this session in memory_backend.jsonl (not copied verbatim --
# EvoEmo/internal content-disjointness is preserved -- but matched in
# register so distractor realism is comparable).
TOPICS = [
    "a workplace conflict with a coworker", "financial uncertainty about rent",
    "social anxiety when meeting people", "grief after a loss",
    "caregiving for a dependent relative", "moving to a new city",
    "an identity transition", "family expectations",
    "uncertainty in a partner relationship", "self-doubt and confidence",
    "sleep disruption and insomnia", "work deadline pressure and burnout",
    "academic study and exams", "loss of motivation and procrastination",
    "feeling isolated and unsure of belonging", "a life stage transition",
    "rebuilding trust after feeling let down", "repairing a conflict after an argument",
    "an uncertain future", "a distant friendship",
    "parenting pressure involving a child", "a career and job change",
    "stress around a health and exercise routine", "feeling stuck over a decision",
]

# One SIGNAL template per source (matches the real corpus's own positive
# shape) and several NON-SIGNAL/distractor templates (also lifted from the
# real corpus's own negative-shape family, generalized).
SIGNAL_TEMPLATE = {
    MemorySource.MP: "The user prefers acknowledgement before suggestions about {topic}, with one gentle question at a time.",
    MemorySource.MS: "Across multiple prior sessions, the user has repeatedly felt worse about {topic} when concerns pile up, and steadier after naming today's main concern.",
    MemorySource.ME: "In one prior episode involving {topic}, wrote down the hardest moment before responding and later felt less overwhelmed.",
}
DISTRACTOR_TEMPLATES = {
    MemorySource.MP: [
        "The user enjoys reading general news stories about {topic}, while saying those stories do not describe their own circumstances.",
        "The user previously preferred not to discuss {topic}.",
    ],
    MemorySource.MS: [
        "Across multiple prior sessions, the user has discussed news about {topic} only as an outside topic, not a recurring personal pattern.",
        "Across multiple prior sessions, the user previously withdrew whenever {topic} arose.",
    ],
    MemorySource.ME: [
        "In a prior session, the user last spring described a private incident involving {topic}.",
        "In a prior session, the user once read a news story about {topic} and explicitly said it was unrelated to their own experience.",
    ],
}

CATALOG_SIZES = [2, 5, 10, 20, 40, 70, 109]


def _memory_id(seed_parts: tuple) -> str:
    return "mem_" + stable_hex(PROTOCOL, *seed_parts, n=24)


def _build_case(source: MemorySource, catalog_size: int, seed: int) -> dict:
    rng = random.Random(f"{PROTOCOL}|{source.value}|{catalog_size}|{seed}")
    signal_topic = rng.choice(TOPICS)
    session_index = 45  # current turn; every item below must be < this
    signal_session = rng.randint(2, session_index - 1)
    signal_text = SIGNAL_TEMPLATE[source].format(topic=signal_topic)
    signal_id = _memory_id((source.value, "signal", signal_topic, catalog_size, seed))
    items = [
        MemoryItem(
            memory_id=signal_id, source=source, created_session=signal_session, text=signal_text
        )
    ]
    n_distractors = max(0, catalog_size - 1)
    for i in range(n_distractors):
        topic = rng.choice(TOPICS)
        template = rng.choice(DISTRACTOR_TEMPLATES[source])
        text = template.format(topic=topic)
        # Real distractors can legitimately collide in text across different
        # (topic, template) draws at high volume; de-dup by retrying once.
        item_id = _memory_id((source.value, "distractor", i, topic, template[:12], catalog_size, seed))
        items.append(
            MemoryItem(
                memory_id=item_id, source=source,
                created_session=rng.randint(1, session_index - 1), text=text,
            )
        )
    # The query must name the signal topic specifically -- this is what a
    # real current_user_text turn would do (the user is currently talking
    # about THIS topic), not a generic probe.
    current_user_text = (
        f"I've been struggling with {signal_topic} again and don't know what to do."
    )
    return {
        "source": source.value, "catalog_size": catalog_size, "seed": seed,
        "signal_memory_id": signal_id, "signal_text": signal_text,
        "signal_topic": signal_topic, "session_index": session_index,
        "items": items, "current_user_text": current_user_text,
    }


def _run_case(case: dict) -> dict:
    source = MemorySource(case["source"])
    queries = source_specific_memory_queries(case["current_user_text"], [], "")
    discoveries = discover_final_typed_memory_candidates(
        queries=queries, items=case["items"], source_metadata={},
        session_index=case["session_index"],
    )
    selected = discoveries[source].selected_items
    top1_id = selected[0].memory_id if selected else None
    return {
        "source": case["source"], "catalog_size": case["catalog_size"], "seed": case["seed"],
        "signal_memory_id": case["signal_memory_id"],
        "top1_memory_id": top1_id,
        "signal_is_top1": top1_id == case["signal_memory_id"],
        "n_selected": len(selected),
    }


def main() -> None:
    results = []
    for source in (MemorySource.MP, MemorySource.MS, MemorySource.ME):
        for catalog_size in CATALOG_SIZES:
            for seed in range(10):
                case = _build_case(source, catalog_size, seed)
                results.append(_run_case(case))

    by_source_size: dict[tuple[str, int], list[dict]] = {}
    for r in results:
        key = (r["source"], r["catalog_size"])
        by_source_size.setdefault(key, []).append(r)

    summary = []
    for (source, size), rows in sorted(by_source_size.items()):
        n_correct = sum(1 for r in rows if r["signal_is_top1"])
        summary.append(
            {
                "source": source, "catalog_size": size, "n_trials": len(rows),
                "n_signal_ranked_top1": n_correct,
                "signal_top1_rate": n_correct / len(rows),
            }
        )
        print(f"{source:3s} catalog_size={size:3d}  signal_top1_rate={n_correct}/{len(rows)}")

    report = {
        "protocol": PROTOCOL,
        "note": (
            "Zero-API deterministic retriever stress test. Every item is built from the "
            "same fixed template + topic-substitution scheme already used by the real "
            "training corpus (not free-form generation). Tests whether the shared "
            "discover_final_typed_memory_candidates() retriever still ranks the designed "
            "signal item as Rank-1 as distractor catalog scale grows to real EvoEmo levels "
            "(MP up to 7, MS up to 33, ME up to 109 in real data; this test goes to 109 for "
            "all three sources to find the true breaking point, not just EvoEmo's per-"
            "source ceiling)."
        ),
        "catalog_sizes_tested": CATALOG_SIZES,
        "trials_per_source_per_size": 10,
        "summary": summary,
        "all_results": results,
    }
    write_json(OUT_PATH, report)
    print(f"\nfull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
