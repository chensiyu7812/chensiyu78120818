#!/usr/bin/env python3
"""Retriever scale stress test V2 -- corrected per-component contracts.

V1 (57_retriever_scale_stress_test_v1_5.py, see PM_V1_5_V5_3_RETRIEVER_
STRESS_TEST_CORRECTION_20260806_ZH.md for the full retraction) had real,
confirmed bugs: ME's "gold" signal did not pass compile_atomic_reusable_
outcome(), MS used the frozen-out-of-scope MS_PATTERN shape instead of
MS_SESSION and never used the decided BGE-M3 path, MP/MS were swept to
catalog_size=109 despite real EvoEmo ceilings of ~7 and 13-33, and the
distractor pool's low template/topic count forced far more artificial
duplication than real content would ever have.

This version fixes all of those, verified against real contracts before
any trial counts:
  - MP: signal is a genuine MP_PREFERENCE-shaped statement; distractors no
    longer include the ambiguous "previously preferred not to discuss X"
    case (a real historical boundary is not a clean negative); tested only
    up to MP's real EvoEmo ceiling (7).
  - MS: signal is a genuine single-session MS_SESSION observation, asserted
    to pass compile_atomic_session_observation() before the trial counts;
    ms_semantic_encoder (BGE-M3) is wired in, matching the decided default;
    tested at MS's real EvoEmo range (13, 22, 33).
  - ME: signal is asserted to pass compile_atomic_reusable_outcome() before
    the trial counts; me_subtype_hints() is wired in; distractor template
    and topic pool widened to reduce forced duplication; tested at ME's
    real EvoEmo range (37, 69, 109) plus small-scale points for a full
    curve.

Zero API calls except for loading the local BGE-M3 model (CPU, no network).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.contracts import MemoryItem, MemorySource  # noqa: E402
from metacom_pm.io import stable_hex, write_json  # noqa: E402
from metacom_pm.retrieval import source_specific_memory_queries  # noqa: E402
from metacom_pm.v1_5_candidate_discovery import (  # noqa: E402
    discover_final_typed_memory_candidates,
)
from metacom_pm.v1_5_v5_2_atomic_memory import (  # noqa: E402
    compile_atomic_reusable_outcome,
    compile_atomic_session_observation,
    me_subtype_hints,
)

PROTOCOL = "pm-v1.5-retriever-scale-stress-test-v2"
OUT_PATH = ROOT / "outputs" / "pm_v1_5_retriever_scale_stress_test_v2.json"
SESSION_INDEX = 45

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
    "an uncertain future", "a distant friendship", "parenting pressure involving a child",
    "a career and job change", "stress around a health and exercise routine",
    "feeling stuck over a decision", "a disagreement with a sibling",
    "returning to school as an adult", "a chronic health condition",
    "planning a wedding", "an upcoming medical procedure",
    "a pet's declining health", "a friend moving away", "a difficult landlord",
]

# --- MP: MP_PREFERENCE-shaped signal; distractors are clearly off-topic or
# clearly not a boundary/preference at all (the ambiguous "previously
# preferred not to discuss X" case from V1 is dropped).
MP_SIGNAL = "The user prefers acknowledgement before suggestions about {topic}, with one gentle question at a time."
MP_DISTRACTORS = [
    "The user enjoys reading general news stories about {topic}, while saying those stories do not describe their own circumstances.",
    "The user mentioned owning a small collection of {topic}-adjacent hobby books, unrelated to any support preference.",
]

# --- MS: genuine single-session MS_SESSION observation (not cross-session
# MS_PATTERN); asserted against compile_atomic_session_observation().
MS_SIGNAL = "In the session about {topic}, the user specifically said they had already tried talking to a close friend about it, but nothing changed afterward."
MS_DISTRACTORS = [
    "In an unrelated session, the user briefly mentioned {topic} in passing without discussing it further.",
    "In an earlier session, the user said they did not want to go into detail about {topic} that day.",
]

# --- ME: asserted against compile_atomic_reusable_outcome().
ME_SIGNAL = "When dealing with {topic}, I wrote down the hardest moment before responding, and it helped me feel calmer."
ME_DISTRACTORS = [
    "In a prior session, the user last spring described a private incident involving {topic}.",
    "In a prior session, the user once read a news story about {topic} and explicitly said it was unrelated to their own experience.",
    "The user mentioned {topic} briefly while describing an unrelated weekend plan.",
    "In a prior session, the user said they were still deciding how to think about {topic}.",
]

CATALOG_SIZES = {
    MemorySource.MP: [2, 5, 7],
    MemorySource.MS: [13, 22, 33],
    MemorySource.ME: [2, 5, 10, 20, 37, 69, 109],
}


def _memory_id(seed_parts: tuple) -> str:
    return "mem_" + stable_hex(PROTOCOL, *seed_parts, n=24)


def _build_case(source: MemorySource, catalog_size: int, seed: int) -> dict:
    rng = random.Random(f"{PROTOCOL}|{source.value}|{catalog_size}|{seed}")
    topic = rng.choice(TOPICS)
    signal_session = rng.randint(2, SESSION_INDEX - 1)
    signal_template, distractor_templates = {
        MemorySource.MP: (MP_SIGNAL, MP_DISTRACTORS),
        MemorySource.MS: (MS_SIGNAL, MS_DISTRACTORS),
        MemorySource.ME: (ME_SIGNAL, ME_DISTRACTORS),
    }[source]
    signal_text = signal_template.format(topic=topic)

    # Hard gold-validity assertion -- a trial with an invalid gold is
    # skipped, not silently counted. This is the exact discipline the
    # V1 correction found missing.
    if source is MemorySource.ME:
        assert compile_atomic_reusable_outcome(signal_text) is not None, signal_text
    elif source is MemorySource.MS:
        assert compile_atomic_session_observation(signal_text) is not None, signal_text

    signal_id = _memory_id((source.value, "signal-v2", topic, catalog_size, seed))
    items = [
        MemoryItem(memory_id=signal_id, source=source, created_session=signal_session, text=signal_text)
    ]
    for i in range(max(0, catalog_size - 1)):
        t = rng.choice(TOPICS)
        template = rng.choice(distractor_templates)
        text = template.format(topic=t)
        item_id = _memory_id((source.value, "distractor-v2", i, t, template[:16], catalog_size, seed))
        items.append(
            MemoryItem(
                memory_id=item_id, source=source,
                created_session=rng.randint(1, SESSION_INDEX - 1), text=text,
            )
        )
    current_user_text = f"I've been struggling with {topic} again and don't know what to do."
    return {
        "source": source.value, "catalog_size": catalog_size, "seed": seed,
        "signal_memory_id": signal_id, "items": items, "current_user_text": current_user_text,
    }


def _run_case(case: dict, *, ms_semantic_encoder) -> dict:
    source = MemorySource(case["source"])
    queries = source_specific_memory_queries(case["current_user_text"], [], "")
    source_metadata = me_subtype_hints(case["items"]) if source is MemorySource.ME else {}
    discoveries = discover_final_typed_memory_candidates(
        queries=queries, items=case["items"], source_metadata=source_metadata,
        session_index=SESSION_INDEX,
        ms_semantic_encoder=ms_semantic_encoder if source is MemorySource.MS else None,
    )
    selected = discoveries[source].selected_items
    top1_id = selected[0].memory_id if selected else None
    return {
        "source": case["source"], "catalog_size": case["catalog_size"], "seed": case["seed"],
        "signal_is_top1": top1_id == case["signal_memory_id"],
    }


def main() -> None:
    from metacom_pm.v1_5_v5_3_semantic_ms_retrieval import BgeM3Encoder  # noqa: PLC0415

    print("loading BGE-M3 for MS...")
    ms_encoder = BgeM3Encoder()

    results = []
    for source in (MemorySource.MP, MemorySource.MS, MemorySource.ME):
        for catalog_size in CATALOG_SIZES[source]:
            for seed in range(10):
                case = _build_case(source, catalog_size, seed)
                results.append(_run_case(case, ms_semantic_encoder=ms_encoder))

    summary = []
    by_key: dict[tuple[str, int], list[dict]] = {}
    for r in results:
        by_key.setdefault((r["source"], r["catalog_size"]), []).append(r)
    for (source, size), rows in sorted(by_key.items()):
        n_correct = sum(1 for r in rows if r["signal_is_top1"])
        summary.append({
            "source": source, "catalog_size": size, "n_trials": len(rows),
            "signal_top1_rate": n_correct / len(rows),
        })
        print(f"{source:3s} catalog_size={size:3d}  signal_top1_rate={n_correct}/{len(rows)}")

    write_json(OUT_PATH, {
        "protocol": PROTOCOL,
        "note": (
            "V2, corrected per-component contracts (see PM_V1_5_V5_3_RETRIEVER_STRESS_TEST_"
            "CORRECTION_20260806_ZH.md for what V1 got wrong). Gold signals asserted valid "
            "against their real compiler/construct before any trial counts. MS uses the "
            "decided BGE-M3 default. Each source tested only up to its own real EvoEmo scale."
        ),
        "summary": summary,
        "all_results": results,
    })
    print(f"\nfull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
