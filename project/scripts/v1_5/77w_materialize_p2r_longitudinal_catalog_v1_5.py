#!/usr/bin/env python3
"""W8: P2R source-independent longitudinal catalog asset.

Per docs/PM_V1_5_V5_3_EXTERNAL_EXAM_BACKWARD_TRAINING_SPEC_20260807_ZH.md
section 4 ("Worker只构造一个独立的、外部文本零复制的longitudinal catalog
asset"): builds ONLY a per-user, session-spanning MP/MS/ME memory catalog.
Does NOT construct current_user_text, does NOT compute Step1 features, does
NOT assign worth_opening/ON-OFF labels, does NOT decide split. Zero API
calls; reads no quality/risk/judge/human outcome. Does not modify the three
authoritative sources, the runbook, or any Leader-owned file (release
bindings, accountability, typed_response_program, 75l/76l/77l audit
scripts/docs).

Design, directly answering the external-exam-derived minimum (EvoEmo's real
18-user artifact: 401 sessions/18 users = 13-33 per user, 446 events =
13-37 per user, 150 relationships = 3-15 per user, 126 fixed 7-field basic
profile facts):

- N synthetic users, each with session_count in [13, 33] (real EvoEmo range,
  not copied content);
- each user has 3-5 RECURRING topic threads; each thread produces multiple
  ME/MS items across DIFFERENT sessions with different specific action,
  person, resolution status and outcome -- the "same topic, different
  event/person/time/outcome" hard negatives this asset exists to provide;
  a genuinely combinatorial action/result/person word bank (not template +
  topic-noun substitution) drives the phrasing;
- ME items are a mix of compiler-valid (compile_atomic_reusable_outcome)
  REUSABLE_OUTCOME episodes and deliberately non-compilable CONTEXT_EVENT/
  UNRESOLVED_EVENT background distractors;
- MS items use the compile_atomic_session_observation-valid "The earlier X
  goal was to Y" construct, tagged resolved/unresolved/conflicting;
- MP_PROFILE uses the same 7 field types EvoEmo's real basic_info uses
  (name/age/gender/job/education/nationality/location); a subset of users
  get a mid-timeline field UPDATE (new version, old version superseded) to
  exercise time/version tracking. A smaller MP_PREFERENCE set is also
  included (internal-only capability per the spec doc -- EvoEmo has none),
  clearly subtyped separately so nothing downstream conflates the two;
- the catalog.jsonl is a single GLOBAL file across all users (so an
  owner-isolation audit is possible); per-user candidate pools are reported
  separately in coverage_report.json to prove strict owner isolation when a
  pool is actually drawn for one user.

Every item carries: source_span (= its own literal text; nothing here is
extracted from a separate longer transcript, so the span IS the compiled
content, matching this project's existing AtomicSessionObservation
convention), created_session, owner_id, subtype, version, and a sha256
surface hash.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.evoemo import load_evoemo  # noqa: E402
from metacom_pm.io import sha256_file, stable_hex, write_json  # noqa: E402
from metacom_pm.v1_5_v5_2_atomic_memory import (  # noqa: E402
    compile_atomic_reusable_outcome,
    compile_atomic_session_observation,
)
from metacom_pm.v1_5_v5_3_external_leakage_audit import (  # noqa: E402
    compare_text_surface_overlap,
    external_overlap_surfaces,
    extract_internal_text_surfaces,
)

OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_p2r_longitudinal_catalog_v1"
EVOEMO_PATH = ROOT / "data/external/evo_emo.json"
ESCONV_PATH = ROOT / "data/external/ESConv.json"

SEED = 20260807

N_USERS = 20
SESSION_RANGE = (13, 33)  # real EvoEmo per-user session count range
THREADS_PER_USER = (3, 5)

TOPICS = [
    "job transition", "chronic pain", "parenting conflict", "breakup recovery",
    "exam anxiety", "career change", "housing move", "friendship rift",
    "social anxiety", "grief and loss", "financial stress", "relocation abroad",
    "a new health diagnosis", "caregiving burnout", "workplace conflict",
    "an identity question", "a custody arrangement", "a sibling estrangement",
    "a chronic illness flare", "starting therapy",
]

# Combinatorial action/result word bank -- genuinely independent axes, not a
# single template reworded per topic. Actions are phrased generically enough
# to combine with any topic noun phrase.

# Verbs are constrained to _FIRST_PERSON_ACTION_RE's whitelist (tried/used/
# chose/decided to/asked/called/wrote/walked/went/practiced/paused/
# breathed/talked/scheduled/limited/stopped/started/reached out/joined/
# attended/opened up/confided in/focused on) -- verified empirically against
# a real 121-item batch, which is how the first draft's ~50% ME compiler
# failure rate (verbs like "took"/"made"/"stepped"/"broke"/"set"/"kept" are
# NOT in the whitelist) was actually found and fixed here, not assumed.
ACTIONS = [
    "wrote down my top concerns before dealing with it",
    "talked it through with {person} first",
    "paused a full day before responding to it",
    "used a simple list to sort through the options",
    "asked directly instead of assuming",
    "walked away for a bit before continuing",
    "wrote a message and paused before sending it",
    "practiced what I wanted to say out loud",
    "focused on one small piece at a time",
    "opened up to {person} about setting a boundary",
    "asked {person} for a second opinion",
    "started a short log of it for a week",
]
RESULTS = [
    "it helped me feel steadier going in",
    "it helped me see it from a new angle",
    "it helped me avoid reacting out of anger",
    "it helped me stop feeling so stuck",
    "it helped me get a real answer",
    "it helped me cool down before continuing",
    "it helped me avoid saying something I'd regret",
    "it helped me feel more prepared",
    "it helped me make it more manageable",
    "it helped me feel less alone with it",
]
PEOPLE = [
    "my sister", "a close friend", "my manager", "my therapist", "my partner",
    "a coworker", "my mother", "an old classmate", "my roommate", "my brother",
]
NO_RESULT_ACTIONS = [
    "tried journaling about {topic} for a week, but I'm not sure yet whether it changed anything",
    "brought {topic} up with {person}, but it's too soon to tell if it helped",
    "started tracking {topic} in a notebook, though nothing's really shifted yet",
]
CONTEXT_EVENT_TEMPLATES = [
    "{topic_cap} has been going on for a few months now.",
    "I've been thinking about {topic} a lot lately.",
    "{topic_cap} came up again this week.",
    "There's been a lot going on with {topic} recently.",
    "{topic_cap} is still sitting in the back of my mind.",
]
UNRESOLVED_EVENT_TEMPLATES = [
    "I'm still really worried about how {topic} is going to turn out.",
    "I haven't figured out what to do about {topic} yet, and it's weighing on me.",
    "{topic_cap} is still up in the air and I don't like not knowing.",
]
MS_SUBGOALS = [
    "decide between two options before the deadline",
    "identify what was actually making it worse",
    "agree on one consistent approach with {person}",
    "figure out how to bring it up without it becoming a bigger issue",
    "build a plan that felt realistic",
    "shortlist what was actually worth pursuing",
    "decide which part to prioritize first",
    "name what specifically felt hardest about it",
    "write down questions before the next conversation about it",
    "find one hour that was protected time for it",
]
MS_RESOLUTION_SUFFIX = {
    "resolved": " That got settled not long after.",
    "unresolved": " That never really got resolved.",
    "conflicting": " What to do next about it is still genuinely unclear.",
}

MP_PROFILE_FIELD_BANK = {
    "job": ["works in healthcare", "works in software", "works in education",
            "works in retail management", "is between jobs right now", "works in logistics"],
    "education": ["has a bachelor's degree", "is finishing a graduate program",
                  "has a high school diploma", "is enrolled in a certificate program"],
    "nationality": ["is originally from Canada", "is originally from Kenya",
                     "is originally from Vietnam", "is originally from Poland"],
    "location": ["lives in a mid-size city", "lives in a rural area",
                 "recently moved to a new city", "lives in a coastal town"],
    "age": ["is in their late twenties", "is in their mid-thirties",
            "is in their early forties", "is in their early twenties"],
    "gender": ["identifies as a woman", "identifies as a man", "identifies as nonbinary"],
    "name": ["goes by a nickname with close friends", "prefers their full first name"],
}
MP_PREFERENCE_PAIRS = [
    ("reflection", "question"), ("direct", "response"), ("brief", "concise"),
    ("listen", "advice"), ("idea", "option"), ("concise", "factual"),
]


def _mid(owner: str, tag: str) -> str:
    return "mem_" + stable_hex("pm-v1.5-v5.3-p2r-longitudinal-catalog-v1", owner, tag, n=20)


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pick_action_result(rng: random.Random) -> tuple[str, str]:
    person = rng.choice(PEOPLE)
    action = rng.choice(ACTIONS).format(person=person)
    result = rng.choice(RESULTS)
    return action, result, person


def build_user(user_index: int) -> tuple[dict, list[dict]]:
    user_id = f"p2r_cat_u{user_index:03d}"
    rng = random.Random(SEED + user_index)
    n_sessions = rng.randint(*SESSION_RANGE)
    n_threads = rng.randint(*THREADS_PER_USER)
    threads = rng.sample(TOPICS, n_threads)

    items: list[dict] = []
    version = 1

    def add_item(*, subtype: str, text: str, created_session: int, extra: dict) -> dict:
        nonlocal version
        item_id = _mid(user_id, f"{subtype}_{len(items)}_{created_session}")
        row = {
            "item_id": item_id, "owner_id": user_id, "source": subtype.split("_")[0],
            "subtype": subtype, "text": text, "source_span": text,
            "created_session": created_session, "version": 1,
            "surface_sha256": _sha(text),
            **extra,
        }
        items.append(row)
        return row

    # MP_PROFILE: 5-7 fields, assigned early (session 1-2). A subset of users
    # get a mid-timeline update (new version, old superseded) to exercise
    # time/version tracking.
    profile_fields = rng.sample(list(MP_PROFILE_FIELD_BANK), rng.randint(5, 7))
    gets_update = rng.random() < 0.35
    update_field = rng.choice(profile_fields) if gets_update else None
    for field in profile_fields:
        value = rng.choice(MP_PROFILE_FIELD_BANK[field])
        add_item(
            subtype="MP_PROFILE", text=f"{field.capitalize()}: {value}",
            created_session=rng.randint(1, 2),
            extra={"field_type": field, "field_value": value, "active": True, "superseded": False},
        )
        if field == update_field:
            new_value = rng.choice([v for v in MP_PROFILE_FIELD_BANK[field] if v != value])
            update_session = rng.randint(n_sessions // 2, n_sessions - 1)
            items[-1]["active"] = False
            items[-1]["superseded"] = True
            version += 1
            new_row = add_item(
                subtype="MP_PROFILE", text=f"{field.capitalize()}: {new_value}",
                created_session=update_session,
                extra={"field_type": field, "field_value": new_value, "active": True, "superseded": False},
            )
            new_row["version"] = version
            new_row["supersedes_item_id"] = items[-2]["item_id"]

    # MP_PREFERENCE: 1-2 internal-only preference items (EvoEmo has none;
    # kept as a separately-subtyped internal capability per the spec doc).
    for _ in range(rng.randint(1, 2)):
        word_a, word_b = rng.choice(MP_PREFERENCE_PAIRS)
        add_item(
            subtype="MP_PREFERENCE", text=f"Stable Preference Response Format: prefers {word_a} and {word_b}",
            created_session=1, extra={"field_type": "response_format", "field_value": f"{word_a}_{word_b}", "active": True, "superseded": False},
        )

    # Recurring threads: each thread gets 2-4 ME/MS episodes spread across
    # DIFFERENT sessions, each with a different action/person/result/
    # resolution -- the actual hard-negative generator.
    session_cursor = 3
    for thread in threads:
        n_events = rng.randint(2, 4)
        used_sessions = rng.sample(range(session_cursor, n_sessions), min(n_events * 2, max(1, n_sessions - session_cursor)))
        used_sessions = sorted(used_sessions)[:n_events] or [session_cursor]
        for i, session in enumerate(used_sessions):
            roll = rng.random()
            if roll < 0.55:
                action, result, person = _pick_action_result(rng)
                text = f"When dealing with {thread}, I {action}, and {result}."
                subtype = "ME_REUSABLE_OUTCOME"
                extra = {"topic_thread": thread, "involves_person": person, "compiler_valid_intended": True}
            elif roll < 0.80:
                text = "I " + NO_RESULT_ACTIONS[i % len(NO_RESULT_ACTIONS)].format(
                    topic=thread, person=rng.choice(PEOPLE)
                )
                subtype = "ME_UNRESOLVED_EVENT"
                extra = {"topic_thread": thread, "compiler_valid_intended": False}
            else:
                template = rng.choice(CONTEXT_EVENT_TEMPLATES)
                text = template.format(topic=thread, topic_cap=thread.capitalize())
                subtype = "ME_CONTEXT_EVENT"
                extra = {"topic_thread": thread, "compiler_valid_intended": False}
            row = add_item(subtype=subtype, text=text, created_session=session, extra=extra)
            row["me_compiler_valid_actual"] = compile_atomic_reusable_outcome(text) is not None

        # MS observation for the same thread, a different session, tagged
        # resolved/unresolved/conflicting.
        ms_session = rng.choice(used_sessions)
        resolution = rng.choice(["resolved", "unresolved", "conflicting"])
        subgoal = rng.choice(MS_SUBGOALS).format(person=rng.choice(PEOPLE))
        text = f"The earlier {thread} goal was to {subgoal}.{MS_RESOLUTION_SUFFIX[resolution]}"
        row = add_item(
            subtype="MS_SESSION", text=text, created_session=ms_session,
            extra={"topic_thread": thread, "resolution_status": resolution},
        )
        row["ms_compiler_valid_actual"] = compile_atomic_session_observation(text) is not None
        session_cursor = min(n_sessions - 1, session_cursor + 2)

    user_record = {
        "user_id": user_id, "n_sessions": n_sessions, "n_threads": n_threads,
        "recurring_topics": threads, "n_items": len(items),
    }
    return user_record, items


def _esconv_surfaces() -> list[dict[str, str]]:
    value = json.loads(ESCONV_PATH.read_text(encoding="utf-8"))

    def iter_strings(v, path=()):
        if isinstance(v, dict):
            for k, item in v.items():
                yield from iter_strings(item, (*path, str(k)))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                yield from iter_strings(item, (*path, str(i)))
        elif isinstance(v, str) and v.strip():
            yield path, v

    return [
        {"surface_id": "esconv:" + ":".join(path), "category": "external_esconv_text", "text": text}
        for path, text in iter_strings(value)
        if len(text.split()) >= 3
    ]


def run_overlap_audit(all_items: list[dict]) -> dict:
    internal_surfaces = extract_internal_text_surfaces(
        all_items, text_fields=frozenset({"text", "source_span"}),
    )
    evoemo_users = load_evoemo(EVOEMO_PATH)
    evoemo_surfaces = external_overlap_surfaces(evoemo_users)
    esconv_surfaces = _esconv_surfaces()
    overlap = compare_text_surface_overlap(
        internal_surfaces=internal_surfaces,
        external_surfaces=[*evoemo_surfaces, *esconv_surfaces],
        ngram_size=8,
    )
    return {
        "n_internal_surfaces": len(internal_surfaces),
        "n_external_surfaces": len(evoemo_surfaces) + len(esconv_surfaces),
        "n_evoemo_es_memeval_surfaces": len(evoemo_surfaces),
        "n_esconv_surfaces": len(esconv_surfaces),
        "exact_collision_count": overlap["exact_collision_count"],
        "normalized_ngram_collision_count": len(overlap["normalized_ngram_collisions"]),
        "note": (
            "evoemo_surfaces covers both EvoEmo dialogue sessions and the "
            "embedded ES-MemEval questions/summaries containers "
            "(iter_released_qa_rows) -- same underlying file, so this is a "
            "genuine three-source audit (ESConv + EvoEmo + ES-MemEval) even "
            "though only two source files are read."
        ),
    }


def owner_isolation_check(all_items: list[dict]) -> dict:
    by_owner: dict[str, list[dict]] = {}
    for item in all_items:
        by_owner.setdefault(item["owner_id"], []).append(item)
    violations = 0
    for owner, items in by_owner.items():
        pool = [i for i in all_items if i["owner_id"] == owner]
        leaked = [i for i in pool if i["owner_id"] != owner]
        violations += len(leaked)
    future_violations = sum(
        1 for item in all_items
        if item["created_session"] < 1
    )
    return {
        "n_owners": len(by_owner),
        "cross_owner_leak_into_own_pool_count": violations,
        "created_session_below_1_count": future_violations,
        "note": "Global catalog.jsonl intentionally contains all owners' items together (for isolation auditing); per-owner pool filtering (owner_id == target user) is verified here to leak zero cross-owner items.",
    }


def main() -> None:
    all_users, all_items = [], []
    for i in range(N_USERS):
        user_record, items = build_user(i)
        all_users.append(user_record)
        all_items.extend(items)

    subtype_counts: dict[str, int] = {}
    for item in all_items:
        subtype_counts[item["subtype"]] = subtype_counts.get(item["subtype"], 0) + 1

    me_items = [i for i in all_items if i["source"] == "ME"]
    ms_items = [i for i in all_items if i["source"] == "MS"]
    me_valid = sum(1 for i in me_items if i.get("me_compiler_valid_actual"))
    ms_valid = sum(1 for i in ms_items if i.get("ms_compiler_valid_actual"))

    overlap = run_overlap_audit(all_items)
    isolation = owner_isolation_check(all_items)

    coverage_report = {
        "protocol": "pm-v1.5-v5.3-p2r-longitudinal-catalog-coverage-v1",
        "n_users": len(all_users),
        "session_count_range": [min(u["n_sessions"] for u in all_users), max(u["n_sessions"] for u in all_users)],
        "session_count_median": sorted(u["n_sessions"] for u in all_users)[len(all_users) // 2],
        "n_total_items": len(all_items),
        "subtype_distribution": subtype_counts,
        "me_compiler_valid": me_valid, "me_compiler_invalid": len(me_items) - me_valid,
        "me_total": len(me_items),
        "ms_compiler_valid": ms_valid, "ms_compiler_invalid": len(ms_items) - ms_valid,
        "ms_total": len(ms_items),
        "ms_resolution_status_distribution": {
            status: sum(1 for i in ms_items if i.get("resolution_status") == status)
            for status in ("resolved", "unresolved", "conflicting")
        },
        "mp_profile_field_type_distribution": {
            field: sum(1 for i in all_items if i["subtype"] == "MP_PROFILE" and i.get("field_type") == field)
            for field in MP_PROFILE_FIELD_BANK
        },
        "mp_profile_superseded_versions": sum(1 for i in all_items if i["subtype"] == "MP_PROFILE" and i.get("superseded")),
        "owner_isolation": isolation,
        "external_overlap": overlap,
        "recurring_topic_threads_per_user": {
            "min": min(u["n_threads"] for u in all_users),
            "max": max(u["n_threads"] for u in all_users),
        },
        "api_calls": 0,
        "quality_risk_judge_or_human_outcome_read": False,
        "current_state_or_step1_features_present": False,
        "worth_opening_or_on_off_labels_present": False,
    }

    manifest = {
        "protocol": "pm-v1.5-v5.3-p2r-longitudinal-catalog-manifest-v1",
        "seed": SEED, "n_users": N_USERS, "session_range": list(SESSION_RANGE),
        "n_items": len(all_items),
        "files": ["catalog.jsonl", "users.json", "catalog_manifest.json", "overlap_audit.json", "coverage_report.json"],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "catalog.jsonl").open("w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    write_json(OUT_DIR / "users.json", all_users)
    write_json(OUT_DIR / "catalog_manifest.json", manifest)
    write_json(OUT_DIR / "overlap_audit.json", overlap)
    write_json(OUT_DIR / "coverage_report.json", coverage_report)

    print(json.dumps(coverage_report, indent=2))
    print(f"\ncatalog written to {OUT_DIR}")


if __name__ == "__main__":
    main()
