#!/usr/bin/env python3
"""Balanced, content-independent qualification set for the ME 3-state
action-readiness observer (v1_5_v5_3_action_readiness.py).

Supersedes/extends script 63. Script 63 established real-world frequency
and false-positive behavior on 70 real EvoEmo turns, but that sample had
only 2 real INVITES_ACTION examples and 0 real DECLINES_ACTION examples --
too sparse to estimate recall for either rare class. Per the correction:
the 70 real samples stay as the real-frequency / false-positive audit and
are NOT reused here to estimate rare-class recall.

This is a separate, deliberately BALANCED set: 3 classes x 6 surface-form
templates x 10 topics = 180 examples, all newly authored (no EvoEmo/
ES-MemEval sentences copied or paraphrased), covering:
  - INVITES_ACTION: direct ("what should I do"), advice-request,
    try-something, INDIRECT resource-seeking ("where can I go for X" --
    the gap script 63 found), openness, and suggest-request phrasings;
  - DECLINES_ACTION: internally constructed natural refusals (EvoEmo has
    structurally none, per script 63's corpus-wide search), covering
    vent-not-fix, don't-tell-me-what-to-do, just-listen, already-know,
    not-yet-ready, and vent-not-fix-2 phrasings;
  - UNKNOWN: disclosure, reflection, gratitude, backstory, observation,
    and venting-update -- conversational functions with no action-readiness
    cue, matching the shape of the real EvoEmo turns that dominate script
    63's sample.

Split hygiene: every row carries semantic_family (topic), template_family
(class + surface-form id), and user_id (one synthetic persona per topic).
No template is reworded across topics into a different semantic_family
label, and no (template_family, semantic_family) pair repeats -- reported
explicitly below rather than assumed.

Zero API calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.io import write_json  # noqa: E402
from metacom_pm.v1_5_v5_3_action_readiness import (  # noqa: E402
    ActionReadiness,
    observe_action_readiness,
)

TOPICS = [
    ("work_stress", "the pressure at work"),
    ("relationship_conflict", "the tension with my partner"),
    ("health_anxiety", "my worries about my health"),
    ("sleep_issues", "how badly I've been sleeping"),
    ("family_caregiving", "taking care of my mother"),
    ("academic_pressure", "the workload in my classes"),
    ("financial_worry", "how tight money has been"),
    ("social_isolation", "how disconnected I feel from friends"),
    ("grief_loss", "losing my grandfather"),
    ("self_esteem", "how I've been feeling about myself"),
]

INVITES_TEMPLATES = [
    ("invite_direct_what_should_i_do", lambda t: f"What should I do about {t}?"),
    ("invite_advice_request", lambda t: f"Do you have any advice for handling {t}?"),
    ("invite_try_something", lambda t: f"Is there something I could try for {t}?"),
    ("invite_indirect_resource", lambda t: f"Where can I go for help with {t}?"),
    ("invite_openness", lambda t: f"I'd be open to hearing an idea for dealing with {t}."),
    ("invite_suggest_request", lambda t: f"Could you suggest a way to approach {t}?"),
]

DECLINES_TEMPLATES = [
    ("decline_vent_not_fix", lambda t: f"I don't want advice about {t} right now, I just need to vent."),
    ("decline_dont_tell_me", lambda t: f"Please don't tell me what to do about {t}, I just need you to listen."),
    ("decline_just_listen", lambda t: f"I'm not looking for suggestions on {t}, just someone to hear me out."),
    ("decline_already_know", lambda t: f"I already know what to do about {t}, I just want to talk it through."),
    ("decline_not_yet_ready", lambda t: f"Let's not jump to solutions for {t} yet, I just need to get this off my chest."),
    ("decline_vent_not_fix_2", lambda t: f"I just need to vent about {t}, not fix it right now."),
]

UNKNOWN_TEMPLATES = [
    ("unknown_disclosure", lambda t: f"{t.capitalize()} has been really hard for me lately."),
    ("unknown_reflection", lambda t: f"I've been thinking a lot about how {t} started."),
    ("unknown_gratitude", lambda t: f"Thanks for listening about {t}, it means a lot."),
    ("unknown_backstory", lambda t: f"My sister doesn't really understand what {t} has been like."),
    ("unknown_observation", lambda t: f"I noticed {t} again today and it caught me off guard."),
    ("unknown_venting_update", lambda t: f"It's been a rollercoaster dealing with {t} this week."),
]

CLASS_TEMPLATES = {
    "INVITES_ACTION": INVITES_TEMPLATES,
    "DECLINES_ACTION": DECLINES_TEMPLATES,
    "UNKNOWN": UNKNOWN_TEMPLATES,
}

OUT_PATH = ROOT / "outputs/pm_v1_5_v5_3_me_action_readiness_balanced_qualification_v1/report.json"


def build_rows() -> list[dict]:
    rows = []
    for topic_idx, (family, topic_phrase) in enumerate(TOPICS):
        user_id = f"synth_u{topic_idx:02d}"
        for label, templates in CLASS_TEMPLATES.items():
            for template_id, fn in templates:
                rows.append({
                    "manual_label": label,
                    "semantic_family": family,
                    "template_family": template_id,
                    "user_id": user_id,
                    "text": fn(topic_phrase),
                })
    return rows


def main() -> None:
    rows = build_rows()

    # Split hygiene: no (template_family, semantic_family) pair repeats.
    pairs = [(r["template_family"], r["semantic_family"]) for r in rows]
    assert len(pairs) == len(set(pairs)), "duplicate (template_family, semantic_family) pair"

    for r in rows:
        r["predicted_label"] = observe_action_readiness(r["text"]).value
        r["agrees"] = r["predicted_label"] == r["manual_label"]

    n = len(rows)
    agree = sum(r["agrees"] for r in rows)
    per_class = {}
    confusion: dict[str, dict[str, int]] = {}
    for label in CLASS_TEMPLATES:
        subset = [r for r in rows if r["manual_label"] == label]
        recall = sum(r["agrees"] for r in subset) / len(subset)
        per_class[label] = {"n": len(subset), "recall": recall}
        confusion[label] = {}
        for r in subset:
            confusion[label][r["predicted_label"]] = confusion[label].get(r["predicted_label"], 0) + 1

    misses = [
        {"text": r["text"], "manual_label": r["manual_label"], "predicted_label": r["predicted_label"],
         "template_family": r["template_family"], "semantic_family": r["semantic_family"]}
        for r in rows if not r["agrees"]
    ]

    report = {
        "protocol": "pm-v1.5-v5.3-me-action-readiness-balanced-qualification-v1",
        "note": (
            "Deliberately balanced (60 per class), newly authored, "
            "content-disjoint from EvoEmo/ES-MemEval and from the real "
            "70-sample audit (script 63). Split hygiene: no "
            "(template_family, semantic_family) pair repeats -- checked "
            "programmatically, not just asserted."
        ),
        "n_total": n,
        "overall_agreement_rate": agree / n,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "n_semantic_families": len(TOPICS),
        "n_template_families": sum(len(v) for v in CLASS_TEMPLATES.values()),
        "n_users": len(TOPICS),
        "misses": misses,
        "rows": rows,
        "api_calls": 0,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUT_PATH, report)
    print(f"n_total={n} overall_agreement={agree/n:.3f}")
    for label, stats in per_class.items():
        print(f"  {label}: n={stats['n']} recall={stats['recall']:.3f}")
    print(f"\nmisses ({len(misses)}):")
    for m in misses:
        print(f"  [{m['manual_label']}->{m['predicted_label']}] {m['text']}")
    print(f"\nfull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
