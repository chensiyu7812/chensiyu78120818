#!/usr/bin/env python3
"""Freeze the fresh post-repair memory-head confirmation partition.

The V1 confirmation was consumed when the lexical-only model exposed a stable
false-on pattern.  This V2 partition uses new users, semantic families, and
surface wording.  It is generated only after BGE was rejected and the
component-specific conservative floor policy was frozen.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-memory-opportunity-fresh-confirmation-v2"
STATUS = "PASS_FROZEN_FRESH_POST_REPAIR_CONFIRMATION_V2"

FRESH_CONFIRM_THEMES = (
    ("ceramics studio", "A ceramics session became discouraging when the clay collapsed twice.", "Working on one small shape made the following studio visit easier to enter.", "The ceramics studio opens again tomorrow, and I feel discouraged before trying the wheel."),
    ("history seminar", "A seminar felt uncomfortable after a comment received no response.", "Preparing one concrete question made the next discussion less intimidating.", "The history seminar meets again soon, and I am hesitant to speak after the earlier silence."),
    ("rowing practice", "A rowing practice felt confusing when the group changed pace without warning.", "Watching one timing cue made the next practice more manageable.", "Rowing practice resumes this week, and the changing pace is already making me tense."),
    ("theater crew", "A theater task became stressful when several backstage requests arrived together.", "Clarifying one backstage responsibility made the next rehearsal smoother.", "The theater crew is busy again, and I feel overwhelmed by what everyone may need from me."),
    ("coding meetup", "A meetup felt exposing after a demo stopped working midway through.", "Explaining one small part first made the next meetup easier to approach.", "There is another coding meetup tonight, and I feel uneasy about showing unfinished work."),
    ("museum group", "A museum outing felt lonely when conversation formed into closed pairs.", "Walking with one approachable person made the next visit less isolating.", "The museum group is meeting again, and I worry I will end up on the edge of the conversation."),
    ("sewing lesson", "A sewing lesson became frustrating when several corrections were given at once.", "Practicing one seam made the following lesson feel possible again.", "My sewing lesson is coming up, and I am tense about being corrected in front of everyone."),
    ("birdwatching walk", "A birdwatching walk felt disappointing when unfamiliar calls were hard to follow.", "Learning one call made the next walk more enjoyable and less pressured.", "The birdwatching walk is this weekend, and I feel behind before it has even begun."),
)


def _builder_module():
    path = ROOT / "scripts/v1_5/24dw_freeze_memory_opportunity_supplement_v1_5.py"
    spec = importlib.util.spec_from_file_location("memory_supplement_builder", path)
    if not spec or not spec.loader:
        raise RuntimeError("cannot load memory supplement builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(*, root: Path = ROOT):
    return _builder_module().build(
        root=root,
        protocol=PROTOCOL,
        status=STATUS,
        out_dir_name="pm_v1_5_memory_opportunity_fresh_confirmation_v2",
        confirmation_themes=FRESH_CONFIRM_THEMES,
        user_prefix="pmv15_memv2",
    )


def main() -> None:
    report = build()
    print(
        {
            "protocol": report["protocol"],
            "status": report["status"],
            "counts": report["counts"],
            "checks": report["checks"],
        }
    )


if __name__ == "__main__":
    main()
