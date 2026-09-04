#!/usr/bin/env python3
"""Freeze the untouched V3 confirmation for the one representation repair.

The content-word representation and all training hyperparameters are fixed
before this partition is built.  The rows are zero-API, outcome-blind, and use
new users, themes, and wording.  Only the confirmation role is emitted; these
rows may never enter fit or threshold selection.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from metacom_pm.v1_5_memory_opportunity_features import (
    SCALE_STABLE_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL,
    build_scale_stable_memory_opportunity_observation,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-memory-opportunity-scale-stable-confirmation-v3"
STATUS = "PASS_FROZEN_UNTOUCHED_SCALE_STABLE_CONFIRMATION_V3"

FRESH_CONFIRM_THEMES = (
    ("pottery market", "A pottery stall felt discouraging when several pieces drew no attention.", "Arranging one small group of pieces made the next market easier to enter.", "The pottery market opens again this weekend, and I feel discouraged before setting up."),
    ("astronomy club", "A club discussion felt exposing after an observation was corrected abruptly.", "Preparing one specific observation made the next meeting less intimidating.", "The astronomy club meets soon, and I am hesitant to speak after the earlier correction."),
    ("kayaking lesson", "A lesson felt confusing when the group changed direction without warning.", "Watching one paddle cue made the following lesson more manageable.", "Kayaking starts again this week, and the changing pace is already making me tense."),
    ("film project", "A film task became stressful when several production requests arrived together.", "Clarifying one production responsibility made the next session smoother.", "The film project is busy again, and I feel overwhelmed by what everyone may need from me."),
    ("robotics circle", "A demonstration felt exposing after the prototype stopped working midway through.", "Explaining one small mechanism first made the next meeting easier to approach.", "There is another robotics meeting tonight, and I feel uneasy about showing unfinished work."),
    ("heritage walk", "A group walk felt lonely when conversation formed into closed pairs.", "Walking with one approachable person made the next visit less isolating.", "The heritage group is meeting again, and I worry I will end up outside the conversation."),
    ("weaving class", "A class became frustrating when several corrections were given at once.", "Practicing one pattern made the following class feel possible again.", "My weaving class is coming up, and I am tense about being corrected in front of everyone."),
    ("nature survey", "A survey felt disappointing when unfamiliar signs were hard to follow.", "Learning one sign made the next survey more enjoyable and less pressured.", "The nature survey is this weekend, and I feel behind before it has even begun."),
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
        out_dir_name="pm_v1_5_memory_opportunity_scale_stable_confirmation_v3",
        confirmation_themes=FRESH_CONFIRM_THEMES,
        user_prefix="pmv15_memv3",
        roles=("untouched_confirmation",),
        observation_builder=build_scale_stable_memory_opportunity_observation,
        feature_protocol=SCALE_STABLE_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL,
        match_feature_name="candidate_content_match_level",
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
