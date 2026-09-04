"""Frozen classification + repair-mode machinery for actual-468's
context_grounding_match audit incident.

Background: the real actual-468 run (development_actual_corpus_semantic_
review) produced 14 unanimous real_case_rejections and 74 real_case_
disagreements, ALL on context_grounding_match. A full manual audit of all 88
of those states (plus 3 more states outside that set caught by an
unconditional garbage-context rule) found two distinct, unrelated root
causes:

1. INSTRUMENT_AMBIGUITY (66/91): the context_grounding_match candidate claim
   is phrased as a self-referential meta-description ("both candidate
   context fields are supported") rather than stating the verification
   target directly. deepseek_official systematically misreads this as
   requiring literal field-name mentions in the evidence text; google_gemini
   does not. This is a prompt-wording bug, fixed by rewording the claim
   (see build_context_grounding_claim_and_evidence), never by touching data.

2. DATA_DEFECT (25/91): the frozen dataset itself has a real defect --
   garbage authorized_user_context, a fabricated/ungrounded fact, or an
   internal inconsistency in the visible dialogue (two different named
   relations across turns, or an unresolved two-move narrative). These
   require repair, split into two repair_mode values:
   - FIELD_ONLY_REPAIR (19/91): current_user_text/history are internally
     consistent; only authorized_user_context and/or session_summary need
     regeneration.
   - VISIBLE_SURFACE_REPAIR (6/91): the visible dialogue itself is
     inconsistent or ambiguous; regenerating context/summary alone cannot
     honestly resolve it.

No CLEAN states were found in this specific 91-state audit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import Field

from .io import iter_jsonl, sha256_file
from .pm_v2_contracts import StrictModel

CONTEXT_GROUNDING_DEFECT_CLASSIFICATION_PROTOCOL = (
    "pm-v1.5-context-grounding-defect-classification-v1"
)

# Canonical, version-controlled location. The classification was originally
# written under outputs/pm_v1_5_actual_corpus_semantic_review_v8_18_
# deepseek_official_v2_candidate/ -- real, but that whole directory is
# git-ignored (like every other paid-run output directory in this project),
# so a fresh checkout or CI would silently lack the file this loader's
# frozen-sha256 gate depends on. Moved here, byte-identical (same sha256),
# specifically so the classification is part of the repository, not only of
# one machine's local disk.
DEFAULT_CLASSIFICATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "pm_v1_5_contracts"
    / "context_grounding_defect_classification_v1.jsonl"
)

# Frozen once, at the time this classification was produced and independently
# reviewed. Any future edit to the classification file must be a deliberate,
# disclosed act (a fresh review), never a silent drift -- so this constant is
# updated only by whoever re-runs and re-freezes the classification, and
# every consumer must fail closed on a mismatch.
FROZEN_CONTEXT_GROUNDING_DEFECT_CLASSIFICATION_SHA256 = (
    "a20a1003e02cccd42a9dedb720073a73172e0228e4638d2a5bbdc1f69d6b9af5"
)
EXPECTED_TOTAL_RECORDS = 91
EXPECTED_DATA_DEFECT_COUNT = 25
EXPECTED_FIELD_ONLY_REPAIR_COUNT = 19
EXPECTED_VISIBLE_SURFACE_REPAIR_COUNT = 6
EXPECTED_INSTRUMENT_AMBIGUITY_COUNT = 66
EXPECTED_CLEAN_COUNT = 0

CLASSIFICATIONS = ("DATA_DEFECT", "INSTRUMENT_AMBIGUITY", "CLEAN")
DEFECT_TYPES = (
    "garbage",
    "unsupported_fact",
    "temporal_contradiction",
    "summary_contradiction",
    "claim_phrasing_self_referential",
    "empty_summary_NA_bug",
)
REPAIR_MODES = ("FIELD_ONLY_REPAIR", "VISIBLE_SURFACE_REPAIR", "AUDIT_CONTRACT_FIX_ONLY")


class ContextGroundingClassificationRecord(StrictModel):
    state_id: str = Field(min_length=1)
    field: Literal["context_grounding_match"]
    bucket: Literal["REJECT", "DISAGREE", "PASS"]
    split: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    regime: str = Field(min_length=1)
    current_user_text: str
    history: list[dict[str, Any]]
    session_summary: str
    authorized_user_context: str
    judges: Mapping[str, Any]
    classification: Literal["DATA_DEFECT", "INSTRUMENT_AMBIGUITY", "CLEAN"]
    defect_type: str
    repair_mode: str
    rationale: str = Field(min_length=1)


def load_context_grounding_defect_classification(
    path: str | Path,
) -> list[ContextGroundingClassificationRecord]:
    """Fail-closed loader: requires the frozen SHA256 to match exactly.

    No repair-generation code may run without successfully loading this
    artifact first -- a changed or hand-edited classification file is
    refused rather than silently trusted.
    """

    path = Path(path)
    real_sha256 = sha256_file(path)
    if real_sha256 != FROZEN_CONTEXT_GROUNDING_DEFECT_CLASSIFICATION_SHA256:
        raise RuntimeError(
            "context_grounding_defect_classification.jsonl does not match the "
            "frozen sha256 -- refusing to trust an unreviewed or altered "
            "classification file"
        )
    records = [
        ContextGroundingClassificationRecord.model_validate(row)
        for row in iter_jsonl(path)
    ]
    if len(records) != EXPECTED_TOTAL_RECORDS:
        raise RuntimeError(
            f"expected {EXPECTED_TOTAL_RECORDS} classification records, found "
            f"{len(records)}"
        )
    if len({r.state_id for r in records}) != len(records):
        raise RuntimeError("classification file has duplicate state_id rows")
    for record in records:
        if record.classification not in CLASSIFICATIONS:
            raise RuntimeError(f"unknown classification: {record.classification}")
        if record.defect_type not in DEFECT_TYPES:
            raise RuntimeError(f"unknown defect_type: {record.defect_type}")
        if record.repair_mode not in REPAIR_MODES:
            raise RuntimeError(f"unknown repair_mode: {record.repair_mode}")
        if record.classification == "DATA_DEFECT" and record.repair_mode not in (
            "FIELD_ONLY_REPAIR",
            "VISIBLE_SURFACE_REPAIR",
        ):
            raise RuntimeError(
                f"DATA_DEFECT record {record.state_id} must have a real repair_mode"
            )
        if record.classification == "INSTRUMENT_AMBIGUITY" and record.repair_mode != (
            "AUDIT_CONTRACT_FIX_ONLY"
        ):
            raise RuntimeError(
                f"INSTRUMENT_AMBIGUITY record {record.state_id} must be "
                "AUDIT_CONTRACT_FIX_ONLY, never a data repair_mode"
            )
    n_defect = sum(1 for r in records if r.classification == "DATA_DEFECT")
    n_ambiguity = sum(1 for r in records if r.classification == "INSTRUMENT_AMBIGUITY")
    n_clean = sum(1 for r in records if r.classification == "CLEAN")
    n_field_only = sum(1 for r in records if r.repair_mode == "FIELD_ONLY_REPAIR")
    n_surface = sum(1 for r in records if r.repair_mode == "VISIBLE_SURFACE_REPAIR")
    if (
        n_defect != EXPECTED_DATA_DEFECT_COUNT
        or n_ambiguity != EXPECTED_INSTRUMENT_AMBIGUITY_COUNT
        or n_clean != EXPECTED_CLEAN_COUNT
        or n_field_only != EXPECTED_FIELD_ONLY_REPAIR_COUNT
        or n_surface != EXPECTED_VISIBLE_SURFACE_REPAIR_COUNT
    ):
        raise RuntimeError(
            "classification file counts drifted from the frozen expectation "
            f"(DATA_DEFECT={n_defect}, INSTRUMENT_AMBIGUITY={n_ambiguity}, "
            f"CLEAN={n_clean}, FIELD_ONLY_REPAIR={n_field_only}, "
            f"VISIBLE_SURFACE_REPAIR={n_surface})"
        )
    return records


def field_only_repair_state_ids(
    records: list[ContextGroundingClassificationRecord],
) -> set[str]:
    return {r.state_id for r in records if r.repair_mode == "FIELD_ONLY_REPAIR"}


def visible_surface_repair_state_ids(
    records: list[ContextGroundingClassificationRecord],
) -> set[str]:
    return {r.state_id for r in records if r.repair_mode == "VISIBLE_SURFACE_REPAIR"}


def data_defect_state_ids(
    records: list[ContextGroundingClassificationRecord],
) -> set[str]:
    return {r.state_id for r in records if r.classification == "DATA_DEFECT"}
