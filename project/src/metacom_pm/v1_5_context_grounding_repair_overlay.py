"""Repair overlay + canonical local recompilation for the 25 context_
grounding DATA_DEFECT states.

Supersedes the narrower recompute_visible_semantic_fields() path in
v1_5_context_grounding_data_repair.py, which only refreshes text_embedding
and provenance.semantic_observation. An independent investigation
(verified directly against the code, not assumed) found MORE derived
fields are also a function of the visible dialogue text:

- inventory.MP/MS/ME.query_similarity_mean (PMV2State's own field,
  pm_v2_contracts.py) -- computed in _catalog_summary (pm_v2_data.py) from
  prepare_visible_semantic_state's state_query_vector.
- step0_observation.memory_sources[*].query_to_source_similarity
  (pm_v1_5_step0.py) -- re-surfaces the same value.

recompute_visible_semantic_fields() alone would leave these two stale.
The correct fix is not another hand-written per-field patch (easy to
miss yet another derived field) but to run the SAME canonical, already-
existing single-state compiler used for the original 468-state corpus:
case_to_state() (pm_v2_data.py), invoked by write_development_dataset()
for every case in every bundle.

write_development_dataset() is whole-corpus atomic (rewrites every output
file from scratch, enforces cross-user/cross-split invariants) -- but this
is a property of a LOCAL, deterministic, network-free function (verified:
its only callees are case_to_state/case_to_memory_backend/case_to_
evaluator_context/state_to_v1_runtime/validate_bundle/validate_split_
manifests and friends; the frozen semantic encoder loads a local snapshot
with local_files_only=True; no LLM/HTTP client appears anywhere in its
call path). It can be pointed at a brand-new output directory without
ever reading from or overwriting the original one, given a bundles list
where 443 of 468 cases are the ORIGINAL, untouched GeneratedStateCase
objects and only the 25 DATA_DEFECT cases are patched -- pm_v2_bundles.
jsonl round-trips losslessly (verified empirically against the real
468-case file) with no unpersisted/ephemeral compilation input (the only
RNG in the pipeline is content-seeded and already materialized in the
stored memory rows; case_to_state itself uses no RNG).

This module builds exactly that: an explicit, hash-bound repair overlay
applied to a copy of the real bundles, never mutating the originals, ready
to be handed to write_development_dataset() targeting a new directory
(e.g. data/pm_v1_5_formal_v8_19_grounding_repaired_candidate/). Producing
the real new directory is a separate, explicit next step -- this module
only builds and tests the overlay/patch mechanism.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import Field, field_validator

from .io import canonical_json, sha256_text
from .pm_v2_contracts import StrictModel

REPAIR_OVERLAY_PROTOCOL = "pm-v1.5-context-grounding-repair-overlay-v1"


class RepairOverlayRecord(StrictModel):
    """One case-level patch, bound to the exact originals it was derived
    from so a stale or mismatched overlay is refused rather than silently
    applied."""

    state_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    repair_mode: str
    original_bundle_case_sha256: str = Field(min_length=64, max_length=64)
    classification_sha256: str = Field(min_length=64, max_length=64)
    authorized_user_context: str = Field(min_length=1, max_length=250)
    session_summary: str | None = Field(default=None, max_length=250)
    recent_dialogue_patch: Mapping[int, str] = Field(default_factory=dict)

    @field_validator("recent_dialogue_patch", mode="before")
    @classmethod
    def _coerce_patch_keys_to_int(cls, value: Any) -> Any:
        """JSON has no integer keys: model_dump(mode="json") always writes
        recent_dialogue_patch with string keys, and this model's inherited
        strict=True config refuses to coerce them back to int on reload --
        meaning every persisted repair_overlays.jsonl would otherwise be
        unloadable by any later consumer (e.g. canonical recompilation)."""

        if isinstance(value, Mapping):
            coerced = {int(key): item for key, item in value.items()}
            if len(coerced) != len(value):
                raise ValueError(
                    "recent_dialogue_patch has non-canonical integer-string "
                    "keys that collide after conversion (e.g. '1' and '01') "
                    "-- refusing to silently drop one"
                )
            return coerced
        return value


def find_bundle_location_for_state(
    *,
    bundles: Sequence[Any],
    user_id: str,
    current_user_text: str,
) -> str:
    """Return the case_id within user_id's bundle whose current_user_text
    matches exactly -- the join key between a classification record's
    state_id and the bundle's own case_id. current_user_text is unique
    per state by the frozen generation contract's cross-user-duplicate
    audit, so an exact match is a safe, unambiguous join."""

    for bundle in bundles:
        if bundle.user_id != user_id:
            continue
        matches = [
            case.case_id
            for case in bundle.cases
            if case.current_user_text == current_user_text
        ]
        if len(matches) > 1:
            raise RuntimeError(
                f"current_user_text is not unique within user {user_id}'s bundle"
            )
        if matches:
            return matches[0]
    raise RuntimeError(
        f"no case in user {user_id}'s bundle matches the given current_user_text"
    )


def build_repair_overlay_record(
    *,
    record: Any,  # ContextGroundingClassificationRecord
    bundles: Sequence[Any],
    classification_sha256: str,
    authorized_user_context: str,
    session_summary: str | None,
    recent_dialogue_patch: Mapping[int, str],
) -> RepairOverlayRecord:
    case_id = find_bundle_location_for_state(
        bundles=bundles,
        user_id=record.user_id,
        current_user_text=record.current_user_text,
    )
    original_case = next(
        case
        for bundle in bundles
        if bundle.user_id == record.user_id
        for case in bundle.cases
        if case.case_id == case_id
    )
    return RepairOverlayRecord(
        state_id=record.state_id,
        user_id=record.user_id,
        case_id=case_id,
        repair_mode=record.repair_mode,
        original_bundle_case_sha256=sha256_text(
            canonical_json(original_case.model_dump(mode="json"))
        ),
        classification_sha256=classification_sha256,
        authorized_user_context=authorized_user_context,
        session_summary=session_summary,
        recent_dialogue_patch=dict(recent_dialogue_patch),
    )


def apply_repair_overlay_to_bundles(
    *,
    bundles: Sequence[Any],
    overlays: Sequence[RepairOverlayRecord],
    classification_records: Sequence[Any],  # ContextGroundingClassificationRecord
    expected_classification_sha256: str,
) -> list[Any]:
    """Return a NEW list of GeneratedUserBundle objects with exactly the
    overlaid cases patched. Bundles containing no overlaid case are
    returned as the SAME object (guaranteeing byte-identical serialization
    for the 443 untouched cases); bundles with at least one overlaid case
    get a fresh .model_copy with only the flagged case(s) replaced.

    classification_records (the full, frozen 91-row classification) is the
    single source of truth this function checks every overlay against --
    it never trusts the overlay's own state_id/user_id/repair_mode/case_id
    or the caller's choice of WHICH cases to patch:

    - overlays must cover EXACTLY the classification's DATA_DEFECT state_ids
      -- no missing state, no extra one, no duplicate.
    - classification_sha256 must equal expected_classification_sha256 (the
      real, frozen artifact hash) for every overlay.
    - case_id is independently RE-DERIVED from the matching classification
      record's (user_id, current_user_text) and must equal the overlay's
      own case_id and user_id/repair_mode.
    - FIELD_ONLY_REPAIR overlays must carry an EMPTY recent_dialogue_patch;
      VISIBLE_SURFACE_REPAIR overlays' patch keys must equal EXACTLY the
      frozen VISIBLE_SURFACE_REPAIR_TURN_INDICES for that state -- no
      subset, no superset, no off-target turn.
    """

    from .v1_5_context_grounding_data_repair import (
        FIELD_ONLY_REPAIR_SUMMARY_ALSO_NEEDED,
        VISIBLE_SURFACE_REPAIR_TURN_INDICES,
    )
    from .v1_5_context_grounding_repair import data_defect_state_ids

    if len({o.state_id for o in overlays}) != len(overlays):
        raise RuntimeError("duplicate state_id among overlays")
    if len({o.case_id for o in overlays}) != len(overlays):
        raise RuntimeError("duplicate case_id among overlays")

    expected_state_ids = data_defect_state_ids(classification_records)
    overlay_state_ids = {o.state_id for o in overlays}
    if overlay_state_ids != expected_state_ids:
        missing = expected_state_ids - overlay_state_ids
        extra = overlay_state_ids - expected_state_ids
        raise RuntimeError(
            "overlays do not exactly match the frozen DATA_DEFECT state_ids -- "
            f"missing={missing or None}, extra={extra or None}"
        )

    records_by_state_id = {r.state_id: r for r in classification_records}
    for overlay in overlays:
        if overlay.classification_sha256 != expected_classification_sha256:
            raise RuntimeError(
                f"overlay for {overlay.state_id} is bound to a different "
                "classification_sha256 than the frozen artifact -- refusing "
                "a stale or mismatched overlay"
            )
        record = records_by_state_id[overlay.state_id]
        if overlay.user_id != record.user_id:
            raise RuntimeError(
                f"overlay for {overlay.state_id} has user_id={overlay.user_id!r}, "
                f"classification says {record.user_id!r}"
            )
        if overlay.repair_mode != record.repair_mode:
            raise RuntimeError(
                f"overlay for {overlay.state_id} has repair_mode="
                f"{overlay.repair_mode!r}, classification says "
                f"{record.repair_mode!r}"
            )
        real_case_id = find_bundle_location_for_state(
            bundles=bundles, user_id=record.user_id, current_user_text=record.current_user_text
        )
        if overlay.case_id != real_case_id:
            raise RuntimeError(
                f"overlay for {overlay.state_id} has case_id={overlay.case_id!r}, "
                f"but the classification's own (user_id, current_user_text) "
                f"resolves to {real_case_id!r}"
            )
        if record.repair_mode == "FIELD_ONLY_REPAIR":
            if overlay.recent_dialogue_patch:
                raise RuntimeError(
                    f"FIELD_ONLY_REPAIR overlay for {overlay.state_id} must not "
                    "carry a recent_dialogue_patch"
                )
            summary_repair_required = (
                overlay.state_id in FIELD_ONLY_REPAIR_SUMMARY_ALSO_NEEDED
            )
            if summary_repair_required and overlay.session_summary is None:
                raise RuntimeError(
                    f"FIELD_ONLY_REPAIR overlay for {overlay.state_id} must "
                    "carry the frozen required session_summary repair"
                )
            if not summary_repair_required and overlay.session_summary is not None:
                raise RuntimeError(
                    f"FIELD_ONLY_REPAIR overlay for {overlay.state_id} must "
                    "leave session_summary byte-identical; this state is not "
                    "in FIELD_ONLY_REPAIR_SUMMARY_ALSO_NEEDED"
                )
        elif record.repair_mode == "VISIBLE_SURFACE_REPAIR":
            expected_indices = set(VISIBLE_SURFACE_REPAIR_TURN_INDICES[overlay.state_id])
            actual_indices = set(overlay.recent_dialogue_patch)
            if actual_indices != expected_indices:
                raise RuntimeError(
                    f"VISIBLE_SURFACE_REPAIR overlay for {overlay.state_id} patches "
                    f"turn indices {sorted(actual_indices)}, but the frozen spec "
                    f"requires exactly {sorted(expected_indices)}"
                )
        else:
            raise RuntimeError(f"unsupported repair_mode: {record.repair_mode}")

    overlay_by_case_id = {overlay.case_id: overlay for overlay in overlays}
    seen_case_ids: set[str] = set()
    patched_bundles = []
    for bundle in bundles:
        overlaid_case_ids = {
            case.case_id for case in bundle.cases if case.case_id in overlay_by_case_id
        }
        if not overlaid_case_ids:
            patched_bundles.append(bundle)
            continue
        new_cases = []
        for case in bundle.cases:
            overlay = overlay_by_case_id.get(case.case_id)
            if overlay is None:
                new_cases.append(case)
                continue
            seen_case_ids.add(case.case_id)
            real_sha256 = sha256_text(canonical_json(case.model_dump(mode="json")))
            if real_sha256 != overlay.original_bundle_case_sha256:
                raise RuntimeError(
                    f"overlay for case {case.case_id} does not match the real "
                    "original bundle case -- refusing to apply a stale or "
                    "mismatched overlay"
                )
            patched = dict(case.model_dump(mode="json"))
            patched["authorized_user_context"] = overlay.authorized_user_context
            if overlay.session_summary is not None:
                # Whether session_summary is present or absent for this
                # (user, case_field) cell is a frozen, counterbalanced
                # corpus-design property enforced by write_development_
                # dataset's summarize_observable_state_support -- confirmed
                # as a real hard-fail ("summary support drifted") by
                # running the actual recompilation pipeline. Never let an
                # overlay turn an originally-empty summary non-empty (or
                # vice versa), regardless of what the overlay says.
                original_summary_present = bool(str(case.session_summary).strip())
                new_summary_present = bool(overlay.session_summary.strip())
                if original_summary_present == new_summary_present:
                    patched["session_summary"] = overlay.session_summary
                elif not original_summary_present:
                    patched["session_summary"] = ""
                else:
                    raise RuntimeError(
                        f"overlay for case {case.case_id} would clear a "
                        "non-empty session_summary to empty -- refusing "
                        "(this would also drift the frozen summary-"
                        "presence design)"
                    )
            if overlay.recent_dialogue_patch:
                new_dialogue = [dict(turn) for turn in patched["recent_dialogue"]]
                for index, content in overlay.recent_dialogue_patch.items():
                    new_dialogue[index]["content"] = content
                patched["recent_dialogue"] = new_dialogue
            # Reconstruct via the real constructor (not .model_copy(), which
            # never re-runs validators) -- this forces GeneratedStateCase's
            # own source_consistency model_validator (roles alternate, ends
            # with assistant, current_user_text not repeated in history,
            # memory-source correctness, etc.) to run again on the patched
            # object, catching a structurally broken repair immediately
            # rather than silently producing an invalid case.
            new_cases.append(type(case).model_validate(patched))
        patched_bundle = dict(bundle.model_dump(mode="json"))
        patched_bundle["cases"] = [c.model_dump(mode="json") for c in new_cases]
        patched_bundles.append(type(bundle).model_validate(patched_bundle))
    missing = {overlay.case_id for overlay in overlays} - seen_case_ids
    if missing:
        raise RuntimeError(f"overlay case_ids not found in any bundle: {missing}")
    return patched_bundles


def split_by_user_from_existing_states(states_path: str | Path) -> dict[str, Any]:
    """Derive split_by_user from the already-compiled, real pm_v2_states.jsonl
    rather than re-deriving generation-time split assignment logic -- split
    is fixed/historical for an already-generated corpus and is recorded on
    every one of its states."""

    from .io import iter_jsonl
    from .pm_v2_contracts import PMV2Split

    split_by_user: dict[str, Any] = {}
    for row in iter_jsonl(states_path):
        user_id = str(row["user_id"])
        split = PMV2Split(str(row["split"]))
        if user_id in split_by_user and split_by_user[user_id] != split:
            raise RuntimeError(f"user {user_id} has inconsistent split across states")
        split_by_user[user_id] = split
    return split_by_user


def recompile_repaired_development_dataset(
    *,
    bundles: Sequence[Any],
    split_by_user: Mapping[str, Any],
    out_dir: str | Path,
    strategy_catalog_count: int,
    strategy_estimated_tokens: int,
    strategy_top_k: int,
    strategy_bank_sha256: str,
    strategy_cards: Sequence[Any],
    memory_min_score: float,
    strategy_min_score: float,
    expected_semantic_families_by_split: Mapping[Any, Sequence[str]],
    semantic_encoder: Any,
    enforce_required_hit_preflight: bool = True,
) -> dict[str, Any]:
    """Thin wrapper around the existing write_development_dataset -- this
    module does not re-derive its own config values; the caller is
    responsible for passing the SAME real strategy_top_k/strategy_bank_
    sha256/expected_semantic_families_by_split etc. used for the original
    v8_18 corpus (scripts/v1_5/20_generate_pm_v2_development_data_v1_5.py),
    so the ONLY thing that differs between this recompilation and the
    original is the 25 overlaid cases' content.
    """

    from .pm_v2_data import write_development_dataset

    return write_development_dataset(
        bundles=bundles,
        split_by_user=dict(split_by_user),
        out_dir=out_dir,
        strategy_catalog_count=strategy_catalog_count,
        strategy_estimated_tokens=strategy_estimated_tokens,
        strategy_top_k=strategy_top_k,
        strategy_bank_sha256=strategy_bank_sha256,
        strategy_cards=strategy_cards,
        enforce_required_hit_preflight=enforce_required_hit_preflight,
        memory_min_score=memory_min_score,
        strategy_min_score=strategy_min_score,
        expected_semantic_families_by_split=dict(expected_semantic_families_by_split),
        semantic_encoder=semantic_encoder,
    )
