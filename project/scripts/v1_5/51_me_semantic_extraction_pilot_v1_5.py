"""ME semantic extraction pilot: can a real LLM call recover more valid
action+outcome candidates than the strict regex compiler (0.40% pass rate,
3/747 real chunks -- PM_V1_5_V5_3_ME_VERIFICATION_FINDINGS_20260806_ZH.md)?

Status: dry-run by default, real --live calls only on explicit request.

This is deliberately a small, cheap PILOT, not the full "semantic ME
compiler" a second Codex session proposed. It tests only the empirical
question that proposal's cost/benefit depends on: does a real extractor
find meaningfully more candidates than 0.4%, on real chunks the current
regex compiler already rejects? It does NOT implement the full compiler
contract (owner_id/session_id/turn_id plumbing, the full hard-validator
suite) -- that is real, separate engineering work, not justified until
this pilot's answer is known.

Backend hard check (the one thing this pilot does implement for real,
because it is cheap and load-bearing): both extracted spans must be exact
substrings of the original chunk text. An extractor claiming a span that
is not verbatim in the source is rejected outright, regardless of how
plausible it looks -- same fail-closed principle already used throughout
this project (describe_memory_candidate's causal-boundary assertion,
compile_atomic_reusable_outcome's literal-span requirement).

2026-08-06 v3: the v2 prompt (few-shot examples) fixed recall (0/15 -> 4/15
claimed+verbatim-passed on a fresh sample) but a manual read of all 4 full
source chunks found only 1 was genuinely sound -- 2 failed on exactly the
dimension flagged above as unverified (a third-party's action attributed to
the user; a future-tense plan treated as a completed action), 1 failed on a
related dimension (the "outcome" was actually a purpose/goal clause, not
something observed to have happened). verify_extracted_span() below closes
this gap with three cheap, deterministic checks -- calibrated against
exactly those 4 real examples (owner: reject if no first-person subject
marker in action_span; completion: reject if action_span contains a future/
modal marker; outcome: reject if outcome_span contains a future/modal
marker or opens with a purpose-clause marker like "to "/"in order to").
This reproduced the manual verdict exactly on all 4 real cases (accepts the
1 genuinely sound one, rejects the other 3, one for each real failure mode
found). Still a heuristic, not a proof -- calibrated on n=4, needs more
real data to know how well it generalizes, and does not catch every way an
extraction could be wrong (e.g. a third-person subject phrased with "I"
somewhere else in the span would slip through the owner check).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.contracts import MemorySource  # noqa: E402
from metacom_pm.evoemo import build_evo_memory  # noqa: E402
from metacom_pm.v1_5_v5_2_atomic_memory import compile_atomic_reusable_outcome  # noqa: E402

EVOEMO = ROOT / "data/external/evo_emo.json"
PANEL_DIR = ROOT / "outputs/pm_v1_5b_corrected_external_split_v1"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_me_extraction_pilot_v1"

# 2026-08-06 v2: the first version of this prompt (no examples) got 0/15 on
# a real pilot. An independent review manually re-read the raw "abstained"
# chunks and found at least 2 genuine misses (confirmed directly against the
# saved chunk text, not taken on trust): a p18 chunk containing "i decided
# to to yoga for better sleep its nice and going good" (messy grammar, but
# a clear action+outcome), and a p9 chunk containing "I used this chat...
# It helps... Yes it helps" (outcome mentioned before the action is named,
# i.e. not in linear order). The recall failure looks like it came from (a)
# no worked examples at all, and (b) messy/informal or non-linear phrasing
# probably reading as "not confident enough" under a bare "abstain if
# unsure" instruction. v2 adds real few-shot examples, including messy ones,
# and says explicitly that bad grammar or informal phrasing is not by
# itself a reason to abstain.
EXTRACTION_SYSTEM_PROMPT = (
    "You will be given one chunk of a user's own past chat turns (their own words, in "
    "order, sometimes informal or with typos). Look for a specific, local instance of: the "
    "user did something (a concrete past action they actually took, not a plan or hope), and "
    "they observed a specific result from it (not a general feeling, not advice they "
    "received, not something a third party did). Both the action and the result must be "
    "about the SAME local event. They do not have to appear in that order in the text, and "
    "messy grammar, typos, or informal phrasing are NOT by themselves a reason to abstain -- "
    "judge the content, not the writing quality.\n\n"
    "Examples of chunks that DO contain one (has_reusable_outcome=true):\n"
    "- \"I tried to focus on engaging one-on-one, which helped a bit, but it was hard to "
    "shake the feeling of being judged.\" -> action_span=\"I tried to focus on engaging "
    "one-on-one\", outcome_span=\"which helped a bit, but it was hard to shake the feeling "
    "of being judged or not doing enough.\"\n"
    "- \"...i decided to to yoga for better sleep its nice and going good ok.thank you\" "
    "(messy grammar, still valid) -> action_span=\"i decided to to yoga for better sleep\", "
    "outcome_span=\"its nice and going good\"\n"
    "- \"It helps to have a person to talk to. I used this chat ... Yes it helps.\" (outcome "
    "mentioned before the action is named -- still valid, look at the whole chunk) -> "
    "action_span=\"I used this chat\", outcome_span=\"Yes it helps\"\n\n"
    "If you find one: return has_reusable_outcome=true, and action_span/outcome_span as "
    "EXACT, VERBATIM substrings copied from the chunk (do not paraphrase, do not fix grammar "
    "or punctuation -- copy the exact characters, including typos).\n\n"
    "If there really is no local action+outcome pair (e.g. only a feeling, a plan, advice "
    "received, or something someone else did), return has_reusable_outcome=false and leave "
    "both spans empty."
)


_FIRST_PERSON_SUBJECT_RE = re.compile(r"\b(?:I|I've|I'm|I'd|my|me|we|we've|we're|our)\b", re.IGNORECASE)
_FUTURE_MODAL_RE = re.compile(
    r"\b(?:will|'ll|going to|gonna|would|plan to|planning to|hope to|hoping to|"
    r"want to|wanting to|intend to|should|might|may|about to)\b",
    re.IGNORECASE,
)
_PURPOSE_CLAUSE_RE = re.compile(r"^\s*(?:to |in order to |so that |so as to )", re.IGNORECASE)


def verify_extracted_span(action_span: str, outcome_span: str) -> tuple[str, ...]:
    """Cheap, deterministic checks calibrated on 4 real extractions (see the
    module docstring's 2026-08-06 v3 note): reproduces a manual read of all
    4 exactly (accepts the 1 sound one, rejects the other 3, one failure
    reason each). Heuristic, not a proof -- n=4 calibration set, will miss
    failure modes it wasn't calibrated against.
    """

    errors: list[str] = []
    if not _FIRST_PERSON_SUBJECT_RE.search(action_span):
        errors.append("ACTION_HAS_NO_FIRST_PERSON_SUBJECT")
    if _FUTURE_MODAL_RE.search(action_span):
        errors.append("ACTION_LOOKS_LIKE_A_FUTURE_PLAN_NOT_A_COMPLETED_ACTION")
    if _FUTURE_MODAL_RE.search(outcome_span):
        errors.append("OUTCOME_LOOKS_LIKE_A_FUTURE_PLAN_NOT_AN_OBSERVED_RESULT")
    if _PURPOSE_CLAUSE_RE.search(outcome_span):
        errors.append("OUTCOME_LOOKS_LIKE_A_PURPOSE_CLAUSE_NOT_AN_OBSERVED_RESULT")
    return tuple(errors)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def panel_user_ids() -> list[str]:
    uids: list[str] = []
    seen = set()
    for fname in ("evoemo_qualification_panel_private.jsonl", "evoemo_lockbox_panel_private.jsonl"):
        for row in load_jsonl(PANEL_DIR / fname):
            uid = row["user_id_private_analysis_only"]
            if uid not in seen:
                seen.add(uid)
                uids.append(uid)
    return uids


def collect_uncompilable_chunks(
    users: dict, uids: list[str], n: int, exclude_memory_ids: frozenset[str] = frozenset()
) -> list[dict]:
    """Real ME chunks (via the real build_evo_memory() compiler) that FAIL
    the current strict regex -- these are the only chunks worth piloting an
    alternative extractor on; chunks that already pass need no pilot.

    Spreads across distinct users round-robin (one chunk per user per pass)
    rather than draining one user's pool first, so a small pilot sample
    isn't accidentally all one person's writing style.

    exclude_memory_ids lets a re-test draw genuinely fresh material -- in
    particular, two chunks from the first pilot round are now baked into
    EXTRACTION_SYSTEM_PROMPT as few-shot examples (v2), so re-testing on
    them would not be a fair read of whether the prompt improvement
    generalizes, only whether the model can recognize its own example.
    """

    per_user: dict[str, list] = {}
    for uid in uids:
        user = users[uid]
        items, _extra = build_evo_memory(user)
        me_items = [it for it in items if it.source is MemorySource.ME]
        per_user[uid] = [
            it for it in me_items
            if compile_atomic_reusable_outcome(it.text) is None
            and it.memory_id not in exclude_memory_ids
        ]

    chunks: list[dict] = []
    max_pool_len = max((len(pool) for pool in per_user.values()), default=0)
    for index in range(max_pool_len):
        for uid in uids:
            if len(chunks) >= n:
                return chunks
            pool = per_user[uid]
            if index < len(pool):
                it = pool[index]
                chunks.append({"user_id": uid, "memory_id": it.memory_id, "text": it.text})
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Make real, billable API calls.")
    parser.add_argument("--n", type=int, default=15, help="Number of uncompilable chunks to pilot.")
    args = parser.parse_args()

    # 2026-08-06 v4 fix: a real re-test found this exclusion logic only ever
    # excluded the IMMEDIATELY PRIOR run's chunks, not the full cross-run
    # history. Consequence, confirmed with real data: two chunks tested (and
    # rejected) in run 1, then baked into EXTRACTION_SYSTEM_PROMPT as v2
    # few-shot examples, then correctly excluded in run 2 -- resurfaced in
    # run 3 because run 3 only read run 2's output file, not run 1's
    # archive. The model "found" both trivially (they are its own worked
    # examples) and both were counted as fresh fully_verified successes,
    # inflating that run's headline rate. A second bug compounded this: the
    # archive step used a single fixed filename and skipped silently once
    # that file existed, so run 2's real 15-chunk results were overwritten
    # by run 3 with no archive ever made -- silent data loss. Fixed by (a)
    # unioning memory_ids across every archived round, not just the latest,
    # and (b) archiving every run under a round-numbered filename so a later
    # run can never collide with an earlier archive.
    exclude_memory_ids: frozenset[str] = frozenset()
    out_path = OUT_DIR / "me_extraction_pilot_results.jsonl"
    if args.live:
        prior_round_files = sorted(OUT_DIR.glob("me_extraction_pilot_results_round*.jsonl"))
        all_prior_ids: set[str] = set()
        for f in prior_round_files:
            for r in load_jsonl(f):
                if "memory_id" in r:
                    all_prior_ids.add(r["memory_id"])
        if out_path.exists():
            current = load_jsonl(out_path)
            for r in current:
                if "memory_id" in r:
                    all_prior_ids.add(r["memory_id"])
            next_round = len(prior_round_files) + 1
            archive_path = OUT_DIR / f"me_extraction_pilot_results_round{next_round}_20260806.jsonl"
            out_path.rename(archive_path)
            print(f"archived round {next_round} ({len(current)} chunks) to {archive_path}")
        exclude_memory_ids = frozenset(all_prior_ids)
        print(f"excluding {len(exclude_memory_ids)} memory_ids tested across all prior rounds "
              f"({len(prior_round_files)} prior archives + current out_path if present)")

    users = {str(u["id"]): u for u in json.loads(EVOEMO.read_text(encoding="utf-8"))}
    uids = panel_user_ids()
    chunks = collect_uncompilable_chunks(users, uids, args.n, exclude_memory_ids=exclude_memory_ids)
    print(f"selected {len(chunks)} real ME chunks that fail the current strict compiler, "
          f"across {len({c['user_id'] for c in chunks})} distinct users "
          f"(excluded {len(exclude_memory_ids)} previously-piloted chunks)")

    if not args.live:
        for c in chunks:
            print(f"\n[{c['user_id']}] {c['text'][:150]}")
        print("\n(dry-run: no extraction calls made)")
        return

    from metacom_pm.api import OpenAICompatibleClient  # noqa: PLC0415
    from metacom_pm.config import endpoint_from_config, load_config  # noqa: PLC0415
    from metacom_pm.contracts import StrictModel  # noqa: PLC0415

    class _ExtractionSchema(StrictModel):
        has_reusable_outcome: bool
        action_span: str
        outcome_span: str

    config = load_config(ROOT / "configs/experiment.yaml")
    endpoint = endpoint_from_config(config, "generator")
    client = OpenAICompatibleClient(endpoint)

    results = []
    n_claimed = n_verbatim_valid = n_fully_verified = 0
    try:
        for i, c in enumerate(chunks, 1):
            messages = [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": c["text"]},
            ]
            result, parsed = client.chat(messages, response_schema=_ExtractionSchema)
            row = {"user_id": c["user_id"], "memory_id": c["memory_id"], "chunk_text": c["text"]}
            if parsed is None:
                row["error"] = str(result)
                print(f"\n[{i}/{len(chunks)}] {c['user_id']}: no structured output")
                results.append(row)
                continue
            extracted = parsed.model_dump()
            row.update(extracted)
            if extracted["has_reusable_outcome"]:
                n_claimed += 1
                action_ok = extracted["action_span"] and extracted["action_span"] in c["text"]
                outcome_ok = extracted["outcome_span"] and extracted["outcome_span"] in c["text"]
                row["action_span_verbatim_in_source"] = bool(action_ok)
                row["outcome_span_verbatim_in_source"] = bool(outcome_ok)
                valid = bool(action_ok and outcome_ok)
                row["verbatim_validated"] = valid
                semantic_errors = ()
                if valid:
                    n_verbatim_valid += 1
                    semantic_errors = verify_extracted_span(
                        extracted["action_span"], extracted["outcome_span"]
                    )
                    row["semantic_verification_errors"] = list(semantic_errors)
                    row["fully_verified"] = not semantic_errors
                    if not semantic_errors:
                        n_fully_verified += 1
                print(f"\n[{i}/{len(chunks)}] {c['user_id']}: claimed=yes, "
                      f"verbatim_validated={valid}, semantic_errors={semantic_errors}")
                print(f"  action: {extracted['action_span']!r}")
                print(f"  outcome: {extracted['outcome_span']!r}")
            else:
                row["verbatim_validated"] = False
                print(f"\n[{i}/{len(chunks)}] {c['user_id']}: claimed=no (abstained)")
            results.append(row)
    finally:
        client.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n=== summary ===")
    print(f"chunks piloted: {len(chunks)}")
    print(f"extractor claimed a reusable outcome: {n_claimed}")
    print(f"claims that passed the verbatim-substring hard check: {n_verbatim_valid}")
    print(f"of those, claims that ALSO passed owner/completion/purpose-clause checks: "
          f"{n_fully_verified}")
    print(f"pilot fully-verified rate: {n_fully_verified}/{len(chunks)} "
          f"({n_fully_verified/len(chunks):.1%})" if chunks else "n/a")
    print("NOTE: fully-verified is still a heuristic (calibrated on n=4), not proof of "
          "correctness -- a human read is still warranted before trusting any candidate.")
    print(f"(for comparison: strict regex compiler's real rate on the full 747-chunk "
          f"panel was 0.40%, 3/747 -- this pilot only sampled chunks that regex already "
          f"rejects, so these two rates are not directly comparable as stated; see the "
          f"analysis doc for the correct framing)")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
