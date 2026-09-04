"""ME verification against the real production ME compiler (same-stack).

Status: EXPLORATORY, same-stack (real functions, not hand-rolled).

Question: is ME's real problem "too few candidates exist" (retrieval/coverage
of raw material) or "candidates exist but the strict atomic-outcome compiler
used by V5.2's locked composer rejects almost all of them" (compiler
coverage)? An earlier claim in this session ("only 4 ME candidates exist in
EvoEmo") was retracted -- it came from running compile_atomic_reusable_outcome
directly over raw dialogue turns, bypassing the real evoemo.build_evo_memory()
chunker. This script uses the real chunker and the real compiler
(v1_5_v5_2_locked_composer._me_clause calls compile_atomic_reusable_outcome on
each ME candidate's literal span; this is a real production call site, not a
diagnostic-only function).

For each of the 12 unique EvoEmo users appearing in the 138-state qualifying
panel (outputs/pm_v1_5b_corrected_external_split_v1), this script:
  1. builds the real ME candidate pool via build_evo_memory(user);
  2. runs compile_atomic_reusable_outcome on each candidate's literal text;
  3. reports the per-user pool size and compile pass rate.

This does not re-run retrieval/ranking -- it isolates the compiler-coverage
question from the ranking question, since V5.2's locked composer raises
ValueError (see _me_clause) for any ME candidate the compiler rejects,
independent of how well that candidate was ranked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.contracts import MemorySource  # noqa: E402
from metacom_pm.evoemo import build_evo_memory  # noqa: E402
from metacom_pm.v1_5_v5_2_atomic_memory import compile_atomic_reusable_outcome  # noqa: E402

EVOEMO = ROOT / "data/external/evo_emo.json"
PANEL_DIR = ROOT / "outputs/pm_v1_5b_corrected_external_split_v1"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_me_verification_v1"


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


def main() -> None:
    users = {str(u["id"]): u for u in json.loads(EVOEMO.read_text(encoding="utf-8"))}
    panel_uids = panel_user_ids()

    per_user_rows = []
    total_items = 0
    total_compilable = 0
    for uid in panel_uids:
        user = users[uid]
        items, _extra = build_evo_memory(user)
        me_items = [it for it in items if it.source is MemorySource.ME]
        compilable = [it for it in me_items if compile_atomic_reusable_outcome(it.text) is not None]
        total_items += len(me_items)
        total_compilable += len(compilable)
        per_user_rows.append(
            {
                "user_id": uid,
                "me_pool_size": len(me_items),
                "me_compilable_count": len(compilable),
                "me_compilable_rate": (len(compilable) / len(me_items)) if me_items else None,
                "me_compilable_texts": [it.text for it in compilable],
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "me_compiler_coverage_by_user.jsonl"
    with out_path.open("w") as f:
        for row in per_user_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    users_with_any_compilable = sum(1 for r in per_user_rows if r["me_compilable_count"] > 0)
    print(f"panel users: {len(panel_uids)}")
    print(f"total ME candidates across panel users (real build_evo_memory chunker): {total_items}")
    print(f"total compilable (compile_atomic_reusable_outcome passes): {total_compilable}")
    print(f"overall compile pass rate: {total_compilable / total_items:.4%}" if total_items else "n/a")
    print(f"users with >=1 compilable ME candidate: {users_with_any_compilable}/{len(panel_uids)}")
    for row in per_user_rows:
        print(
            f"  {row['user_id']}: pool={row['me_pool_size']} "
            f"compilable={row['me_compilable_count']}"
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
