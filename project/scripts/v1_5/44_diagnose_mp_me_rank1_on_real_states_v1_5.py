"""Extend the MS lexical-vs-BGE-M3 diagnostic to MP and ME.

Status: EXPLORATORY / NOT A FROZEN CONTRACT ARTIFACT.

MS was tested (scripts/v1_5/43) and found BGE-M3 wins by a wide margin. This
script asks whether that generalizes to MP (short static profile facts:
job/education/age/gender/nationality/location -- basic_info in
data/external/evo_emo.json, ~5 candidates per user, fixed regardless of
current turn) and ME (episodic action-result spans, extracted from seeker
dialogue turns with the exact production compiler
metacom_pm.v1_5_v5_2_atomic_memory.compile_atomic_reusable_outcome -- not an
approximation).

MP and ME are structurally different retrieval problems from MS: MP has a
tiny, non-narrative candidate pool (picking which of ~5 short tags applies,
not finding one relevant narrative among many); ME candidates are short
first-person action/result spans, not full session summaries. Do not assume
the MS result transfers -- that is exactly what this script checks, echoing
V15-OBS-19 (BGE lost to lexical for a different content type: RS cards).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.retrieval import context_query  # noqa: E402  (MP/ME use full context)
from metacom_pm.text import lexical_score  # noqa: E402
from metacom_pm.v1_5_v5_2_atomic_memory import compile_atomic_reusable_outcome  # noqa: E402

EVOEMO = ROOT / "data/external/evo_emo.json"
PANEL_DIR = ROOT / "outputs/pm_v1_5b_corrected_external_split_v1"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_ms_retrieval_scoring_diagnostic_v1"
BGE_M3 = Path(
    "/home/tokkio/.cache/huggingface/hub/models--BAAI--bge-m3/"
    "snapshots/5617a9f61b028005a4858fdac845db406aefb181"
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def build_mp_documents() -> dict[str, list[dict]]:
    users = json.loads(EVOEMO.read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    for user in users:
        info = user.get("basic_info") or {}
        docs = []
        for field in ("job", "education", "age", "gender", "nationality", "location"):
            value = info.get(field)
            if value in (None, ""):
                continue
            docs.append({"memory_id": f"mp_{field}", "text": f"{field}: {value}"})
        out[str(user["id"])] = docs
    return out


def build_me_documents() -> dict[str, list[dict]]:
    users = json.loads(EVOEMO.read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    for user in users:
        docs = []
        seen = set()
        for session in user.get("dialog_history") or []:
            for turn in session.get("dialogue") or []:
                if str(turn.get("role") or "").lower() != "seeker":
                    continue
                atomic = compile_atomic_reusable_outcome(str(turn.get("content") or ""))
                if atomic is None or atomic.literal_evidence_span in seen:
                    continue
                seen.add(atomic.literal_evidence_span)
                docs.append(
                    {
                        "memory_id": f"me_{len(docs)}",
                        "session_id": str(session.get("id") or ""),
                        "text": atomic.literal_evidence_span,
                        "polarity": atomic.polarity,
                    }
                )
        out[str(user["id"])] = docs
    return out


def load_states() -> list[dict]:
    states = []
    for fname in ("evoemo_qualification_panel_private.jsonl", "evoemo_lockbox_panel_private.jsonl"):
        for row in load_jsonl(PANEL_DIR / fname):
            rt = row["runtime_state"]
            states.append(
                {
                    "state_id": row["state_id"],
                    "user_id": row["user_id_private_analysis_only"],
                    "current_user_text": rt.get("current_user_text", ""),
                    "current_session_history": rt.get("current_session_history", []),
                    "current_session_summary": rt.get("current_session_summary", ""),
                }
            )
    return states


def main() -> None:
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BGE_M3, local_files_only=True)
    model = AutoModel.from_pretrained(BGE_M3, local_files_only=True)
    model.eval()

    def encode(texts: list[str]) -> np.ndarray:
        pieces = []
        with torch.inference_mode():
            for start in range(0, len(texts), 8):
                batch = tokenizer(
                    texts[start : start + 8], padding=True, truncation=True,
                    max_length=8192, return_tensors="pt",
                )
                vector = model(**batch).last_hidden_state[:, 0]
                vector = torch.nn.functional.normalize(vector, p=2, dim=1)
                pieces.append(vector.numpy())
        return np.concatenate(pieces, axis=0)

    states = load_states()
    mp_docs = build_mp_documents()
    me_docs = build_me_documents()
    print(f"states: {len(states)}")
    print(f"users with MP candidates: {sum(1 for v in mp_docs.values() if v)}, "
          f"mean MP candidates/user: {sum(len(v) for v in mp_docs.values())/len(mp_docs):.1f}")
    print(f"users with ME candidates: {sum(1 for v in me_docs.values() if v)}, "
          f"total ME candidates extracted: {sum(len(v) for v in me_docs.values())}")

    def run(component: str, docs_by_user: dict[str, list[dict]], query_fn) -> None:
        doc_vecs_by_user = {
            uid: encode([d["text"] for d in docs]) for uid, docs in docs_by_user.items() if docs
        }
        rows_out = []
        agree = disagree = skipped = 0
        for state in states:
            docs = docs_by_user.get(state["user_id"]) or []
            if len(docs) < 2:
                skipped += 1
                continue
            query = query_fn(state)
            lex_top1 = max(docs, key=lambda d: (lexical_score(query, d["text"]), d["memory_id"]))
            query_vec = encode([query])[0]
            sims = doc_vecs_by_user[state["user_id"]] @ query_vec
            bge_top1 = docs[int(np.argmax(sims))]
            same = lex_top1["memory_id"] == bge_top1["memory_id"]
            agree += int(same)
            disagree += int(not same)
            rows_out.append(
                {
                    "state_id": state["state_id"],
                    "user_id": state["user_id"],
                    "current_user_text": state["current_user_text"],
                    "candidate_pool_size": len(docs),
                    "lexical_top1": lex_top1["text"],
                    "bge_top1": bge_top1["text"],
                    "scorers_agree_on_top1": same,
                }
            )
        out_path = OUT_DIR / f"{component.lower()}_rank1_real_states_lexical_vs_bge.jsonl"
        with out_path.open("w") as f:
            for row in rows_out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {
            "component": component,
            "states_scored": len(rows_out),
            "states_skipped_fewer_than_2_candidates": skipped,
            "agree": agree,
            "disagree": disagree,
            "disagreement_rate": disagree / len(rows_out) if rows_out else None,
        }
        print(json.dumps(summary, indent=2))
        (OUT_DIR / f"{component.lower()}_rank1_real_states_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )

    run("MP", mp_docs, lambda s: context_query(
        s["current_user_text"], s["current_session_history"], s["current_session_summary"]))
    run("ME", me_docs, lambda s: context_query(
        s["current_user_text"], s["current_session_history"], s["current_session_summary"]))


if __name__ == "__main__":
    main()
