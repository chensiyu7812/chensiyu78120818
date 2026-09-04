"""Diagnostic: lexical vs BGE-M3 Rank-1 MS candidate, on the REAL 138 EvoEmo
response-generation states (not the ES-MemEval proxy task).

Status: EXPLORATORY / NOT A FROZEN CONTRACT ARTIFACT. Follow-up to
scripts/v1_5/42_diagnose_ms_retrieval_scoring_v1_5.py, which showed BGE-M3 beats
the production lexical_score by a wide margin on ES-MemEval's session-retrieval
task. That task's candidate "documents" were full raw dialogue transcripts; this
script instead uses each session's `summary` field from data/external/evo_emo.json
(dialog_history[i]["summary"]) -- a short third-person distillation, which is
much closer to what the production MS candidate literal text actually looks like
(compare: composer._ms_clause's quoted "An earlier session recorded: ..." text).

This does NOT reconstruct the production Top-k discovery pipeline exactly (owner/
version/eligibility filtering, embedding calibration floors, etc.) -- it is a
best-effort, directly-comparable reproduction: same query-construction function
the real retriever uses (seeker_only_context_query), same user pool, same
per-session summary documents, two scoring functions swapped in.

No API calls. BGE-M3 encoding runs locally (CPU), reusing the same already-
downloaded snapshot the project's formal pipeline uses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.retrieval import seeker_only_context_query  # noqa: E402
from metacom_pm.text import lexical_score  # noqa: E402

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


def build_user_documents() -> dict[str, list[dict]]:
    users = json.loads(EVOEMO.read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    for user in users:
        docs = []
        for session in user.get("dialog_history") or []:
            summary = str(session.get("summary") or "").strip()
            if not summary:
                continue
            docs.append(
                {
                    "session_id": str(session.get("id") or ""),
                    "date": str(session.get("timestamp") or ""),
                    "text": summary,
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
                    "panel_id": row["panel_id"],
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
                    texts[start : start + 8],
                    padding=True,
                    truncation=True,
                    max_length=8192,
                    return_tensors="pt",
                )
                vector = model(**batch).last_hidden_state[:, 0]
                vector = torch.nn.functional.normalize(vector, p=2, dim=1)
                pieces.append(vector.numpy())
        return np.concatenate(pieces, axis=0)

    user_documents = build_user_documents()
    states = load_states()
    print(f"states: {len(states)}, users with documents: {len(user_documents)}")

    # Embed every user's session-summary documents once.
    doc_vectors_by_user: dict[str, np.ndarray] = {}
    for user_id, docs in user_documents.items():
        if not docs:
            continue
        doc_vectors_by_user[user_id] = encode([d["text"] for d in docs])

    rows_out = []
    agree = 0
    disagree = 0
    for state in states:
        user_id = state["user_id"]
        docs = user_documents.get(user_id) or []
        if not docs:
            continue
        query = seeker_only_context_query(
            state["current_user_text"],
            state["current_session_history"],
            state["current_session_summary"],
        )

        lex_ranked = sorted(
            docs, key=lambda d: (lexical_score(query, d["text"]), d["session_id"]), reverse=True
        )
        lex_top1 = lex_ranked[0]

        query_vec = encode([query])[0]
        doc_vecs = doc_vectors_by_user[user_id]
        sims = doc_vecs @ query_vec
        bge_order = np.argsort(-sims)
        bge_top1 = docs[int(bge_order[0])]

        same = lex_top1["session_id"] == bge_top1["session_id"]
        agree += int(same)
        disagree += int(not same)

        rows_out.append(
            {
                "state_id": state["state_id"],
                "user_id": user_id,
                "current_user_text": state["current_user_text"],
                "lexical_top1_session_id": lex_top1["session_id"],
                "lexical_top1_summary": lex_top1["text"],
                "bge_top1_session_id": bge_top1["session_id"],
                "bge_top1_summary": bge_top1["text"],
                "scorers_agree_on_top1": same,
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "ms_rank1_real_states_lexical_vs_bge.jsonl"
    with out_path.open("w") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "status": "EXPLORATORY_NOT_A_FROZEN_CONTRACT_ARTIFACT",
        "states_scored": len(rows_out),
        "lexical_and_bge_pick_same_top1": agree,
        "lexical_and_bge_pick_different_top1": disagree,
        "disagreement_rate": disagree / len(rows_out) if rows_out else None,
        "note": (
            "Disagreement alone does not say which scorer is right -- see the "
            "per-state jsonl for manual side-by-side topical-relevance review."
        ),
    }
    (OUT_DIR / "ms_rank1_real_states_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
