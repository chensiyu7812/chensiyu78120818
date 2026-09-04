from __future__ import annotations

from typing import Any, Mapping, Sequence

from .prompts import SELECTIVE_ESMEM_SYSTEM


EXTERNAL_RAW_MEMORY_ADAPTER_PROTOCOL = (
    "pm-v1.5-external-raw-session-response-adapter-v1"
)
EXTERNAL_QA_PROMPT_PROTOCOL = "pm-v1.5-esmemeval-official-aligned-qa-prompt-v1"


QA_SYSTEM_PROMPT = """## Task Description
You are given a user question and a set of retrieved memory fragments.
Your task is to filter and summarize the relevant information from the memory fragments and generate a concise, accurate answer to the user's question based on the most pertinent details.
You may need to evaluate the relevance and accuracy of each memory fragment, and if needed, disregard irrelevant or incorrect information.
If the question cannot be answered with the available information, return "unknown."

## Input Format
Question: What did Sarah experience on her birthday in 2024?
Relevant Memory:
1. [2024-08-15] Sarah spent her birthday with her family at a beach resort.
2. [2024-08-15] Sarah was surprised with a birthday cake from her friends.
3. [2024-08-14] Sarah was stressed at work before her birthday, dealing with tight deadlines.
4. [2024-08-15] Sarah enjoyed a quiet dinner with close friends on her birthday evening.

## Output Format
Answer: On her birthday in 2024, Sarah celebrated with family at a beach resort and was surprised with a birthday cake from her friends."""


def render_evoemo_session_document(
    session: Mapping[str, Any], *, human_name: str
) -> dict[str, Any]:
    """Render one same-user session using the released ES-MemEval unit.

    The official QA implementation indexes one complete dialogue session as
    one document.  This adapter mirrors that unit while keeping the rendering
    local and deterministic.  It never reads observations, summaries, events,
    questions, answers, or evidence.
    """

    lines: list[str] = []
    for turn in session.get("dialogue") or []:
        role = str(turn.get("role") or "").strip().lower()
        if role == "seeker":
            label = human_name
        elif role == "supporter":
            label = "Supporter"
        else:
            label = role.capitalize() or "Unknown"
        content = " ".join(str(turn.get("content") or "").split())
        if content:
            lines.append(f"{label}: {content}")
    return {
        "session_id": str(session.get("id") or ""),
        "date": str(session.get("timestamp") or ""),
        "text": "\n".join(lines),
    }


def render_indexed_session_fragments(
    documents: Sequence[Mapping[str, Any]], *, indexed: bool
) -> list[str]:
    fragments: list[str] = []
    for index, document in enumerate(documents, start=1):
        prefix = f"{index}. " if indexed else ""
        fragments.append(
            f"{prefix}[{str(document['date'])}]\n{str(document['text'])}"
        )
    return fragments


def qa_messages(
    *, question: str, memory_fragments: Sequence[str]
) -> list[dict[str, str]]:
    content = f"Question: {question.strip()}"
    if memory_fragments:
        content += "\nRelevant Memory:\n" + "\n".join(memory_fragments)
    return [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def raw_response_messages(
    *,
    current_context: str,
    session_fragments: Sequence[str],
    strategy_instruction: str = "",
) -> list[dict[str, str]]:
    sections = [current_context]
    if session_fragments:
        sections.append(
            "Strictly prior same-user dialogue sessions. Use only a detail that "
            "materially helps the present exchange; keep it attributed to the "
            "past and do not turn one episode into a current fact or stable "
            "pattern:\n" + "\n".join(session_fragments)
        )
    if strategy_instruction.strip():
        sections.append(strategy_instruction.strip())
    sections.append(
        "Write only the counselor's concise next response. Do not mention "
        "retrieval, memory stores, strategy cards, scores, or system conditions."
    )
    return [
        {"role": "system", "content": SELECTIVE_ESMEM_SYSTEM},
        {"role": "user", "content": "\n\n".join(sections)},
    ]
