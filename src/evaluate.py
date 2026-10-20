"""Step 1 vs Step 2 evaluation on practice/questions.csv.

    python -m src.evaluate

COLUMN NAMES BELOW ARE GUESSES (we haven't seen practice/questions.csv yet). Adjust Q_COLS / GOLD_COLS
after reading DATA_DICTIONARY.md. Grading uses an LLM judge against the gold answer/decision.
"""
import json

import pandas as pd

from . import llm
from .agent import Agent
from .config import PRACTICE_DIR
from .kb import KnowledgeBase

Q_COLS = ["question", "text", "customer_question"]
GOLD_COLS = ["expected_answer", "gold_answer", "expected_decision", "expected"]


def pick(df, names):
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"None of {names} in {list(df.columns)}; update evaluate.py")


def judge(question, gold, answer) -> bool:
    m = llm.chat([{"role": "user", "content":
                   f"Question: {question}\nGold answer/decision: {gold}\nCandidate: {answer}\n"
                   "Does the candidate agree with the gold on the key fact or decision? Reply YES or NO only."}],
                 max_tokens=3, temperature=0)
    return (m.content or "").strip().upper().startswith("Y")


def main():
    df = pd.read_csv(PRACTICE_DIR / "questions.csv", dtype=str).fillna("")
    qc, gc = pick(df, Q_COLS), pick(df, GOLD_COLS)
    kb = KnowledgeBase()
    rows, s1, s2 = [], 0, 0
    for _, r in df.iterrows():
        a1 = Agent("prompt").respond(r[qc])
        ag2 = Agent("rag", kb)
        a2 = ag2.respond(r[qc])
        ok1, ok2 = judge(r[qc], r[gc], a1), judge(r[qc], r[gc], a2)
        s1, s2 = s1 + ok1, s2 + ok2
        rows.append({"question": r[qc], "gold": r[gc], "step1": a1, "step1_ok": ok1, "step2": a2, "step2_ok": ok2,
                     "sources": json.dumps([c["section_id"] for c in ag2.last_sources])})
    pd.DataFrame(rows).to_csv("sample_outputs/eval_step1_vs_step2.csv", index=False)
    n = len(df)
    print(f"Step 1 (prompt only): {s1}/{n}   Step 2 (RAG): {s2}/{n}")


if __name__ == "__main__":
    main()
