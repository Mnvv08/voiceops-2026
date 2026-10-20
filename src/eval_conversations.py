"""Replay practice/conversations.csv through the chat agent, turn by turn.

    python -m src.eval_conversations

COLUMN NAMES ARE GUESSES (dataset not seen yet): conversation_id, turn, and a customer-text column.
Adjust the constants below after reading DATA_DICTIONARY.md. For each conversation a fresh Agent is
used; we save every reply, the sources, and the tool calls so you can compare them with the expected
decisions / tool calls / gold_evidence columns by eye and with the judge.
"""
import json

import pandas as pd

from . import llm
from .agent import Agent
from .config import PRACTICE_DIR
from .kb import KnowledgeBase

CONV, TURN = "conversation_id", "turn"
TEXT_COLS = ["customer_text", "customer_turn", "asr_text", "text", "utterance"]
EXPECT_COLS = ["expected_decision", "expected_behavior", "expected", "expected_response"]


def pick(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def judge(text, expected, reply):
    m = llm.chat([{"role": "user", "content":
                   f"Customer said: {text}\nExpected behaviour/decision: {expected}\nAgent replied: {reply}\n"
                   "Does the agent's reply match the expected behaviour? Reply YES or NO only."}],
                 max_tokens=3, temperature=0)
    return (m.content or "").strip().upper().startswith("Y")


def main():
    df = pd.read_csv(PRACTICE_DIR / "conversations.csv", dtype=str).fillna("")
    tc, ec = pick(df, TEXT_COLS), pick(df, EXPECT_COLS)
    if tc is None:
        raise KeyError(f"Set TEXT_COLS; columns are {list(df.columns)}")
    df["_t"] = df[TURN].astype(int)
    kb, rows, ok, n = KnowledgeBase(), [], 0, 0
    for cid, g in df.sort_values(["_t"]).groupby(CONV, sort=False):
        agent = Agent("chat", kb)
        for _, r in g.iterrows():
            reply = agent.respond(r[tc])
            verdict = judge(r[tc], r[ec], reply) if ec and r[ec] else None
            if verdict is not None:
                n, ok = n + 1, ok + verdict
            rows.append({"conversation_id": cid, "turn": r[TURN], "customer": r[tc], "reply": reply,
                         "expected": r[ec] if ec else "", "judge_ok": verdict,
                         "sources": json.dumps([c["section_id"] for c in agent.last_sources]),
                         "tool_calls": json.dumps(agent.tools.log, default=str)})
    pd.DataFrame(rows).to_csv("sample_outputs/eval_conversations.csv", index=False)
    print(f"Judged turns correct: {ok}/{n}")


if __name__ == "__main__":
    main()
