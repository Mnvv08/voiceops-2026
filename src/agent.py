"""The support agent. One class, three modes that map to the brief's steps:

  mode="prompt"  Step 1  system prompt only, no documents
  mode="rag"     Step 2  answers from retrieved KB chunks, single turn
  mode="chat"    Step 3  RAG + conversation memory + lookup_order / create_ticket
"""
import json

from . import llm
from .config import TODAY, TOP_K
from .kb import KnowledgeBase
from .tools_adapter import TOOL_SCHEMAS, Tools

BASE = f"""You are ZentroMart's customer support assistant. ZentroMart is an Indian online store for groceries and electronics.
Today's date is {TODAY}. Business days are Monday to Saturday excluding listed holidays; return windows use calendar days.
Your first reply in a conversation must say you are an AI assistant.
Replies are spoken aloud on a phone call: keep them short (1-3 sentences), plain language, no markdown, no lists.
Write dates with the year. Never invent a policy, date, amount or order detail."""

RAG_RULES = """
Answer ONLY from the CONTEXT below. If the context does not cover the question, say you don't have that
information and offer to connect the customer to a human agent. Do not guess.
If chunks conflict, trust the higher-precedence source: policy > agent guide > help > FAQ > marketing.
Never say you did something (checked an order, created a ticket) unless a tool result in this conversation shows it."""

CHAT_RULES = """
Conversation rules:
- Remember the order ID and problem the customer already gave; resolve follow-ups like "how long will that take" from context.
- If you need a detail (order ID, or phone last 4 digits / pincode to verify), ask ONE clear question.
- Before revealing any order detail, the customer must be verified; use lookup_order with their details.
- If the request breaks policy (for example demanding compensation the policy does not allow), refuse politely, explain the rule briefly,
  then call create_ticket with a short summary and hand over to a human. Only say a ticket exists if the tool returned created=true."""


class Agent:
    def __init__(self, mode: str = "chat", kb: KnowledgeBase | None = None):
        assert mode in ("prompt", "rag", "chat")
        self.mode = mode
        self.kb = kb if mode != "prompt" else None
        self.tools = Tools() if mode == "chat" else None
        self.history: list[dict] = []
        self.last_sources: list[dict] = []
        self.turns = 0

    # ---------- helpers ----------
    def _query(self, text: str) -> str:
        """Retrieval query: include recent customer turns so follow-ups still retrieve well."""
        prev = [m["content"] for m in self.history if m["role"] == "user"][-2:]
        return " ".join(prev + [text])

    def _system(self, context: str = "") -> str:
        s = BASE
        if self.mode in ("rag", "chat"):
            s += RAG_RULES
        if self.mode == "chat":
            s += CHAT_RULES
        if context:
            s += "\n\nCONTEXT:\n" + context
        return s

    # ---------- main entry ----------
    def respond(self, text: str) -> str:
        context = ""
        self.last_sources = []
        if self.kb:
            self.last_sources = self.kb.retrieve(self._query(text), TOP_K)
            context = "\n\n".join(f"[{c['doc']} | {c['section_id']}]\n{c['text']}" for c in self.last_sources)

        msgs = [{"role": "system", "content": self._system(context)}]
        if self.mode == "chat":
            msgs += self.history
        msgs.append({"role": "user", "content": text})

        reply = ""
        for _ in range(5):  # tool loop
            m = llm.chat(msgs, tools=TOOL_SCHEMAS if self.tools else None)
            if not getattr(m, "tool_calls", None):
                reply = (m.content or "").strip()
                break
            msgs.append({"role": "assistant", "content": m.content or "",
                         "tool_calls": [{"id": t.id, "type": "function",
                                         "function": {"name": t.function.name, "arguments": t.function.arguments}}
                                        for t in m.tool_calls]})
            for t in m.tool_calls:
                result = self.tools.run(t.function.name, json.loads(t.function.arguments or "{}"))
                msgs.append({"role": "tool", "tool_call_id": t.id, "content": result})

        if self.turns == 0 and "ai assistant" not in reply.lower():
            reply = "Hi, I'm ZentroMart's AI assistant. " + reply
        self.turns += 1
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        return reply
