# VoiceOps 2026 Report (max 4 pages)

**Team / name:** ...  **Stack:** Groq (LLM), OpenAI embeddings, numpy index, Gnani STT/TTS

## 1. What I built
- **Step 1, prompt-only:** system prompt, AI-assistant disclosure, no documents.
- **Step 2, RAG:** chunking by [SECTION-ID], embedding model, top-k, source precedence, "I don't know" rule.
- **Step 3, conversation:** memory, query rewriting for follow-ups, verification, lookup_order / create_ticket, refusals and handover.
- **Step 4, voice:** Gnani STT -> agent -> Gnani TTS, empty-audio fallback, text prepared for speech.

## 2. Step 1 vs Step 2
| | Step 1 (prompt only) | Step 2 (RAG) |
|---|---|---|
| Practice questions correct | __/52 | __/52 |
Example where RAG fixed a wrong answer: ...
Example where it still failed and why: ...

## 3. Conversation results
Practice conversations: __/__ judged turns correct. Notes on follow-ups, clarifying questions, handovers.

## 4. What went wrong and what I would fix
- ...

## 5. Bonus work
- ...
