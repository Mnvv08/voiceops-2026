# VoiceOps 2026: ZentroMart Voice Support Agent

A customer-support agent that answers from company documents (RAG), holds a multi-turn conversation
with order lookup and human handover, and speaks via Gnani STT/TTS.

**Stack:** Groq (LLM) · OpenAI embeddings · numpy cosine index · Gnani STT/TTS (only speech service used).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill GROQ_API_KEY, OPENAI_API_KEY, GNANI_API_KEY
# download + extract the organiser dataset, then set DATASET_DIR in .env
python validate_dataset.py  # inside the dataset folder
```

## Run (batch, as used by organisers)
```bash
python run.py --audio_dir ./test_audio --out_dir ./outputs
```
Saves per file: `<name>.json` (transcript, reply, retrieved chunks, tool calls) and `<name>_reply.wav`.

## Live demo (record your voice)
```bash
streamlit run app.py
```

## Evaluate Step 1 vs Step 2
```bash
python -m src.evaluate               # Step 1 vs Step 2 on practice/questions.csv
python -m src.eval_conversations    # multi-turn practice conversations
```

## Layout
- `src/kb.py` chunking by `[SECTION-ID]`, embeddings, retrieval with source precedence
- `src/agent.py` modes: `prompt` (Step 1), `rag` (Step 2), `chat` (Step 3 with tools)
- `src/tools_adapter.py` wraps the provided `lookup_order` / `create_ticket`
- `src/voice.py` Gnani STT -> agent -> Gnani TTS
- `run.py` batch runner · `app.py` Streamlit demo · `REPORT_TEMPLATE.md` report outline
