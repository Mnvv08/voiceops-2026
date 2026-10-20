"""Batch run: one command for the organisers.

    python run.py --audio_dir ./test_audio --out_dir ./outputs

Each audio file is treated as an independent customer call (fresh agent). For every file we save
<name>.json (transcript, reply, retrieved chunks, tool calls) and <name>_reply.wav.
"""
import argparse
import json
from pathlib import Path

from src.agent import Agent
from src.kb import KnowledgeBase
from src.voice import Voice

EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--lang", default="en-IN")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kb, voice = KnowledgeBase(), Voice()
    files = sorted(p for p in Path(a.audio_dir).iterdir() if p.suffix.lower() in EXTS)

    for p in files:
        agent = Agent("chat", kb)
        rec = {"file": p.name}
        try:
            heard, reply = voice.turn(str(p), agent, str(out / f"{p.stem}_reply.wav"), a.lang)
            rec.update(transcript=heard, reply=reply, audio=f"{p.stem}_reply.wav",
                       sources=[{k: c[k] for k in ("doc", "section_id", "score")} for c in agent.last_sources],
                       chunks=[c["text"] for c in agent.last_sources],
                       tool_calls=agent.tools.log if agent.tools else [])
        except Exception as e:  # never crash the whole batch on one bad file
            rec["error"] = str(e)
        (out / f"{p.stem}.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
        print(p.name, "->", rec.get("reply") or rec.get("error"))


if __name__ == "__main__":
    main()
