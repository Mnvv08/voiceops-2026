"""Knowledge base: chunk kb/*.md by [SECTION-ID], embed with OpenAI, retrieve with cosine similarity.

ASSUMPTIONS (verify against the real files once the dataset is downloaded):
  * sections start with a line containing a bracketed ID, e.g. "## [RET-04] Title"
  * documents may carry `status:` and `effective_from:` lines near the top
"""
import hashlib
import json
import os
import re
from dataclasses import dataclass, asdict

import numpy as np
from openai import OpenAI

from .config import CACHE_DIR, EMBED_MODEL, KB_DIR, TOP_K

# Precedence from the brief: policy > agent guide > help > faq > marketing
PRECEDENCE = {"pol": 0, "agent": 1, "help": 2, "faq": 3, "mkt": 4, "00": 5}
SECTION_RE = re.compile(r"^\s*#*\s*\[([A-Za-z0-9_.\-]+)\]")


@dataclass
class Chunk:
    doc: str
    section_id: str
    status: str
    effective_from: str
    rank: int
    text: str


def _meta(text: str, key: str) -> str:
    m = re.search(rf"^\W*{key}\W*[:=]\s*(.+)$", text, re.I | re.M)
    return m.group(1).strip() if m else ""


def load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(KB_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        status, eff = _meta(raw, "status"), _meta(raw, "effective_from")
        # superseded documents are history, not policy: do not index them
        if "supersed" in status.lower() or "superseded" in path.name:
            continue
        rank = PRECEDENCE.get(path.name.split("_")[0], 5)
        sections, cur_id, cur = [], path.stem, []
        for line in raw.splitlines():
            m = SECTION_RE.match(line)
            if m and cur:
                sections.append((cur_id, "\n".join(cur)))
                cur = []
            if m:
                cur_id = m.group(1)
            cur.append(line)
        if cur:
            sections.append((cur_id, "\n".join(cur)))
        for sid, body in sections:
            body = body.strip()
            if len(body) < 40:
                continue
            # split very long sections so embeddings stay focused
            for i in range(0, len(body), 1800):
                chunks.append(Chunk(path.name, sid, status, eff, rank, body[i:i + 1800]))
    return chunks


class KnowledgeBase:
    def __init__(self):
        self.chunks = load_chunks()
        if not self.chunks:
            raise RuntimeError(f"No chunks found in {KB_DIR}. Is DATASET_DIR set correctly?")
        self.vecs = self._embeddings()

    def _embeddings(self) -> np.ndarray:
        CACHE_DIR.mkdir(exist_ok=True)
        key = hashlib.md5(("".join(c.text for c in self.chunks) + EMBED_MODEL).encode()).hexdigest()[:12]
        f = CACHE_DIR / f"emb_{key}.npy"
        if f.exists():
            return np.load(f)
        vecs = self._embed([f"[{c.section_id}] {c.text}" for c in self.chunks])
        np.save(f, vecs)
        return vecs

    @staticmethod
    def _embed(texts: list[str]) -> np.ndarray:
        oa = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        out = []
        for i in range(0, len(texts), 64):
            r = oa.embeddings.create(model=EMBED_MODEL, input=texts[i:i + 64])
            out += [d.embedding for d in r.data]
        v = np.array(out, dtype="float32")
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        q = self._embed([query])[0]
        sims = self.vecs @ q
        # small bonus for higher-precedence documents (policy beats FAQ/marketing)
        sims = sims - 0.01 * np.array([c.rank for c in self.chunks])
        out = []
        for i in np.argsort(-sims)[:k]:
            c = self.chunks[i]
            out.append({**asdict(c), "score": float(sims[i])})
        return out
