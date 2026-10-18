#!/usr/bin/env python3
"""
Semantic topic de-duplication (local embeddings, free, CPU only).

  python dedupe.py check "Fall of the US dollar"   # quick manual test
  python dedupe.py commit                          # append build/topic_entry.json to used_topics.json

Heavy imports (torch / sentence-transformers) are lazy, so `commit` is instant.
"""
import datetime as dt
import json
import os
import sys
from pathlib import Path

DB_PATH = Path(os.getenv("USED_TOPICS_PATH") or "used_topics.json")
ENTRY_PATH = Path("build/topic_entry.json")
MODEL_NAME = os.getenv("EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
THRESHOLD = float(os.getenv("DEDUPE_THRESHOLD") or "0.78")  # >= this = duplicate

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def embed(text: str) -> list:
    vec = _get_model().encode(text, normalize_embeddings=True)
    return [round(float(x), 5) for x in vec]


def load(path: Path = DB_PATH) -> dict:
    if not path.exists():
        return {"entries": []}
    return json.loads(path.read_text(encoding="utf-8"))


def check(text: str, db: dict, threshold: float = THRESHOLD):
    """Return (is_duplicate, best_score, nearest_topic, vector)."""
    import numpy as np

    vec = embed(text)
    entries = db.get("entries", [])
    if not entries:
        return False, 0.0, None, vec
    mat = np.array([e["vector"] for e in entries], dtype="float32")
    scores = mat @ np.array(vec, dtype="float32")  # cosine (vectors are normalized)
    i = int(scores.argmax())
    return bool(scores[i] >= threshold), float(scores[i]), entries[i]["topic"], vec


def make_entry(topic: str, text: str, vec: list, title: str = "") -> dict:
    return {
        "date": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"),
        "topic": topic,
        "title": title,
        "text": text,
        "vector": vec,
    }


def save_entry_file(entry: dict) -> None:
    ENTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENTRY_PATH.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")


def commit() -> None:
    if not ENTRY_PATH.exists():
        print("No build/topic_entry.json - nothing to commit.")
        return
    entry = json.loads(ENTRY_PATH.read_text(encoding="utf-8"))
    db = load()
    if any(e["topic"] == entry["topic"] for e in db["entries"]):
        print("Topic already recorded.")
        return
    db["entries"].append(entry)
    DB_PATH.write_text(json.dumps(db, ensure_ascii=False), encoding="utf-8")
    print(f"Recorded topic: {entry['topic']} (total {len(db['entries'])})")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "commit":
        commit()
    elif len(sys.argv) >= 3 and sys.argv[1] == "check":
        dup, score, near, _ = check(" ".join(sys.argv[2:]), load())
        print(f"duplicate={dup} score={score:.3f} nearest={near!r} threshold={THRESHOLD}")
    else:
        print(__doc__)
