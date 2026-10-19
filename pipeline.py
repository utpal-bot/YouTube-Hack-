#!/usr/bin/env python3
"""
Daily documentary pipeline.

  python pipeline.py prepare                  # topic -> script -> voice -> images   (build/)
  python pipeline.py render --target long     # 1920x1080 long video + thumbnail     (out/long.mp4)
  python pipeline.py render --target short1   # 1080x1920 Shorts: short1..short3     (out/shortN.mp4)
  python pipeline.py voicetest                # sample Hinglish clips for every voice option (out/)
"""
import argparse
import asyncio
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import dedupe

BUILD, OUT, TMP = Path("build"), Path("out"), Path("tmp")
AUDIO, IMG = BUILD / "audio", BUILD / "img"
MUSIC = Path("assets/music.mp3")
SFX_DIR = Path("assets/sfx")


def env(name: str, default: str) -> str:
    return os.getenv(name) or default


# ---- LLM Configuration ----
LLM_BASE = env("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_KEY = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY") or ""
MODEL = env("LLM_MODEL", "openai/gpt-oss-120b")
FAST_MODEL = env("LLM_FAST_MODEL", "openai/gpt-oss-20b")
FALLBACKS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
LLM_TPM = int(env("LLM_TPM", "7500"))
WRITE_PARTS = int(env("WRITE_PARTS", "2"))
TRANSLIT_BATCH = int(env("TRANSLIT_BATCH", "10"))
NICHE = env("NICHE", "mysteries of technology, science and the human mind")
LANGUAGE = env("LANGUAGE", "English")

IS_HINGLISH = "hinglish" in LANGUAGE.lower()
USE_DEVA = IS_HINGLISH and env("HINGLISH_MODE", "devanagari").lower() != "roman"
if IS_HINGLISH:
    _default_voice = "hi-IN-MadhurNeural" if USE_DEVA else "en-IN-PrabhatNeural"
else:
    _default_voice = "en-US-AndrewMultilingualNeural"
VOICE = env("VOICE", _default_voice)
VOICE_RATE = env("VOICE_RATE", "+0%")
LANG_DESC = (
    "Hinglish: Hindi written in simple Roman (English) letters, natural Indian spoken style, "
    "mixing common English words" if IS_HINGLISH else LANGUAGE
)

FPS = int(env("FPS", "30"))
PAD = 0.25
SILENCE_EXTRA = 0.6
BORDERLINE = 0.65

MUSIC_GAIN = float(env("MUSIC_GAIN", "0.12"))
SFX_GAIN = {
    "riser": float(env("SFX_GAIN_RISER", "1.2")),
    "braam": float(env("SFX_GAIN_BRAAM", "0.7")),
    "glitch": float(env("SFX_GAIN_GLITCH", "0.5")),
}
SFX_CAPS = {"riser": 6, "braam": 4, "glitch": 5, "silence": 4}
SFX_EXT = (".wav", ".mp3", ".ogg", ".flac", ".m4a")

TARGETS = {
    "long": (1920, 1080),
    "short1": (1080, 1920),
    "short2": (1080, 1920),
    "short3": (1080, 1920),
}

TOPIC_SYSTEM = (
    "You pick fresh topics for a documentary-style faceless YouTube channel. "
    "Return ONLY JSON. Topics must be specific, intriguing and clearly different "
    "in core concept from anything already covered (even if worded differently)."
)

PLAN_GUIDE = """You are the head writer of a documentary-style faceless YouTube channel (suspense, curiosity-driven, factual).
Plan ONE video as a list of beats. Do NOT write narration yet. Write beats in English.

STRUCTURE: cold_open (hook + promise) -> past -> present -> future -> resolution.

RETENTION RULES:
- Open a loop in the cold open and CLOSE it in the resolution.
- End every section with a new open loop that is PAID OFF later. Never leave a cliffhanger unresolved.
- A new fact, twist or question every 2-3 beats (~20-30 seconds).

FACTS: Only well-established facts. Never invent statistics, quotes, studies or names.
Anything a human should double-check goes into "verify".

SFX: "sfx" is a list using ONLY these tags: riser, braam, silence, glitch (use [] for none).
- riser: on the LAST beat of a section (builds into the next beat's cut). Max 6 per video.
- braam: on the FIRST beat after a riser = the big reveal. Max 4 per video.
- silence: on the beat right before the single biggest reveal (music drops out). Max 4.
- glitch: on unsettling / tension moments. Max 5.

OUTPUT: ONLY valid JSON (no markdown fences):
{
  "title": "<= 70 chars, curiosity-driven, NOT misleading",
  "description": "2-3 sentences + 3 hashtags",
  "tags": ["8-12 tags"],
  "topic_summary": "one sentence",
  "thumbnail_text": "2-4 words, ALL CAPS, curiosity, Roman letters only",
  "thumbnail_scene_id": 1,
  "verify": ["claims a human should fact-check before publish"],
  "beats": [
    {"id": 1, "section": "cold_open|past|present|future|resolution",
     "beat": "max 18 words: what this scene says",
     "visual_query": "2-4 word stock photo search", "sfx": ["riser"]}
  ],
  "shorts": [
    {"title": "...", "hook": "max 8 words, shown on screen in first 3 seconds",
     "thumb_text": "2-4 words, ALL CAPS", "scene_ids": [3,4,5,6]}
  ]
}
RULES: 38-44 beats (each beat = one ~10 second scene, ~1200 words once written). Exactly 3 shorts,
each using 3-4 CONSECUTIVE beat ids (~30-50 seconds), standalone-friendly: one from past, one from present, one from future.
"""

WRITE_GUIDE = """You are the narrator-writer of a documentary-style faceless YouTube channel.
Write the narration in __LANG__.
For each requested beat write ONE scene: 25-35 words, conversational spoken rhythm, short sentences (max ~15 words).
It must flow from the previous scene, follow the outline's open loops and pay them off, and never invent
statistics, quotes, studies or names.
Return ONLY JSON (no markdown): {"scenes":[{"id":N,"narration":"..."}]} - exactly the requested ids.
""".replace("__LANG__", LANG_DESC)


def sh(cmd):
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd[:8]), "...", flush=True)
    subprocess.run(cmd, check=True)


def probe_duration(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(out.strip())


_window = []


def _budget_wait(est: int):
    while True:
        now = time.time()
        while _window and now - _window[0][0] >= 60:
            _window.pop(0)
        used = sum(t for _, t in _window)
        if used + est <= LLM_TPM or not _window:
            return
        wait = 60 - (now - _window[0][0]) + 0.5
        print(f"  [tpm] {used}+{est} > {LLM_TPM}, waiting {wait:.0f}s", flush=True)
        time.sleep(max(wait, 1))


def claude(model: str, system: str, user: str, max_tokens: int) -> str:
    import requests

    if not LLM_KEY:
        raise SystemExit("GROQ_API_KEY (or LLM_API_KEY) is not set.")
    chain = [model] + [m for m in FALLBACKS if m != model]
    est = int((len(system) + len(user)) / 3.2) + max_tokens
    last = ""
    for m in chain:
        for attempt in range(5):
            _budget_wait(est)
            body = {
                "model": m,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }
            if "gpt-oss" in m:
                body["reasoning_effort"] = "low"
            r = requests.post(f"{LLM_BASE}/chat/completions", timeout=180,
                              headers={"Authorization": f"Bearer {LLM_KEY}"}, json=body)
            if r.status_code == 404 or (r.status_code == 400 and "decommission" in r.text):
                last = r.text
                print(f"  model {m} unavailable -> trying next", flush=True)
                break
            if r.status_code == 429:
                wait = float(r.headers.get("retry-after") or 10 * (attempt + 1))
                print(f"  429 rate limited, sleeping {wait:.0f}s", flush=True)
                _window.append((time.time(), LLM_TPM))
                time.sleep(wait)
                continue
            if r.status_code == 413:
                raise RuntimeError("Request too large for your TPM tier. Set WRITE_PARTS=3 (or 4) and retry.")
            r.raise_for_status()
            data = r.json()
            _window.append((time.time(), int((data.get("usage") or {}).get("total_tokens") or est)))
            return data["choices"][0]["message"].get("content") or ""
    raise RuntimeError(f"No usable model. Last error: {last}")


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def claude_json(model, system, user, max_tokens, tries=3) -> dict:
    last = None
    for i in range(tries):
        try:
            return parse_json(claude(model, system, user, int(max_tokens * (1 + 0.3 * i))))
        except SystemExit:
            raise
        except Exception as e:
            last = e
            print(f"  claude_json attempt {i + 1} failed: {e}", flush=True)
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"LLM JSON failed after {tries} tries: {last}")


def judge_same(a: str, b: str) -> bool:
    out = claude_json(
        FAST_MODEL, "Answer ONLY JSON.",
        f'Are these two video topics about the same core concept?\nA: {a}\nB: {b}\n'
        'Return {"same": true|false}', 500,
    )
    return bool(out.get("same"))


def choose_topic(db: dict):
    recent = [e["topic"] for e in db["entries"][-30:]]
    for attempt in range(6):
        listing = "\n- ".join(recent) if recent else "(none yet)"
        out = claude_json(
            FAST_MODEL, TOPIC_SYSTEM,
            f"Niche: {NICHE}\nAlready covered (never repeat or rephrase):\n- {listing}\n"
            'Return {"topic": "...", "angle": "..."}', 1200,
        )
        topic, angle = out["topic"].strip(), out.get("angle", "").strip()
        text = f"{topic}. {angle}"
        dup, score, near, vec = dedupe.check(text, db)
        print(f"[topic try {attempt + 1}] {topic!r} score={score:.2f} nearest={near!r}", flush=True)
        if dup or (score >= BORDERLINE and near and judge_same(topic, near)):
            recent.append(topic)
            continue
        return topic, angle, text, vec
    raise SystemExit("Could not find a unique topic after 6 attempts.")


def parse_sfx(value) -> list:
    text = " ".join(value) if isinstance(value, (list, tuple)) else str(value or "")
    return [n for n in ("riser", "braam", "glitch", "silence") if re.search(rf"\b{n}\b", text, re.I)]


def cap_sfx(scenes):
    counts = {}
    for s in scenes:
        if "silence" in s["sfx"] and "riser" in s["sfx"]:
            s["sfx"].remove("riser")
        keep = []
        for tag in s["sfx"]:
            if counts.get(tag, 0) < SFX_CAPS[tag]:
                keep.append(tag)
                counts[tag] = counts.get(tag, 0) + 1
        s["sfx"] = keep


def fill_tts_text(scenes):
    missing = [s for s in scenes if not (s.get("tts_text") or "").strip()]
    for i in range(0, len(missing), TRANSLIT_BATCH):
        batch = missing[i:i + TRANSLIT_BATCH]
        print(f"  transliterating scenes {batch[0]['id']}-{batch[-1]['id']} to Devanagari", flush=True)
        out = claude_json(
            FAST_MODEL, "Transliterate. Answer ONLY JSON.",
            "Rewrite each Roman Hinglish sentence in Devanagari script, word for word (same number of "
            "words, same order); transliterate English words into Devanagari too.\n"
            'Return {"items":[{"id":1,"tts_text":"..."}]}\n'
            + json.dumps([{"id": s["id"], "text": s["narration"]} for s in batch], ensure_ascii=False),
            1000 + 260 * len(batch),
        )
        mapping = {}
        for it in out.get("items", []):
            try:
                mapping[int(it["id"])] = (it.get("tts_text") or "").strip()
            except (KeyError, TypeError, ValueError):
                continue
        for s in batch:
            s["tts_text"] = mapping.get(s["id"]) or s["narration"]


def outline_text(beats) -> str:
    return "\n".join(f"{b['id']}. [{b['section']}] {b['beat']}" for b in beats)


def split_ids(n: int, parts: int):
    size = -(-n // max(1, parts))
    return [list(range(i + 1, min(i + size, n) + 1)) for i in range(0, n, size)]


def write_part(plan: dict, beats: list, ids: list, tail: str) -> dict:
    got = {}
    for _ in range(2):
        want = [i for i in ids if i not in got]
        if not want:
            break
        user = (
            f"Title: {plan['title']}\nSummary: {plan.get('topic_summary', '')}\n"
            f"FULL OUTLINE:\n{outline_text(beats)}\n\n"
            + (f"Previous scene narration (continue from it): {tail}\n\n" if tail else "")
            + "Write narration ONLY for these beat ids: " + ", ".join(map(str, want)) + "\n"
            + 'Return {"scenes":[{"id":N,"narration":"..."}]}'
        )
        out = claude_json(MODEL, WRITE_GUIDE, user, max_tokens=900 + 130 * len(want))
        for it in out.get("scenes", []):
            try:
                i, text = int(it["id"]), (it.get("narration") or "").strip()
            except (KeyError, TypeError, ValueError):
                continue
            if i in want and text:
                got[i] = text
    return got


def generate_script(topic: str, angle: str) -> dict:
    plan = claude_json(
        MODEL, PLAN_GUIDE,
        f"Topic: {topic}\nAngle: {angle or 'your choice'}\nPlan the video now.", max_tokens=4500,
    )
    beats = plan.get("beats") or []
    if len(beats) < 10:
        raise RuntimeError(f"Plan too short: {len(beats)} beats")

    idmap = {}
    for new, b in enumerate(beats, 1):
        try:
            idmap[int(b.get("id", new))] = new
        except (TypeError, ValueError):
            pass
        b["id"] = new
        b["section"] = b.get("section") or "present"
        b["beat"] = (b.get("beat") or "").strip()
        b["visual_query"] = b.get("visual_query") or "dark abstract technology"
        b["sfx"] = parse_sfx(b.get("sfx"))
    n = len(beats)
    narr, tail = {}, ""
    for ids in split_ids(n, WRITE_PARTS):
        print(f"  writing scenes {ids[0]}-{ids[-1]} of {n}", flush=True)
        narr.update(write_part(plan, beats, ids, tail))
        tail = " ".join(narr[i] for i in ids[-2:] if i in narr)[-400:]

    scenes = []
    for b in beats:
        text = narr.get(b["id"])
        if not text:
            print(f"  WARNING: scene {b['id']} missing narration, using the beat text")
            text = b["beat"]
        scenes.append({"id": b["id"], "section": b["section"], "narration": text,
                       "visual_query": b["visual_query"], "sfx": b["sfx"]})
    cap_sfx(scenes)
    if USE_DEVA:
        fill_tts_text(scenes)

    ids = {s["id"] for s in scenes}
    shorts = plan.get("shorts") or []
    fixed = []
    for k in range(3):
        src = shorts[k] if k < len(shorts) else {}
        sid = []
        for i in src.get("scene_ids", []):
            try:
                j = idmap.get(int(i))
            except (TypeError, ValueError):
                j = None
            if j in ids:
                sid.append(j)
        if not sid:
            start = min(k * (n // 3) + 1, max(1, n - 3))
            sid = list(range(start, min(start + 4, n + 1)))
        fixed.append({
            "title": src.get("title") or f"{plan['title']} (Part {k + 1})",
            "hook": src.get("hook") or plan["title"],
            "thumb_text": src.get("thumb_text") or "",
            "scene_ids": sid,
        })
    try:
        thumb_scene = idmap.get(int(plan.get("thumbnail_scene_id") or 1), 1)
    except (TypeError, ValueError):
        thumb_scene = 1
    return {
        "title": plan["title"],
        "description": plan.get("description", ""),
        "tags": plan.get("tags", []),
        "topic_summary": plan.get("topic_summary", ""),
        "thumbnail_text": plan.get("thumbnail_text", ""),
        "thumbnail_scene_id": thumb_scene,
        "verify": plan.get("verify", []),
        "scenes": scenes,
        "shorts": fixed,
    }


def spread_words(marks, text: str, dur: float):
    if not marks:
        marks = [(0.0, dur, text)]
    out = []
    for off, d, t in marks:
        words = t.split()
        if not words:
            continue
        total = sum(len(w) + 1 for w in words)
        cur = off
        for w in words:
            span = d * (len(w) + 1) / total
            out.append([round(cur, 3), round(cur + span, 3), w])
            cur += span
    return out


def retarget_words(spoken, caption_text: str):
    cw = caption_text.split()
    if not spoken or not cw:
        return spoken
    out, ns = [], len(spoken)
    for j, w in enumerate(cw):
        if len(cw) == ns:
            s, e = spoken[j][0], spoken[j][1]
        else:
            a = min(int(j * ns / len(cw)), ns - 1)
            b = min(max(a, int((j + 1) * ns / len(cw)) - 1), ns - 1)
            s, e = spoken[a][0], spoken[b][1]
        out.append([s, e, w])
    return out


async def synth_voice(text: str, path: Path, voice: str, rate: str):
    import edge_tts
    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=20), reraise=True)
    async def run():
        try:
            comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
        except TypeError:
            comm = edge_tts.Communicate(text, voice, rate=rate)
        marks = []
        with open(path, "wb") as f:
            async for ch in comm.stream():
                if ch["type"] == "audio":
                    f.write(ch["data"])
                elif ch["type"] in ("WordBoundary", "SentenceBoundary"):
                    marks.append((ch["offset"] / 1e7, ch["duration"] / 1e7, ch["text"]))
        return marks

    return await run()


async def tts_scene(sem, scene: dict) -> dict:
    path = AUDIO / f"s{scene['id']:03d}.mp3"
    spoken_text = scene["tts_text"] if USE_DEVA and scene.get("tts_text") else scene["narration"]
    async with sem:
        marks = await synth_voice(spoken_text, path, VOICE, VOICE_RATE)
    dur = probe_duration(path)
    words = spread_words(marks, spoken_text, dur)
    if USE_DEVA:
        words = retarget_words(words, scene["narration"])
    pad = PAD + (SILENCE_EXTRA if "silence" in scene.get("sfx", []) else 0.0)
    return {
        "id": scene["id"],
        "audio": str(path),
        "dur": round(dur + pad, 3),
        "pad": round(pad, 3),
        "words": words,
    }


async def tts_all(scenes):
    sem = asyncio.Semaphore(4)
    return await asyncio.gather(*(tts_scene(sem, s) for s in scenes))


def cmd_voicetest():
    OUT.mkdir(exist_ok=True)
    roman = "Socho agar ek din tumhara dimaag khud sochna band kar de, aur tumhe pata bhi na chale."
    deva = "सोचो अगर एक दिन तुम्हारा दिमाग खुद सोचना बंद कर दे, और तुम्हें पता भी न चले।"
    jobs = [
        ("A_hi-IN-MadhurNeural_devanagari", "hi-IN-MadhurNeural", deva),
        ("B_hi-IN-SwaraNeural_devanagari", "hi-IN-SwaraNeural", deva),
        ("C_en-IN-PrabhatNeural_roman", "en-IN-PrabhatNeural", roman),
        ("D_en-IN-NeerjaNeural_roman", "en-IN-NeerjaNeural", roman),
        ("E_hi-IN-MadhurNeural_roman_BROKEN_CASE", "hi-IN-MadhurNeural", roman),
    ]

    async def run():
        for name, voice, text in jobs:
            await synth_voice(text, OUT / f"voicetest_{name}.mp3", voice, "+0%")
            print("wrote", OUT / f"voicetest_{name}.mp3")

    asyncio.run(run())
    print("Listen to A/B vs C/D. Set VOICE and HINGLISH_MODE accordingly.")


# --------------------------------------------------------------------------- #
# Images (Robust Pixabay Fetcher + Ladder of Fallbacks)
# --------------------------------------------------------------------------- #
PIX_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://pixabay.com/",
}
GENERIC_QUERIES = [
    "abstract technology", "futuristic technology", "space galaxy", "science laboratory",
    "neural network", "digital data", "night city lights", "abstract light",
    "microscope science", "planet earth",
]
SECTION_QUERIES = {
    "cold_open": ["mystery dark", "abstract dark"],
    "past": ["vintage history", "old library"],
    "present": ["modern city", "technology"],
    "future": ["futuristic city", "space galaxy"],
    "resolution": ["sunrise horizon", "light rays"],
}
STOPWORDS = {"the", "and", "that", "with", "from", "this", "what", "your", "into", "will", "have", "about",
             "when", "they", "them", "than", "then", "which", "while", "there", "their", "here", "just"}
_pix = {"calls": [], "cache": {}, "disabled": None, "warned_nokey": False}


def topic_queries(script: dict, topic: str = "") -> list:
    words = [w for w in re.findall(r"[a-z]{4,}", f"{script.get('title', '')} {topic}".lower()) if w not in STOPWORDS]
    uniq = list(dict.fromkeys(words))
    out = []
    if len(uniq) >= 2:
        out.append(" ".join(uniq[:2]))
    if len(uniq) >= 4:
        out.append(" ".join(uniq[2:4]))
    return out + uniq[:2]


def _pix_throttle():
    now = time.time()
    _pix["calls"] = [t for t in _pix["calls"] if now - t < 60]
    if len(_pix["calls"]) >= 85:
        time.sleep(max(1.0, 60 - (now - _pix["calls"][0]) + 0.5))
    _pix["calls"].append(time.time())


def pixabay_search(key: str, query: str, strict: bool = True) -> list:
    import requests

    ck = (query, strict)
    if ck in _pix["cache"]:
        return _pix["cache"][ck]
    params = {"key": key, "q": query[:100], "safesearch": "true", "per_page": 30}
    if strict:
        params.update({"image_type": "photo", "orientation": "horizontal", "min_width": 1280})
    for attempt in range(3):
        _pix_throttle()
        r = requests.get("https://pixabay.com/api/", params=params, headers=PIX_HEADERS, timeout=30)
        if r.status_code == 429:
            print(f"  pixabay 429, sleeping {20 * (attempt + 1)}s", flush=True)
            time.sleep(20 * (attempt + 1))
            continue
        if r.status_code in (400, 401, 403):
            _pix["disabled"] = f"HTTP {r.status_code}: {r.text[:150]}"
            print(f"  PIXABAY ERROR -> {_pix['disabled']}", flush=True)
            return []
        r.raise_for_status()
        hits = r.json().get("hits", [])
        _pix["cache"][ck] = hits
        return hits
    return []


def _download_image(hit: dict, dst: Path) -> bool:
    import requests
    from PIL import Image
    from io import BytesIO

    for url in (hit.get("fullHDURL"), hit.get("largeImageURL"), hit.get("webformatURL")):
        if not url:
            continue
        try:
            data = requests.get(url, headers=PIX_HEADERS, timeout=60)
            data.raise_for_status()
            im = Image.open(BytesIO(data.content)).convert("RGB")
            if im.width < 600:
                continue
            im.save(dst, quality=92)
            return True
        except Exception as e:
            print(f"  image download failed ({url[:60]}...): {e}", flush=True)
    return False


def gradient_card(dst: Path, seed: int):
    from PIL import Image, ImageDraw

    rnd = random.Random(seed)
    top = (rnd.randint(5, 40), rnd.randint(5, 40), rnd.randint(30, 90))
    w, h = 1920, 1080
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / h
        d.line([(0, y), (w, y)], fill=tuple(int(c * (1 - t)) for c in top))
    img.save(dst, quality=92)


def fetch_image(scene: dict, used: set, extra_queries=None) -> str:
    dst = IMG / f"{scene['id']:03d}.jpg"
    key = os.getenv("PIXABAY_API_KEY")
    if not key:
        if not _pix["warned_nokey"]:
            print("WARNING: PIXABAY_API_KEY is empty/not passed to this step -> gradient cards only!", flush=True)
            _pix["warned_nokey"] = True
        gradient_card(dst, scene["id"])
        return "gradient:no_key"

    q = (scene.get("visual_query") or "").strip()
    i = scene["id"]
    ladder = [(q, True), (" ".join(q.split()[:2]), True)]
    ladder += [(x, True) for x in (extra_queries or [])]
    ladder += [(x, True) for x in SECTION_QUERIES.get(scene.get("section", ""), [])]
    ladder += [(GENERIC_QUERIES[i % len(GENERIC_QUERIES)], True), (GENERIC_QUERIES[(i + 3) % len(GENERIC_QUERIES)], True)]
    ladder += [(q, False), ("abstract background", False)]

    seen = set()
    for query, strict in ladder:
        if not query or (query, strict) in seen:
            continue
        seen.add((query, strict))
        if _pix["disabled"]:
            break
        try:
            hits = pixabay_search(key, query, strict)
        except Exception as e:
            print(f"  pixabay request failed for {query!r}: {e}", flush=True)
            continue
        hits = [h for h in hits if h.get("largeImageURL") not in used]
        if not hits:
            continue
        pool = hits[:10]
        for k in range(min(3, len(pool))):
            h = pool[(i + k) % len(pool)]
            if _download_image(h, dst):
                used.add(h.get("largeImageURL"))
                return f"pixabay:{query}"
    gradient_card(dst, scene["id"])
    return f"gradient:{_pix['disabled'] or 'no_results'}"


def ass_time(t: float) -> str:
    cs = int(round(max(t, 0) * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def clean(w: str) -> str:
    return w.replace("{", "(").replace("}", ")").replace("\\", "")


def chunk_words(words, n):
    line = []
    for w in words:
        line.append(w)
        end_punct = w[2].rstrip("\"'").endswith((".", "?", "!", ":", ";"))
        if len(line) >= n or (end_punct and len(line) >= 2):
            yield line
            line = []
    if line:
        yield line


def make_ass(words, W: int, H: int, short: bool, hook: str = "") -> str:
    fs, outline, mv = (88, 6, 520) if short else (64, 4, 70)
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {W}\nPlayResY: {H}\nWrapStyle: 2\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Cap,Liberation Sans,{fs},&H0000E5FF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,{outline},2,2,60,60,{mv},1\n"
        f"Style: Hook,Liberation Sans,{int(fs * 1.15)},&H0000E5FF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,{outline},3,8,60,60,260,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = []
    if hook:
        events.append(
            f"Dialogue: 0,{ass_time(0)},{ass_time(3)},Hook,,0,0,0,,{{\\fad(150,150)}}{clean(hook).upper()}"
        )
    for line in chunk_words(words, 4 if short else 6):
        parts = []
        for i, (s, e, w) in enumerate(line):
            nxt = line[i + 1][0] if i + 1 < len(line) else e
            cs = max(1, round((nxt - s) * 100))
            word = clean(w).upper() if short else clean(w)
            parts.append("{\\k%d}%s" % (cs, word))
        events.append(
            f"Dialogue: 0,{ass_time(line[0][0])},{ass_time(line[-1][1] + 0.05)},Cap,,0,0,0,,{' '.join(parts)}"
        )
    return header + "\n".join(events) + "\n"


def synth_sfx(name: str, dst: Path):
    TMP.mkdir(exist_ok=True)
    if name == "riser":
        src = "anoisesrc=color=pink:amplitude=1.0:duration=4:sample_rate=48000"
        af = "highpass=f=250,volume='pow(t/4,2.2)':eval=frame,afade=t=out:st=3.92:d=0.08"
    elif name == "braam":
        expr = ("(0.5*sin(2*PI*55*t)+0.35*sin(2*PI*55.8*t)+0.25*sin(2*PI*110*t)"
                "+0.2*sin(2*PI*165*t)+0.15*sin(2*PI*220.5*t))*exp(-1.1*t)*min(1,t/0.02)")
        src, af = f"aevalsrc='{expr}':d=3:s=48000", "lowpass=f=1400,volume=0.9"
    else:
        src = "anoisesrc=color=white:duration=0.4:sample_rate=48000"
        af = ("volume='if(lt(mod(t,0.08),0.04),1,0.15)':eval=frame,highpass=f=600,"
              "acrusher=bits=5:mode=log:mix=0.9,afade=t=out:st=0.3:d=0.1")
    sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", src,
        "-af", af, "-ac", "2", dst])


def get_sfx(name: str, i: int = 0) -> Path:
    variants = [p for p in sorted(SFX_DIR.glob(f"{name}*")) if p.suffix.lower() in SFX_EXT]
    if variants:
        return variants[i % len(variants)]
    dst = TMP / f"synth_{name}.wav"
    if not dst.exists():
        print(f"  assets/sfx/{name}* not found -> synthesising fallback")
        synth_sfx(name, dst)
    return dst


def prep_image(src: Path, W: int, H: int, dst: Path):
    from PIL import Image

    im = Image.open(src).convert("RGB")
    tw, th = W * 2, H * 2
    scale = max(tw / im.width, th / im.height)
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    left, top = (im.width - tw) // 2, (im.height - th) // 2
    im.crop((left, top, left + tw, top + th)).save(dst, quality=92)


def ken_burns(idx: int, n_frames: int):
    """Smooth ease-in/out camera move, a different one per scene so the video never feels static."""
    N = max(n_frames - 1, 1)
    e = f"((on/{N})*(on/{N})*(3-2*on/{N}))"
    cx, cy = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    mode = idx % 6
    if mode == 0:
        return f"1+0.2*{e}", cx, cy
    if mode == 1:
        return f"1.2-0.2*{e}", cx, cy
    if mode == 2:
        return "1.16", f"(iw-iw/zoom)*{e}", cy
    if mode == 3:
        return "1.16", f"(iw-iw/zoom)*(1-{e})", cy
    if mode == 4:
        return f"1+0.18*{e}", f"(iw-iw/zoom)*{e}", f"(ih-ih/zoom)*(1-{e})"
    return f"1+0.18*{e}", f"(iw-iw/zoom)*(1-{e})", f"(ih-ih/zoom)*{e}"


def render_scene(img: Path, audio: str, n_frames: int, idx: int, W: int, H: int, out: Path):
    z, x, y = ken_burns(idx, n_frames)
    vf = f"zoompan=z='{z}':x='{x}':y='{y}':d={n_frames}:s={W}x{H}:fps={FPS},format=yuv420p"
    sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", img, "-i", audio,
        "-vf", vf, "-af", "apad,aresample=48000", "-ac", "2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k",
        "-frames:v", n_frames, "-shortest", out])


def finalize(concat: Path, ass: Path, out: Path, events: list, silences: list):
    inputs = ["-i", concat]
    chains = [f"[0:v]ass={ass}[v]"]
    labels = ["[0:a]"]
    idx = 1

    if MUSIC.exists():
        duck = "".join(f",volume=enable='between(t,{a:.3f},{b:.3f})':volume=0" for a, b in silences)
        inputs += ["-stream_loop", "-1", "-i", MUSIC]
        chains.append(
            f"[{idx}:a]aformat=sample_rates=48000:channel_layouts=stereo,volume={MUSIC_GAIN}{duck}[mus]"
        )
        labels.append("[mus]")
        idx += 1

    for k, ev in enumerate(events):
        path = get_sfx(ev["kind"], k)
        length = probe_duration(path)
        f = "aformat=sample_rates=48000:channel_layouts=stereo"
        if ev["kind"] == "riser":
            d = min(length, ev["end"] - ev["start"])
            at = ev["end"] - d
            f += f",atrim=start={length - d:.3f},asetpts=PTS-STARTPTS"
        else:
            at = max(0.0, ev["start"] - (0.05 if ev["kind"] == "glitch" else 0.0))
        f += f",adelay={int(at * 1000)}:all=1,volume={SFX_GAIN[ev['kind']]}"
        inputs += ["-i", path]
        chains.append(f"[{idx}:a]{f}[x{k}]")
        labels.append(f"[x{k}]")
        idx += 1

    if len(labels) > 1:
        chains.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=first:normalize=0,alimiter=limit=0.95[a]")
        amap = "[a]"
    else:
        amap = "0:a"

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
           "-filter_complex", ";".join(chains), "-map", "[v]", "-map", amap,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out]
    sh(cmd)


def cmd_render(target: str):
    W, H = TARGETS[target]
    script = json.loads((BUILD / "script.json").read_text(encoding="utf-8"))
    timings = {t["id"]: t for t in json.loads((BUILD / "timings.json").read_text(encoding="utf-8"))}
    by_id = {s["id"]: s for s in script["scenes"]}
    short = target != "long"
    hook = ""
    if short:
        spec = script["shorts"][int(target[-1]) - 1]
        scenes, hook = [by_id[i] for i in spec["scene_ids"]], spec["hook"]
    else:
        scenes = script["scenes"]

    TMP.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    clips, words, events, silences, cursor = [], [], [], [], 0.0
    for idx, sc in enumerate(scenes):
        t = timings[sc["id"]]
        if short and clips and cursor + t["dur"] > 58:
            break
        n = max(1, round(t["dur"] * FPS))
        end = cursor + n / FPS
        img = TMP / f"{target}_{idx:03d}.jpg"
        prep_image(IMG / f"{sc['id']:03d}.jpg", W, H, img)
        clip = TMP / f"{target}_{idx:03d}.mp4"
        render_scene(img, t["audio"], n, idx, W, H, clip)
        img.unlink(missing_ok=True)
        for s, e, w in t["words"]:
            words.append([cursor + s, cursor + e, w])
        for kind in sc.get("sfx", []):
            if kind == "silence":
                silences.append((end - t.get("pad", PAD), end))
            else:
                events.append({"kind": kind, "start": cursor, "end": end})
        clips.append(clip.resolve())
        cursor = end

    lst = TMP / f"{target}_list.txt"
    lst.write_text("".join(f"file '{c}'\n" for c in clips), encoding="utf-8")
    concat = TMP / f"{target}_concat.mp4"
    sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", lst, "-c", "copy", concat])

    ass = TMP / f"{target}.ass"
    ass.write_text(make_ass(words, W, H, short, hook), encoding="utf-8")
    finalize(concat, ass, OUT / f"{target}.mp4", events, silences)
    print(f"Rendered {target}: {cursor:.1f}s | sfx={[e['kind'] for e in events]} silences={len(silences)}")

    try:
        import thumbnail
        thumbnail.make_for_target(target)
    except Exception as e:
        print(f"WARNING: thumbnail failed: {e}")


def cmd_prepare():
    for d in (BUILD, AUDIO, IMG):
        d.mkdir(parents=True, exist_ok=True)

    db = dedupe.load()
    override = (os.getenv("TOPIC") or "").strip()
    if override:
        topic, angle, text = override, "", override
        dup, score, near, vec = dedupe.check(text, db)
        if dup:
            print(f"WARNING: topic is close to {near!r} (score {score:.2f})")
    else:
        topic, angle, text, vec = choose_topic(db)

    script = generate_script(topic, angle)
    (BUILD / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
    (BUILD / "verify.txt").write_text("\n".join(script.get("verify", [])), encoding="utf-8")

    timings = asyncio.run(tts_all(script["scenes"]))
    (BUILD / "timings.json").write_text(json.dumps(timings, ensure_ascii=False), encoding="utf-8")

    used, report = set(), {}
    extra = topic_queries(script, topic)
    for sc in script["scenes"]:
        report[sc["id"]] = fetch_image(sc, used, extra)
        time.sleep(0.3)
    (BUILD / "img_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    real = sum(v.startswith("pixabay") for v in report.values())
    print(f"Images: {real}/{len(report)} real photos, {len(report) - real} gradient fallbacks", flush=True)
    for sid, src in report.items():
        print(f"  scene {sid:>2}: {src}")
    if real < 0.3 * len(report) and os.getenv("ALLOW_GRADIENT") != "1":
        raise SystemExit(
            "Too many gradient fallbacks. Check that the PIXABAY_API_KEY secret exists and is passed to the "
            f"'prepare' step. Last Pixabay error: {_pix['disabled'] or 'none (no search results)'}. "
            "Set ALLOW_GRADIENT=1 to ignore."
        )

    dedupe.save_entry_file(dedupe.make_entry(topic, text, vec, script["title"]))
    total = sum(t["dur"] for t in timings)
    print(f"Prepared {script['title']!r}: {len(script['scenes'])} scenes, {total / 60:.1f} min, voice={VOICE}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("voicetest")
    r = sub.add_parser("render")
    r.add_argument("--target", required=True, choices=list(TARGETS))
    a = ap.parse_args()
    if a.cmd == "prepare":
        cmd_prepare()
    elif a.cmd == "voicetest":
        cmd_voicetest()
    else:
        cmd_render(a.target)


if __name__ == "__main__":
    sys.exit(main())
