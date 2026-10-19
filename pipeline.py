#!/usr/bin/env python3
"""
Daily documentary pipeline.

  python pipeline.py prepare                  # topic -> script -> voice -> visuals   (build/)
  python pipeline.py render --target long     # 1920x1080 long video + thumbnail     (out/long.mp4)
  python pipeline.py render --target short1   # 1080x1920 Shorts: short1..short3     (out/shortN.mp4)
  python pipeline.py voicetest                # sample Hinglish clips for every voice option (out/)

LLM: any OpenAI-compatible endpoint (default: Groq free tier, openai/gpt-oss-*). Script generation is
CHUNKED (1 plan call + WRITE_PARTS write calls + small transliteration batches) and paced by a
tokens-per-minute limiter (LLM_TPM), so the free-tier limits are never exceeded.
Visuals (zero API key): Wikimedia Commons images + B-roll -> Openverse -> NASA -> procedural animated background.

Design: audio-first timeline. Every scene's length = its TTS audio length, so visuals and
narration can never drift. Scenes are rendered one by one with FFmpeg (low RAM), concatenated
with stream copy, then captions + music + SFX are mixed/burned in ONE final pass.

SFX tags (per scene, from the script's "sfx" list), files looked up in assets/sfx/:
  riser   -> ENDS exactly at the scene's cut (builds into the next scene)
  braam   -> HITS at the scene's first frame (the reveal)
  glitch  -> short digital burst at the scene's first frame
  silence -> the scene gets extra tail pause and the background music is muted for that tail
Missing files are synthesised with FFmpeg as a fallback, so the pipeline never breaks.
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
AUDIO, IMG, VID = BUILD / "audio", BUILD / "img", BUILD / "vid"
MUSIC = Path("assets/music.mp3")  # optional background music (you must own the license)
SFX_DIR = Path("assets/sfx")      # riser*.wav, braam*.wav, glitch*.wav (variants rotate: riser_1, riser_2 ...)


def env(name: str, default: str) -> str:
    return os.getenv(name) or default  # treats empty string (unset GH var) as default


# ---- LLM (OpenAI-compatible; Groq by default) ----
LLM_BASE = env("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_KEY = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY") or ""
MODEL = env("LLM_MODEL", "openai/gpt-oss-120b")        # script writing
FAST_MODEL = env("LLM_FAST_MODEL", "openai/gpt-oss-20b")  # topics, judge, transliteration
FALLBACKS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]  # tried in order on 404 / decommissioned
LLM_TPM = int(env("LLM_TPM", "7500"))          # your tokens-per-minute budget (free tier is small; check console)
WRITE_PARTS = int(env("WRITE_PARTS", "2"))     # 2 => plan + 2 write calls = the "3-call" script
TRANSLIT_BATCH = int(env("TRANSLIT_BATCH", "10"))
NICHE = env("NICHE", "mysteries of technology, science and the human mind")
LANGUAGE = env("LANGUAGE", "English")

# ---- Hinglish handling ------------------------------------------------------
# Hindi neural voices stumble on Roman-script Hinglish. Fix: the LLM writes every scene twice:
#   narration -> Roman Hinglish (used for captions)       tts_text -> Devanagari, word-for-word (used for the voice)
# Caption timing is mapped 1:1 from the Devanagari words. HINGLISH_MODE=roman skips this and
# reads Roman text with an Indian-English voice instead.
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
PAD = 0.25            # seconds of breathing room after each scene
SILENCE_EXTRA = 0.6   # extra pause appended to scenes tagged "silence"
BORDERLINE = 0.65     # similarity band where an LLM judge double-checks topics

MUSIC_GAIN = float(env("MUSIC_GAIN", "0.12"))
SFX_GAIN = {
    "riser": float(env("SFX_GAIN_RISER", "1.2")),
    "braam": float(env("SFX_GAIN_BRAAM", "0.7")),
    "glitch": float(env("SFX_GAIN_GLITCH", "0.5")),
}
SFX_CAPS = {"riser": 6, "braam": 4, "glitch": 5, "silence": 4}  # per video, keeps effects meaningful
SFX_EXT = (".wav", ".mp3", ".ogg", ".flac", ".m4a")
VISUAL_PROVIDERS = env("VISUAL_PROVIDERS", "wikimedia,openverse,nasa")  # all keyless: no API key needed
BROLL_EVERY = int(env("BROLL_EVERY", "3"))      # every Nth scene tries a B-roll video first (0 = never)
MOTION_BLUR = env("MOTION_BLUR", "1") == "1"    # subtle 3-frame blend on Ken Burns shots
WIKI_CONTACT = env("WIKI_CONTACT", "https://github.com")  # Wikimedia asks for contact info in User-Agent

TARGETS = {
    "long": (1920, 1080),
    "short1": (1080, 1920),
    "short2": (1080, 1920),
    "short3": (1080, 1920),
}

# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
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

VISUAL QUERIES ("visual_query" = a stock-photo search, 2-3 words):
- Concrete and filmable only. Show what a camera can see: a place, an object, a person doing something, light, texture.
- Never abstract concepts. BAD: "quantum entanglement", "future of AI", "human mind".
  GOOD: "neon fiber optic", "observatory telescope night", "macro circuit board".
- Subject + setting or light works best. Do not repeat the same subject in consecutive beats.
- No names, brands, logos, text, or quotes inside a query.

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
     "visual_query": "2-3 concrete visual words", "sfx": ["riser"]}
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


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
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


_window = []  # (timestamp, tokens) used in the last 60 s


def _budget_wait(est: int):
    """Block until `est` more tokens fit into the LLM_TPM budget of the rolling 60 s window."""
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
    """Chat completion on any OpenAI-compatible endpoint (name kept so nothing else changes).
    Features: TPM pacing, 429 retry-after, model fallback on 404/decommissioned."""
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
                "max_tokens": max_tokens,  # for gpt-oss this ALSO covers hidden reasoning tokens
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
                _window.append((time.time(), LLM_TPM))  # assume the minute is spent
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
        try:  # on retry give the model more room (truncated JSON / reasoning ate the budget)
            return parse_json(claude(model, system, user, int(max_tokens * (1 + 0.3 * i))))
        except SystemExit:
            raise
        except Exception as e:
            last = e
            print(f"  claude_json attempt {i + 1} failed: {e}", flush=True)
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"LLM JSON failed after {tries} tries: {last}")


# --------------------------------------------------------------------------- #
# Topic + script
# --------------------------------------------------------------------------- #
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
    """Accepts ["riser","braam"], "riser, braam" or "[SFX: riser]" and returns known tags."""
    text = " ".join(value) if isinstance(value, (list, tuple)) else str(value or "")
    return [n for n in ("riser", "braam", "glitch", "silence") if re.search(rf"\b{n}\b", text, re.I)]


def cap_sfx(scenes):
    counts = {}
    for s in scenes:
        if "silence" in s["sfx"] and "riser" in s["sfx"]:
            s["sfx"].remove("riser")  # silence scene = music drops; a riser would fight it
        keep = []
        for tag in s["sfx"]:
            if counts.get(tag, 0) < SFX_CAPS[tag]:
                keep.append(tag)
                counts[tag] = counts.get(tag, 0) + 1
        s["sfx"] = keep


def fill_tts_text(scenes):
    """Devanagari twin for every scene (batched to stay far below the TPM budget)."""
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
            s["tts_text"] = mapping.get(s["id"]) or s["narration"]  # last resort: Roman text


def outline_text(beats) -> str:
    return "\n".join(f"{b['id']}. [{b['section']}] {b['beat']}" for b in beats)


def split_ids(n: int, parts: int):
    size = -(-n // max(1, parts))
    return [list(range(i + 1, min(i + size, n) + 1)) for i in range(0, n, size)]


def write_part(plan: dict, beats: list, ids: list, tail: str) -> dict:
    """One narration call for the given beat ids. Returns {id: narration}."""
    got = {}
    for _ in range(2):  # second pass only asks for what is still missing
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
    """Chunked script: 1 PLAN call, WRITE_PARTS narration calls, then small transliteration batches."""
    plan = claude_json(
        MODEL, PLAN_GUIDE,
        f"Topic: {topic}\nAngle: {angle or 'your choice'}\nPlan the video now.", max_tokens=4500,
    )
    beats = plan.get("beats") or []
    if len(beats) < 10:
        raise RuntimeError(f"Plan too short: {len(beats)} beats")

    idmap = {}  # renumber beats 1..n and remap every reference
    for new, b in enumerate(beats, 1):
        try:
            idmap[int(b.get("id", new))] = new
        except (TypeError, ValueError):
            pass
        b["id"] = new
        b["section"] = b.get("section") or "present"
        b["beat"] = (b.get("beat") or "").strip()
        vq = " ".join(str(b.get("visual_query") or "").split()[:3])  # keep stock searches short
        b["visual_query"] = vq or "night city lights"
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
        if not sid:  # fallback: 4 consecutive scenes spread across the video
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


# --------------------------------------------------------------------------- #
# Voice (edge-tts) + word timings
# --------------------------------------------------------------------------- #
def spread_words(marks, text: str, dur: float):
    """Turn boundary events (word- or sentence-level) into per-word [start, end, word]."""
    if not marks:  # fallback: spread evenly by character length
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
    """Map timings of the spoken (Devanagari) words onto the caption (Roman) words.
    1:1 when the word counts match, proportional by index otherwise."""
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
    """Write mp3 to `path`, return boundary marks [(start, dur, text)]."""
    import edge_tts
    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=20), reraise=True)
    async def run():
        try:
            comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
        except TypeError:  # older edge-tts without `boundary`
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
    """Generate comparison clips so you can LISTEN and pick the best Hinglish voice."""
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
    print("Listen to A/B (Devanagari mode) vs C/D (roman mode). Set VOICE and HINGLISH_MODE accordingly.")


# --------------------------------------------------------------------------- #
# Visuals (ZERO API KEY): Wikimedia Commons (images + B-roll video) -> Openverse (commercial-licence images)
#                         -> NASA Image and Video Library (public domain) -> procedural animated background
# --------------------------------------------------------------------------- #
UA = f"yt-documentary-pipeline/3.0 (zero-key visual fetcher; contact: {WIKI_CONTACT})"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
OPENVERSE_API = "https://api.openverse.org/v1/images/"
NASA_API = "https://images-api.nasa.gov/search"
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

# keyless public APIs: polite per-provider limits (calls, window in seconds)
LIMITS = {"wikimedia": (50, 60), "openverse": (15, 60), "nasa": (30, 60)}
_prov = {n: {"calls": [], "cache": {}, "disabled": None, "fails": 0} for n in LIMITS}


def topic_queries(script: dict, topic: str = "") -> list:
    """Search terms taken from the video's own title/topic (used when a scene's query finds nothing)."""
    words = [w for w in re.findall(r"[a-z]{4,}", f"{script.get('title', '')} {topic}".lower()) if w not in STOPWORDS]
    uniq = list(dict.fromkeys(words))
    out = []
    if len(uniq) >= 2:
        out.append(" ".join(uniq[:2]))
    if len(uniq) >= 4:
        out.append(" ".join(uniq[2:4]))
    return out + uniq[:2]


def license_ok(text: str) -> bool:
    """Commercial-use check on a licence string. Any NonCommercial (nc) or NoDerivatives (nd) licence is rejected.
    Accepts public domain / CC0 / CC BY and CC BY-SA variants."""
    t = re.sub(r"[-_/]", " ", (text or "").lower())
    if not t.strip():
        return False
    if re.search(r"\b(nc|nd)\b", t):
        return False
    return bool(re.search(r"public domain|\bpdm\b|\bcc0\b|\bcc\b|\bby\b|\bpd\b", t))


def _allow_call(name: str) -> bool:
    maxn, window = LIMITS[name]
    st, now = _prov[name], time.time()
    st["calls"] = [t for t in st["calls"] if now - t < window]
    if len(st["calls"]) >= maxn:
        time.sleep(max(1.0, window - (now - st["calls"][0]) + 0.5))
    st["calls"].append(time.time())
    return True


def _get_json(name: str, url: str, params=None):
    """Keyless GET with throttling. Repeated failures disable the provider for the rest of the run."""
    import requests

    st = _prov[name]
    for attempt in range(3):
        if st["disabled"] or not _allow_call(name):
            return None
        try:
            r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=30)
        except Exception as e:
            st["fails"] += 1
            print(f"  {name} request failed: {e}", flush=True)
            if st["fails"] >= 3:
                st["disabled"] = "network errors x3"
            return None
        if r.status_code == 429:
            time.sleep(20 * (attempt + 1))
            continue
        if r.status_code >= 400:
            st["fails"] += 1
            print(f"  {name} HTTP {r.status_code} (failure {st['fails']})", flush=True)
            if st["fails"] >= 3:
                st["disabled"] = f"HTTP {r.status_code} x3"
            return None
        st["fails"] = 0
        try:
            return r.json()
        except ValueError:
            return None
    return None


def _pages(data) -> list:
    return list(((data or {}).get("query") or {}).get("pages", {}).values())


# ---- search functions: each returns candidates {kind, id, urls | manifest, credit} ----
def search_wikimedia_image(_key: str, query: str) -> list:
    """Wikimedia Commons image search (no key). Keeps JPEG/PNG >= 1280 px with a commercial-use licence."""
    data = _get_json("wikimedia", COMMONS_API, {
        "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
        "gsrsearch": f"{query} filetype:bitmap", "gsrlimit": 20,
        "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 1920,
    })
    out = []
    for pg in _pages(data):
        info = (pg.get("imageinfo") or [{}])[0]
        if info.get("mime") not in ("image/jpeg", "image/png") or int(info.get("width") or 0) < 1280:
            continue
        meta = info.get("extmetadata") or {}
        lic = (meta.get("LicenseShortName") or {}).get("value", "")
        if not license_ok(lic):
            continue
        artist = re.sub(r"<[^>]+>", "", (meta.get("Artist") or {}).get("value", "") or "").strip()
        title = pg.get("title", "")
        out.append({
            "kind": "image", "id": f"wm:{title}", "urls": [info.get("thumburl"), info.get("url")],
            "credit": f"{title.replace('File:', '')} by {artist or 'unknown'} ({lic}), Wikimedia Commons",
        })
    return out


def search_wikimedia_video(_key: str, query: str) -> list:
    """Wikimedia Commons B-roll video search (no key). WebM/Ogg, >= 1280 px wide, >= 6 s, <= 80 MB."""
    data = _get_json("wikimedia", COMMONS_API, {
        "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
        "gsrsearch": f"{query} filetype:video", "gsrlimit": 15,
        "prop": "videoinfo", "viprop": "url|size|mime|duration|dimensions|extmetadata",
    })
    out = []
    for pg in _pages(data):
        info = (pg.get("videoinfo") or [{}])[0]
        if info.get("mime") not in ("video/webm", "video/ogg"):
            continue
        if int(info.get("width") or 0) < 1280 or float(info.get("duration") or 0) < 6:
            continue
        if int(info.get("size") or 0) > 80 * 1024 * 1024:
            continue
        meta = info.get("extmetadata") or {}
        lic = (meta.get("LicenseShortName") or {}).get("value", "")
        if not license_ok(lic):
            continue
        artist = re.sub(r"<[^>]+>", "", (meta.get("Artist") or {}).get("value", "") or "").strip()
        title = pg.get("title", "")
        out.append({
            "kind": "video", "id": f"wmv:{title}", "urls": [info.get("url")],
            "credit": f"{title.replace('File:', '')} by {artist or 'unknown'} ({lic}), Wikimedia Commons",
        })
    return out


def search_openverse(_key: str, query: str) -> list:
    """Openverse (aggregator of CC-licensed images). license_type=commercial plus our own licence check."""
    data = _get_json("openverse", OPENVERSE_API, {
        "q": query[:100], "license_type": "commercial", "aspect_ratio": "wide", "page_size": 20,
    })
    out = []
    for r in (data or {}).get("results", []):
        url = r.get("url") or ""
        if not url.lower().split("?")[0].endswith((".jpg", ".jpeg", ".png")):
            continue
        if int(r.get("width") or 0) < 1200:
            continue
        lic = r.get("license") or ""
        if not license_ok(lic):
            continue
        out.append({
            "kind": "image", "id": f"ov:{r.get('id')}", "urls": [url],
            "credit": f"{r.get('title') or 'Untitled'} by {r.get('creator') or 'unknown'} "
                      f"({lic.upper()}, Openverse; source: {r.get('foreign_landing_url') or 'n/a'})",
        })
    return out


def search_nasa(_key: str, query: str) -> list:
    """NASA Image and Video Library (no key). Mostly public domain; NASA usage guidelines still apply."""
    data = _get_json("nasa", NASA_API, {"q": query[:100], "media_type": "image", "page_size": 20})
    out = []
    for item in ((data or {}).get("collection") or {}).get("items", []):
        d = (item.get("data") or [{}])[0]
        if not item.get("href"):
            continue
        out.append({
            "kind": "image", "id": f"nasa:{d.get('nasa_id')}", "urls": [], "manifest": item["href"],
            "credit": f"{d.get('title') or 'NASA image'} (NASA Image and Video Library; NASA usage guidelines apply)",
        })
    return out


def _candidate_urls(cand: dict) -> list:
    """Direct URLs for a candidate. NASA items carry a manifest, expanded lazily to the largest sizes."""
    if cand.get("urls"):
        return cand["urls"]
    if cand.get("manifest"):
        data = _get_json("nasa", cand["manifest"])
        if isinstance(data, list):
            files = [u for u in data if isinstance(u, str)]
            ranked = []
            for tag in ("~orig", "~large", "~medium"):
                ranked += [u for u in files if tag in u and u.lower().endswith((".jpg", ".jpeg", ".png"))]
            return ranked[:3]
    return []


SEARCH = {
    "wikimedia": {"image": search_wikimedia_image, "video": search_wikimedia_video},
    "openverse": {"image": search_openverse},
    "nasa": {"image": search_nasa},
}


def active_providers() -> list:
    out = []
    for n in [x.strip() for x in VISUAL_PROVIDERS.split(",") if x.strip() in SEARCH]:
        if not _prov[n]["disabled"]:
            out.append(n)
    return out


def search_cached(name: str, kind: str, query: str) -> list:
    st = _prov[name]
    if st["disabled"]:
        return []
    ck = (kind, query)
    if ck not in st["cache"]:
        fn = SEARCH[name].get(kind)
        try:
            st["cache"][ck] = fn(None, query) if fn else []
        except Exception as e:
            print(f"  {name} search failed for {query!r}: {e}", flush=True)
            st["cache"][ck] = []
    return st["cache"][ck]


def provider_reasons() -> str:
    bits = [f"{n}: {st['disabled']}" for n, st in _prov.items() if st["disabled"]]
    return "; ".join(bits) or "no provider returned a usable result"


# ---- downloads + probing ----
def _stream_to(url: str, dst: Path, max_mb: int):
    import requests

    with requests.get(url, headers={"User-Agent": UA}, timeout=120, stream=True) as r:
        r.raise_for_status()
        total = 0
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                total += len(chunk)
                if total > max_mb * 1024 * 1024:
                    raise ValueError(f"file larger than {max_mb} MB")
                f.write(chunk)


def probe_video(path: Path):
    """(width, height, duration) via ffprobe, or None if unreadable."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration",
         "-of", "json", str(path)], capture_output=True, text=True)
    if out.returncode:
        return None
    d = json.loads(out.stdout or "{}")
    st = (d.get("streams") or [{}])[0]
    return int(st.get("width") or 0), int(st.get("height") or 0), float((d.get("format") or {}).get("duration") or 0)


def _materialize_image(cand: dict, i: int) -> bool:
    from PIL import Image

    tmp = IMG / f"_tmp_{i:03d}"
    for url in _candidate_urls(cand):
        if not url:
            continue
        try:
            _stream_to(url, tmp, 40)
            im = Image.open(tmp).convert("RGB")
            if im.width < 1200:
                continue
            im.save(IMG / f"{i:03d}.jpg", quality=92)
            tmp.unlink(missing_ok=True)
            return True
        except Exception as e:
            print(f"  image download failed ({url[:60]}...): {e}", flush=True)
    tmp.unlink(missing_ok=True)
    return False


def _materialize_video(cand: dict, i: int) -> bool:
    dst = VID / f"{i:03d}.mp4"
    for url in cand.get("urls") or []:
        if not url:
            continue
        try:
            _stream_to(url, dst, 80)
            info = probe_video(dst)
            if info and info[0] >= 1280 and info[2] >= 3:
                return True
        except Exception as e:
            print(f"  video download failed ({url[:60]}...): {e}", flush=True)
    dst.unlink(missing_ok=True)
    return False


def fetch_visual(scene: dict, used: set, extra_queries=None) -> dict:
    """Resolve one scene to a visual asset, with zero API keys. For every query of the ladder the providers
    are asked in order (Wikimedia -> Openverse -> NASA). Every BROLL_EVERY-th scene tries a video clip first.
    The procedural animated background is the final tier, so the pipeline always has something to render.
    Returns {"type": image|video|procedural, "path": relative path under build/ or None, "source", "credit"}."""
    i = scene["id"]
    want_video = BROLL_EVERY > 0 and i % BROLL_EVERY == 0
    kinds = ["video", "image"] if want_video else ["image", "video"]
    q = (scene.get("visual_query") or "").strip()
    ladder = [q, " ".join(q.split()[:2])]
    ladder += list(extra_queries or [])
    ladder += SECTION_QUERIES.get(scene.get("section", ""), [])
    ladder += [GENERIC_QUERIES[i % len(GENERIC_QUERIES)], GENERIC_QUERIES[(i + 3) % len(GENERIC_QUERIES)],
               "abstract background"]

    seen = set()
    for query in ladder:
        if not query or query in seen:
            continue
        seen.add(query)
        for name in active_providers():
            for kind in kinds:
                pool = [c for c in search_cached(name, kind, query) if c["id"] not in used]
                for k in range(min(3, len(pool))):  # up to 3 different candidates if a download fails
                    cand = pool[(i + k) % len(pool)]
                    ok = _materialize_video(cand, i) if kind == "video" else _materialize_image(cand, i)
                    if not ok:
                        continue
                    used.add(cand["id"])
                    rel = f"vid/{i:03d}.mp4" if kind == "video" else f"img/{i:03d}.jpg"
                    return {"type": kind, "path": rel, "source": f"{name}:{query}", "credit": cand.get("credit")}
    return {"type": "procedural", "path": None, "source": f"procedural ({provider_reasons()})", "credit": None}


def check_visual_providers():
    """Informational preflight. Never blocks: the procedural tier guarantees the run continues."""
    for n in [x.strip() for x in VISUAL_PROVIDERS.split(",") if x.strip() in SEARCH]:
        hits = search_cached(n, "image", "mountain")
        if _prov[n]["disabled"]:
            print(f"  [visual check] {n}: FAILED ({_prov[n]['disabled']})", flush=True)
        else:
            print(f"  [visual check] {n}: OK ({len(hits)} results)", flush=True)
    print("  [visual check] procedural animated background: always available as the last tier", flush=True)


# ---- renderers for the three asset types (each writes a clip of exactly n_frames/FPS seconds with audio) ----
def render_video_scene(src: Path, audio: str, n_frames: int, W: int, H: int, out: Path):
    """B-roll clip: fill the frame (cover-crop), loop if shorter than the narration, cut to the scene length."""
    dur = n_frames / FPS
    fc = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H},"
          f"fps={FPS},format=yuv420p[v];[1:a]apad,aresample=48000,aformat=channel_layouts=stereo[a]")
    sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stream_loop", "-1", "-i", src, "-i", audio,
        "-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k", out])


PROC_STYLES = ["waves", "mesh", "grid"]  # animated FFmpeg generators, one style per seed (rotates per scene)


def render_procedural_scene(audio: str, n_frames: int, seed: int, W: int, H: int, out: Path):
    """Final-tier animated background, generated by FFmpeg (no files, no keys):
      waves -> moving colour interference waves      mesh -> pulsing node lattice      grid -> sweeping grid lines
    Rendered at 480x270 and upscaled, so it is cheap on a CPU runner."""
    rnd = random.Random(seed * 7919 + 13)
    style = PROC_STYLES[seed % len(PROC_STYLES)]
    sp = rnd.uniform(0.6, 1.4)
    ha, hb = rnd.uniform(0, 6.28), rnd.uniform(0, 6.28)
    dur = n_frames / FPS
    if style == "waves":
        fx, fy = rnd.uniform(0.02, 0.05), rnd.uniform(0.02, 0.045)
        lum = (f"36+26*sin(X*{fx:.4f}+T*{sp:.3f})*cos(Y*{fy:.4f}-T*{sp * 0.7:.3f})"
               f"+18*sin((X+Y)*0.012+T*0.4)")
        cb = f"128+38*sin(X*{fy:.4f}-T*{sp * 0.5:.3f}+{ha:.3f})"
        cr = f"128+34*cos(Y*{fx:.4f}+T*{sp * 0.6:.3f}+{hb:.3f})"
    elif style == "mesh":
        ax, ay = rnd.uniform(0.07, 0.12), rnd.uniform(0.07, 0.12)
        lum = (f"30+70*pow(0.5+0.5*cos(X*{ax:.4f}+T*{sp:.3f}),10)"
               f"*pow(0.5+0.5*cos(Y*{ay:.4f}-T*{sp * 0.8:.3f}),10)")
        cb = f"128+25*sin(X*0.004+T*0.2+{ha:.3f})"
        cr = f"128+25*cos(Y*0.004-T*0.15+{hb:.3f})"
    else:  # grid
        ax = rnd.uniform(0.10, 0.16)
        lum = (f"28+60*pow(0.5+0.5*cos(X*{ax:.4f}+T*{sp * 2:.3f}),40)"
               f"+60*pow(0.5+0.5*cos(Y*{ax:.4f}-T*{sp * 1.5:.3f}),40)")
        cb = f"120+20*sin(T*0.4+{ha:.3f})"
        cr = f"140+20*cos(T*0.3+{hb:.3f})"
    src = (f"nullsrc=s=480x270:r={FPS}:d={dur:.3f},geq=lum='{lum}':cb='{cb}':cr='{cr}',"
           f"noise=alls=8:allf=t,scale={W}:{H}:flags=lanczos,format=yuv420p")
    sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", src, "-i", audio,
        "-map", "0:v", "-map", "1:a", "-af", "apad,aresample=48000,aformat=channel_layouts=stereo",
        "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", out])


# --------------------------------------------------------------------------- #
# Captions (ASS karaoke, burned in one pass)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# SFX assets (+ synthetic fallback)
# --------------------------------------------------------------------------- #
def synth_sfx(name: str, dst: Path):
    """Rough FFmpeg-synthesised stand-ins. Replace with real files in assets/sfx/ for better quality."""
    TMP.mkdir(exist_ok=True)
    if name == "riser":
        src = "anoisesrc=color=pink:amplitude=1.0:duration=4:sample_rate=48000"
        af = "highpass=f=250,volume='pow(t/4,2.2)':eval=frame,afade=t=out:st=3.92:d=0.08"
    elif name == "braam":
        expr = ("(0.5*sin(2*PI*55*t)+0.35*sin(2*PI*55.8*t)+0.25*sin(2*PI*110*t)"
                "+0.2*sin(2*PI*165*t)+0.15*sin(2*PI*220.5*t))*exp(-1.1*t)*min(1,t/0.02)")
        src, af = f"aevalsrc='{expr}':d=3:s=48000", "lowpass=f=1400,volume=0.9"
    else:  # glitch
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


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def prep_image(src: Path, W: int, H: int, dst: Path):
    """Cover-crop to the target aspect and upscale 2x so zoompan stays smooth (no jitter)."""
    from PIL import Image

    im = Image.open(src).convert("RGB")
    tw, th = W * 2, H * 2
    scale = max(tw / im.width, th / im.height)
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    left, top = (im.width - tw) // 2, (im.height - th) // 2
    im.crop((left, top, left + tw, top + th)).save(dst, quality=92)


def ken_burns(idx: int, n_frames: int):
    """Smooth ease-in/out camera move, a different one per scene so the video never feels static.
    Returns zoompan (z, x, y) expressions. 0 zoom-in, 1 zoom-out, 2 pan L->R, 3 pan R->L,
    4 zoom-in drifting up-right, 5 zoom-in drifting down-left."""
    N = max(n_frames - 1, 1)
    e = f"((on/{N})*(on/{N})*(3-2*on/{N}))"  # smoothstep easing 0 -> 1
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
    mb = ",tmix=frames=3:weights='1 2 1'" if MOTION_BLUR else ""
    vf = f"zoompan=z='{z}':x='{x}':y='{y}':d={n_frames}:s={W}x{H}:fps={FPS}{mb},format=yuv420p"
    sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", img, "-i", audio,
        "-vf", vf, "-af", "apad,aresample=48000", "-ac", "2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k",
        "-frames:v", n_frames, "-shortest", out])


def finalize(concat: Path, ass: Path, out: Path, events: list, silences: list):
    """One pass: burn captions, mix narration + music (muted in silence windows) + SFX at exact times."""
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
        if ev["kind"] == "riser":  # align the END of the riser with the scene cut
            d = min(length, ev["end"] - ev["start"])
            at = ev["end"] - d
            f += f",atrim=start={length - d:.3f},asetpts=PTS-STARTPTS"
        else:  # braam / glitch hit on the scene's first frame
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
    visuals = {}
    if (BUILD / "visuals.json").exists():
        visuals = json.loads((BUILD / "visuals.json").read_text(encoding="utf-8"))
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
    kinds = {"image": 0, "video": 0, "procedural": 0}
    for idx, sc in enumerate(scenes):
        t = timings[sc["id"]]
        if short and clips and cursor + t["dur"] > 58:  # keep Shorts under 60s
            break
        n = max(1, round(t["dur"] * FPS))
        end = cursor + n / FPS
        clip = TMP / f"{target}_{idx:03d}.mp4"

        asset = visuals.get(str(sc["id"])) or {"type": "procedural"}
        if asset["type"] != "procedural" and not (BUILD / asset["path"]).exists():
            print(f"  WARNING: scene {sc['id']} asset missing, using procedural background")
            asset = {"type": "procedural"}
        kinds[asset["type"]] += 1
        if asset["type"] == "video":
            render_video_scene(BUILD / asset["path"], t["audio"], n, W, H, clip)
        elif asset["type"] == "image":
            img = TMP / f"{target}_{idx:03d}.jpg"
            prep_image(BUILD / asset["path"], W, H, img)
            render_scene(img, t["audio"], n, idx, W, H, clip)
            img.unlink(missing_ok=True)
        else:
            render_procedural_scene(t["audio"], n, sc["id"], W, H, clip)

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
    print(f"Rendered {target}: {cursor:.1f}s | visuals={kinds} | sfx={[e['kind'] for e in events]} "
          f"silences={len(silences)}")

    try:  # thumbnail is a bonus: never fail the render because of it
        import thumbnail
        thumbnail.make_for_target(target)
    except Exception as e:
        print(f"WARNING: thumbnail failed: {e}")


# --------------------------------------------------------------------------- #
# Prepare
# --------------------------------------------------------------------------- #
def make_poster(asset: dict, sid: int, audio: str):
    """Still frame for every scene so thumbnail.py always has build/img/NNN.jpg (photos already have one)."""
    dst = IMG / f"{sid:03d}.jpg"
    if asset["type"] == "image" and dst.exists():
        return
    if asset["type"] == "video":
        sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", "1", "-i", BUILD / asset["path"],
            "-frames:v", "1", "-vf", "scale=1920:-2", "-q:v", "2", dst])
        return
    TMP.mkdir(exist_ok=True)
    tmp = TMP / f"poster_{sid:03d}.mp4"
    render_procedural_scene(audio, 2 * FPS, sid, 1920, 1080, tmp)
    sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", "0.5", "-i", tmp,
        "-frames:v", "1", "-q:v", "2", dst])


def cmd_prepare():
    from collections import Counter

    for d in (BUILD, AUDIO, IMG, VID):
        d.mkdir(parents=True, exist_ok=True)

    print("Checking visual providers...", flush=True)
    check_visual_providers()  # informational; the procedural tier means the run never stops here

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
    timings = asyncio.run(tts_all(script["scenes"]))
    (BUILD / "timings.json").write_text(json.dumps(timings, ensure_ascii=False), encoding="utf-8")

    used, visuals, credits = set(), {}, []
    extra = topic_queries(script, topic)
    for sc in script["scenes"]:
        asset = fetch_visual(sc, used, extra)
        visuals[str(sc["id"])] = asset
        if asset.get("credit"):
            credits.append(f"Scene {sc['id']}: {asset['credit']}")
        print(f"  scene {sc['id']:>2}: {asset['type']:<10} {asset['source']}", flush=True)
        time.sleep(0.2)

    audio_by_id = {t["id"]: t["audio"] for t in timings}
    for sc in script["scenes"]:  # poster frames: needed by thumbnail.py for every scene
        try:
            make_poster(visuals[str(sc["id"])], sc["id"], audio_by_id[sc["id"]])
        except Exception as e:
            print(f"  WARNING: poster for scene {sc['id']} failed: {e}", flush=True)

    counts = Counter(a["type"] for a in visuals.values())
    (BUILD / "visuals.json").write_text(json.dumps(visuals, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Visuals: {dict(counts)}", flush=True)
    if counts.get("procedural", 0) > 0.3 * len(visuals):
        print("NOTE: many scenes use the procedural animated background. A more specific visual_query "
              "usually finds more photos; see the [visual check] lines above.", flush=True)

    if credits:  # attribution goes into the description (required for CC-BY-style Wikimedia images)
        script["description"] = (script.get("description", "") + "\n\nVisual credits:\n" + "\n".join(credits)).strip()
    (BUILD / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
    (BUILD / "verify.txt").write_text("\n".join(script.get("verify", [])), encoding="utf-8")

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
