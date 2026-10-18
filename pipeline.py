#!/usr/bin/env python3
"""
Daily documentary pipeline (Groq + Pixabay + Edge-TTS).

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


# Both set to the verified active Groq model
MODEL = env("GROQ_MODEL", "llama-3.3-70b-versatile")
FAST_MODEL = env("GROQ_FAST_MODEL", "llama-3.3-70b-versatile")
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

_TTS_FIELD = (
    '"tts_text": "SAME sentence in Devanagari script, word-for-word (same number of words, same order); '
    'English loanwords are transliterated into Devanagari",\n     '
    if USE_DEVA else ""
)
_TTS_RULE = (
    "- narration is Roman Hinglish (for captions). tts_text MUST be its Devanagari twin, 1:1 word order.\n"
    if USE_DEVA else ""
)

STYLE_GUIDE = """You are a documentary scriptwriter for a faceless YouTube channel (suspense, curiosity-driven, factual).
Write the narration in __LANG__. Conversational, spoken rhythm, short sentences (max ~15 words).

STRUCTURE: cold_open (hook + promise) -> past -> present -> future -> resolution.

RETENTION RULES:
- Open a loop in the cold open and CLOSE it in the resolution.
- End every section with a new open loop that is PAID OFF later. Never leave a cliffhanger unresolved.
- A new fact, twist or question roughly every 20-30 seconds.

FACTS: Use only well-established facts. Never invent statistics, quotes, studies or names.
Anything a human should double-check goes into "verify".

SFX: "sfx" is a list using ONLY these tags: riser, braam, silence, glitch (use [] for none).
- riser: on the LAST scene of a section (it builds into the next scene's cut). Max 6 per video.
- braam: on the FIRST scene after a riser = the big reveal. Max 4 per video.
- silence: on the scene right before the single biggest reveal (music drops out). Max 4.
- glitch: on unsettling / tension moments. Max 5.
__TTS_RULE__
OUTPUT: ONLY valid JSON (no markdown fences) with this schema:
{
  "title": "<= 70 chars, curiosity-driven, NOT misleading",
  "description": "2-3 sentences + 3 hashtags",
  "tags": ["8-12 tags"],
  "topic_summary": "one sentence",
  "thumbnail_text": "2-4 words, ALL CAPS, curiosity, Roman letters only",
  "thumbnail_scene_id": 1,
  "verify": ["claims a human should fact-check before publish"],
  "scenes": [
    {"id": 1, "section": "cold_open|past|present|future|resolution",
     "narration": "25-35 words",
     __TTS_FIELD__"visual_query": "2-4 word stock photo search",
     "sfx": ["riser"]}
  ],
  "shorts": [
    {"title": "...", "hook": "max 8 words, shown on screen in first 3 seconds",
     "thumb_text": "2-4 words, ALL CAPS", "scene_ids": [3,4,5,6]}
  ]
}
RULES: 38-44 scenes (~1200 words total, about 8 minutes). Exactly 3 shorts, each using 3-4
CONSECUTIVE scene ids (~30-50 seconds), starting on a strong line and making sense standalone:
one from the past section, one from present, one from future.
"""
STYLE_GUIDE = (
    STYLE_GUIDE.replace("__LANG__", LANG_DESC)
    .replace("__TTS_RULE__", _TTS_RULE)
    .replace("__TTS_FIELD__", _TTS_FIELD)
)


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


def claude(model: str, system: str, user: str, max_tokens: int) -> str:
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set!")

    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=model,
        temperature=0.7,
        max_tokens=min(max_tokens, 8000),
        response_format={"type": "json_object"},
    )
    return chat_completion.choices[0].message.content


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def claude_json(model, system, user, max_tokens, tries=3) -> dict:
    last = None
    for i in range(tries):
        try:
            return parse_json(claude(model, system, user, max_tokens))
        except Exception as e:
            last = e
            print(f"  groq_json attempt {i + 1} failed: {e}", flush=True)
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"Groq JSON failed after {tries} tries: {last}")


def judge_same(a: str, b: str) -> bool:
    out = claude_json(
        FAST_MODEL, "Answer ONLY JSON.",
        f'Are these two video topics about the same core concept?\nA: {a}\nB: {b}\n'
        'Return {"same": true|false}', 100,
    )
    return bool(out.get("same"))


def choose_topic(db: dict):
    recent = [e["topic"] for e in db["entries"][-30:]]
    for attempt in range(6):
        listing = "\n- ".join(recent) if recent else "(none yet)"
        out = claude_json(
            FAST_MODEL, TOPIC_SYSTEM,
            f"Niche: {NICHE}\nAlready covered (never repeat or rephrase):\n- {listing}\n"
            'Return {"topic": "...", "angle": "..."}', 600,
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
    if not missing:
        return
    print(f"  transliterating {len(missing)} scenes to Devanagari", flush=True)
    out = claude_json(
        FAST_MODEL, "Transliterate. Answer ONLY JSON.",
        "Rewrite each Roman Hinglish sentence in Devanagari script, word for word (same number of "
        "words, same order); transliterate English words into Devanagari too.\n"
        'Return {"items":[{"id":1,"tts_text":"..."}]}\n'
        + json.dumps([{"id": s["id"], "text": s["narration"]} for s in missing], ensure_ascii=False),
        8000,
    )
    mapping = {i["id"]: i.get("tts_text", "") for i in out.get("items", [])}
    for s in missing:
        s["tts_text"] = mapping.get(s["id"]) or s["narration"]


def generate_script(topic: str, angle: str) -> dict:
    user = f"Topic: {topic}\nAngle: {angle or 'your choice'}\nWrite the full script JSON now."
    script = claude_json(MODEL, STYLE_GUIDE, user, max_tokens=8000)
    scenes = script.get("scenes") or []
    if len(scenes) < 10:
        raise RuntimeError(f"Script too short: {len(scenes)} scenes")
    for i, s in enumerate(scenes, 1):
        s["id"] = i
        s["narration"] = s["narration"].strip()
        s["visual_query"] = s.get("visual_query") or "dark abstract technology"
        s["sfx"] = parse_sfx(s.get("sfx"))
    cap_sfx(scenes)
    if USE_DEVA:
        fill_tts_text(scenes)

    n = len(scenes)
    ids = {s["id"] for s in scenes}
    shorts = script.get("shorts") or []
    fixed = []
    for k in range(3):
        src = shorts[k] if k < len(shorts) else {}
        sid = [i for i in src.get("scene_ids", []) if i in ids]
        if not sid:
            start = min(k * (n // 3) + 1, max(1, n - 3))
            sid = list(range(start, min(start + 4, n + 1)))
        fixed.append({
            "title": src.get("title") or f"{script['title']} (Part {k + 1})",
            "hook": src.get("hook") or script["title"],
            "thumb_text": src.get("thumb_text") or "",
            "scene_ids": sid,
        })
    script["shorts"] = fixed
    return script


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


def fetch_image(scene: dict, used: set):
    """Fetches images using Pixabay API."""
    import requests
    from PIL import Image
    from io import BytesIO

    dst = IMG / f"{scene['id']:03d}.jpg"
    key = os.getenv("PIXABAY_API_KEY") or os.getenv("PEXELS_API_KEY")
    if key:
        try:
            r = requests.get(
                "https://pixabay.com/api/",
                params={
                    "key": key,
                    "q": scene["visual_query"],
                    "image_type": "photo",
                    "orientation": "horizontal",
                    "per_page": 10,
                    "safesearch": "true"
                },
                timeout=30,
            )
            r.raise_for_status()
            hits = [h for h in r.json().get("hits", []) if h["largeImageURL"] not in used]
            if hits:
                p = hits[scene["id"] % len(hits)]
                used.add(p["largeImageURL"])
                data = requests.get(p["largeImageURL"], timeout=60)
                data.raise_for_status()
                Image.open(BytesIO(data.content)).convert("RGB").save(dst, quality=92)
                return
        except Exception as e:
            print(f"  image fetch failed for {scene['visual_query']!r}: {e}")
    gradient_card(dst, scene["id"])


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


def render_scene(img: Path, audio: str, n_frames: int, idx: int, W: int, H: int, out: Path):
    z = f"1+0.18*on/{n_frames}" if idx % 2 == 0 else f"1.18-0.18*on/{n_frames}"
    vf = (
        f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={n_frames}:s={W}x{H}:fps={FPS},format=yuv420p"
    )
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

    used = set()
    for sc in script["scenes"]:
        fetch_image(sc, used)
        time.sleep(0.25)

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
