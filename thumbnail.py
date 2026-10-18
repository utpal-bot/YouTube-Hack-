#!/usr/bin/env python3
"""
Thumbnail generator (Pillow only).

  python thumbnail.py                 # all targets from build/script.json
  python thumbnail.py --target long   # out/thumb_long.jpg   (1280x720)
  python thumbnail.py --target short2 # out/thumb_short2.jpg (1080x1920)

Look: graded background photo + vignette + dark gradient behind the text,
huge uppercase bold text with thick outline and shadow, last line in yellow,
accent underline. Text comes from the script ("thumbnail_text" / Short "thumb_text").

Tip: drop a bold display font (e.g. Anton or Bebas Neue, .ttf) into assets/fonts/ for a stronger look.
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

BUILD, OUT = Path("build"), Path("out")
SIZES = {"long": (1280, 720), "short1": (1080, 1920), "short2": (1080, 1920), "short3": (1080, 1920)}
YELLOW, WHITE = (255, 214, 0), (255, 255, 255)

FONT_CANDIDATES = [
    *sorted(Path("assets/fonts").glob("*.ttf")),
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def load_font(size: int):
    for p in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(str(p), size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def cover(img: Image.Image, W: int, H: int) -> Image.Image:
    scale = max(W / img.width, H / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    left, top = (img.width - W) // 2, (img.height - H) // 2
    return img.crop((left, top, left + W, top + H))


def grade(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Contrast(img).enhance(1.25)
    img = ImageEnhance.Color(img).enhance(1.25)
    return ImageEnhance.Brightness(img).enhance(0.9)


def darken(img: Image.Image, mask: Image.Image) -> Image.Image:
    return Image.composite(Image.new("RGB", img.size, (0, 0, 0)), img, mask)


def text_gradient_mask(W: int, H: int, portrait: bool) -> Image.Image:
    g = Image.linear_gradient("L")  # black (top) -> white (bottom)
    if portrait:
        g = ImageOps.invert(g)  # strongest at the top, where the text sits
        g = g.resize((W, H))
    else:
        g = ImageOps.invert(g.rotate(90, expand=True))  # strongest at the left
        g = g.resize((W, H))
    return g.point(lambda v: int(255 * ((v / 255) ** 1.6) * 0.88))


def vignette_mask(W: int, H: int) -> Image.Image:
    return Image.radial_gradient("L").resize((W, H)).point(lambda v: int(v * 0.6))


def split_lines(words, n):
    n = min(n, len(words))
    size = -(-len(words) // n)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def fit_text(draw, words, max_w, max_h, max_size):
    """Pick the line layout (1-3 lines) that allows the biggest font."""
    best = None
    for n in (1, 2, 3):
        lines = split_lines(words, n)
        for size in range(max_size, 39, -4):
            font = load_font(size)
            line_h = int(size * 1.1)
            widest = max(draw.textlength(l, font=font) for l in lines)
            if widest <= max_w and line_h * len(lines) <= max_h:
                cand = (size, len(lines), lines, font, line_h, widest)
                if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
                    best = cand
                break
    if best is None:  # extremely long text: shrink hard
        lines = split_lines(words, 3)
        font = load_font(40)
        best = (40, 3, lines, font, 44, max(draw.textlength(l, font=font) for l in lines))
    return best


def make_thumbnail(src: Path, text: str, size, out: Path):
    W, H = size
    portrait = H > W
    img = grade(cover(Image.open(src).convert("RGB"), W, H))
    img = darken(img, vignette_mask(W, H))
    img = darken(img, text_gradient_mask(W, H, portrait))

    words = text.upper().split()[:6] or ["WATCH", "THIS"]
    probe = ImageDraw.Draw(img)
    if portrait:
        max_w, max_h, max_size = int(W * 0.88), int(H * 0.30), 200
    else:
        max_w, max_h, max_size = int(W * 0.58), int(H * 0.72), 190
    fsize, n_lines, lines, font, line_h, widest = fit_text(probe, words, max_w, max_h, max_size)

    total_h = line_h * n_lines
    y0 = int(H * 0.12) if portrait else (H - total_h) // 2
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d, ds = ImageDraw.Draw(layer), ImageDraw.Draw(shadow)
    stroke = max(4, fsize // 12)
    off = max(4, fsize // 14)
    for i, line in enumerate(lines):
        lw = d.textlength(line, font=font)
        x = (W - lw) / 2 if portrait else int(W * 0.05)
        y = y0 + i * line_h
        color = YELLOW if (i == n_lines - 1) else WHITE
        ds.text((x + off, y + off), line, font=font, fill=(0, 0, 0, 255), stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        d.text((x, y), line, font=font, fill=color, stroke_width=stroke, stroke_fill=(0, 0, 0))
    # accent underline
    bar_w = int(widest * 0.45)
    bar_x = (W - bar_w) // 2 if portrait else int(W * 0.05)
    bar_y = y0 + total_h + max(10, fsize // 8)
    d.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + max(8, fsize // 12)], radius=6, fill=YELLOW)

    shadow = shadow.filter(ImageFilter.GaussianBlur(max(4, fsize // 18)))
    composed = Image.alpha_composite(Image.alpha_composite(img.convert("RGBA"), shadow), layer).convert("RGB")

    out.parent.mkdir(parents=True, exist_ok=True)
    q = 92
    while True:  # YouTube limit: 2 MB
        composed.save(out, "JPEG", quality=q, optimize=True)
        if out.stat().st_size <= 1_900_000 or q <= 60:
            break
        q -= 8
    return out


def short_text(s: str) -> str:
    return " ".join(s.split()[:4])


def resolve(target: str, script: dict):
    by_id = {s["id"]: s for s in script["scenes"]}
    if target == "long":
        try:
            sid = int(script.get("thumbnail_scene_id") or 1)
        except (TypeError, ValueError):
            sid = 1
        text = script.get("thumbnail_text") or short_text(script["title"])
    else:
        spec = script["shorts"][int(target[-1]) - 1]
        sid = spec["scene_ids"][0]
        text = spec.get("thumb_text") or short_text(spec["hook"])
    if sid not in by_id:
        sid = 1
    return BUILD / "img" / f"{sid:03d}.jpg", text


def make_for_target(target: str):
    script = json.loads((BUILD / "script.json").read_text(encoding="utf-8"))
    src, text = resolve(target, script)
    out = make_thumbnail(src, text, SIZES[target], OUT / f"thumb_{target}.jpg")
    print(f"Thumbnail {target}: {out} ({out.stat().st_size // 1024} KB) text={text!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="all", choices=["all", *SIZES])
    a = ap.parse_args()
    for t in (SIZES if a.target == "all" else [a.target]):
        make_for_target(t)


if __name__ == "__main__":
    sys.exit(main())
