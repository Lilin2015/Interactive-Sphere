"""Drawing/rendering utilities (visualization only, not part of the
algorithm)."""

import os

import cv2
import numpy as np


# Native system fonts for panel text (PIL does no font fallback, so
# separate chains for Latin and CJK)
_FONT_LATIN = ["/System/Library/Fonts/SFNS.ttf",
               "C:/Windows/Fonts/segoeui.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
_FONT_CJK = ["/System/Library/Fonts/Hiragino Sans GB.ttc",
             "C:/Windows/Fonts/msyh.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
_FONT_CACHE = {}


def _resolve_font(candidates):
    """First candidate path that exists; None if none do."""
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _font(size, text=""):
    """Pick the font by whether text has non-ASCII characters; PIL is
    lazily loaded so the module imports without PIL installed."""
    from PIL import ImageFont
    path = _resolve_font(_FONT_CJK if any(ord(c) > 127 for c in text)
                         else _FONT_LATIN)
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = (ImageFont.truetype(path, size) if path
                            else ImageFont.load_default())
    return _FONT_CACHE[key]


def draw_texts_pil(img, texts):
    """Batch PIL text in one format conversion.
    texts: [(x, y, text, size, bgr_color)]."""
    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for x, y, t, size, color in texts:
        d.text((x, y), t, font=_font(size, t),
               fill=(color[2], color[1], color[0]))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def id_color(i):
    """ID pseudo-color: golden-angle hue spacing (adjacent IDs get
    well-separated colors)."""
    hsv = np.uint8([[[int((i * 0.618 % 1.0) * 179), 230, 255]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def stat_bar(img, x, y, frac, color=(0, 255, 0), w=100, h=8):
    """Metric progress bar; frac is clamped to [0, 1]."""
    frac = max(0.0, min(1.0, frac))
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
    if frac > 0:
        cv2.rectangle(img, (x, y), (x + int(w * frac), y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)


def bar_color(frac):
    """Green below half, yellow above half, red when full."""
    if frac >= 1.0:
        return (0, 0, 255)
    if frac > 0.5:
        return (0, 255, 255)
    return (0, 255, 0)
