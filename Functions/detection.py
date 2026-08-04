"""Marker dot detection."""

import math
import time

import cv2
import numpy as np

# Fallback defaults; the main flow derives thresholds via dot_params
BLOCK_SIZE = 51
THRESH_C = 10  # a pixel is black only if gray < neighborhood mean - C
MIN_AREA = 50

# Below this area circularity statistics are meaningless; skip the check
SHAPE_CHECK_AREA = 20


def preprocess(frame, ksize=3):
    """Grayscale + Gaussian blur (ksize=1 means no blur)."""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY)
    if ksize <= 1:
        return gray
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


def process(frame, ksize=3, block=BLOCK_SIZE):
    """Local adaptive threshold: pixels darker than the neighborhood mean
    by THRESH_C become white."""
    gray = preprocess(frame, ksize)
    mean = cv2.boxFilter(gray, -1, (block, block), normalize=True)
    return ((gray.astype(np.int16) - mean.astype(np.int16))
            < -THRESH_C).astype(np.uint8) * 255


def detect_dots(binary, min_area=MIN_AREA, max_area=None, min_circ=0.5,
                solid_thr=0.9, prof=None):
    """Connected components -> area/hull/circularity filtering -> solidity
    (filled/hollow) classification. Returns (dots, labels, solid_ids,
    hollow_ids, hollow_map); dot centers are convex-hull centroids, which
    are stabler than ring centroids for hollow dots."""
    # Per-stage timing (zero overhead when prof is None)
    def _mark(name, t0):
        if prof is not None:
            prof[name] = prof.get(name, 0.0) + time.perf_counter() - t0
        return time.perf_counter()

    _t = time.perf_counter()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    _t = _mark("cc", _t)

    # raw-area pre-filter; the minArea/2 lower bound is deliberately loose,
    # the real bound is the hull check below
    keep = stats[1:, cv2.CC_STAT_AREA] >= min_area / 2
    if max_area is not None:
        keep &= stats[1:, cv2.CC_STAT_AREA] <= max_area
    cand = np.nonzero(keep)[0] + 1  # CC labels start at 1 (0 is background)

    dots = []
    solid_ids = []
    hollow_ids = []
    hollow_map = {}
    for i in cand:
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        roi_lab = labels[y:y + h, x:x + w]
        area = int(stats[i, cv2.CC_STAT_AREA])

        mask = (roi_lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)

        # hull fill pixel count: the raw area of a hollow/defective dot
        # shrinks naturally; the hull measures the whole dot disk
        hull = cv2.convexHull(cnt)
        hull_mask = np.zeros(roi_lab.shape, dtype=np.uint8)
        cv2.fillPoly(hull_mask, [hull], 1)
        hull_px = int(hull_mask.sum())
        if hull_px < min_area:
            continue

        if area >= SHAPE_CHECK_AREA:
            peri = cv2.arcLength(cnt, True)
            if peri <= 0 or 4 * np.pi * area / (peri * peri) < min_circ:
                continue
        _t = _mark("filter", _t)

        m = cv2.moments(hull)
        if m["m00"] <= 0:
            continue
        cx = x + m["m10"] / m["m00"]
        cy = y + m["m01"] / m["m00"]
        is_hollow = (area / hull_px) < solid_thr
        dots.append((cx, cy, float(area)))
        (hollow_ids if is_hollow else solid_ids).append(i)
        # round to 0.1: Subdiv2D turns coordinates into float32, which
        # hashes differently from float64 with the same value
        hollow_map[(round(float(cx), 1), round(float(cy), 1))] = is_hollow
        _t = _mark("classify", _t)
    return dots, labels, solid_ids, hollow_ids, hollow_map




# Derived parameters (GUI sliders -> pixel thresholds), Length = image
# short side (makes the ratios resolution independent):
#   sphereDia = Length * sphereDiaRatio; dotDia = sphereDia * dotDiaRatio
#   blockSize = sphereDia / 4 (neighborhood: larger than a dot, smaller
#               than the shading gradient scale)
#   maxArea   = dotArea * 2 * redundant (central dots can reach 3x
#               dotArea when the sphere is close; x2 alone would kill them)
#   minArea   = dotArea / redundant (judged on hull-fill pixels)
def dot_params(length, sphere_dia_ratio=0.3, dot_dia_ratio=0.05,
               redundant=2):
    sphere_dia = length * sphere_dia_ratio
    dot_dia = sphere_dia * dot_dia_ratio
    dot_area = math.pi * dot_dia ** 2 / 4
    return {"sphere_dia": sphere_dia, "dot_dia": dot_dia,
            "block": max(3, round(sphere_dia / 4)),
            "min_area": dot_area / redundant,
            "max_area": dot_area * 2 * redundant}
