"""Single-frame pipeline (detection -> identification -> pose measurement)
and the sphereDiaRatio auto-tuning behind the Auto Set button."""

import time

import numpy as np

import detection
import identification
from pose_measurement import estimate_pose, proj_diameter


def run_pipeline(frame, sd=None, calib=None, *, ksize=3, block,
                 min_area, max_area=None, min_circ=0.5, solid_thr=0.9,
                 prof=None):
    """Full single-frame pipeline. sd=None skips identification/pose,
    calib=None skips pose. prof accumulates per-stage times (measurement
    only, no behavior change)."""
    h, w = frame.shape[:2]
    t0 = time.perf_counter()
    binary = detection.process(frame, ksize, block)
    if prof is not None:
        prof["binarize"] = (prof.get("binarize", 0.0)
                            + time.perf_counter() - t0)
    dots, labels, sids, hids, hmap = detection.detect_dots(
        binary, min_area=min_area, max_area=max_area,
        min_circ=min_circ, solid_thr=solid_thr, prof=prof)

    t0 = time.perf_counter()
    edges, longest, remaining = identification.mesh_remaining_edges(dots, w, h)
    if prof is not None:
        prof["mesh"] = (prof.get("mesh", 0.0) + time.perf_counter() - t0)
    t0 = time.perf_counter()
    cells, struct_edges, struct_nodes = \
        identification.find_cells_and_structure(remaining)
    if prof is not None:
        prof["cells"] = (prof.get("cells", 0.0) + time.perf_counter() - t0)

    obs = match = pose = None
    if sd is not None:
        t0 = time.perf_counter()
        obs = identification.build_obs_graph(struct_nodes, struct_edges,
                                       cells, hmap)
        if prof is not None:
            prof["obs"] = (prof.get("obs", 0.0) + time.perf_counter() - t0)
        match = identification.match_frame(sd, obs, prof=prof)
    if match is not None and calib is not None:
        pose = estimate_pose(sd, match, calib, prof=prof)

    return {"binary": binary, "dots": dots, "labels": labels,
            "sids": sids, "hids": hids, "hmap": hmap,
            "edges": edges, "longest": longest, "remaining": remaining,
            "cells": cells, "struct_edges": struct_edges,
            "struct_nodes": struct_nodes, "obs": obs, "match": match,
            "pose": pose}


def autoset(frame, sd, calib, dot_dia_ratio=0.05, redundant=2.0,
            min_circ=0.5, solid_thr=0.9, ratios=None, current=0.3,
            prof=None):
    """Auto-tune sphereDiaRatio: runs the full pipeline at 10 sampled
    ratios and returns (written, results, info). Written value prefers
    the valid pose with the smallest reprojection error (source="pose");
    falls back to the dot-richest sample ("dots") or the current value
    ("current"). Always writes back a value clamped to [0.1, 0.5].
    """
    if ratios is None:
        ratios = np.linspace(0.1, 0.5, 10)
    h0, w0 = frame.shape[:2]
    length = min(h0, w0)
    results = []
    best_dots = []  # dots of the most-populated sample (dots fallback)
    for sr in ratios:
        err = dia = None
        try:
            dp = detection.dot_params(length, sr, dot_dia_ratio,
                                      redundant)
            r = run_pipeline(frame, sd, calib, block=dp["block"],
                             min_area=dp["min_area"],
                             max_area=dp["max_area"], min_circ=min_circ,
                             solid_thr=solid_thr, prof=prof)
            if len(r["dots"]) > len(best_dots):
                best_dots = r["dots"]
            if r["pose"] is not None:
                err = r["pose"]["med_err"]
                dia = proj_diameter(sd, r["pose"], calib)
        except Exception:
            err = dia = None  # any exception = failed sample (not blocked)
        results.append((float(sr), err, dia))
    valid = [(sr, e, d) for sr, e, d in results if e is not None]
    if valid:
        best_sr, _, best_d = min(valid, key=lambda x: x[1])
        measured = best_d / length
        source = "pose"
    elif len(best_dots) >= 10:
        pts = np.array([[d[0], d[1]] for d in best_dots])
        dd = np.linalg.norm(pts - pts.mean(0), axis=1)
        measured = float(np.percentile(dd, 95) * 2) / length
        best_sr = None
        source = "dots"
    else:
        measured = current
        best_sr = None
        source = "current"
    written = float(np.clip(measured, 0.1, 0.5))
    return written, results, {"measured": measured, "best_sr": best_sr,
                              "clamped": written != measured,
                              "source": source}
