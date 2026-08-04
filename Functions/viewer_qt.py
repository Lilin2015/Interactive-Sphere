"""Interactive sphere pose-measurement viewer (PySide6 GUI)."""

import argparse
import math
import os
import sys
import threading
import time

import cv2
import numpy as np

import detection
from identification import DICT_PATH, SphereDict
from pose_measurement import CALIB_PATH, draw_model_view, load_calib
from pipeline import autoset, run_pipeline
from render import bar_color, draw_texts_pil, id_color, stat_bar

try:  # fall back to stub classes if PySide6 is not installed
    from PySide6.QtCore import QObject, Qt, QTimer, Signal
    from PySide6.QtGui import QImage, QKeyEvent, QPixmap
    from PySide6.QtWidgets import (QApplication, QCheckBox, QFileDialog,
                                   QHBoxLayout, QLabel, QMainWindow,
                                   QFrame, QPushButton, QRadioButton, QSlider,
                                   QToolButton, QVBoxLayout, QWidget)
    _QT_OK = True
except ImportError:
    _QT_OK = False
    # stubs so the Bridge/MainWindow class bodies still execute at import
    class _QtStub:
        def __getattr__(self, name):
            return object
    Qt = _QtStub()
    QObject = QMainWindow = QWidget = QApplication = QTimer = object
    QImage = QKeyEvent = QPixmap = QCheckBox = QFileDialog = object
    QFrame = QHBoxLayout = QLabel = QPushButton = object
    QRadioButton = QSlider = QToolButton = QVBoxLayout = object

    def Signal(*args, **kwargs):
        return None


# views are scaled to fit this box; the algorithm always runs at full resolution
FIT_W, FIT_H = 1080, 720


PANEL_W = 250

VIDEO_FILTER = "Video Files (*.mp4 *.mov *.avi *.mkv *.m4v)"

DEFAULT_VIDEO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "Material", "test.mp4")


# ------------------------------------------------------------ Shared state

class Shared:
    """State written by the UI thread, read by the worker (under lock)."""

    def __init__(self, video):
        self.lock = threading.Lock()
        self.video_path = video     # initial video (may be None)
        self.frame_idx = 0
        self.playing = False
        self.sphere_dia_ratio = 0.3  # sphere diameter / short image side (0.1~0.5)
        self.dot_dia_ratio = 0.05    # dot diameter / sphere diameter (0.02~0.2)
        self.redundant = 2.0         # redundancy factor for area decision (1.2~3.0)
        self.circ = 50               # circularity threshold Circ% (40~90, used as /100)
        self.solid_thr = 0.9         # solidity threshold (0.50~0.95)
        self.mirror = False
        self.debug_view = "pose measurement"  # one of the three Debug views
        self.glyphs_active = False  # parameter glyphs while a slider is dragged
        self.quit = False
        self.req = []               # [("video", path) / ("calib", path) /
                                    #  ("dict", path)]

    def snapshot(self):
        with self.lock:
            return {"frame_idx": self.frame_idx, "playing": self.playing,
                    "sphere_dia_ratio": self.sphere_dia_ratio,
                    "dot_dia_ratio": self.dot_dia_ratio,
                    "redundant": self.redundant, "circ": self.circ,
                    "solid_thr": self.solid_thr,
                    "mirror": self.mirror, "debug_view": self.debug_view,
                    "glyphs_active": self.glyphs_active, "quit": self.quit}

    def push_req(self, kind, path=None):
        with self.lock:
            self.req.append((kind, path))


# ---------------------------------------------------------- Worker thread

def worker(sh, publish):
    """Algorithm thread: grab frame -> pipeline -> composite -> publish."""
    sd = SphereDict() if os.path.exists(DICT_PATH) else None
    calib = load_calib() if sd is not None else None
    calib_ver = sd_ver = 0
    src_ver = 0  # bumped on source switch; UI resets the Frame slider on change
    names = {"video": "-", "calib": os.path.basename(CALIB_PATH),
             "dict": os.path.basename(DICT_PATH)}

    cap = None          # video file
    total, fps = 1, 30.0
    last_key = None
    anchor = None       # playback anchor (start frame, start wall clock)
    prev_playing = False

    def open_video(path):
        nonlocal cap, total, fps, last_key, src_ver
        new_cap = cv2.VideoCapture(path)
        if not new_cap.isOpened():
            new_cap.release()
            print(f"Cannot open video: {path}", file=sys.stderr)
            return
        if cap is not None:
            cap.release()
        cap = new_cap
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        names["video"] = os.path.basename(path)
        with sh.lock:
            sh.frame_idx = 0
            # do not reset `playing`: the initial load must keep --autoplay
        src_ver += 1
        last_key = None

    if sh.video_path:
        open_video(sh.video_path)

    while True:
        snap = sh.snapshot()
        if snap["quit"]:
            break
        # ---- handle requests enqueued by the UI
        with sh.lock:
            reqs, sh.req = sh.req, []
        autoset_out = None
        for kind, path in reqs:
            if kind == "autoset":
                # autoset runs offline on the current frame without touching live display state
                fi_req = snap["frame_idx"]
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi_req)
                ok_a, frame_a = cap.read()
                if ok_a:
                    if snap["mirror"]:
                        frame_a = cv2.flip(frame_a, 1)
                    t_as = time.perf_counter()
                    chosen, res, info = autoset(
                        frame_a, sd, calib, snap["dot_dia_ratio"],
                        snap["redundant"], snap["circ"] / 100,
                        snap["solid_thr"],
                        current=snap["sphere_dia_ratio"])
                    autoset_out = {"chosen": chosen, "results": res,
                                   "ms": (time.perf_counter() - t_as)
                                   * 1000}
                    print(f"[autoset] frame {fi_req}:")
                    for r, e, d in res:
                        print(f"  ratio {r:.3f} -> "
                              + (f"{e:.2f}px, D={d:.0f}px"
                                 if e is not None else "failed"))
                    src_name = {"pose": "pose-derived", "dots": "dot-estimate",
                                "current": "fallback-current"}[info["source"]]
                    msg = f"  [{src_name}] ratio={info['measured']:.3f}"
                    if info["best_sr"] is not None:
                        msg += f" (winning sample {info['best_sr']:.3f})"
                    msg += f" -> write back {chosen:.3f}"
                    if info["clamped"]:
                        msg += " (measured ratio out of range, clamped;" \
                               " consider widening the slider range)"
                    print(msg + f"  ({autoset_out['ms']:.0f}ms)")
                    last_key = None  # force re-render this cycle to publish the result
            elif kind == "video":
                open_video(path)
                with sh.lock:  # user-initiated switch: restart paused
                    sh.playing = False
            elif kind == "calib":
                c = load_calib(path)
                if c is not None:
                    calib = c
                    calib_ver += 1
                    names["calib"] = os.path.basename(path)
                    last_key = None
            elif kind == "dict":
                try:
                    sd = SphereDict(path)
                    sd_ver += 1
                    names["dict"] = os.path.basename(path)
                    last_key = None
                except Exception:
                    pass  # invalid dictionary file: keep the old model

        if cap is None:
            time.sleep(0.05)
            continue

        fi, playing = snap["frame_idx"], snap["playing"]
        sph_r, dot_r = snap["sphere_dia_ratio"], snap["dot_dia_ratio"]
        redundant = snap["redundant"]
        min_circ = snap["circ"] / 100
        solid_thr = snap["solid_thr"]
        mirror, debug = snap["mirror"], snap["debug_view"]
        glyphs = snap["glyphs_active"]

        # ---- grab a frame (seek + read)
        # frame chasing: target = anchor frame + elapsed x fps; playback
        # tracks real fps even when the algorithm is slower; stops at end
        if playing and not prev_playing:
            anchor = (fi, time.perf_counter())
        prev_playing = playing
        if not playing:
            anchor = None
        if playing:
            target = anchor[0] + int(
                (time.perf_counter() - anchor[1]) * fps)
            if target >= total:
                fi = total - 1
                playing = False
                with sh.lock:
                    sh.frame_idx = fi
                    sh.playing = False
            elif target > fi:
                fi = target
                with sh.lock:
                    sh.frame_idx = fi
        key = (fi, sph_r, dot_r, redundant, min_circ,
               solid_thr, mirror, debug, glyphs, calib_ver, sd_ver)
        if key == last_key:  # nothing changed: idle (low power while paused)
            time.sleep(0.01)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        if mirror:
            frame = cv2.flip(frame, 1)
        last_key = key

        # ---- algorithm pipeline
        hh0, ww0 = frame.shape[:2]
        dp = detection.dot_params(min(hh0, ww0), sph_r, dot_r, redundant)
        t0 = time.perf_counter()
        r = run_pipeline(frame, sd, calib, block=dp["block"],
                         min_area=dp["min_area"], max_area=dp["max_area"],
                         min_circ=min_circ, solid_thr=solid_thr)
        algo_ms = (time.perf_counter() - t0) * 1000
        binary, labels, sids, hids = (r["binary"], r["labels"],
                                      r["sids"], r["hids"])
        mr, pose = r["match"], r["pose"]

        # ---- composite the 1080x720 view
        h0, w0 = binary.shape
        ds = min(FIT_W / w0, FIT_H / h0, 1.0)
        dw, dh = round(w0 * ds), round(h0 * ds)
        canvas = np.zeros((FIT_H, FIT_W, 3), dtype=np.uint8)
        x_off = (FIT_W - dw) // 2
        y_off = (FIT_H - dh) // 2
        if debug == "detection":
            base = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            base[np.isin(labels, sids)] = (0, 200, 0)
            base[np.isin(labels, hids)] = (0, 0, 220)
            canvas[y_off:y_off + dh, x_off:x_off + dw] = \
                cv2.resize(base, (dw, dh))
        else:
            v1 = cv2.resize(frame, (dw, dh))
            if pose is not None and debug != "identification":
                draw_model_view(v1, sd, pose, calib, scale=ds)
            if debug == "identification":
                # draw only edges whose both endpoints are match-confirmed;
                # endpoints use detection coordinates, not reprojection
                if mr is not None and mr.get("accepted"):
                    dm = {did: (x, y) for (x, y), did in mr["map"].items()}
                    drawn = set()
                    for did, (x, y) in dm.items():
                        for B in sd.adj.get(did, ()):
                            if B in dm and (B, did) not in drawn:
                                drawn.add((did, B))
                                x2, y2 = dm[B]
                                cv2.line(v1, (int(x * ds), int(y * ds)),
                                         (int(x2 * ds), int(y2 * ds)),
                                         (0, 200, 0), 1, cv2.LINE_AA)
                    for did, (x, y) in dm.items():
                        cv2.circle(v1, (int(x * ds), int(y * ds)), 5,
                                   id_color(did), -1, cv2.LINE_AA)
            canvas[y_off:y_off + dh, x_off:x_off + dw] = v1

        # metrics area (top-left of the view)
        BAR_X = 210
        BAR_W, BAR_H = 80, 6
        frame_txt = f"frame {fi}/{total - 1}"
        texts = [(10, 4, f"processing resolution {w0} x {h0}   "
                 f"{frame_txt}", 14, (0, 255, 0)),
                 (10, 30, f"timeCost {algo_ms:.0f} ms", 14,
                  (0, 255, 0))]
        stat_bar(canvas, BAR_X, 37, algo_ms / 50,
                        bar_color(algo_ms / 50), w=BAR_W, h=BAR_H)
        max_texts = [(BAR_X + BAR_W + 8, 30, "50 ms", 14,
                      (150, 150, 150))]
        if pose is not None:
            out_pct = (1.0 - pose["n_in"] / pose["n_total"]) / 0.2
            err_frac = pose["med_err"] / 1.0
            texts += [
                (10, 56, f"outlierPct "
                 f"{(1.0 - pose['n_in'] / pose['n_total']) * 100:.1f} %",
                 14, (0, 255, 0)),
                (10, 82, f"meanReprojError {pose['med_err']:.2f} px",
                 14, (0, 255, 0))]
            stat_bar(canvas, BAR_X, 63, out_pct,
                            bar_color(out_pct), w=BAR_W, h=BAR_H)
            stat_bar(canvas, BAR_X, 89, err_frac,
                            bar_color(err_frac), w=BAR_W, h=BAR_H)
        else:
            texts += [(10, 56, "outlierPct -", 14, (0, 255, 0)),
                      (10, 82, "meanReprojError -", 14, (0, 255, 0))]
            for by in (63, 89):  # no data: thin placeholder line
                cv2.line(canvas, (BAR_X, by + 3),
                         (BAR_X + BAR_W, by + 3),
                         (0, 255, 0), 1, cv2.LINE_AA)
        max_texts += [(BAR_X + BAR_W + 8, 56, "20 %", 14,
                       (150, 150, 150)),
                      (BAR_X + BAR_W + 8, 82, "1 px", 14,
                       (150, 150, 150))]
        draw_texts_pil(canvas, texts)
        draw_texts_pil(canvas, max_texts)

        # ---- parameter glyphs (slider-drag preview): concentric circles at
        # view center; labels staggered at angles to avoid overlap
        if glyphs:
            cy_c, cx_c = FIT_H // 2, FIT_W // 2
            R = max(1, int(round(dp["sphere_dia"] / 2 * ds)))
            r = max(1, int(round(dp["dot_dia"] / 2 * ds)))
            b = max(2, int(round(dp["block"] * ds)))
            rd = snap["redundant"]
            YEL = (0, 255, 255)
            RED = (0, 0, 255)
            cv2.circle(canvas, (cx_c, cy_c), R, YEL, 1, cv2.LINE_AA)
            cv2.circle(canvas, (cx_c, cy_c), r, YEL, 1, cv2.LINE_AA)
            bx0 = cx_c - R - 30 - b
            cv2.rectangle(canvas, (bx0, cy_c - b // 2),
                          (bx0 + b, cy_c + b // 2), YEL, 1, cv2.LINE_AA)
            texts = [
                (cx_c + R + 6, cy_c - R // 2 - 7, "sphereDia", 14, YEL),
                (cx_c + r + 6, cy_c - 7, "dotDia", 14, YEL),
                (bx0 + b + 6, cy_c - 7, "blockSize", 14, YEL)]
            if glyphs == "redundant":
                BLU = (255, 0, 0)
                for dia, name, ang, col in (
                        (dp["sphere_dia"] * rd, f"sphereDia×{rd:.1f}",
                         -55, BLU),
                        (dp["sphere_dia"] / rd, f"sphereDia/{rd:.1f}",
                         -15, BLU),
                        (dp["dot_dia"] * rd, f"dotDia×{rd:.1f}", 25, RED),
                        (dp["dot_dia"] / rd, f"dotDia/{rd:.1f}", 70, RED)):
                    rr = max(1, int(round(dia / 2 * ds)))
                    cv2.circle(canvas, (cx_c, cy_c), rr, col, 1,
                               cv2.LINE_AA)
                    a = math.radians(ang)
                    tx = cx_c + int((rr + 8) * math.cos(a))
                    ty = cy_c + int((rr + 8) * math.sin(a)) - 7
                    texts.append((tx, ty, name, 14, col))
            draw_texts_pil(canvas, texts)

        # ---- publish the result (cross-thread Qt Signal)
        meta = {"img": canvas, "frame_idx": fi, "playing": playing,
                "total": total, "algo_ms": algo_ms, "names": dict(names),
                "src_ver": src_ver}
        if autoset_out is not None:  # autoset result rides along once
            meta["autoset"] = autoset_out
            autoset_out = None
        publish(meta)

    if cap is not None:
        cap.release()


# --------------------------------------------------------------- UI thread

class Bridge(QObject):
    """Worker -> main thread channel (emit is queued across threads)."""
    new_frame = Signal(object)


class MainWindow(QMainWindow):
    def __init__(self, sh, shots, shot_frames, sh_debug="pose measurement"):
        super().__init__()
        self.sh = sh
        self.shot_paths = shots
        self.shot_frames = shot_frames
        self.n_results = 0
        self._src_ver = None
        self.setWindowTitle("Interactive Sphere Viewer")
        self._pixmap = None  # original 1080x720, resampled on window resize

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        # ---- left column: image area + Frame slider + Play/Pause
        left = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 427)
        self.image_label.setStyleSheet("background: black;")
        self.image_label.setAlignment(Qt.AlignCenter)
        left.addWidget(self.image_label, stretch=1)
        bottom = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(1)
        self.frame_slider.valueChanged.connect(self._on_frame_slider)
        bottom.addWidget(self.frame_slider, stretch=1)
        help_lab = self._label(
            "Space: play   A/D / ←→: step   Q: quit",
            "color: #969696; font-size: 11px;")
        bottom.addWidget(help_lab)
        self.play_btn = QPushButton("Play/Pause")
        self.play_btn.clicked.connect(self._on_play)
        bottom.addWidget(self.play_btn)
        left.addLayout(bottom)
        root.addLayout(left, stretch=1)

        # ---- right column: fixed-width native panel
        panel = QVBoxLayout()
        panel.setContentsMargins(6, 4, 6, 4)
        panel.setSpacing(4)
        head_style = "color: #00c853; font-weight: bold;"
        gray_style = "color: #969696; font-size: 11px;"

        # Debug container created early; added to the panel after the divider
        self.debug_host = QWidget()
        dbg = QVBoxLayout(self.debug_host)
        dbg.setContentsMargins(0, 0, 0, 0)
        dbg.setSpacing(4)
        panel.addWidget(self._label("Parameters", head_style))
        # float ratios use int x1000 sliders; glyphs show while dragging
        self.sr_val = self._label("sphereDiaRatio 0.300", gray_style)
        self.sr_slider = QSlider(Qt.Horizontal)
        self.sr_slider.setMinimum(100)   # 0.100
        self.sr_slider.setMaximum(500)   # 0.500
        self.sr_slider.setValue(300)     # 0.300 default
        self.sr_slider.valueChanged.connect(
            lambda v: self._on_ratio("sphereDiaRatio",
                                     "sphere_dia_ratio", v, self.sr_val, 3))
        self.sr_slider.sliderPressed.connect(
            lambda: self._on_glyph("ratio"))
        self.sr_slider.sliderReleased.connect(
            lambda: self._on_glyph(False))
        sr_row = QHBoxLayout()
        sr_row.addWidget(self.sr_val, stretch=1)
        self.autoset_btn = QPushButton("Auto Set")
        self.autoset_btn.clicked.connect(self._on_autoset)
        sr_row.addWidget(self.autoset_btn)
        panel.addLayout(sr_row)
        panel.addWidget(self.sr_slider)
        self.dr_val = self._label("dotDiaRatio 0.050", gray_style)
        self.dr_slider = QSlider(Qt.Horizontal)
        self.dr_slider.setMinimum(20)    # 0.020
        self.dr_slider.setMaximum(200)   # 0.200
        self.dr_slider.setValue(50)      # 0.050 default
        self.dr_slider.valueChanged.connect(
            lambda v: self._on_ratio("dotDiaRatio",
                                     "dot_dia_ratio", v, self.dr_val, 3))
        self.dr_slider.sliderPressed.connect(
            lambda: self._on_glyph("ratio"))
        self.dr_slider.sliderReleased.connect(
            lambda: self._on_glyph(False))
        dbg.addWidget(self.dr_val)
        dbg.addWidget(self.dr_slider)
        self.rd_val = self._label("redundant 2.0", gray_style)
        self.rd_slider = QSlider(Qt.Horizontal)
        self.rd_slider.setMinimum(12)   # 1.2
        self.rd_slider.setMaximum(30)   # 3.0
        self.rd_slider.setValue(20)     # 2.0 default
        self.rd_slider.setSingleStep(1)
        self.rd_slider.valueChanged.connect(self._on_redundant)
        self.rd_slider.sliderPressed.connect(
            lambda: self._on_glyph("redundant"))
        self.rd_slider.sliderReleased.connect(
            lambda: self._on_glyph(False))
        dbg.addWidget(self.rd_val)
        dbg.addWidget(self.rd_slider)
        self.circ_val = self._label("Circularity 50", gray_style)
        self.circ_slider = QSlider(Qt.Horizontal)
        self.circ_slider.setMinimum(40)
        self.circ_slider.setMaximum(90)
        self.circ_slider.setValue(50)
        self.circ_slider.valueChanged.connect(self._on_circ)
        dbg.addWidget(self.circ_val)
        dbg.addWidget(self.circ_slider)
        self.st_val = self._label("Solidity 0.90", gray_style)
        self.st_slider = QSlider(Qt.Horizontal)
        self.st_slider.setMinimum(50)
        self.st_slider.setMaximum(95)
        self.st_slider.setValue(90)
        self.st_slider.valueChanged.connect(self._on_solid_thr)
        dbg.addWidget(self.st_val)
        dbg.addWidget(self.st_slider)
        # macOS Aqua draws no ticks, so these sliders use the Fusion style
        from PySide6.QtWidgets import QStyleFactory
        self._fusion = QStyleFactory.create("Fusion")  # keep a reference to prevent GC
        for slider, interval in ((self.sr_slider, 50),
                                 (self.dr_slider, 10),
                                 (self.rd_slider, 2),
                                 (self.circ_slider, 10),
                                 (self.st_slider, 5)):
            slider.setStyle(self._fusion)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(interval)
        self.mirror_box = QCheckBox("mirror")
        self.mirror_box.toggled.connect(self._on_mirror)
        panel.addWidget(self.mirror_box)
        panel.addSpacing(8)

        panel.addWidget(self._label("Video Source", head_style))
        self.name_video = self._label("-", gray_style)
        panel.addWidget(self.name_video)
        vrow = QHBoxLayout()
        btn_vf = QPushButton("File...")
        btn_vf.clicked.connect(self._on_pick_video)
        vrow.addWidget(btn_vf)
        panel.addLayout(vrow)
        panel.addSpacing(8)

        panel.addWidget(self._label("Intrinsic Matrix", head_style))
        self.name_calib = self._label("-", gray_style)
        panel.addWidget(self.name_calib)
        btn_cf = QPushButton("File...")
        btn_cf.clicked.connect(
            lambda: self._pick_file("Select Intrinsic Matrix file",
                                    "MAT Files (*.mat)", "calib"))
        panel.addWidget(btn_cf)
        panel.addSpacing(8)

        panel.addWidget(self._label("Sphere Model", head_style))
        self.name_dict = self._label("-", gray_style)
        panel.addWidget(self.name_dict)
        btn_df = QPushButton("File...")
        btn_df.clicked.connect(
            lambda: self._pick_file("Select Sphere Model file",
                                    "JSON Files (*.json)", "dict"))
        panel.addWidget(btn_df)
        panel.addSpacing(8)

        btn_row = QHBoxLayout()
        btn_reset = QPushButton("Reset")
        btn_reset.clicked.connect(self._on_reset)
        btn_exit = QPushButton("Exit")
        btn_exit.clicked.connect(self._on_quit)
        btn_row.addWidget(btn_reset)
        btn_row.addWidget(btn_exit)
        panel.addLayout(btn_row)
        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setFrameShadow(QFrame.Sunken)
        panel.addWidget(hline)
        # ---- Debug collapsible area (expanded by default)
        self.debug_btn = QToolButton()
        self.debug_btn.setText("Debug")
        self.debug_btn.setCheckable(True)
        self.debug_btn.setChecked(True)
        self.debug_btn.setArrowType(Qt.DownArrow)
        self.debug_btn.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon)
        self.debug_btn.setStyleSheet(
            "QToolButton { color: #00c853; font-weight: bold;"
            " font-size: 13px; border: none; }")
        panel.addWidget(self.debug_btn)
        self.debug_host.setVisible(True)
        panel.addWidget(self.debug_host)
        _radio_style = "font-size: 11px;"
        self.radio_origin = QRadioButton("pose measurement")
        self.radio_origin.setChecked(True)
        self.radio_det = QRadioButton("detection")
        self.radio_id = QRadioButton("identification")
        self.radio_origin.toggled.connect(
            lambda on: on and self._set_debug("pose measurement"))
        self.radio_det.toggled.connect(
            lambda on: on and self._set_debug("detection"))
        self.radio_id.toggled.connect(
            lambda on: on and self._set_debug("identification"))
        for _rb in (self.radio_origin, self.radio_det, self.radio_id):
            _rb.setStyleSheet(_radio_style)
        dbg.addWidget(self.radio_origin)
        dbg.addWidget(self.radio_det)
        dbg.addWidget(self.radio_id)
        if sh_debug == "detection":
            self.radio_det.setChecked(True)
        elif sh_debug == "identification":
            self.radio_id.setChecked(True)
        self.debug_btn.toggled.connect(self._on_debug_toggle)

        panel.addStretch(1)


        panel_host = QWidget()
        panel_host.setLayout(panel)
        panel_host.setFixedWidth(PANEL_W)
        root.addWidget(panel_host)
        self.setCentralWidget(central)
        self.resize(FIT_W + PANEL_W + 60, FIT_H + 110)

    # ---- control callbacks: only write shared state, never touch the algorithm
    @staticmethod
    def _label(text, style=""):
        lab = QLabel(text)
        if style:
            lab.setStyleSheet(style)
        return lab

    def _on_frame_slider(self, value):
        with self.sh.lock:
            self.sh.frame_idx = value
            self.sh.playing = False  # dragging/clicking the slider pauses playback

    def _on_play(self):
        with self.sh.lock:
            self.sh.playing = not self.sh.playing

    def _on_ratio(self, disp, name, value, label, ndigits):
        label.setText(f"{disp} {value / 1000:.{ndigits}f}")
        with self.sh.lock:
            setattr(self.sh, name, value / 1000)

    def _hide_glyphs(self):
        """Hide the autoset indicator circles (QTimer.singleShot callback)."""
        with self.sh.lock:
            self.sh.glyphs_active = False

    def _on_autoset(self):
        """Auto Set: ask the worker to run autoset on the current frame."""
        self.autoset_btn.setEnabled(False)  # re-enabled when the result arrives
        self.sh.push_req("autoset")

    def _on_glyph(self, active):
        """Show parameter glyphs while a slider is dragged."""
        with self.sh.lock:
            self.sh.glyphs_active = active

    def _on_redundant(self, value):
        self.rd_val.setText(f"redundant {value / 10:.1f}")
        with self.sh.lock:
            self.sh.redundant = value / 10

    def _on_circ(self, value):
        self.circ_val.setText(f"Circularity {value}")
        with self.sh.lock:
            self.sh.circ = value

    def _on_solid_thr(self, value):
        self.st_val.setText(f"Solidity {value / 100:.2f}")
        with self.sh.lock:
            self.sh.solid_thr = value / 100

    def _on_reset(self):
        """Reset all sliders to defaults (setValue triggers the write-back)."""
        self.sr_slider.setValue(300)   # sphereDiaRatio 0.300
        self.dr_slider.setValue(50)    # dotDiaRatio 0.050
        self.rd_slider.setValue(20)    # redundant 2.0
        self.circ_slider.setValue(50)  # Circ% 50
        self.st_slider.setValue(90)    # SolidThr 0.9
        self.mirror_box.setChecked(False)

    def _on_mirror(self, checked):
        with self.sh.lock:
            self.sh.mirror = bool(checked)

    def _on_debug_toggle(self, checked):
        self.debug_host.setVisible(checked)
        self.debug_btn.setArrowType(
            Qt.DownArrow if checked else Qt.RightArrow)

    def _set_debug(self, mode):
        with self.sh.lock:
            self.sh.debug_view = mode  # worker re-renders the current frame next cycle

    def _on_quit(self):
        with self.sh.lock:
            self.sh.quit = True
        QApplication.quit()

    def _on_pick_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video file",
                                              "", VIDEO_FILTER)
        if path:
            self.sh.push_req("video", path)

    def _pick_file(self, title, filt, kind):
        path, _ = QFileDialog.getOpenFileName(self, title, "", filt)
        if path:
            self.sh.push_req(kind, path)

    def _step(self, delta):
        with self.sh.lock:
            self.sh.frame_idx = max(0, self.sh.frame_idx + delta)
            self.sh.playing = False

    def keyPressEvent(self, ev: QKeyEvent):
        key = ev.key()
        if key == Qt.Key_Space:
            self._on_play()
        elif key in (Qt.Key_Left, Qt.Key_A):
            self._step(-1)
        elif key in (Qt.Key_Right, Qt.Key_D):
            self._step(1)
        elif key == Qt.Key_O:
            self._on_pick_video()
        elif key == Qt.Key_Q:
            self._on_quit()
        else:
            super().keyPressEvent(ev)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._show_pixmap()

    def _show_pixmap(self):
        if self._pixmap is not None:
            self.image_label.setPixmap(self._pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))

    # ---- result slot (main thread)
    def on_result(self, r):
        bgr = np.ascontiguousarray(r["img"][..., ::-1])  # BGR -> RGB
        img = QImage(bgr.data, FIT_W, FIT_H, FIT_W * 3,
                     QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(img)  # deep copy, so bgr can be released
        self._show_pixmap()
        # source switched: reset the Frame slider (worker already zeroed frame_idx)
        if r.get("src_ver") != self._src_ver:
            self._src_ver = r.get("src_ver")
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(0)
            self.frame_slider.blockSignals(False)
        # blockSignals prevents feedback loops into the worker
        if self.frame_slider.maximum() != max(1, r["total"] - 1):
            self.frame_slider.setMaximum(max(1, r["total"] - 1))
        if r["playing"]:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(r["frame_idx"])
            self.frame_slider.blockSignals(False)
        if "autoset" in r:  # write back to the slider + 2 s of glyphs
            chosen = r["autoset"]["chosen"]
            self.sr_slider.setValue(int(round(chosen * 1000)))
            with self.sh.lock:
                self.sh.glyphs_active = "ratio"
            QTimer.singleShot(2000, self._hide_glyphs)
            self.autoset_btn.setEnabled(True)
        self.name_video.setText(r["names"]["video"])
        self.name_calib.setText(r["names"]["calib"])
        self.name_dict.setText(r["names"]["dict"])
        # self-check screenshots: write the composite image directly to disk
        self.n_results += 1
        if self.shot_paths and self.n_results in self.shot_frames:
            path = self.shot_paths[self.shot_frames.index(self.n_results)]
            cv2.imwrite(path, r["img"])
            print(f"[self-check] screenshot saved {path} (frame {r['frame_idx']})")
            if self.n_results == self.shot_frames[-1]:
                with self.sh.lock:
                    self.sh.quit = True
                QApplication.quit()


def main():
    if not _QT_OK:
        raise SystemExit(
            "PySide6 not available: algorithm functions can still be "
            "imported and reused, but the GUI requires PySide6 "
            "(python3 -m pip install pyside6)")
    parser = argparse.ArgumentParser(
        description="Interactive Sphere Viewer (PySide6/Qt frontend)")
    parser.add_argument("video", nargs="?", default=None,
                        help="video path; if omitted, loads the default video "
                             "DEFAULT_VIDEO (if missing, press O or File... "
                             "after startup to choose)")
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--shots", default=None, metavar="P1,P2,P3")
    parser.add_argument("--debug", default="pose measurement",
                        choices=["pose measurement", "detection", "identification"],
                        help="self-check: initial Debug view "
                             "(default: pose measurement)")
    args = parser.parse_args()

    video = args.video
    if video is None and os.path.exists(DEFAULT_VIDEO):
        video = DEFAULT_VIDEO
    sh = Shared(video)
    if args.autoplay:
        sh.playing = True
    sh.debug_view = args.debug
    shot_paths = args.shots.split(",") if args.shots else []
    shot_frames = [30 + 60 * i for i in range(len(shot_paths))]

    app = QApplication(sys.argv)
    bridge = Bridge()
    win = MainWindow(sh, shot_paths, shot_frames, sh_debug=args.debug)
    bridge.new_frame.connect(win.on_result, Qt.QueuedConnection)

    th = threading.Thread(target=worker,
                          args=(sh, bridge.new_frame.emit), daemon=True)
    th.start()
    win.show()
    app.exec()

    with sh.lock:
        sh.quit = True
    th.join(timeout=2)


if __name__ == "__main__":
    main()
