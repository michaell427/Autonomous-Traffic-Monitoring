"""Desktop viewer: one window with Add files, playlist, before/after, and frame stepping.

Run from project root (venv active):

  python demo_upload_window.py --weights yolov8n.pt

- **Add files…** — pick multiple images/videos (you can add again anytime).
- **List** — click a row to jump to that item.
- **◀ media / media ▶** — previous / next file in the queue.
- **◀ frame / frame ▶** — previous / next frame (videos only).
- **Play / Pause** — auto-advance frames at the clip’s FPS (before or after view; **Space** still toggles).
- **Tracking IDs** — optional video mode for persistent IDs (ByteTrack/BoT-SORT).
- **Before / After** — toggle original vs YOLO overlay (or **Space**).

Keyboard: **Space** = before/after; **P** = play/pause (video); **← / →** = frame (video); **[** / **]** = prev/next media.

Video defaults to per-frame detection. Enable **Tracking IDs** to use sequential
multi-object tracking with persistent IDs in the viewer.

Preview uses a **Tk Canvas** + Pillow (not a Label), so the image does not drive window
geometry. **opencv-python-headless** is fine (no separate OpenCV window).
"""

from __future__ import annotations

import argparse
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

from demo_before_after import IMAGE_EXTS, annotate_frame

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
_MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS
_MAX_VIDEO_CACHE_FRAMES = 120

# Fallback bounds when the canvas is not mapped yet (avoid oversized PhotoImage).
_PREVIEW_FALLBACK_MAX_W = 512
_PREVIEW_FALLBACK_MAX_H = 384
_PREVIEW_ABSOLUTE_CAP_W = 1600
_PREVIEW_ABSOLUTE_CAP_H = 1200
# Windows sends many Configure events during maximize/fullscreen restore; debounce avoids
# blocking the UI thread with repeated PIL resizes (looks like the window "shrinks slowly").
_PREVIEW_RESIZE_DEBOUNCE_MS = 200
_PREVIEW_MAP_RETRY_MAX = 12
_PREVIEW_CANVAS_PAD = 8


def resolve_weights(raw: str, project_root: Path) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = (project_root / p).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Weights not found: {p}")
    return p


def bgr_to_photo(bgr: np.ndarray, max_w: int, max_h: int) -> ImageTk.PhotoImage:
    h, w = bgr.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError("empty image")
    scale = min(max_w / w, max_h / h, 1.0)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    if (nw, nh) != (w, h):
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil)


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


class UploadViewerApp:
    def __init__(
        self,
        root: tk.Tk,
        model: YOLO,
        weights_path: Path,
        tracker: str,
        conf: float,
        imgsz: int,
        initial_files: Optional[List[Path]] = None,
    ) -> None:
        self.root = root
        self.model = model
        self.weights_path = weights_path
        self.tracker = tracker
        self.conf = conf
        self.imgsz = imgsz

        self.paths: List[Path] = []
        self.media_index: int = 0
        self.show_before: bool = True
        self.video_frame_index: int = 0
        self._cap: Optional[cv2.VideoCapture] = None
        self._video_n_frames: int = 0
        self._video_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._video_track_cache: Dict[int, np.ndarray] = {}
        self._track_ready_upto: int = -1
        self._track_model: Optional[YOLO] = None

        self._image_before: Optional[np.ndarray] = None
        self._image_after: Optional[np.ndarray] = None

        self._photo: Optional[ImageTk.PhotoImage] = None
        self._preview_resize_after_id: Optional[str] = None
        self._preview_map_retry_count: int = 0
        self._preview_map_after_id: Optional[str] = None

        self._playing: bool = False
        self._play_after_id: Optional[str] = None
        self._video_fps: float = 24.0
        self.use_tracking_ids = tk.BooleanVar(value=False)

        root.title("Traffic — upload & before/after")
        root.minsize(720, 560)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()

        root.bind("<space>", lambda e: self._toggle_before_after())
        root.bind("p", lambda e: self._toggle_play())
        root.bind("P", lambda e: self._toggle_play())
        root.bind("<Left>", lambda e: self._prev_frame())
        root.bind("<Right>", lambda e: self._next_frame())
        root.bind("<bracketleft>", lambda e: self._prev_media())
        root.bind("<bracketright>", lambda e: self._next_media())

        if initial_files:
            self._append_paths(initial_files)
        self._refresh_all()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        ttk.Button(top, text="Add files…", command=self._on_add_files).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(top, text="Clear queue", command=self._on_clear_queue).pack(side=tk.LEFT, padx=(0, 8))

        self.status_var = tk.StringVar(value="Add images or videos to begin.")
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, padx=8)

        mid = ttk.Frame(self.root, padding=(8, 0))
        mid.pack(fill=tk.BOTH, expand=True)

        list_frame = ttk.LabelFrame(mid, text="Queue", padding=4)
        list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

        self.listbox = tk.Listbox(list_frame, width=36, height=14, exportselection=False)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=sb.set)
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)

        right = ttk.Frame(mid)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ctrl = ttk.Frame(right, padding=(0, 0, 0, 8))
        ctrl.pack(fill=tk.X)

        self.btn_prev_media = ttk.Button(ctrl, text="◀ media", command=self._prev_media)
        self.btn_prev_media.pack(side=tk.LEFT, padx=(0, 4))
        self.btn_next_media = ttk.Button(ctrl, text="media ▶", command=self._next_media)
        self.btn_next_media.pack(side=tk.LEFT, padx=(0, 12))

        self.btn_prev_frame = ttk.Button(ctrl, text="◀ frame", command=self._prev_frame)
        self.btn_prev_frame.pack(side=tk.LEFT, padx=(0, 4))
        self.btn_next_frame = ttk.Button(ctrl, text="frame ▶", command=self._next_frame)
        self.btn_next_frame.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_play = ttk.Button(ctrl, text="Play", command=self._toggle_play, state=tk.DISABLED)
        self.btn_play.pack(side=tk.LEFT, padx=(0, 12))

        self.btn_toggle = ttk.Button(ctrl, text="Before / After", command=self._toggle_before_after)
        self.btn_toggle.pack(side=tk.LEFT)
        self.chk_tracking = ttk.Checkbutton(
            ctrl,
            text="Tracking IDs",
            variable=self.use_tracking_ids,
            command=self._on_toggle_tracking_ids,
        )
        self.chk_tracking.pack(side=tk.LEFT, padx=(12, 0))

        self.preview_wrap = ttk.Frame(right, relief=tk.SUNKEN, borderwidth=1)
        self.preview_wrap.pack(fill=tk.BOTH, expand=True)
        # Canvas (not Label): image does not become the widget's minimum size, so fullscreen /
        # restore does not fight the geometry manager with huge PhotoImages.
        self.preview_canvas = tk.Canvas(
            self.preview_wrap,
            highlightthickness=0,
            borderwidth=0,
            background="#2e2e2e",
        )
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.preview_canvas.bind("<Configure>", self._on_preview_configure)

    def _on_preview_configure(self, event: tk.Event) -> None:
        if str(getattr(event, "widget", "")) != str(self.preview_canvas):
            return
        # Ignore transient tiny sizes during WM transitions.
        if event.width < 32 or event.height < 32:
            return
        # Real layout: allow mapped retries to use winfo from now on.
        self._preview_map_retry_count = 0
        if self._preview_resize_after_id is not None:
            try:
                self.root.after_cancel(self._preview_resize_after_id)
            except tk.TclError:
                pass
        self._preview_resize_after_id = self.root.after(
            _PREVIEW_RESIZE_DEBOUNCE_MS, self._finish_preview_resize
        )

    def _finish_preview_resize(self) -> None:
        self._preview_resize_after_id = None
        self._update_preview_image()

    def _preview_display_bounds(self) -> Tuple[int, int]:
        """Max width/height for the scaled PhotoImage (fits inside the canvas)."""
        try:
            self.root.update_idletasks()
        except tk.TclError:
            pass
        try:
            pw = int(self.preview_canvas.winfo_width())
            ph = int(self.preview_canvas.winfo_height())
        except tk.TclError:
            pw, ph = 1, 1
        pad = 2 * _PREVIEW_CANVAS_PAD
        if pw > pad + 1 and ph > pad + 1:
            mw = max(160, pw - pad)
            mh = max(120, ph - pad)
        else:
            mw, mh = _PREVIEW_FALLBACK_MAX_W, _PREVIEW_FALLBACK_MAX_H
        mw = min(mw, _PREVIEW_ABSOLUTE_CAP_W)
        mh = min(mh, _PREVIEW_ABSOLUTE_CAP_H)
        return mw, mh

    def _draw_canvas_placeholder(self, message: str) -> None:
        self.preview_canvas.delete("all")
        try:
            w = max(2, int(self.preview_canvas.winfo_width()))
            h = max(2, int(self.preview_canvas.winfo_height()))
        except tk.TclError:
            w, h = 400, 300
        self.preview_canvas.create_text(
            w // 2,
            h // 2,
            text=message,
            fill="#b0b0b0",
            width=max(80, w - 32),
        )

    def _cancel_preview_map_timer(self) -> None:
        if self._preview_map_after_id is not None:
            try:
                self.root.after_cancel(self._preview_map_after_id)
            except tk.TclError:
                pass
            self._preview_map_after_id = None

    def _reset_preview_map_state(self) -> None:
        self._cancel_preview_map_timer()
        self._preview_map_retry_count = 0

    def _retry_preview_after_map(self) -> None:
        self._preview_map_after_id = None
        self._update_preview_image()

    def _on_add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Add images or videos",
            filetypes=[
                (
                    "Media",
                    "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv *.webm *.m4v",
                ),
                ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Videos", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                ("All", "*.*"),
            ],
        )
        if not paths:
            return
        added: List[Path] = []
        for s in paths:
            p = Path(s).resolve()
            if not p.is_file():
                continue
            if p.suffix.lower() not in _MEDIA_EXTS:
                continue
            added.append(p)
        if not added:
            return
        self._append_paths(added)

    def _append_paths(self, new_paths: List[Path]) -> None:
        had_any = bool(self.paths)
        for p in new_paths:
            if p not in self.paths:
                self.paths.append(p)
        self._rebuild_listbox()
        if not self.paths:
            return
        if not had_any:
            self.media_index = 0
            self._load_current_media()
        elif self.media_index >= len(self.paths):
            self.media_index = len(self.paths) - 1
            self._load_current_media()
        else:
            self._refresh_status()
            self._update_preview_image()

    def _on_clear_queue(self) -> None:
        self.paths.clear()
        self.media_index = 0
        self._release_video()
        self._image_before = self._image_after = None
        self.listbox.delete(0, tk.END)
        self.status_var.set("Queue cleared. Add files…")
        self._set_frame_buttons_state(False)
        self._draw_canvas_placeholder("No file loaded")
        self._photo = None
        self.btn_play.configure(text="Play", state=tk.DISABLED)
        self.chk_tracking.configure(state=tk.DISABLED)

    def _on_close(self) -> None:
        self._reset_preview_map_state()
        if self._preview_resize_after_id is not None:
            try:
                self.root.after_cancel(self._preview_resize_after_id)
            except tk.TclError:
                pass
            self._preview_resize_after_id = None
        self._release_video()
        self.root.destroy()

    def _stop_playback(self) -> None:
        self._playing = False
        if self._play_after_id is not None:
            try:
                self.root.after_cancel(self._play_after_id)
            except tk.TclError:
                pass
            self._play_after_id = None
        if getattr(self, "btn_play", None) is not None:
            try:
                self.btn_play.configure(text="Play")
            except tk.TclError:
                pass

    def _rebuild_listbox(self) -> None:
        self.listbox.delete(0, tk.END)
        for p in self.paths:
            self.listbox.insert(tk.END, p.name)

    def _on_list_select(self, _event: Optional[tk.Event] = None) -> None:
        sel = self.listbox.curselection()
        if not sel or not self.paths:
            return
        i = int(sel[0])
        if i == self.media_index:
            return
        self.media_index = i
        self._load_current_media()

    def _release_video(self) -> None:
        self._stop_playback()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._video_cache.clear()
        self._reset_tracking_state()
        self._video_n_frames = 0

    def _reset_tracking_state(self) -> None:
        self._video_track_cache.clear()
        self._track_ready_upto = -1
        self._track_model = None

    def _load_current_media(self) -> None:
        self._release_video()
        self._image_before = self._image_after = None
        self.show_before = True
        self.video_frame_index = 0

        if not self.paths:
            return

        self.media_index = max(0, min(self.media_index, len(self.paths) - 1))
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(self.media_index)
        self.listbox.see(self.media_index)

        path = self.paths[self.media_index]
        try:
            if is_video(path):
                self._set_frame_buttons_state(True)
                self.chk_tracking.configure(state=tk.NORMAL)
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    self.status_var.set(f"Could not open video: {path.name}")
                    self._set_frame_buttons_state(False)
                    return
                self._cap = cap
                self._video_n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
                if fps < 1.0 or fps > 120.0:
                    fps = 24.0
                self._video_fps = fps
                self._ensure_video_pair(0)
            else:
                self._set_frame_buttons_state(False)
                self.chk_tracking.configure(state=tk.DISABLED)
                bgr = cv2.imread(str(path))
                if bgr is None:
                    self.status_var.set(f"Could not read image: {path.name}")
                    return
                after = annotate_frame(self.model, bgr, self.conf, self.imgsz)
                if bgr.shape != after.shape:
                    after = cv2.resize(after, (bgr.shape[1], bgr.shape[0]))
                self._image_before = bgr
                self._image_after = after
        except Exception as ex:  # noqa: BLE001
            self.status_var.set(f"Error loading {path.name}: {ex}")
            return

        self._refresh_status()
        self._update_preview_image()

    def _set_frame_buttons_state(self, video: bool) -> None:
        state = tk.NORMAL if video else tk.DISABLED
        self.btn_prev_frame.configure(state=state)
        self.btn_next_frame.configure(state=state)
        self.btn_play.configure(state=state)
        if not video:
            self._stop_playback()
            self.chk_tracking.configure(state=tk.DISABLED)

    def _ensure_video_pair(self, frame_index: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._cap is None:
            return None, None
        if frame_index in self._video_cache:
            return self._video_cache[frame_index]
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None, None
        after = annotate_frame(self.model, frame, self.conf, self.imgsz)
        if frame.shape != after.shape:
            after = cv2.resize(after, (frame.shape[1], frame.shape[0]))
        if len(self._video_cache) >= _MAX_VIDEO_CACHE_FRAMES:
            del self._video_cache[min(self._video_cache.keys())]
        self._video_cache[frame_index] = (frame, after)
        return frame, after

    def _get_track_model(self) -> YOLO:
        if self._track_model is None:
            self._track_model = YOLO(str(self.weights_path))
        return self._track_model

    def _ensure_video_tracked(self, frame_index: int) -> Optional[np.ndarray]:
        if frame_index in self._video_track_cache:
            return self._video_track_cache[frame_index]

        # If user seeks backward beyond prepared tracker history, rebuild from frame 0.
        if frame_index <= self._track_ready_upto:
            self._reset_tracking_state()

        track_model = self._get_track_model()
        start = self._track_ready_upto + 1
        for i in range(start, frame_index + 1):
            before, _ = self._ensure_video_pair(i)
            if before is None:
                return None
            results = track_model.track(
                source=before,
                conf=self.conf,
                imgsz=self.imgsz,
                tracker=self.tracker,
                persist=True,
                verbose=False,
            )
            tracked = results[0].plot()
            if tracked.shape != before.shape:
                tracked = cv2.resize(tracked, (before.shape[1], before.shape[0]))
            if len(self._video_track_cache) >= _MAX_VIDEO_CACHE_FRAMES:
                del self._video_track_cache[min(self._video_track_cache.keys())]
            self._video_track_cache[i] = tracked
            self._track_ready_upto = i
        return self._video_track_cache.get(frame_index)

    def _current_display_bgr(self) -> Optional[np.ndarray]:
        if not self.paths:
            return None
        path = self.paths[self.media_index]
        if is_video(path):
            before, det_after = self._ensure_video_pair(self.video_frame_index)
            if before is None or det_after is None:
                return None
            if self.show_before:
                return before
            if self.use_tracking_ids.get():
                tracked_after = self._ensure_video_tracked(self.video_frame_index)
                return tracked_after if tracked_after is not None else det_after
            return det_after
        if self._image_before is None or self._image_after is None:
            return None
        return self._image_before if self.show_before else self._image_after

    def _refresh_status(self) -> None:
        if not self.paths:
            self.status_var.set("Add images or videos to begin.")
            return
        path = self.paths[self.media_index]
        n = len(self.paths)
        pos = self.media_index + 1
        if is_video(path):
            total = self._video_n_frames if self._video_n_frames > 0 else "?"
            fr = self.video_frame_index + 1
            mode = "BEFORE" if self.show_before else "AFTER"
            tracking_tag = " | TRACK" if self.use_tracking_ids.get() else ""
            self.status_var.set(
                f"{mode}{tracking_tag}  |  {path.name}  |  media {pos}/{n}  |  frame {fr}/{total}"
            )
        else:
            mode = "BEFORE" if self.show_before else "AFTER"
            self.status_var.set(f"{mode}  |  {path.name}  |  media {pos}/{n}")

    def _update_preview_image(self) -> None:
        bgr = self._current_display_bgr()
        if bgr is None:
            self._reset_preview_map_state()
            self._photo = None
            if self.paths:
                self._draw_canvas_placeholder("Could not show frame")
            else:
                self._draw_canvas_placeholder("Add files…")
            return
        try:
            self.root.update_idletasks()
            pw = int(self.preview_canvas.winfo_width())
            ph = int(self.preview_canvas.winfo_height())
        except tk.TclError:
            pw, ph = 1, 1

        if pw > 1 and ph > 1:
            self._preview_map_retry_count = 0
            max_w, max_h = self._preview_display_bounds()
        else:
            self._preview_map_retry_count += 1
            if self._preview_map_retry_count > _PREVIEW_MAP_RETRY_MAX:
                self._reset_preview_map_state()
                max_w, max_h = _PREVIEW_FALLBACK_MAX_W, _PREVIEW_FALLBACK_MAX_H
            else:
                self._cancel_preview_map_timer()
                self._preview_map_after_id = self.root.after(16, self._retry_preview_after_map)
                return

        try:
            self._photo = bgr_to_photo(bgr, max_w, max_h)
        except Exception:
            self._draw_canvas_placeholder("Display error")
            self._photo = None
            return

        self.preview_canvas.delete("all")
        cx = max(1, pw // 2)
        cy = max(1, ph // 2)
        self.preview_canvas.create_image(cx, cy, image=self._photo, anchor=tk.CENTER)

    def _refresh_all(self) -> None:
        self._refresh_status()
        self._update_preview_image()

    def _toggle_before_after(self) -> None:
        if not self.paths:
            return
        self.show_before = not self.show_before
        self._refresh_all()

    def _on_toggle_tracking_ids(self) -> None:
        if not self.paths:
            self.use_tracking_ids.set(False)
            return
        path = self.paths[self.media_index]
        if not is_video(path):
            self.use_tracking_ids.set(False)
            return
        self._stop_playback()
        self._reset_tracking_state()
        self._refresh_all()

    def _toggle_play(self) -> None:
        if not self.paths or not is_video(self.paths[self.media_index]):
            return
        if self._playing:
            self._stop_playback()
            return
        self._playing = True
        self.btn_play.configure(text="Pause")
        self._schedule_play_tick()

    def _schedule_play_tick(self) -> None:
        if not self._playing:
            return
        self._play_after_id = self.root.after(1, self._play_tick)

    def _play_tick(self) -> None:
        self._play_after_id = None
        if not self._playing:
            return
        if not self.paths or not is_video(self.paths[self.media_index]):
            self._stop_playback()
            return

        t0 = time.monotonic()
        n = self._video_n_frames
        if n > 0 and self.video_frame_index >= n - 1:
            self._stop_playback()
            self._refresh_all()
            return

        before, after = self._ensure_video_pair(self.video_frame_index + 1)
        if before is None:
            self._stop_playback()
            self._refresh_all()
            return

        self.video_frame_index += 1
        self._refresh_status()
        self._update_preview_image()

        elapsed = time.monotonic() - t0
        gap = 1.0 / self._video_fps
        delay_ms = max(1, int((gap - elapsed) * 1000))
        self._play_after_id = self.root.after(delay_ms, self._play_tick)

    def _prev_media(self) -> None:
        if not self.paths:
            return
        if self.media_index <= 0:
            return
        self.media_index -= 1
        self._load_current_media()

    def _next_media(self) -> None:
        if not self.paths:
            return
        if self.media_index >= len(self.paths) - 1:
            return
        self.media_index += 1
        self._load_current_media()

    def _prev_frame(self) -> None:
        if not self.paths or not is_video(self.paths[self.media_index]):
            return
        self._stop_playback()
        if self.video_frame_index <= 0:
            return
        self.video_frame_index -= 1
        self.show_before = True
        self._refresh_all()

    def _next_frame(self) -> None:
        if not self.paths or not is_video(self.paths[self.media_index]):
            return
        self._stop_playback()
        before, after = self._ensure_video_pair(self.video_frame_index + 1)
        if before is None:
            return
        self.video_frame_index += 1
        self.show_before = True
        self._refresh_all()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tk window: add multiple media files, before/after, navigate media and frames."
    )
    parser.add_argument("--weights", type=str, default="yolov8n.pt")
    parser.add_argument(
        "--file",
        action="append",
        default=None,
        help="Pre-load file(s); can be passed multiple times",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    weights = resolve_weights(args.weights, project_root)

    initial: List[Path] = []
    if args.file:
        for f in args.file:
            p = Path(f).resolve()
            if p.is_file() and p.suffix.lower() in _MEDIA_EXTS:
                initial.append(p)
            else:
                print(f"Skip (missing or unsupported): {f}", file=sys.stderr)

    print("Loading model…")
    model = YOLO(str(weights))

    root = tk.Tk()
    app = UploadViewerApp(
        root,
        model,
        weights,
        args.tracker,
        args.conf,
        args.imgsz,
        initial_files=initial or None,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
