"""Desktop viewer: one window with Add files, playlist, before/after, and frame stepping.

Run from project root (venv active):

  python demo_upload_window.py --weights yolov8n.pt

- **Add files…** — pick multiple images/videos (you can add again anytime).
- **List** — click a row to jump to that item.
- **◀ media / media ▶** — previous / next file in the queue.
- **◀ frame / frame ▶** — previous / next frame (videos only).
- **Play / Pause** — auto-advance frames at the clip’s FPS (before or after view; **Space** still toggles).
- **Before / After** — toggle original vs YOLO overlay (or **Space**).

Keyboard: **Space** = before/after; **P** = play/pause (video); **← / →** = frame (video); **[** / **]** = prev/next media.

Video uses per-frame detection (cached, bounded). For full-sequence tracking use
``python src/inference.py ... --track``.

Display uses Tk + Pillow, so **opencv-python-headless** is fine (no separate OpenCV window).
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

# Preview max size (widget grows with window; we scale image to fit)
_PREVIEW_MAX_W = 960
_PREVIEW_MAX_H = 720


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
        conf: float,
        imgsz: int,
        initial_files: Optional[List[Path]] = None,
    ) -> None:
        self.root = root
        self.model = model
        self.conf = conf
        self.imgsz = imgsz

        self.paths: List[Path] = []
        self.media_index: int = 0
        self.show_before: bool = True
        self.video_frame_index: int = 0
        self._cap: Optional[cv2.VideoCapture] = None
        self._video_n_frames: int = 0
        self._video_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        self._image_before: Optional[np.ndarray] = None
        self._image_after: Optional[np.ndarray] = None

        self._photo: Optional[ImageTk.PhotoImage] = None
        self._preview_max_w = _PREVIEW_MAX_W
        self._preview_max_h = _PREVIEW_MAX_H

        self._playing: bool = False
        self._play_after_id: Optional[str] = None
        self._video_fps: float = 24.0

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

        preview_wrap = ttk.Frame(right, relief=tk.SUNKEN, borderwidth=1)
        preview_wrap.pack(fill=tk.BOTH, expand=True)
        self.preview_label = ttk.Label(preview_wrap, anchor=tk.CENTER, text="No file loaded")
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        preview_wrap.bind("<Configure>", self._on_preview_configure)

    def _on_preview_configure(self, event: tk.Event) -> None:
        # inner padding ~8
        self._preview_max_w = max(320, event.width - 16)
        self._preview_max_h = max(240, event.height - 16)
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
        self.preview_label.configure(image="", text="No file loaded")
        self._photo = None
        self.btn_play.configure(text="Play", state=tk.DISABLED)

    def _on_close(self) -> None:
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
        self._video_n_frames = 0

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

    def _current_display_bgr(self) -> Optional[np.ndarray]:
        if not self.paths:
            return None
        path = self.paths[self.media_index]
        if is_video(path):
            before, after = self._ensure_video_pair(self.video_frame_index)
            if before is None or after is None:
                return None
            return before if self.show_before else after
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
            self.status_var.set(
                f"{mode}  |  {path.name}  |  media {pos}/{n}  |  frame {fr}/{total}"
            )
        else:
            mode = "BEFORE" if self.show_before else "AFTER"
            self.status_var.set(f"{mode}  |  {path.name}  |  media {pos}/{n}")

    def _update_preview_image(self) -> None:
        bgr = self._current_display_bgr()
        if bgr is None:
            if self.paths:
                self.preview_label.configure(image="", text="Could not show frame")
            self._photo = None
            return
        try:
            self._photo = bgr_to_photo(bgr, self._preview_max_w, self._preview_max_h)
        except Exception:
            self.preview_label.configure(image="", text="Display error")
            self._photo = None
            return
        self.preview_label.configure(image=self._photo, text="")

    def _refresh_all(self) -> None:
        self._refresh_status()
        self._update_preview_image()

    def _toggle_before_after(self) -> None:
        if not self.paths:
            return
        self.show_before = not self.show_before
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
    app = UploadViewerApp(root, model, args.conf, args.imgsz, initial_files=initial or None)
    root.mainloop()


if __name__ == "__main__":
    main()
