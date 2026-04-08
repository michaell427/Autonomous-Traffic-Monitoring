"""Toggle before/after detection in one OpenCV window.

- **Space** — switch between original frame and YOLO overlay (same image).
- **Right arrow** — next image.
- **Left arrow** — previous image (stops at the first image).
- **Q** or **Esc** — quit.

Requires a display **and** a full OpenCV build with HighGUI (not headless). From project root:

  python demo_before_after.py --weights yolov8n.pt --source bdd100k_yolo_format/val/images

If you see "function is not implemented" for cvNamedWindow, use **opencv-python** (not
**opencv-python-headless**), or run with **--save-dir** to write images instead of a window:

  pip uninstall opencv-python-headless -y
  pip install "opencv-python>=4.8.0"

  python demo_before_after.py --weights yolov8n.pt --source bdd100k_yolo_format/val/images --save-dir outputs/demo_before_after
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Arrow codes differ by OS / OpenCV backend; accept several.
_KEY_LEFT = frozenset({65361, 2424832, 81, 63234})
_KEY_RIGHT = frozenset({65363, 2555904, 83, 63235})


def _wait_key() -> int:
    """Read one key (including arrow keys on Windows via waitKeyEx)."""
    if hasattr(cv2, "waitKeyEx"):
        k = cv2.waitKeyEx(0)
    else:
        k = cv2.waitKey(0)
    return int(k) if k != -1 else -1


def collect_images(source: Path, limit: Optional[int]) -> List[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Not a supported image file: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Not a file or directory: {source}")
    paths = sorted(
        p
        for p in source.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {source}")
    if limit is not None:
        paths = paths[:limit]
    return paths


def opencv_gui_available() -> bool:
    try:
        cv2.namedWindow("__cv_gui_probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__cv_gui_probe__")
        return True
    except cv2.error:
        return False


def annotate_frame(model: YOLO, bgr: np.ndarray, conf: float, imgsz: int) -> np.ndarray:
    results = model.predict(
        source=bgr,
        conf=conf,
        imgsz=imgsz,
        verbose=False,
    )
    return results[0].plot()


def run_save_mode(
    paths: List[Path],
    model: YOLO,
    out_dir: Path,
    conf: float,
    imgsz: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(paths):
        before = cv2.imread(str(path))
        if before is None:
            print(f"Skip (unreadable): {path}")
            continue
        after = annotate_frame(model, before, conf, imgsz)
        if before.shape != after.shape:
            after = cv2.resize(after, (before.shape[1], before.shape[0]))
        stem = f"{i:04d}_{path.stem}"
        cv2.imwrite(str(out_dir / f"{stem}_before.jpg"), before)
        cv2.imwrite(str(out_dir / f"{stem}_after.jpg"), after)
        pair = np.hstack([before, after])
        cv2.putText(
            pair,
            "BEFORE",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            pair,
            "AFTER",
            (before.shape[1] + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(out_dir / f"{stem}_pair.jpg"), pair)
        print(f"Wrote {stem}_*.jpg")
    print(f"Done. Output directory: {out_dir}")


def run_gui_mode(
    paths: List[Path],
    model: YOLO,
    conf: float,
    imgsz: int,
) -> None:
    win = "Before / After (Space toggles)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    idx = 0
    show_before = True

    while 0 <= idx < len(paths):
        path = paths[idx]
        before = cv2.imread(str(path))
        if before is None:
            print(f"Skip (unreadable): {path}")
            idx += 1
            show_before = True
            continue

        after = annotate_frame(model, before, conf, imgsz)
        if before.shape != after.shape:
            after = cv2.resize(after, (before.shape[1], before.shape[0]))

        while True:
            vis = before.copy() if show_before else after.copy()
            mode = "BEFORE (raw)" if show_before else "AFTER (detections)"
            bar = f"{mode}  |  {idx + 1}/{len(paths)}  {path.name}"
            cv2.putText(
                vis,
                bar,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if show_before else (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                "Space: toggle   Left/Right: prev/next   Q/Esc: quit",
                (10, vis.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(win, vis)
            key = _wait_key()
            key_lo = key & 0xFF

            if key_lo in (ord("q"), ord("Q"), 27) or key == 27:
                cv2.destroyAllWindows()
                return
            if key in _KEY_RIGHT or key_lo in _KEY_RIGHT:
                idx += 1
                show_before = True
                break
            if key in _KEY_LEFT or key_lo in _KEY_LEFT:
                if idx > 0:
                    idx -= 1
                    show_before = True
                    break
                continue
            if key_lo == 32 or key == 32:  # Space
                show_before = not show_before

    cv2.destroyAllWindows()
    print("End of list.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Before/after demo: Space toggles, arrows change image, Q quits."
    )
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt weights")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image file or directory of images",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max images when source is a directory (default: all)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="If set, skip GUI and write before/after/pair JPEGs here (works with opencv-python-headless)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    weights = Path(args.weights)
    if not weights.is_file():
        weights = (root / args.weights).resolve()
    if not weights.is_file():
        sys.exit(f"Weights not found: {args.weights}")

    source = Path(args.source)
    if not source.is_absolute():
        source = (root / source).resolve()
    paths = collect_images(source, args.limit)

    print(f"Loaded {len(paths)} image(s).")
    model = YOLO(str(weights))

    if args.save_dir:
        out = Path(args.save_dir)
        if not out.is_absolute():
            out = (root / out).resolve()
        run_save_mode(paths, model, out, args.conf, args.imgsz)
        return

    if not opencv_gui_available():
        print(
            "\nOpenCV has no GUI support in this environment (common if "
            "'opencv-python-headless' is installed instead of 'opencv-python').\n\n"
            "Fix (recommended):\n"
            "  pip uninstall opencv-python-headless -y\n"
            "  pip install \"opencv-python>=4.8.0\"\n\n"
            "Or skip the window and export images:\n"
            "  python demo_before_after.py --weights ... --source ... --save-dir outputs/demo_before_after\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Keys: Space = toggle | Left/Right = prev/next image | Q/Esc = quit")
    run_gui_mode(paths, model, args.conf, args.imgsz)


if __name__ == "__main__":
    main()
