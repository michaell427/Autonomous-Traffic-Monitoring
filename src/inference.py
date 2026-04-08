"""YOLO detection inference on images, video folders, video files, or webcam.

With ``--track``, runs multi-object tracking (default ByteTrack via Ultralytics).
Tracking needs a *sequence* of frames: a video file, webcam, or a folder of images
processed in order—not a single still (a single image still works but IDs are not meaningful).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Union

from ultralytics import YOLO

Source = Union[str, int, Path]


def resolve_source(raw: str) -> Source:
    s = raw.strip()
    if s.isdigit():
        return int(s)
    p = Path(s)
    if p.exists():
        return p.resolve()
    return s


def run_inference(
    weights: Path,
    source: Source,
    output_parent: Path,
    name: str,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    device: Optional[str] = None,
    half: bool = False,
    show: bool = False,
    save_txt: bool = False,
    stream: bool = False,
    track: bool = False,
    tracker: str = "bytetrack.yaml",
) -> Path:
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    output_parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))

    kwargs = dict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        save=True,
        save_txt=save_txt,
        project=str(output_parent),
        name=name,
        exist_ok=True,
        verbose=True,
        stream=stream,
        half=half,
    )
    if device:
        kwargs["device"] = device
    if show:
        kwargs["show"] = True

    if track:
        kwargs["persist"] = True
        kwargs["tracker"] = tracker
        results = model.track(**kwargs)
    else:
        results = model.predict(**kwargs)

    if stream:
        for _ in results:
            pass

    out_dir = output_parent / name
    mode = "tracking" if track else "detection"
    print(f"Done ({mode}). Outputs: {out_dir}")
    return out_dir


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO on image, video, folder, or webcam. "
            "Use --track for multi-object tracking on frame sequences (video/webcam/folder)."
        )
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to .pt checkpoint (e.g. outputs/yolo_training/<run>/weights/best.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image path, video path, directory of images/videos, or 0 for default webcam",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/inference",
        help="Directory where Ultralytics will create the run folder",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="predict",
        help="Run name (subfolder under --output-dir)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda, cpu, or device id (default: Ultralytics auto)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="FP16 inference on CUDA (faster, slightly less precise)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display window (local GUI only; fails on headless servers)",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save one YOLO-format txt per image (labels dir)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream frames for video (lower memory on long clips)",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Multi-object tracking (ByteTrack by default). Use video, webcam, or image folder.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Ultralytics tracker config (e.g. bytetrack.yaml, botsort.yaml) or path to custom YAML",
    )

    args = parser.parse_args(argv)
    weights = Path(args.weights).resolve()
    output_parent = Path(args.output_dir).resolve()
    source = resolve_source(args.source)

    run_inference(
        weights=weights,
        source=source,
        output_parent=output_parent,
        name=args.name,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        half=args.half,
        show=args.show,
        save_txt=args.save_txt,
        stream=args.stream,
        track=args.track,
        tracker=args.tracker,
    )


if __name__ == "__main__":
    main()
