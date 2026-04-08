"""Local upload demo (image/video) with before/after outputs.

Run:
  python app_upload_before_after.py --weights yolov8n.pt

Then open the printed local URL in your browser.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from uuid import uuid4

import gradio as gr
import numpy as np
from ultralytics import YOLO

from src.inference import run_inference

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def resolve_weights(raw: str, project_root: Path) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = (project_root / p).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Weights not found: {p}")
    return p


def find_annotated_video(out_dir: Path) -> Path:
    candidates = []
    for p in out_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No annotated video file found in {out_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def build_app(default_weights: Path, output_dir: Path) -> gr.Blocks:
    model = YOLO(str(default_weights))
    output_dir.mkdir(parents=True, exist_ok=True)

    def run_image(
        image: np.ndarray | None,
        conf: float,
        imgsz: int,
    ):
        if image is None:
            return None, None, "Upload an image first."
        results = model.predict(source=image, conf=conf, imgsz=imgsz, verbose=False)
        after = results[0].plot()
        return image, after, "Done."

    def run_video(
        video_path: str | None,
        conf: float,
        imgsz: int,
        track: bool,
        tracker: str,
    ):
        if not video_path:
            return None, None, "Upload a video first."

        src = Path(video_path).resolve()
        run_name = f"upload_{uuid4().hex[:8]}"
        out_dir = run_inference(
            weights=default_weights,
            source=src,
            output_parent=output_dir,
            name=run_name,
            imgsz=imgsz,
            conf=conf,
            track=track,
            tracker=tracker,
            stream=True,
        )
        annotated = find_annotated_video(out_dir)

        # Copy original into run folder so both before/after are persisted together.
        before_copy = out_dir / f"before{src.suffix.lower()}"
        if not before_copy.exists():
            shutil.copy2(src, before_copy)

        mode = "tracking" if track else "detection"
        msg = f"Done ({mode}). Saved under: {out_dir}"
        return str(before_copy), str(annotated), msg

    with gr.Blocks(title="Traffic Monitoring Upload Demo") as demo:
        gr.Markdown(
            "## Upload Before/After Demo\n"
            "- Upload an **image** or **video**.\n"
            "- The app runs YOLO with your selected weights.\n"
            "- For video, optionally enable tracking (ByteTrack by default)."
        )
        gr.Markdown(f"**Weights:** `{default_weights}`")

        with gr.Tab("Image"):
            with gr.Row():
                inp_img = gr.Image(type="numpy", label="Upload image")
            with gr.Row():
                conf_img = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence")
                imgsz_img = gr.Slider(320, 1280, value=640, step=32, label="Image size")
            run_img_btn = gr.Button("Run image annotation")
            img_status = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                out_before_img = gr.Image(label="Before")
                out_after_img = gr.Image(label="After")
            run_img_btn.click(
                fn=run_image,
                inputs=[inp_img, conf_img, imgsz_img],
                outputs=[out_before_img, out_after_img, img_status],
            )

        with gr.Tab("Video"):
            vid_in = gr.Video(label="Upload video")
            with gr.Row():
                conf_vid = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence")
                imgsz_vid = gr.Slider(320, 1280, value=640, step=32, label="Image size")
            with gr.Row():
                track = gr.Checkbox(value=False, label="Enable tracking")
                tracker = gr.Dropdown(
                    choices=["bytetrack.yaml", "botsort.yaml"],
                    value="bytetrack.yaml",
                    label="Tracker config",
                )
            run_vid_btn = gr.Button("Run video annotation")
            vid_status = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                out_before_vid = gr.Video(label="Before")
                out_after_vid = gr.Video(label="After")
            run_vid_btn.click(
                fn=run_video,
                inputs=[vid_in, conf_vid, imgsz_vid, track, tracker],
                outputs=[out_before_vid, out_after_vid, vid_status],
            )

        gr.Markdown(
            "Outputs are saved under `outputs/upload_app/` by default. "
            "Use your trained checkpoint with `--weights outputs/yolo_training/<run>/weights/best.pt`."
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch local upload before/after demo.")
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO .pt weights",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/upload_app",
        help="Directory for generated outputs",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    weights = resolve_weights(args.weights, project_root)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()

    app = build_app(weights, out_dir)
    app.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
