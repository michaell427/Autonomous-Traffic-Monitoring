"""Multi-object tracking via Ultralytics (ByteTrack / BoT-SORT).

Call :func:`run_tracking` or use the CLI::

    python src/inference.py --weights ... --source video.mp4 --track

For custom tracker YAML, pass ``--tracker path/to.yaml`` (see Ultralytics tracker configs).
"""

from pathlib import Path
from typing import Optional, Union

from ..inference import run_inference, resolve_source

Source = Union[str, int, Path]


def run_tracking(
    weights: Path,
    source: str,
    output_parent: Path,
    name: str = "track",
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    device: Optional[str] = None,
    half: bool = False,
    show: bool = False,
    save_txt: bool = False,
    stream: bool = False,
    tracker: str = "bytetrack.yaml",
) -> Path:
    """Run YOLO + multi-object tracker on a frame sequence (video, webcam, or image folder)."""
    return run_inference(
        weights=weights,
        source=resolve_source(source),
        output_parent=output_parent,
        name=name,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        half=half,
        show=show,
        save_txt=save_txt,
        stream=stream,
        track=True,
        tracker=tracker,
    )
