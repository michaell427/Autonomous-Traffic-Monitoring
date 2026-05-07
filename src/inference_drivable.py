"""Run drivable segmentation on images or folders; save masks and optional overlay.

Writes a BDD-style colored ``*_mask.png`` (easy to compare to GT color_labels).
Also writes ``*_mask_ids.png`` with raw class indices 0/1/2 (looks nearly black
in viewers — use for programmatic use). Predictions are mapped back to the
original resolution by inverting the same letterbox padding used in training.
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import argparse
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from src.data.augmentation import drivable_letterbox_params, get_segmentation_augmentation
from src.data.drivable_dataset import DRIVABLE_BGR_TO_CLASS
from src.models.train_drivable_seg import _build_deeplabv3


def _default_colors_bgr(num_classes: int) -> np.ndarray:
    """BGR colors for visualization (same order as DRIVABLE_BGR_TO_CLASS classes 0,1,2)."""
    palette = np.zeros((max(num_classes, 3), 3), dtype=np.uint8)
    for bgr, c in DRIVABLE_BGR_TO_CLASS.items():
        if c < num_classes:
            palette[c] = np.array(bgr, dtype=np.uint8)
    return palette


@torch.no_grad()
def predict_mask(
    model: nn.Module,
    image_rgb: np.ndarray,
    image_size: Tuple[int, int],
    device: torch.device,
) -> np.ndarray:
    """Class map H×W aligned with ``image_rgb`` (inverse of letterbox val transform)."""
    h, w = image_rgb.shape[:2]
    new_h, new_w, pad_top, pad_left = drivable_letterbox_params(h, w, image_size)

    tf = get_segmentation_augmentation(image_size=image_size, training=False)
    t = tf(image=image_rgb)
    batch = t["image"].unsqueeze(0).to(device)
    out = model(batch)
    logits = out["out"] if isinstance(out, dict) else out
    logits = logits[0].cpu()
    pred_sq = logits.argmax(dim=0).numpy().astype(np.uint8)
    cropped = pred_sq[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser(description="Drivable segmentation inference")
    parser.add_argument("--weights", type=str, required=True, help="best.pth from train_drivable_seg")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image file, or directory of .jpg",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/drivable_inference",
    )
    parser.add_argument("--name", type=str, default="run")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Also save a semi-transparent BGR overlay PNG",
    )
    parser.add_argument("--alpha", type=float, default=0.35)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    try:
        ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.weights, map_location=device)
    cfg = ckpt["config"]
    num_classes = cfg["num_classes"]
    image_size = tuple(cfg["image_size"])

    model = _build_deeplabv3(num_classes, pretrained=False)
    state = ckpt["model_state"]
    try:
        model.load_state_dict(state)
    except RuntimeError as ex:
        msg = str(ex)
        if "Unexpected key(s) in state_dict" in msg and "aux_classifier" in msg:
            model.load_state_dict(state, strict=False)
        else:
            raise
    model.to(device)
    model.eval()

    src = Path(args.source)
    if src.is_dir():
        paths: List[Path] = sorted(src.glob("*.jpg")) + sorted(src.glob("*.jpeg"))
    else:
        paths = [src]

    out_root = Path(args.output_dir) / args.name
    out_root.mkdir(parents=True, exist_ok=True)
    palette = _default_colors_bgr(num_classes)

    for p in tqdm(paths, desc="drivable"):
        im = Image.open(p).convert("RGB")
        rgb = np.array(im)
        mask = predict_mask(model, rgb, image_size, device)
        color = palette[mask]
        cv2.imwrite(str(out_root / f"{p.stem}_mask.png"), color)
        cv2.imwrite(str(out_root / f"{p.stem}_mask_ids.png"), mask)

        if args.overlay:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            blend = (args.alpha * color.astype(np.float32) + (1 - args.alpha) * bgr.astype(np.float32))
            blend = np.clip(blend, 0, 255).astype(np.uint8)
            cv2.imwrite(str(out_root / f"{p.stem}_overlay.png"), blend)


if __name__ == "__main__":
    main()
