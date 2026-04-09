"""Run drivable segmentation on images or folders; save class-ID PNGs and optional color overlay."""

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

from src.data.augmentation import get_segmentation_augmentation
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
    tf = get_segmentation_augmentation(image_size=image_size, training=False)
    t = tf(image=image_rgb)
    batch = t["image"].unsqueeze(0).to(device)
    out = model(batch)
    logits = out["out"] if isinstance(out, dict) else out
    logits = logits[0].cpu()
    h, w = image_rgb.shape[:2]
    up = torch.nn.functional.interpolate(
        logits.unsqueeze(0).float(),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )[0]
    return up.argmax(dim=0).numpy().astype(np.uint8)


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
    model.load_state_dict(ckpt["model_state"])
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
        mask_path = out_root / f"{p.stem}_mask.png"
        cv2.imwrite(str(mask_path), mask)

        if args.overlay:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            color = palette[mask]
            blend = (args.alpha * color.astype(np.float32) + (1 - args.alpha) * bgr.astype(np.float32))
            blend = np.clip(blend, 0, 255).astype(np.uint8)
            cv2.imwrite(str(out_root / f"{p.stem}_overlay.png"), blend)


if __name__ == "__main__":
    main()
