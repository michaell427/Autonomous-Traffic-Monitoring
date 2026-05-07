"""Recompute validation loss / mIoU for a drivable checkpoint (no training)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

import torch

from src.data.drivable_dataset import build_drivable_dataloaders
from src.models.train_drivable_seg import _build_deeplabv3, validate
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate drivable segmentation on val split")
    parser.add_argument("--config", type=str, default="configs/drivable_seg_config.yaml")
    parser.add_argument("--weights", type=str, required=True, help="best.pth from training")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    classes_cfg = config["classes"]
    num_classes = classes_cfg["num_classes"]
    ignore_index = classes_cfg["ignore_index"]
    names = classes_cfg["names"]

    device_s = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)

    print("Building validation DataLoader…", flush=True)
    _, val_loader = build_drivable_dataloaders(config, device_hint=device_s)
    if len(val_loader.dataset) == 0:
        raise SystemExit("Validation dataset is empty. Check configs/drivable_seg_config.yaml paths.")
    print(
        f"Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches "
        f"(batch_size={config['data']['batch_size']})",
        flush=True,
    )
    print("(Forward-only; no backward — roughly one training epoch’s *val* phase, not full train.)", flush=True)

    try:
        ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.weights, map_location=device)

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

    print("Running validation…", flush=True)
    va_loss, miou, per_iou, gt_px = validate(
        model, val_loader, device, num_classes, ignore_index, progress=True
    )

    print(f"val_loss={va_loss:.4f}")
    print(f"val_mIoU (mean over classes present in GT)={miou:.4f}")
    for i in range(num_classes):
        print(f"  IoU {names[i]}={per_iou[i]:.4f}  GT_pixels={int(gt_px[i])}")
    n_present = int((gt_px > 0).sum().item())
    print(f"classes with GT pixels: {n_present}/{num_classes}")
    if ckpt.get("miou") is not None:
        print(f"(checkpoint recorded miou={ckpt['miou']!r} at epoch {ckpt.get('epoch')!r}; "
              "older runs may have used a looser mIoU definition.)")


if __name__ == "__main__":
    main()
