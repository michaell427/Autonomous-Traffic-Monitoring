"""Train semantic drivable segmentation (BDD100K color_labels) with DeepLabV3-ResNet50."""

from __future__ import annotations

import sys
from pathlib import Path

# Project root on path (same pattern as train_yolo.py) so `python src/models/...` works
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from src.data.drivable_dataset import build_drivable_dataloaders
from src.utils.config import load_config


def _build_deeplabv3(num_classes: int, pretrained: bool) -> nn.Module:
    from torchvision.models.segmentation import deeplabv3_resnet50

    if pretrained:
        try:
            from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

            weights = DeepLabV3_ResNet50_Weights.DEFAULT
            model = deeplabv3_resnet50(weights=weights)
        except Exception:
            model = deeplabv3_resnet50(weights="DEFAULT")
    else:
        model = deeplabv3_resnet50(weights=None)

    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(
            256, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
    return model


@torch.no_grad()
def _accumulate_iou(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    inter: torch.Tensor,
    union: torch.Tensor,
) -> None:
    pred = logits.argmax(dim=1)
    valid = target != ignore_index
    pred = pred[valid]
    tgt = target[valid]
    for c in range(num_classes):
        pc = pred == c
        tc = tgt == c
        inter[c] += (pc & tc).sum().item()
        union[c] += (pc | tc).sum().item()


def _miou_from_counts(inter: torch.Tensor, union: torch.Tensor) -> Tuple[float, torch.Tensor]:
    iou = inter / (union + 1e-6)
    valid = union > 0
    if valid.any():
        m = iou[valid].mean().item()
    else:
        m = 0.0
    return m, iou


def _forward_loss(
    model: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    criterion: nn.Module,
    aux_w: float,
) -> torch.Tensor:
    out = model(images)
    if isinstance(out, dict):
        main = out["out"]
        loss = criterion(main, masks)
        aux = out.get("aux")
        if aux is not None and aux_w > 0:
            aux_tgt = F.interpolate(
                masks.unsqueeze(1).float(),
                size=aux.shape[2:],
                mode="nearest",
            ).squeeze(1).long()
            loss = loss + aux_w * criterion(aux, aux_tgt)
        return loss
    return criterion(out, masks)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler],
    aux_w: float,
    log_interval: int,
) -> float:
    model.train()
    total = 0.0
    n = 0
    t0 = time.perf_counter()
    for i, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = _forward_loss(model, images, masks, criterion, aux_w)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = _forward_loss(model, images, masks, criterion, aux_w)
            loss.backward()
            optimizer.step()

        total += loss.item()
        n += 1
        if log_interval and (i + 1) % log_interval == 0:
            elapsed = time.perf_counter() - t0
            sec_per_step = elapsed / (i + 1)
            tqdm.write(
                f"  step {i + 1}  loss {total / n:.4f}  "
                f"elapsed {elapsed:.1f}s  {sec_per_step:.2f}s/step"
            )
    return total / max(n, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
) -> Tuple[float, float, torch.Tensor]:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss_sum = 0.0
    n_batches = 0
    inter = torch.zeros(num_classes)
    union = torch.zeros(num_classes)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        out = model(images)
        logits = out["out"] if isinstance(out, dict) else out
        loss_sum += criterion(logits, masks).item()
        n_batches += 1
        _accumulate_iou(logits, masks, num_classes, ignore_index, inter, union)

    miou, per_iou = _miou_from_counts(inter, union)
    return loss_sum / max(n_batches, 1), miou, per_iou


def _append_experiment_log(
    repo_root: Path,
    row: Dict[str, object],
) -> None:
    path = repo_root / "docs" / "experiment_log.md"
    if not path.exists():
        return
    line = (
        f"| {row.get('date', '')} | {row.get('run', '')} | {row.get('miou', '')} | "
        f"{row.get('checkpoint', '')} | {row.get('notes', '')} |\n"
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train drivable segmentation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/drivable_seg_config.yaml",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument(
        "--log-experiment",
        action="store_true",
        help="Append one markdown table row to docs/experiment_log.md",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    config = load_config(str(cfg_path))
    paths_cfg = config["paths"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    out_cfg = config["output"]
    classes_cfg = config["classes"]

    root = Path(paths_cfg["root_dir"]).resolve()
    if args.epochs is not None:
        train_cfg = {**train_cfg, "epochs": args.epochs}
    if args.batch is not None:
        data_cfg = {**data_cfg, "batch_size": args.batch}
    if args.lr is not None:
        train_cfg = {**train_cfg, "lr": args.lr}

    project = args.project or out_cfg["project"]
    run_name = args.name or out_cfg["run_name"]
    out_dir = (root / project / run_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "paths": paths_cfg,
                "data": data_cfg,
                "model": model_cfg,
                "training": train_cfg,
                "classes": classes_cfg,
            },
            f,
            indent=2,
        )

    device_s = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)

    try:
        train_loader, val_loader = build_drivable_dataloaders(
            {**config, "data": data_cfg},
            device_hint=device_s,
        )
    except RuntimeError as e:
        raise SystemExit(str(e)) from e

    num_classes = classes_cfg["num_classes"]
    ignore_index = classes_cfg["ignore_index"]
    model = _build_deeplabv3(num_classes, pretrained=model_cfg.get("pretrained", True))
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    aux_w = train_cfg.get("aux_loss_weight", 0.4)
    if model.aux_classifier is None:
        aux_w = 0.0

    use_amp = train_cfg.get("amp", True) and device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None
    log_int = train_cfg.get("log_interval", 50)
    es_cfg = train_cfg.get("early_stopping", {})
    es_enabled = bool(es_cfg.get("enabled", False))
    es_patience = int(es_cfg.get("patience", 3))
    es_min_delta = float(es_cfg.get("min_delta", 0.0))
    es_monitor = str(es_cfg.get("monitor", "val_loss")).strip().lower()
    if es_monitor not in {"val_loss", "val_miou"}:
        raise SystemExit(
            f"Unsupported early_stopping.monitor={es_monitor!r}. "
            "Use 'val_loss' or 'val_miou'."
        )
    monitor_best = float("inf") if es_monitor == "val_loss" else float("-inf")
    bad_epochs = 0

    best_miou = -1.0
    best_path = out_dir / "best.pth"

    for epoch in range(1, train_cfg["epochs"] + 1):
        tqdm.write(f"\nEpoch {epoch}/{train_cfg['epochs']}")
        t_train = time.perf_counter()
        tr_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            aux_w,
            log_int,
        )
        train_secs = time.perf_counter() - t_train
        t_val = time.perf_counter()
        va_loss, miou, per_iou = validate(
            model,
            val_loader,
            device,
            num_classes,
            ignore_index,
        )
        val_secs = time.perf_counter() - t_val
        tqdm.write(
            f"  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_mIoU={miou:.4f}"
        )
        tqdm.write(
            f"  epoch time: train {train_secs:.1f}s  val {val_secs:.1f}s  "
            f"total {train_secs + val_secs:.1f}s"
        )
        tqdm.write(
            "  per-class IoU: "
            + ", ".join(f"{classes_cfg['names'][i]}={per_iou[i]:.4f}" for i in range(num_classes))
        )

        if miou > best_miou:
            best_miou = miou
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "num_classes": num_classes,
                        "ignore_index": ignore_index,
                        "image_size": data_cfg["image_size"],
                        "class_names": classes_cfg["names"],
                    },
                    "miou": miou,
                    "epoch": epoch,
                },
                best_path,
            )
            tqdm.write(f"  saved best → {best_path}")

        if es_enabled:
            monitor_now = va_loss if es_monitor == "val_loss" else miou
            if es_monitor == "val_loss":
                improved = monitor_now < (monitor_best - es_min_delta)
            else:
                improved = monitor_now > (monitor_best + es_min_delta)
            if improved:
                monitor_best = monitor_now
                bad_epochs = 0
            else:
                bad_epochs += 1
                tqdm.write(
                    "  early-stop monitor "
                    f"({es_monitor}) no significant improvement: "
                    f"{bad_epochs}/{es_patience} bad epochs"
                )
                if bad_epochs >= es_patience:
                    tqdm.write(
                        "  early stopping triggered on "
                        f"{es_monitor} (best={monitor_best:.4f}, "
                        f"min_delta={es_min_delta})"
                    )
                    break

    tqdm.write(f"\nDone. Best val mIoU={best_miou:.4f}  checkpoint={best_path}")

    if args.log_experiment:
        _append_experiment_log(
            root,
            {
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "run": run_name,
                "miou": f"{best_miou:.4f}",
                "checkpoint": str(best_path.relative_to(root)),
                "notes": f"drivable_seg {cfg_path.name}",
            },
        )


if __name__ == "__main__":
    main()
