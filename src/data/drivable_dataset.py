"""BDD100K drivable-area dataset: RGB images + color PNG masks → class indices.

Official label colors (RGB) are defined in bdd100k/bdd100k ``label/label.py`` (``drivables``).
PNG files under ``color_labels/{split}/`` are read with OpenCV (BGR). BDD100K releases often
name masks ``<image_stem>_drivable_color.png`` (we also accept plain ``<image_stem>.png``).
Unknown pixels map to ``ignore_index`` (default 255) for cross-entropy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.augmentation import get_segmentation_augmentation

# BGR tuples as returned by cv2.imread; corresponds to RGB in bdd100k label.py drivables
DRIVABLE_BGR_TO_CLASS: Dict[Tuple[int, int, int], int] = {
    (86, 94, 219): 0,  # direct   — RGB (219, 94, 86)
    (219, 211, 86): 1,  # alternative — RGB (86, 211, 219)
    (0, 0, 0): 2,  # background
}


def color_drivable_mask_to_classes(
    bgr: np.ndarray, ignore_index: int = 255
) -> np.ndarray:
    """Map H×W×3 BDD drivable color PNG to H×W int64 class indices."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 BGR image, got shape {bgr.shape}")
    h, w = bgr.shape[:2]
    flat = bgr.reshape(-1, 3)
    out = np.full(flat.shape[0], ignore_index, dtype=np.int64)
    for bgr_key, cls in DRIVABLE_BGR_TO_CLASS.items():
        match = np.all(flat == np.array(bgr_key, dtype=np.uint8), axis=1)
        out[match] = cls
    return out.reshape(h, w)


def resolve_drivable_mask_path(msk_root: Path, image_stem: str) -> Optional[Path]:
    """Return mask path for a JPEG stem, or None if neither naming convention exists."""
    for name in (f"{image_stem}.png", f"{image_stem}_drivable_color.png"):
        p = msk_root / name
        if p.is_file():
            return p
    return None


def collect_drivable_pairs(
    images_dir: Path,
    masks_dir: Path,
    split: str,
) -> List[Tuple[Path, Path]]:
    """All (image, mask) pairs for a split."""
    img_root = Path(images_dir) / split
    msk_root = Path(masks_dir) / split
    pairs: List[Tuple[Path, Path]] = []
    for img in sorted(img_root.glob("*.jpg")):
        mpath = resolve_drivable_mask_path(msk_root, img.stem)
        if mpath is not None:
            pairs.append((img, mpath))
    return pairs


def collate_drivable(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "image_id": [b["image_id"] for b in batch],
    }


class BDD100KDrivableDataset(Dataset):
    """Paired BDD100K frames and drivable ``color_labels`` PNG masks."""

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        split: str = "train",
        image_size: Tuple[int, int] = (640, 640),
        transform: Optional[Any] = None,
        ignore_index: int = 255,
        max_samples: Optional[int] = None,
        pairs: Optional[List[Tuple[Path, Path]]] = None,
    ):
        self.split = split
        self.ignore_index = ignore_index
        self.transform = transform

        if pairs is not None:
            resolved: List[Tuple[Path, Path]] = list(pairs)
        else:
            resolved = collect_drivable_pairs(images_dir, masks_dir, split)
        if max_samples is not None:
            resolved = resolved[: max_samples]
        self.pairs = resolved
        self._image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        mask = color_drivable_mask_to_classes(bgr, self.ignore_index)

        if self.transform is not None:
            t = self.transform(image=image, mask=mask)
            image = t["image"]
            mask = t["mask"]
            if isinstance(mask, torch.Tensor):
                mask = mask.long()
            else:
                mask = torch.as_tensor(np.asarray(mask), dtype=torch.long)
        else:
            mask = torch.as_tensor(mask, dtype=torch.long)

        return {
            "image": image,
            "mask": mask,
            "image_id": img_path.stem,
        }


def format_drivable_split_diagnostic(
    images_dir: Path,
    masks_dir: Path,
    split: str,
) -> str:
    """Human-readable counts for one split (for error messages)."""
    img_sp = Path(images_dir) / split
    msk_sp = Path(masks_dir) / split
    if img_sp.is_dir():
        jpgs = list(img_sp.glob("*.jpg"))
        n_jpg = len(jpgs)
        jpg_note = f"{n_jpg} *.jpg"
    else:
        jpgs = []
        n_jpg = 0
        jpg_note = "directory missing"
    if msk_sp.is_dir():
        n_png = len(list(msk_sp.glob("*.png")))
        png_note = f"{n_png} *.png"
    else:
        n_png = 0
        png_note = "directory missing"
    n_pairs = sum(1 for img in jpgs if resolve_drivable_mask_path(msk_sp, img.stem))
    mismatch = ""
    if n_jpg > 0 and n_png > 0 and n_pairs == 0 and msk_sp.is_dir():
        mask_names = sorted(p.name for p in msk_sp.glob("*.png"))
        img_stems = sorted({p.stem for p in jpgs})
        ex_m = mask_names[0] if mask_names else "?"
        ex_i = f"{img_stems[0]}.jpg" if img_stems else "?"
        mismatch = (
            f"\n    No pairs after trying <stem>.png and <stem>_drivable_color.png "
            f"(e.g. mask file {ex_m!r} vs image {ex_i!r}). "
            "Use matching BDD100K image + drivable map releases."
        )
    return (
        f"  split={split!r}\n"
        f"    images: {img_sp.resolve()}\n"
        f"              → {jpg_note}\n"
        f"    masks:  {msk_sp.resolve()}\n"
        f"              → {png_note}\n"
        f"    pairs (<stem>.jpg + <stem>.png or <stem>_drivable_color.png): {n_pairs}"
        f"{mismatch}"
    )


def build_drivable_dataloaders(
    config: Dict[str, Any],
    device_hint: str = "cuda",
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train/val DataLoaders from a loaded drivable_seg_config dict."""
    import random
    from torch.utils.data import DataLoader

    paths = config["paths"]
    data_cfg = config["data"]
    aug_cfg = config.get("augmentation", {})
    classes_cfg = config["classes"]

    root = Path(paths["root_dir"])
    images_dir = root / paths["images_dir"]
    masks_dir = root / paths["drivable_masks_dir"]
    image_size = tuple(data_cfg["image_size"])
    ignore_index = classes_cfg["ignore_index"]

    train_tf = None
    val_tf = None
    if aug_cfg.get("enabled", True):
        train_tf = get_segmentation_augmentation(
            image_size=image_size,
            training=True,
            horizontal_flip=aug_cfg.get("horizontal_flip", 0.5),
            rotation=aug_cfg.get("rotation", 15),
            brightness=aug_cfg.get("brightness", 0.2),
            contrast=aug_cfg.get("contrast", 0.2),
            saturation=aug_cfg.get("saturation", 0.2),
            hue=aug_cfg.get("hue", 0.1),
            blur=aug_cfg.get("blur", 0.1),
            noise=aug_cfg.get("noise", 0.05),
        )
        val_tf = get_segmentation_augmentation(
            image_size=image_size,
            training=False,
        )
    else:
        val_tf = get_segmentation_augmentation(
            image_size=image_size,
            training=False,
        )
        train_tf = val_tf

    max_train = data_cfg.get("max_train_samples")
    max_val = data_cfg.get("max_val_samples")

    train_pairs = collect_drivable_pairs(images_dir, masks_dir, data_cfg["train_split"])
    val_pairs = collect_drivable_pairs(images_dir, masks_dir, data_cfg["val_split"])

    if len(train_pairs) == 0:
        raise RuntimeError(
            "Drivable training set is empty (no matching image+mask pairs).\n"
            + format_drivable_split_diagnostic(
                images_dir, masks_dir, data_cfg["train_split"]
            )
            + "\n"
            + format_drivable_split_diagnostic(
                images_dir, masks_dir, data_cfg["val_split"]
            )
            + "\n\n"
            "Hints: BDD100K image/mask trees are often gitignored—download locally. "
            "Paths come from configs/drivable_seg_config.yaml (paths.root_dir, "
            "images_dir, drivable_masks_dir). Masks: "
            "drivable_maps_dir/<split>/<stem>.png or <stem>_drivable_color.png ."
        )

    if len(val_pairs) == 0:
        frac = data_cfg.get("val_holdout_from_train_fraction")
        if frac is not None and isinstance(frac, (int, float)) and 0 < float(frac) < 1:
            seed = int(data_cfg.get("val_holdout_seed", 42))
            rng = random.Random(seed)
            order = list(range(len(train_pairs)))
            rng.shuffle(order)
            n_val = max(1, int(len(train_pairs) * float(frac)))
            val_pairs = [train_pairs[i] for i in order[:n_val]]
            train_pairs = [train_pairs[i] for i in order[n_val:]]
            print(
                f"[drivable] No masks for split {data_cfg['val_split']!r}; "
                f"held out {len(val_pairs)} pairs from train (fraction={frac}, seed={seed})."
            )
        else:
            raise RuntimeError(
                "Drivable validation set is empty.\n"
                + format_drivable_split_diagnostic(
                    images_dir, masks_dir, data_cfg["val_split"]
                )
                + "\n\nEither add bdd100k_drivable_maps/color_labels/<val_split>/ or set "
                "data.val_holdout_from_train_fraction in configs/drivable_seg_config.yaml "
                "(e.g. 0.05) to hold out part of train."
            )

    if max_train is not None:
        train_pairs = train_pairs[: max_train]
    if max_val is not None:
        val_pairs = val_pairs[: max_val]

    train_ds = BDD100KDrivableDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        split=data_cfg["train_split"],
        image_size=image_size,
        transform=train_tf,
        ignore_index=ignore_index,
        max_samples=None,
        pairs=train_pairs,
    )
    val_ds = BDD100KDrivableDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        split=data_cfg["val_split"],
        image_size=image_size,
        transform=val_tf,
        ignore_index=ignore_index,
        max_samples=None,
        pairs=val_pairs,
    )

    pin = device_hint.startswith("cuda")
    batch = data_cfg["batch_size"]
    workers = data_cfg["num_workers"]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        collate_fn=collate_drivable,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        collate_fn=collate_drivable,
    )
    return train_loader, val_loader
