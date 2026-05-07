"""Data augmentation pipeline using Albumentations"""

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional, Tuple


def drivable_letterbox_params(
    orig_h: int,
    orig_w: int,
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Geometry for val/test preprocessing: LongestMaxSize + center PadIfNeeded.

    Matches ``get_segmentation_augmentation(..., training=False)`` before
    normalize/totensor. Returns ``(new_h, new_w, pad_top, pad_left)`` after
    resize (before padding); padding is split evenly (remainder to bottom/right).
    """
    max_side = max(image_size)
    scale = max_side / max(orig_h, orig_w)
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))
    pad_h = image_size[0] - new_h
    pad_w = image_size[1] - new_w
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    return new_h, new_w, pad_top, pad_left


def get_detection_augmentation(
    image_size: Tuple[int, int] = (640, 640),
    training: bool = True,
    horizontal_flip: float = 0.5,
    rotation: int = 15,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
    blur: float = 0.1,
    noise: float = 0.05
) -> A.Compose:
    """Get augmentation pipeline for detection task.
    
    Args:
        image_size: Target image size (height, width)
        training: Whether to apply training augmentations
        horizontal_flip: Probability of horizontal flip
        rotation: Maximum rotation angle in degrees
        brightness: Brightness variation range
        contrast: Contrast variation range
        saturation: Saturation variation range
        hue: Hue variation range
        blur: Probability of applying blur
        noise: Probability of adding noise
        
    Returns:
        Albumentations compose transform
    """
    if training:
        transform = A.Compose([
            A.LongestMaxSize(max_size=max(image_size)),
            A.PadIfNeeded(
                min_height=image_size[0],
                min_width=image_size[1],
                border_mode=cv2.BORDER_CONSTANT
            ),
            A.HorizontalFlip(p=horizontal_flip),
            A.Rotate(limit=rotation, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=brightness,
                contrast_limit=contrast,
                p=0.5
            ),
            A.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                p=0.5
            ),
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=blur),
            A.GaussNoise(std_range=(0.1, 0.3), p=noise),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    else:
        # Validation/test transforms (only resize and normalize)
        transform = A.Compose([
            A.LongestMaxSize(max_size=max(image_size)),
            A.PadIfNeeded(
                min_height=image_size[0],
                min_width=image_size[1],
                border_mode=cv2.BORDER_CONSTANT
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels']
        ))
    
    return transform


def get_segmentation_augmentation(
    image_size: Tuple[int, int] = (640, 640),
    training: bool = True,
    horizontal_flip: float = 0.5,
    rotation: int = 15,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
    blur: float = 0.1,
    noise: float = 0.05,
    pad_mask_value: float = 0,
) -> A.Compose:
    """Get augmentation pipeline for segmentation task.
    
    Args:
        image_size: Target image size (height, width)
        training: Whether to apply training augmentations
        horizontal_flip: Probability of horizontal flip
        rotation: Maximum rotation angle in degrees
        brightness: Brightness variation range
        contrast: Contrast variation range
        saturation: Saturation variation range
        hue: Hue variation range
        blur: Probability of applying blur
        noise: Probability of adding noise
        pad_mask_value: ``fill_mask`` for ``PadIfNeeded`` (use ``ignore_index``, e.g. 255,
            so letterbox padding is not labeled as class 0).
        
    Returns:
        Albumentations compose transform
    """
    if training:
        transform = A.Compose([
            A.LongestMaxSize(max_size=max(image_size)),
            A.PadIfNeeded(
                min_height=image_size[0],
                min_width=image_size[1],
                border_mode=cv2.BORDER_CONSTANT,
                fill_mask=pad_mask_value,
            ),
            A.HorizontalFlip(p=horizontal_flip),
            A.Rotate(limit=rotation, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=brightness,
                contrast_limit=contrast,
                p=0.5
            ),
            A.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                p=0.5
            ),
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=blur),
            A.GaussNoise(std_range=(0.1, 0.3), p=noise),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        # Validation/test transforms
        transform = A.Compose([
            A.LongestMaxSize(max_size=max(image_size)),
            A.PadIfNeeded(
                min_height=image_size[0],
                min_width=image_size[1],
                border_mode=cv2.BORDER_CONSTANT,
                fill_mask=pad_mask_value,
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    return transform

