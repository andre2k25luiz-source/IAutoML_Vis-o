import albumentations as A
import cv2
import numpy as np

def get_safe_transform():
    """Transformações que NÃO eliminam bounding boxes"""
    return A.Compose([
        # Transformações seguras (não alteram geometria)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.CLAHE(p=0.2),
        
        # Flip é seguro (bboxes são espelhadas corretamente)
        A.HorizontalFlip(p=0.5),
        
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.5,  # Mais tolerante
        min_area=0  # Não descarta por área mínima
    ))

def get_geometric_transform():
    """Transformações geométricas leves (ainda seguras)"""
    return A.Compose([
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.4),
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=5, p=0.3),
        A.RandomScale(scale_limit=0.1, p=0.3),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.4
    ))