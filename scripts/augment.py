import albumentations as A

transform = A.Compose([
    # 🔁 Geometria básica
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.2),
    A.Rotate(limit=25, border_mode=0, p=0.5),

    # 🔍 Zoom e corte (simula câmera)
    A.RandomResizedCrop(size=(640, 640), scale=(0.7, 1.0), ratio=(0.75, 1.33), p=0.5),
    A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.2, rotate_limit=15, p=0.5),

    # 💡 Iluminação e cor
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.5),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
    A.CLAHE(p=0.3),

    # 🌫️ Condições reais
    A.RandomFog(p=0.2),
    A.RandomRain(p=0.2),
    A.RandomShadow(p=0.3),
    A.RandomSunFlare(p=0.2),

    # 📷 Qualidade de imagem
    A.Blur(blur_limit=5, p=0.2),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.ImageCompression(quality_lower=30, quality_upper=100, p=0.3),

    # 🧱 Oclusão (muito importante)
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),

], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_visibility=0.3  # evita bbox inválida
))