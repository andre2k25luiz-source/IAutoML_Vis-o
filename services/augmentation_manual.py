import cv2
import random
import numpy as np


# =========================
# FLIP HORIZONTAL
# =========================
def flip_horizontal(image, boxes):
    h, w = image.shape[:2]

    new_boxes = []
    for cls, x, y, bw, bh in boxes:
        x = 1 - x
        new_boxes.append([cls, x, y, bw, bh])

    return cv2.flip(image, 1), new_boxes


# =========================
# BRIGHTNESS
# =========================
def brightness(image):
    value = random.randint(-40, 40)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)
    v = np.clip(v + value, 0, 255)

    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


# =========================
# ROTATION SIMPLES (SEM PERDER BBOX - versão segura)
# =========================
def rotate_image(image, angle=10):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)

    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


# =========================
# ESCOLHA ALEATÓRIA
# =========================
def augment(image, boxes):
    choice = random.choice(["flip", "bright", "rotate"])

    if choice == "flip":
        return flip_horizontal(image, boxes)

    if choice == "bright":
        return brightness(image), boxes

    if choice == "rotate":
        angle = random.randint(-10, 10)
        return rotate_image(image, angle), boxes