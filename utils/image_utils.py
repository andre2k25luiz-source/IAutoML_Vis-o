import base64
import io
from PIL import Image
import numpy as np
import cv2

def decode_image(img_b64):
    img_bytes = base64.b64decode(img_b64)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(image)

def save_image(path, img):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))