import os
from ultralytics import YOLO
from config import MODEL_PATH

model = None

def get_model(force_reload=False):
    global model

    if force_reload:
        model = None

    if model is None:
        if not os.path.exists(MODEL_PATH):
            return None
        model = YOLO(MODEL_PATH)

    return model