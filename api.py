from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import os
import base64
import cv2
import numpy as np
import albumentations as A
import shutil
import random

from ultralytics import YOLO

app = Flask(__name__)

# =========================
# CONFIG
# =========================
DATASET_BASE = "dataset"
DATASET_AUG = "dataset_aug"
DATASET_FINAL = "dataset_final"

MODEL_PATH = "runs/detect/train/weights/best.pt"

model = None

# =========================
# AUGMENTATION
# =========================

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

# =========================
# UTILS
# =========================
def decode_image(img_b64):
    img_bytes = base64.b64decode(img_b64)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(image)

def save_image(path, img):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def save_yolo(path, boxes, labels):
    with open(path, "w") as f:
        for c, b in zip(labels, boxes):
            f.write(f"{c} {b[0]} {b[1]} {b[2]} {b[3]}\n")

def parse_yolo(lines):
    boxes, labels = [], []
    for line in lines:
        c, x, y, w, h = map(float, line.split())
        boxes.append([x, y, w, h])
        labels.append(int(c))
    return boxes, labels

def ensure_dirs():
    for p in [
        f"{DATASET_BASE}/images",
        f"{DATASET_BASE}/labels",
        f"{DATASET_AUG}/images",
        f"{DATASET_AUG}/labels",
    ]:
        os.makedirs(p, exist_ok=True)

# =========================
# SPLIT + MERGE
# =========================
def build_final_dataset():
    IMG_SRC = [
        f"{DATASET_BASE}/images",
        f"{DATASET_AUG}/images"
    ]
    LBL_SRC = [
        f"{DATASET_BASE}/labels",
        f"{DATASET_AUG}/labels"
    ]

    all_images = []

    for folder in IMG_SRC:
        for f in os.listdir(folder):
            if f.endswith(".jpg"):
                all_images.append((folder, f))

    random.shuffle(all_images)

    n = len(all_images)
    train_end = int(n * 0.7)
    val_end = int(n * 0.9)

    splits = {
        "train": all_images[:train_end],
        "val": all_images[train_end:val_end],
        "test": all_images[val_end:]
    }

    for split in splits:
        os.makedirs(f"{DATASET_FINAL}/{split}/images", exist_ok=True)
        os.makedirs(f"{DATASET_FINAL}/{split}/labels", exist_ok=True)

    for split, files in splits.items():
        for folder, img in files:
            name = img.replace(".jpg", "")

            # descobrir label correto
            lbl_folder = folder.replace("images", "labels")

            shutil.copy(
                f"{folder}/{img}",
                f"{DATASET_FINAL}/{split}/images/{img}"
            )

            shutil.copy(
                f"{lbl_folder}/{name}.txt",
                f"{DATASET_FINAL}/{split}/labels/{name}.txt"
            )

# =========================
# MODEL
# =========================
def get_model():
    global model
    if model is None and os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
    return model

def create_data_yaml():

    import yaml

    path = f"{DATASET_FINAL}/data.yaml"

    labels_dir = f"{DATASET_BASE}/labels"

    class_ids = set()

    for file in os.listdir(labels_dir):
        with open(f"{labels_dir}/{file}") as f:
            for line in f:
                c = int(line.split()[0])
                class_ids.add(c)

    nc = max(class_ids) + 1 if class_ids else 1

    data = {
        "path": DATASET_FINAL,  # 🔥 ESSENCIAL
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": nc,
        "names": [f"class_{i}" for i in range(nc)]
    }

    with open(path, "w") as f:
        yaml.dump(data, f)

    print("✔ data.yaml corrigido FINAL")

def get_first_val_image():
    val_dir = "dataset_final/val"

    if not os.path.exists(val_dir):
        return None

    files = sorted([
        f for f in os.listdir(val_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if len(files) == 0:
        return None

    img_path = os.path.join(val_dir, files[0])
    return img_path

# =========================
# ROUTES
# =========================

@app.route('/')
def index():
    return render_template('index.html')

# SALVAR + AUGMENTAR
@app.route('/save_yolo', methods=['POST'])
def save_yolo_route():

    data = request.json

    image = decode_image(data['image_b64'])
    filename = data['filename']
    yolo_labels = data['labels']

    ensure_dirs()

    # salvar original
    save_image(f"{DATASET_BASE}/images/{filename}.jpg", image)

    with open(f"{DATASET_BASE}/labels/{filename}.txt", "w") as f:
        f.write("\n".join(yolo_labels))

    boxes, labels = parse_yolo(yolo_labels)

    # augmentação
    for i in range(3):
        aug = transform(image=image, bboxes=boxes, class_labels=labels)

        new_img = aug["image"]
        new_boxes = aug["bboxes"]
        new_labels = aug["class_labels"]

        new_name = f"{filename}_aug{i}"

        save_image(f"{DATASET_AUG}/images/{new_name}.jpg", new_img)
        save_yolo(f"{DATASET_AUG}/labels/{new_name}.txt", new_boxes, new_labels)

    return jsonify({"status": "ok", "msg": "salvo + augmentado"})


# BUILD DATASET
@app.route('/build_dataset', methods=['POST'])
def build():
    build_final_dataset()
    return {"status": "dataset pronto"}


# TREINAR
@app.route('/train', methods=['POST'])
def train():

    build_final_dataset()
    create_data_yaml()

    trainer = YOLO("yolov8n.pt")

    trainer.train(
        data=f"{DATASET_FINAL}/data.yaml",
        epochs=10,
        imgsz=640,
        batch=8
    )

    # resetar modelo carregado
    model = None

    return {"status": "treinado"}


# PREDICT
@app.route('/predict', methods=['POST'])
def predict():
    global model

    data = request.json or {}

    # =========================
    # CARREGAR MODELO
    # =========================
    if model is None:
        if not os.path.exists(MODEL_PATH):
            return jsonify({"error": "modelo não treinado"}), 400
        model = YOLO(MODEL_PATH)

    # =========================
    # IMAGEM (VAL OU BASE64)
    # =========================
    image_to_predict = None
    image_b64_to_send = None

    if "image_b64" in data and data["image_b64"]:
        # Se o usuário mandou uma imagem do canvas
        image_b64_to_send = data["image_b64"]
        image_to_predict = decode_image(image_b64_to_send)
    else:
        # Se não mandou, pega a primeira da pasta /val
        img_path = get_first_val_image()
        if img_path is None:
            return jsonify({"error": "nenhuma imagem na pasta dataset_final/val"}), 404
        
        image_to_predict = img_path
        # Converte a imagem do disco para base64 para exibir no HTML
        with open(img_path, "rb") as f:
            image_b64_to_send = base64.b64encode(f.read()).decode('utf-8')

    # =========================
    # INFERÊNCIA
    # =========================
    results = model(image_to_predict)[0]
    detections = []

    if results.boxes is not None:
        for box in results.boxes:
            # .tolist() converte os tensores do YOLO para listas comuns do Python
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            detections.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class": cls,
                "conf": conf
            })

    # Retornamos as detecções E a imagem (para o canvas atualizar o fundo)
    return jsonify({
        "detections": detections, 
        "image_b64": image_b64_to_send
    })


# =========================
# METRICS
# =========================

@app.route('/metrics', methods=['GET'])
def metrics():

    import pandas as pd
    from glob import glob
    import os

    runs = glob("runs/detect/train*")

    if not runs:
        return {"error": "sem métricas ainda"}

    latest = max(runs, key=os.path.getctime)

    csv_path = os.path.join(latest, "results.csv")

    if not os.path.exists(csv_path):
        return {"error": "sem métricas ainda"}

    df = pd.read_csv(csv_path)

    last = df.iloc[-1].to_dict()

    return {
        "epochs": len(df),
        "metrics": last
    }

@app.route('/metrics_plot', methods=['GET'])
def metrics_plot():

    from glob import glob
    import os
    import base64

    runs = glob("runs/detect/train*")

    if not runs:
        return {"error": "sem gráfico"}

    latest = max(runs, key=os.path.getctime)

    # tentar vários nomes possíveis
    possible_files = [
        "results.png",
        "results.jpg",
        "confusion_matrix.png"
    ]

    for fname in possible_files:
        img_path = os.path.join(latest, fname)
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            return {"image": img_b64}

    return {"error": "sem gráfico"}

# =========================
# RUN
# =========================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)