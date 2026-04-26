from flask import Blueprint, request, jsonify
from services.model import get_model
from utils.image_utils import decode_image
import base64
import cv2

predict_routes = Blueprint("predict", __name__)

@predict_routes.route('/predict', methods=['POST'])
def predict():
    model = get_model()

    if model is None:
        return jsonify({"error": "modelo não treinado"}), 400

    data = request.json
    image = decode_image(data["image_b64"])

    results = model(image)[0]

    detections = []

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            detections.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class_id": cls,
                "confidence": conf
            })

    # 🔥 imagem com boxes desenhadas pelo YOLO
    plotted = results.plot()

    _, buffer = cv2.imencode(".jpg", plotted)
    img_b64 = base64.b64encode(buffer).decode()

    return jsonify({
        "image": img_b64,
        "detections": detections
    })