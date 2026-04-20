from flask import Flask, request, jsonify
from PIL import Image
import io
import os
import base64


app = Flask(__name__)


@app.route('/save_yolo', methods=['POST'])
def save_yolo():
    data = request.json

    # Decodificar a imagem enviada
    img_bytes = base64.b64decode(data['image_b64'])
    image = Image.open(io.BytesIO(img_bytes))

    filename = data['filename']
    yolo_labels = data['labels']

    # Criar pastas
    os.makedirs("dataset/images", exist_ok=True)
    os.makedirs("dataset/labels", exist_ok=True)

    # Salvar Imagem e Label
    image.save(f"dataset/images/{filename}.jpg")
    with open(f"dataset/labels/{filename}.txt", 'w') as f:
        f.write("\n".join(yolo_labels))

    return jsonify({"status": "success", "message": f"Arquivo {filename} salvo com sucesso!"})


if __name__ == '__main__':
    # host='0.0.0.0' garante que ele aceite conexões de rede local também
    app.run(host='0.0.0.0', port=5000, debug=True)

