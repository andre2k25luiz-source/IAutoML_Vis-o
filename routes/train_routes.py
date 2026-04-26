from flask import Blueprint, request, jsonify
import random

from ultralytics import YOLO

from services.dataset import ensure_dirs, build_final_dataset, create_data_yaml
from services.augumentation import get_safe_transform, get_geometric_transform

from utils.image_utils import decode_image, save_image
from utils.yolo_utils import parse_yolo, save_yolo

from config import DATASET_BASE, DATASET_AUG, DATASET_FINAL

train_routes = Blueprint("train", __name__)


# =========================
# SAVE YOLO + AUGMENT
# =========================
@train_routes.route('/save_yolo', methods=['POST'])
def save_yolo_route():
    data = request.json
    image = decode_image(data['image_b64'])
    filename = data['filename']
    yolo_labels = data['labels']

    ensure_dirs()

    # Salvar original
    save_image(f"{DATASET_BASE}/images/{filename}.jpg", image)
    with open(f"{DATASET_BASE}/labels/{filename}.txt", "w") as f:
        f.write("\n".join(yolo_labels))

    boxes, labels = parse_yolo(yolo_labels)
    
    # Verificação inicial
    if len(boxes) == 0:
        print(f"[ERRO] {filename} não tem bounding boxes!")
        return jsonify({"status": "erro", "msg": "Imagem sem bboxes"}), 400
    
    print(f"\n[INFO] {filename}: {len(boxes)} bboxes encontradas")
    
    # Configurações
    safe_transform = get_safe_transform()
    geo_transform = get_geometric_transform()
    
    num_augmentations = random.randint(8, 15)
    created = 0
    attempts = 0
    max_attempts = num_augmentations * 2  # Menos tentativas necessárias
    
    print(f"[INFO] Objetivo: gerar {num_augmentations} imagens aumentadas")
    
    while created < num_augmentations and attempts < max_attempts:
        attempts += 1
        
        # Alterna entre transformações seguras e geométricas
        if attempts % 2 == 0:
            transform = geo_transform
            transform_type = "geométrica"
        else:
            transform = safe_transform
            transform_type = "segura"
        
        try:
            # Aplica augmentação
            augmented = transform(
                image=image.copy(),  # Copia para evitar problemas
                bboxes=boxes.copy(),
                class_labels=labels.copy()
            )
            
            new_img = augmented.get("image")
            new_boxes = augmented.get("bboxes", [])
            new_labels = augmented.get("class_labels", [])
            
            # Log de debug
            print(f"[Tentativa {attempts}/{max_attempts}] {transform_type}: {len(new_boxes)} bboxes", end=" ")
            
            # Validação
            if new_img is None or len(new_boxes) == 0:
                print("❌ FALHOU")
                continue
            
            # Sucesso!
            new_name = f"{filename}_aug{created}_{random.randint(1000, 9999)}"
            
            # Salvar imagem
            save_image(f"{DATASET_AUG}/images/{new_name}.jpg", new_img)
            
            # Salvar labels
            with open(f"{DATASET_AUG}/labels/{new_name}.txt", "w") as f:
                for box, label in zip(new_boxes, new_labels):
                    # Garante formato YOLO correto
                    line = f"{int(label)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
                    f.write(line + "\n")
            
            created += 1
            print(f"✅ GERADO ({created}/{num_augmentations})")
            
            # Reseta tentativas após sucesso
            attempts = 0
            
        except Exception as e:
            print(f"❌ ERRO: {str(e)[:50]}")
            continue
    
    print(f"\n{'='*50}")
    print(f"[RESULTADO FINAL] {filename}")
    print(f"  ✓ Geradas: {created}/{num_augmentations}")
    print(f"  ✓ Taxa de sucesso: {created/num_augmentations*100:.1f}%")
    print(f"{'='*50}\n")
    
    return jsonify({
        "status": "ok",
        "generated": created,
        "requested": num_augmentations,
        "success_rate": f"{created/num_augmentations*100:.1f}%"
    })

# =========================
# BUILD DATASET
# =========================
@train_routes.route('/build_dataset', methods=['POST'])
def build():
    build_final_dataset()

    return jsonify({
        "status": "dataset pronto"
    })

# TREINAR

@train_routes.route('/train', methods=['POST'])
def train_route():
    build_final_dataset()
    create_data_yaml()

    trainer = YOLO("yolov8n.pt")

    trainer.train(
        data=f"{DATASET_FINAL}/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8
    )

    return jsonify({
        "status": "treino concluído"
    })