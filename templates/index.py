import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import requests
import base64
import io
import json
import cv2
import os
import random

st.set_page_config(layout="wide")
st.title("IAutoML_Visão_V2 - Auto ML")

API_URL = "http://127.0.0.1:5000"

# Inicializar session state
if "reset" not in st.session_state:
    st.session_state.reset = False
if "boxes" not in st.session_state:
    st.session_state.boxes = []
if "current_class_id" not in st.session_state:
    st.session_state.current_class_id = 0
if "drawing_mode" not in st.session_state:
    st.session_state.drawing_mode = "rect"
if "load_boxes" not in st.session_state:
    st.session_state.load_boxes = False
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

# Cores para diferentes IDs
COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFA500", "#800080",
    "#FF00FF", "#00FFFF", "#FFFF00", "#FFC0CB", "#A52A2A",
    "#808080", "#000000", "#FF4500", "#2E8B57", "#9400D3",
]

def get_color_for_id(class_id):
    return COLORS[class_id % len(COLORS)]

def boxes_to_canvas_objects(boxes):
    """Converte boxes do session_state para objetos do canvas"""
    objects = []
    for i, box in enumerate(boxes):
        objects.append({
            "type": "rect",
            "left": float(box["left"]),
            "top": float(box["top"]),
            "width": float(box["width"]),
            "height": float(box["height"]),
            "strokeWidth": 3,
            "stroke": get_color_for_id(box["class_id"]),
            "fill": "rgba(255, 255, 255, 0)",
            "name": f"box_{i}"
        })
    return objects

uploaded_file = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"], key=f"up_{st.session_state.reset}")

def resize_with_padding(image, size=640):
    img_w, img_h = image.size
    scale = min(size / img_w, size / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    image_resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    new_image = Image.new("RGB", (size, size), (128, 128, 128))
    paste_x, paste_y = (size - new_w) // 2, (size - new_h) // 2
    new_image.paste(image_resized, (paste_x, paste_y))
    return new_image, scale, paste_x, paste_y

if uploaded_file:
    image = Image.open(uploaded_file)
    resized_image, scale, offset_x, offset_y = resize_with_padding(image, 640)
    bg_image_array = np.array(resized_image)
    
    # Sidebar
    with st.sidebar:
        st.header("🎨 Controles")
        
        # Seleção do modo
        st.subheader("Modo")
        mode_option = st.radio(
            "Selecione:",
            ["✏️ Desenhar", "🖱️ Mover/Redimensionar"],
            horizontal=True
        )
        
        if mode_option == "✏️ Desenhar":
            st.session_state.drawing_mode = "rect"
            st.info("🔴 Modo Desenho: Clique e arraste para criar novas boxes")
        else:
            st.session_state.drawing_mode = "transform"
            st.info("🟢 Modo Edição: Clique e arraste nas boxes para mover/redimensionar")
        
        st.divider()

        # Seleção do ID (apenas para novas boxes)
        if st.session_state.drawing_mode == "rect":
            st.subheader("ID para nova box")
            new_id = st.number_input("ID:", min_value=0, max_value=99, 
                                     value=st.session_state.current_class_id, step=1)
            st.session_state.current_class_id = new_id
            st.markdown(f"**Cor:** <span style='color:{get_color_for_id(new_id)}'>●</span>", 
                       unsafe_allow_html=True)
        
        st.divider()
        
        # Botão para carregar boxes existentes no canvas
        if st.session_state.boxes and st.session_state.drawing_mode == "transform":
            if st.button("🔄 Recarregar Boxes no Canvas", use_container_width=True):
                st.session_state.load_boxes = True
                st.session_state.canvas_key += 1
                st.rerun()
        
        # Estatísticas
        if st.session_state.boxes:
            st.subheader(f"📦 Boxes ({len(st.session_state.boxes)})")
            id_counts = {}
            for box in st.session_state.boxes:
                id_counts[box["class_id"]] = id_counts.get(box["class_id"], 0) + 1
            for cid, count in sorted(id_counts.items()):
                st.markdown(f"<span style='color:{get_color_for_id(cid)}'>●</span> ID {cid}: {count}", 
                           unsafe_allow_html=True)
            
            st.divider()
            
            # Edição rápida de ID
            st.subheader("Editar ID da Box")
            box_options = [f"Box {i+1} (ID: {box['class_id']})" for i, box in enumerate(st.session_state.boxes)]
            selected_idx = st.selectbox("Selecione:", range(len(box_options)), format_func=lambda x: box_options[x])
            
            if selected_idx is not None:
                new_id = st.number_input("Novo ID:", min_value=0, max_value=99,
                                         value=st.session_state.boxes[selected_idx]["class_id"], step=1,
                                         key="edit_id")
                if st.button("✅ Aplicar ID", use_container_width=True):
                    st.session_state.boxes[selected_idx]["class_id"] = new_id
                    st.rerun()
                
                if st.button("🗑️ Remover Box", use_container_width=True):
                    st.session_state.boxes.pop(selected_idx)
                    st.rerun()
            
            if st.button("🗑️ Limpar Todas", type="secondary", use_container_width=True):
                st.session_state.boxes = []
                st.rerun()
    
    # Layout principal
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Determinar o initial_drawing baseado no estado
        initial_drawing = None
        if st.session_state.load_boxes and st.session_state.boxes:
            initial_drawing = json.dumps({"objects": boxes_to_canvas_objects(st.session_state.boxes)})
            st.session_state.load_boxes = False
        
        # Canvas interativo
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=3,
            stroke_color=get_color_for_id(st.session_state.current_class_id),
            background_image=Image.fromarray(bg_image_array),
            update_streamlit=True,
            height=640,
            width=640,
            drawing_mode=st.session_state.drawing_mode,
            initial_drawing=initial_drawing,
            key=f"canvas_{st.session_state.reset}_{st.session_state.canvas_key}",
        )
        
        # Processar alterações do canvas
        if canvas_result.json_data is not None:
            current_objects = canvas_result.json_data.get("objects", [])
            current_rects = [obj for obj in current_objects if obj.get("type") == "rect"]
            
            # Modo Desenho: Adicionar novas boxes
            if st.session_state.drawing_mode == "rect":
                if len(current_rects) > len(st.session_state.boxes):
                    # Encontrar a nova box
                    for rect in current_rects:
                        is_new = True
                        for box in st.session_state.boxes:
                            if (abs(box["left"] - rect["left"]) < 5 and
                                abs(box["top"] - rect["top"]) < 5):
                                is_new = False
                                break
                        if is_new:
                            st.session_state.boxes.append({
                                "left": rect["left"],
                                "top": rect["top"],
                                "width": rect["width"],
                                "height": rect["height"],
                                "class_id": st.session_state.current_class_id
                            })
                            st.rerun()
            
            # Modo Transform: Sincronizar alterações
            elif st.session_state.drawing_mode == "transform":
                if len(current_rects) == len(st.session_state.boxes) and len(current_rects) > 0:
                    # Atualizar coordenadas das boxes existentes
                    changed = False
                    for i, rect in enumerate(current_rects):
                        if i < len(st.session_state.boxes):
                            if (abs(st.session_state.boxes[i]["left"] - rect["left"]) > 1 or
                                abs(st.session_state.boxes[i]["top"] - rect["top"]) > 1 or
                                abs(st.session_state.boxes[i]["width"] - rect["width"]) > 1 or
                                abs(st.session_state.boxes[i]["height"] - rect["height"]) > 1):
                                st.session_state.boxes[i]["left"] = rect["left"]
                                st.session_state.boxes[i]["top"] = rect["top"]
                                st.session_state.boxes[i]["width"] = rect["width"]
                                st.session_state.boxes[i]["height"] = rect["height"]
                                changed = True
                    if changed:
                        st.rerun()
    
    with col2:
        if st.session_state.boxes:
            st.subheader("📋 Labels YOLO")
            
            yolo_labels = []
            with st.container(height=400):
                for i, box in enumerate(st.session_state.boxes):
                    w_box, h_box = box["width"] / scale, box["height"] / scale
                    x_c = ((box["left"] - offset_x) / scale + w_box/2) / image.size[0]
                    y_c = ((box["top"] - offset_y) / scale + h_box/2) / image.size[1]
                    w_norm = w_box / image.size[0]
                    h_norm = h_box / image.size[1]
                    
                    yolo_line = f"{box['class_id']} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}"
                    yolo_labels.append(yolo_line)
                    
                    st.markdown(f"<span style='color:{get_color_for_id(box['class_id'])}'>●</span> **Box {i+1}**", 
                               unsafe_allow_html=True)
                    st.code(yolo_line, language="text")
            
            if st.button("💾 Enviar para o Backend", type="primary", use_container_width=True):
                buffered = io.BytesIO()
                resized_image.save(buffered, format="JPEG", quality=70)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                payload = {
                    "filename": str(uploaded_file.name.split('.')[0]),
                    "image_b64": img_str,
                    "labels": yolo_labels
                }
                
                try:
                    # url_flask = "http://127.0.0.1:5000/save_yolo"
                    res = requests.post(f"{API_URL}/save_yolo", json=payload, timeout=100)
                    if res.status_code == 200:
                        st.success(f"✅ Salvo! {len(yolo_labels)} boxes")
                        st.session_state.boxes = []
                        st.session_state.reset = not st.session_state.reset
                        st.rerun()
                    else:
                        st.error(f"❌ Erro: {res.status_code}")
                except Exception as e:
                    st.error(f"❌ Erro: {e}")
        else:
            st.info("ℹ️ Como usar:")
            st.markdown("""
            **Modo Desenho:**
            - Selecione o ID
            - Desenhe retângulos arrastando o mouse
            
            **Modo Mover/Redimensionar:**
            - **Clique em "Recarregar Boxes"** para carregar as boxes
            - Depois clique e arraste nas boxes para mover
            - Use os cantos para redimensionar
            - As coordenadas atualizam automaticamente
            
            **Editar ID:**
            - Selecione a box na lista lateral
            - Altere o ID e clique em "Aplicar"
            """)
        
        st.divider()
st.subheader("🤖 Modelo")

# =========================
# BOTÃO TREINAR
# =========================
if st.button("🚀 Treinar Modelo", use_container_width=True):

    with st.spinner("Treinando modelo... ⏳"):
        try:
            res = requests.post(f"{API_URL}/train", timeout=None)

            if res.status_code == 200:
                st.success("✅ Modelo treinado!")
            else:
                st.error("Erro no treino")
        except Exception as e:
            st.error(str(e))


# =========================
# BOTÃO VER MÉTRICAS
# =========================
if st.button("📊 Ver Estatísticas", use_container_width=True):

    try:
        res = requests.get("http://127.0.0.1:5000/metrics")

        if res.status_code == 200:
            data = res.json()

            if "error" in data:
                st.warning(data["error"])
            else:
                m = data["metrics"]

                st.metric("Epochs", data["epochs"])
                st.metric("mAP50", f"{m.get('metrics/mAP50(B)', 0):.4f}")
                st.metric("mAP50-95", f"{m.get('metrics/mAP50-95(B)', 0):.4f}")
                st.metric("Precision", f"{m.get('metrics/precision(B)', 0):.4f}")
                st.metric("Recall", f"{m.get('metrics/recall(B)', 0):.4f}")

        else:
            st.error("Erro ao buscar métricas")

    except Exception as e:
        st.error(str(e))


# =========================
# GRÁFICO DO TREINO
# =========================
if st.button("📈 Ver Gráfico de Treino", use_container_width=True):

    try:
        res = requests.get("http://127.0.0.1:5000/metrics_plot")

        if res.status_code == 200:
            data = res.json()

            if "image" in data:
                img = base64.b64decode(data["image"])
                st.image(img, caption="Treinamento YOLO")
            else:
                st.warning("Sem gráfico ainda")

        else:
            st.error("Erro ao buscar gráfico")

    except Exception as e:
        st.error(str(e))

st.divider()
st.subheader("🔍 Teste com dataset val")

val_path = "dataset_final/val/images"

if st.button("📂 Rodar em 1 imagem da validação"):

    files = [f for f in os.listdir(val_path) if f.lower().endswith((".jpg",".png",".jpeg"))]
    img_name = random.choice(files)
    img_path = os.path.join(val_path, img_name)

    if len(files) == 0:
        st.error("Nenhuma imagem encontrada no val")
    else:
        if len(files) == 0:
            st.error("Nenhuma imagem encontrada no val")
        else:
            img_name = random.choice(files)
            img_path = os.path.join(val_path, img_name)

        image = Image.open(img_path).convert("RGB")
        resized_val, scale, offset_x, offset_y = resize_with_padding(image)

        st.caption(f"Imagem usada: {img_name}")

        # preparar request
        buffered = io.BytesIO()
        resized_val.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        res = requests.post(
            "http://127.0.0.1:5000/predict",
            json={"image_b64": img_str}
        )

        data = res.json()

        if "error" in data:
            st.error(data["error"])

        else:

            # 🔥 imagem do YOLO (backend)
            if "image" in data:
                img = base64.b64decode(data["image"])
                st.image(img, caption="Resultado YOLO")

            # 🔥 fallback: desenhar manual
            elif "detections" in data:

                img_np = np.array(resized_val)

                for det in data["detections"]:

                    x1, y1, x2, y2 = map(int, [
                        det["x1"], det["y1"], det["x2"], det["y2"]
                    ])

                    cls = det.get("class_id", -1)
                    conf = det.get("confidence", 0)

                    cv2.rectangle(img_np, (x1, y1), (x2, y2), (0,255,0), 2)

                    label = f"{cls} {conf:.2f}"
                    cv2.putText(img_np, label, (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

                st.image(img_np, caption="Fallback OpenCV")

            else:
                st.error("Resposta inesperada do backend")
        
        # 🔥 DEBUG (importante)
        st.write(data)