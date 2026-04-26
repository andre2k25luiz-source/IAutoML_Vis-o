# 🧠 YOLO AutoML Vision Pipeline

Projeto completo de **detecção de objetos com YOLOv8**, incluindo:

- 🏷️ Anotação visual (Streamlit + bounding boxes)
- 🔄 Data augmentation automático e manual
- 🤖 Treinamento de modelo YOLOv8
- 📊 Métricas e visualização de treino
- 🔍 Inferência com visualização de resultados
- 🌐 API Flask para integração

---

## 📁 Estrutura do projeto

---

## 🚀 Como rodar o projeto

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```
### 2. Rodar o backend (Flask API)

```bash
python app.py
```

## 📌 Isso inicia:

* API de treino
* API de predição
* Pipeline de dataset

```bash
streamlit run templates/index.py
```

## 📌 Isso abre:

* Editor de bounding boxes
* Upload de imagens
* Visualização de dataset
* Treinamento do modelo
* Testes de inferência

## 🧠 Como o sistema funciona

# 1. Anotação de imagens

Você desenha bounding boxes no Streamlit e envia para o backend.

# 2. Data augmentation

## O sistema gera automaticamente várias variações da imagem:

* Flip horizontal
* Rotação
* Brilho/contraste
* Ruído
* Zoom e corte

# 3. Treinamento YOLOv8

O modelo é treinado automaticamente:

```bash
yolo.train(
    data="dataset_final/data.yaml",
    epochs=50,
    imgsz=640
)
```

# 4. Inferência (predição)

O modelo retorna:

* Bounding boxes
* Classe do objeto
* Confiança
* Imagem anotada

## 📊 Endpoints principais (Flask API)

| Endpoint        | Função                 |
| --------------- | ---------------------- |
| `/save_yolo`    | Salva imagens + labels |
| `/train`        | Treina modelo YOLO     |
| `/predict`      | Faz inferência         |
| `/metrics`      | Retorna métricas       |
| `/metrics_plot` | Gráficos do treino     |


## 📦 requirements.txt (exemplo)

flask
ultralytics
opencv-python
numpy
pillow
albumentations
streamlit
streamlit-drawable-canvas
requests
pandas

## 🧪 Exemplo de uso

1. Abra Streamlit
2. Faça upload de uma imagem
3. Marque bounding boxes
4. Clique em "Enviar"
5. Treine o modelo
6. Teste a detecção

## 🎯 Objetivo do projeto

Este projeto foi criado para:

* Automatizar criação de datasets YOLO
* Reduzir tempo de anotação manual
* Facilitar treino de modelos de visão computacional
* Criar pipeline completo de IA end-to-end

## ⚡ Tecnologias usadas

* YOLOv8 (Ultralytics)
* Flask
* Streamlit
* OpenCV
* Albumentations
* Python 3.12

## 👨‍💻 Autor

Projeto desenvolvido para estudos avançados de:

* Deep Learning
* Computer Vision
* MLOps básico
