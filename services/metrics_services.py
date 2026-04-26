import os
import base64
import pandas as pd
from glob import glob


def get_metrics():
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


def get_metrics_plot():
    runs = glob("runs/detect/train*")

    if not runs:
        return {"error": "sem gráfico"}

    latest = max(runs, key=os.path.getctime)

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