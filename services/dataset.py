import os
import shutil
import random
from config import DATASET_BASE, DATASET_AUG, DATASET_FINAL

def ensure_dirs():
    for p in [
        f"{DATASET_BASE}/images",
        f"{DATASET_BASE}/labels",
        f"{DATASET_AUG}/images",
        f"{DATASET_AUG}/labels",
    ]:
        os.makedirs(p, exist_ok=True)

def build_final_dataset():
    IMG_SRC = [
        f"{DATASET_BASE}/images",
        f"{DATASET_AUG}/images"
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
            lbl_folder = folder.replace("images", "labels")

            shutil.copy(f"{folder}/{img}", f"{DATASET_FINAL}/{split}/images/{img}")
            shutil.copy(f"{lbl_folder}/{name}.txt", f"{DATASET_FINAL}/{split}/labels/{name}.txt")

def create_data_yaml():
    import yaml
    import os

    path = f"{DATASET_FINAL}/data.yaml"

    # 🔥 pegar labels de TODOS os datasets
    label_dirs = [
        f"{DATASET_BASE}/labels",
        f"{DATASET_AUG}/labels",
        f"{DATASET_FINAL}/train/labels"
    ]

    class_ids = set()

    for labels_dir in label_dirs:
        if not os.path.exists(labels_dir):
            continue

        for file in os.listdir(labels_dir):
            file_path = os.path.join(labels_dir, file)

            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()

                    if len(parts) == 0:
                        continue

                    try:
                        class_ids.add(int(parts[0]))
                    except:
                        continue

    nc = max(class_ids) + 1 if class_ids else 1

    data = {
        "path": DATASET_FINAL,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": nc,
        "names": [f"class_{i}" for i in range(nc)]
    }

    os.makedirs(DATASET_FINAL, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print("✔ data.yaml gerado corretamente")