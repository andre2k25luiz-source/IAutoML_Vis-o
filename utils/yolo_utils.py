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