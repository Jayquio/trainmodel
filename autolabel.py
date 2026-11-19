from ultralytics import YOLOWorld
import cv2, os, glob

IMG_DIR = "raw/bottle"
PREVIEW_DIR = "raw/bottle_preview"
CONF = 0.25                      # lower = more boxes, higher = fewer
CLASS_NAME = "PLASTIC BOTTLE"    # must match your Roboflow class name

os.makedirs(PREVIEW_DIR, exist_ok=True)

model = YOLOWorld("yolov8s-worldv2.pt")      # downloads on first run
model.set_classes(["plastic bottle"])

no_box = []
for path in sorted(glob.glob(f"{IMG_DIR}/*.jpg")):
    res = model.predict(path, conf=CONF, verbose=False)[0]
    name = os.path.splitext(os.path.basename(path))[0]

    kept = 0
    with open(f"{IMG_DIR}/{name}.txt", "w") as f:
        for box in res.boxes:
            x, y, w, h = box.xywhn[0].tolist()
            if w * h > 0.9:                  # skip "whole picture" boxes
                continue
            f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            kept += 1

    cv2.imwrite(f"{PREVIEW_DIR}/{name}.jpg", res.plot())
    if kept == 0:
        no_box.append(name)

with open(f"{IMG_DIR}/data.yaml", "w") as f:
    f.write(f"names:\n  0: {CLASS_NAME}\n")

print(f"Done. {len(no_box)} photos got no box:")
for n in no_box:
    print("  ", n)