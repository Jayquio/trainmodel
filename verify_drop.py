from ultralytics import YOLO
import cv2, numpy as np, requests, serial, time, os
from datetime import datetime

# ============== CONFIG ==============
SERIAL_PORT = "COM5"            # the Arduino's port (Arduino IDE -> Tools -> Port)
ESP32_IP = "192.168.254.136"    # only the IP, no http:// and no slashes
MODEL_PATH = "best.pt"
CONF = 0.6                      # stricter = fewer wrong approvals
SNAPSHOTS = 5                   # photos taken after each drop
GAP = 0.4                       # seconds between photos
DEADLINE = 8                    # max seconds spent on one check
NEED_BOTTLE = 2                 # frames with a bottle needed to approve
MIN_FRAMES = 2                  # fewer good photos than this -> NOCAM
# ====================================

URL = f"http://{ESP32_IP}/capture"

os.makedirs("logs", exist_ok=True)
model = YOLO(MODEL_PATH)
bottle_ids = [i for i, n in model.names.items()
              if "bottle" in n.lower() and "not" not in n.lower()]
print("classes:", model.names, "| bottle ids:", bottle_ids)
model(np.zeros((240, 320, 3), np.uint8), verbose=False)   # warm-up, so the first drop is not slow

ser = serial.Serial(SERIAL_PORT, 9600, timeout=1)
time.sleep(2)                   # opening the port resets the Arduino
print("Ready. Press START on the machine and drop items...")


def get_photo():
    try:
        r = requests.get(URL, timeout=3, headers={"Connection": "close"})
        return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def check_drop(grams):
    start = time.time()
    good, hits, best = 0, 0, None
    for _ in range(SNAPSHOTS):
        if time.time() - start > DEADLINE:
            break
        img = get_photo()
        if img is not None:
            good += 1
            res = model(img, conf=CONF, verbose=False)[0]
            if any(int(c) in bottle_ids for c in res.boxes.cls):
                hits += 1
                best = res.plot()
            elif best is None:
                best = res.plot()
        time.sleep(GAP)

    if good < MIN_FRAMES:
        verdict = "NOCAM"
    elif hits >= NEED_BOTTLE:
        verdict = "BOTTLE"
    else:
        verdict = "NOTBOTTLE"

    if best is not None:        # keep a photo of every decision for review and retraining
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"logs/{stamp}_{verdict}_{grams:.0f}g.jpg", best)
    print(f"{verdict}  (bottle frames {hits}/{good}, weight {grams:.0f} g)")
    return verdict


while True:
    line = ser.readline().decode(errors="ignore").strip()
    if not line:
        continue
    print("arduino:", line)
    if line.startswith("DROP"):
        try:
            grams = float(line.split(",")[1])
        except Exception:
            grams = 0.0
        ser.write((check_drop(grams) + "\n").encode())