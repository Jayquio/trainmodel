from ultralytics import YOLO
import cv2, numpy as np, requests, threading, time

# ==================== CONFIG ====================
ESP32_IP = "192.168.254.136"     # only the IP, no http:// and no slashes
URL = f"http://{ESP32_IP}/capture"

MODEL_PATH = "best.pt"
CONF = 0.5                       # raise (0.6-0.7) to cut false boxes, lower (0.3-0.4) if bottles are missed
IMGSZ = 640                      # 640 for accuracy, 320 if it is too slow
FRAME_INTERVAL = 0.3
REQUEST_TIMEOUT = 10           # seconds to wait for each photo
STALE_SECONDS = 5                # warn if no new photo for this long
# ================================================

model = YOLO(MODEL_PATH)

latest = None
frame_id = 0
last_frame_time = 0
lock = threading.Lock()
running = True
fails = 0


def grabber():
    """Background thread: keeps fetching the newest photo from the ESP32-CAM."""
    global latest, frame_id, last_frame_time, fails
    wait = 1.0
    while running:
        try:
            r = requests.get(URL, timeout=REQUEST_TIMEOUT,
                             headers={"Connection": "close"})
            if r.status_code != 200:
                raise Exception(f"HTTP {r.status_code}")
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise Exception("could not decode image")
            with lock:
                latest = img
                frame_id += 1
                last_frame_time = time.time()
            fails = 0
            wait = 1.0
            time.sleep(FRAME_INTERVAL)
        except Exception as e:
            fails += 1
            print(f"frame error ({fails}): {type(e).__name__}")
            if fails in (5, 15):
                print("ESP32-CAM not responding. Close browser tabs showing it, "
                      "power-cycle the board, use a 5V 2A wall adapter.")
            time.sleep(wait)
            wait = min(wait * 2, 8)


threading.Thread(target=grabber, daemon=True).start()

print(f"Reading from {URL}")
print("Press ESC in the video window to quit.")

seen = -1
last = time.time()
waiting = np.zeros((240, 320, 3), np.uint8)
cv2.putText(waiting, "Waiting for camera...", (30, 125),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

while True:
    with lock:
        new_frame = latest is not None and frame_id != seen
        frame = latest.copy() if new_frame else None
        seen = frame_id
        age = time.time() - last_frame_time if last_frame_time else None

    if not new_frame:
        # no new photo yet: keep the window responsive
        if latest is None:
            cv2.imshow("Waste2Goods", waiting)
        if cv2.waitKey(20) == 27:
            break
        continue

    try:
        results = model(frame, conf=CONF, imgsz=IMGSZ, verbose=False)
        out = results[0].plot()
        count = len(results[0].boxes)
    except Exception as e:
        print(f"inference error: {e}")
        continue

    now = time.time()
    fps = 1 / (now - last) if (now - last) > 0 else 0
    last = now

    status = f"{fps:.1f} FPS | bottles: {count}"
    cv2.putText(out, status, (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 0), 2)
    if age is not None and age > STALE_SECONDS:
        cv2.putText(out, "NO NEW FRAME", (5, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 255), 2)

    cv2.imshow("Waste2Goods", out)
    if cv2.waitKey(1) == 27:      # ESC
        break

running = False
cv2.destroyAllWindows()