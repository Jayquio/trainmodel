import cv2, os, time, threading, requests, numpy as np

# ============== CONFIG ==============
ESP32_IP = "192.168.254.136"     # only the IP
LABEL = "background"             # change to "bottle" for bottle photos
AUTO_SECONDS = 0                 # 0 = press SPACE to save; e.g. 1.5 = auto-save every 1.5 s
SET_QVGA = True                  # 320x240, same size you'll use for detection
# ====================================

URL = f"http://{ESP32_IP}/capture"
os.makedirs(f"raw/{LABEL}", exist_ok=True)
n = len(os.listdir(f"raw/{LABEL}"))

latest = None
lock = threading.Lock()
running = True

def grabber():
    global latest
    fails = 0
    while running:
        try:
            r = requests.get(URL, timeout=6, headers={"Connection": "close"})
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise Exception("could not decode image")
            with lock:
                latest = img
            fails = 0
            time.sleep(0.15)
        except Exception as e:
            fails += 1
            print(f"frame error ({fails}): {e}")
            if fails >= 5:
                print("Not responding: close browser tabs, power-cycle the board, check the IP.")
            time.sleep(min(fails, 5))

if SET_QVGA:
    try:
        requests.get(f"http://{ESP32_IP}/control?var=framesize&val=5", timeout=5)
        print("Camera set to QVGA (320x240)")
    except Exception as e:
        print("Could not set resolution:", e)

threading.Thread(target=grabber, daemon=True).start()

print(f"Saving to raw/{LABEL}/   SPACE = save, ESC = quit")
last_save = 0

while True:
    with lock:
        frame = None if latest is None else latest.copy()
    if frame is None:
        if cv2.waitKey(30) == 27:
            break
        continue

    show = frame.copy()
    cv2.putText(show, f"{LABEL}: {n} saved", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("capture", show)
    key = cv2.waitKey(1)

    auto = AUTO_SECONDS > 0 and (time.time() - last_save) >= AUTO_SECONDS
    if key == 32 or auto:
        cv2.imwrite(f"raw/{LABEL}/{LABEL}_{n:04d}.jpg", frame)
        n += 1
        last_save = time.time()
        print("saved", n)
    elif key == 27:
        break

running = False
cv2.destroyAllWindows()