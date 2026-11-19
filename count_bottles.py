from ultralytics import YOLO
import cv2, numpy as np, requests, serial, threading, time, os, glob
from datetime import datetime

# ==================== CONFIG ====================
SERIAL_PORT = "COM5"            # the Arduino's port (Arduino IDE -> Tools -> Port)
ESP32_IP = "192.168.254.136"    # only the IP, no http:// and no slashes
URL = f"http://{ESP32_IP}/capture"

MODEL_PATH = "best.pt"
CONF = 0.5                      # same as detect_esp32.py. Raise to cut false bottles, lower if real bottles are missed
IMGSZ = 640

NEED_BOTTLE_FRAMES = 2          # frames in a row with a bottle before it counts (use 1 if bottles pass fast)
EMPTY_NEEDED = 2                # empty frames needed before the next item can count
ENABLE_WARN = True              # warn when something that is not a bottle is in the tunnel
NEED_OTHER_FRAMES = 3           # frames in a row with a non-bottle before warning (raise to 4-5 if real bottles get WARN)
PRESENT_FRACTION = 0.05         # share of pixels that must change to count as "something there"
PIXEL_DIFF = 30                 # how different a pixel must be (0-255)

FRAME_PAUSE = 0.2               # pause between photo requests (the ESP32-CAM needs time to breathe)
REQUEST_TIMEOUT = 3          # seconds to wait for each photo
STALE_SECONDS = 4               # show "CAMERA STALLED" if no new photo for this long
MAX_LOG_FILES = 300             # old photos in logs/ are deleted when there are more than this
# ================================================

os.makedirs("logs", exist_ok=True)

# keep the logs folder from growing forever
old = sorted(glob.glob("logs/*.jpg"), key=os.path.getmtime)
for f in old[:-MAX_LOG_FILES]:
    try:
        os.remove(f)
    except OSError:
        pass

model = YOLO(MODEL_PATH)
bottle_ids = [i for i, n in model.names.items()
              if "bottle" in n.lower() and "not" not in n.lower()]
print("classes:", model.names, "| bottle ids:", bottle_ids)
if not bottle_ids:
    raise SystemExit("No bottle class found in model.names - check the class names above")
model(np.zeros((240, 320, 3), np.uint8), verbose=False)    # warm-up

ser = serial.Serial(SERIAL_PORT, 9600, timeout=0.05)
time.sleep(2)                                              # opening the port resets the Arduino

latest, frame_id, last_frame_time = None, 0, time.time()
lock, running = threading.Lock(), True


def grabber():
    """Background thread: keeps fetching the newest photo from the ESP32-CAM."""
    global latest, frame_id, last_frame_time
    fails = 0
    while running:
        try:
            r = requests.get(URL, timeout=REQUEST_TIMEOUT, headers={"Connection": "close"})
            if r.status_code != 200:
                raise Exception(f"HTTP {r.status_code}")
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise Exception("could not decode image")
            with lock:
                latest, frame_id = img, frame_id + 1
                last_frame_time = time.time()
            if fails >= 5:
                print("camera recovered")
            fails = 0
            time.sleep(FRAME_PAUSE)
        except Exception as e:
            fails += 1
            if fails in (1, 5, 20) or fails % 50 == 0:
                print(f"camera error ({fails}): {type(e).__name__} {e if str(e) else ''}".strip())
            time.sleep(min(fails, 5))


def small_gray(img):
    g = cv2.cvtColor(cv2.resize(img, (160, 120)), cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (5, 5), 0)


def changed(a, b):
    return (cv2.absdiff(a, b) > PIXEL_DIFF).mean()


def send(msg):
    ser.write((msg + "\n").encode())
    print("-> Arduino:", msg)


def log_photo(img, tag):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f"logs/{stamp}_{tag}.jpg", img)


threading.Thread(target=grabber, daemon=True).start()

print("Keep the tunnel EMPTY. Capturing the empty reference in 5 seconds...")
time.sleep(5)
while latest is None:
    time.sleep(0.2)
with lock:
    ref = small_gray(latest)
print("Reference saved.")
print("Keys: ESC = quit | R = retake empty reference | S = save current frame | A = test ADD | W = test WARN")

armed = True
bottle_streak = other_streak = empty_streak = 0
count = 0
seen = -1
last = time.time()
last_out = None
stalled = False

try:
    while True:
        # show whatever the Arduino prints (START, TRANSACTION, ...)
        while ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                print("arduino:", line)

        with lock:
            frame = None if latest is None or frame_id == seen else latest.copy()
            seen = frame_id
            age = time.time() - last_frame_time

        if frame is None:
            # no new photo yet: if it has been too long, say so on the window
            if age > STALE_SECONDS and last_out is not None:
                if not stalled:
                    stalled = True
                    bottle_streak = other_streak = empty_streak = 0   # don't mix old and new frames
                shown = last_out.copy()
                cv2.putText(shown, "CAMERA STALLED", (5, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("Waste2Goods tunnel", shown)
            key = cv2.waitKey(20)
        else:
            stalled = False
            res = model(frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
            has_bottle = any(int(c) in bottle_ids for c in res.boxes.cls)
            gray = small_gray(frame)
            present = has_bottle or changed(gray, ref) > PRESENT_FRACTION

            if armed:
                if has_bottle:
                    bottle_streak += 1
                    other_streak = 0
                else:
                    bottle_streak = 0
                    other_streak = other_streak + 1 if present else 0

                if bottle_streak >= NEED_BOTTLE_FRAMES:
                    send("ADD")
                    count += 1
                    log_photo(res.plot(), "ADD")
                    log_photo(frame, "ADD_raw")
                    armed = False
                    bottle_streak = other_streak = empty_streak = 0
                elif ENABLE_WARN and other_streak >= NEED_OTHER_FRAMES:
                    send("WARN")
                    log_photo(res.plot(), "WARN")
                    log_photo(frame, "WARN_raw")
                    armed = False
                    bottle_streak = other_streak = empty_streak = 0
            else:
                empty_streak = empty_streak + 1 if not present else 0
                if empty_streak >= EMPTY_NEEDED:
                    armed = True
                    empty_streak = 0

            out = res.plot()
            now = time.time()
            fps = 1 / (now - last) if now > last else 0
            last = now
            status = "READY" if armed else "WAIT: clear the tunnel"
            cv2.putText(out, f"{status} | counted: {count} | {fps:.1f} FPS", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            last_out = out
            cv2.imshow("Waste2Goods tunnel", out)
            key = cv2.waitKey(1)

        if key == 27:
            break
        if key in (ord('r'), ord('R')) and frame is not None:
            ref = small_gray(frame)
            print("Reference retaken")
        if key in (ord('s'), ord('S')) and frame is not None:
            log_photo(frame, "manual_raw")
            print("Saved current frame to logs/")
        if key in (ord('a'), ord('A')):
            send("ADD")
        if key in (ord('w'), ord('W')):
            send("WARN")
finally:
    running = False
    cv2.destroyAllWindows()
    ser.close()