from ultralytics import YOLO
import cv2, numpy as np, requests, serial, threading, time, os, glob, json, uuid
from datetime import datetime, timezone

# ==================== CONFIG ====================
SERIAL_PORT = "COM4"            # the Arduino's port (Arduino IDE -> Tools -> Port)
ESP32_IP = "192.168.254.136"    # only the IP, no http:// and no slashes
URL = f"http://{ESP32_IP}/capture"

MODEL_PATH = "best.pt"
CONF = 0.5                      # same as detect_esp32.py. Raise to cut false bottles, lower if real bottles are missed
IMGSZ = 640

REQUIRE_START = True            # only send ADD/WARN after the physical START button was pressed
NEED_BOTTLE_FRAMES = 1          # frames in a row with a bottle before it counts (use 1 if bottles pass fast)
EMPTY_NEEDED = 2                # empty frames needed before the next item can count
ENABLE_WARN = True              # warn when something that is not a bottle is in the tunnel
NEED_OTHER_FRAMES = 4           # frames in a row with a non-bottle before warning (raise to 5-6 if empty tunnel still warns)
PRESENT_FRACTION = 0.12         # share of pixels that must change to count as "something there" (raise to 0.2 if empty tunnel still warns)
PIXEL_DIFF = 40                 # how different a pixel must be (0-255)

FRAME_PAUSE = 1.5               # pause between photo requests (the ESP32-CAM needs time to breathe)
REQUEST_TIMEOUT = 4             # seconds to wait for each photo (a healthy ESP32-CAM answers in about 1 s)
STALE_SECONDS = 4               # show "CAMERA STALLED" if no new photo for this long
MAX_LOG_FILES = 300             # old photos in logs/ are deleted when there are more than this
ACK_WAIT = 1.5                  # seconds to wait for the Arduino to confirm a message
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

# state reported by the Arduino
arduino_ready = False           # got ARDUINO_READY
machine_active = False          # START button pressed, bottles are being counted
last_got = None                 # (text, time) of the last "GOT:..." confirmation


# ---------- server sync: sends finished transactions to the Waste2Goods system ----------
HERE = os.path.dirname(os.path.abspath(__file__))


def load_env(path):
    env = {}
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if raw and not raw.startswith("#") and "=" in raw:
                    k, v = raw.split("=", 1)
                    env[k.strip()] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return env


ENV = load_env(os.path.join(HERE, ".env"))
API_URL = ENV.get("API_URL", "")                    # e.g. https://waste2goods.site/api/machine/transactions
MACHINE_KEY = ENV.get("MACHINE_API_KEY", "")        # secret key, same value as on the server
MACHINE_ID = ENV.get("MACHINE_ID", "w2g-001")
SYNC_ENABLED = bool(API_URL and MACHINE_KEY)
PENDING_FILE = os.path.join(HERE, "pending_transactions.jsonl")
pending_lock = threading.Lock()


def queue_transaction(bottles, points, code):
    """Save the transaction to disk first, so it is never lost if the internet is down."""
    rec = {
        "transaction_id": str(uuid.uuid4()),        # the server uses this to ignore duplicates on retry
        "machine_id": MACHINE_ID,
        "bottles": bottles,
        "points": points,
        "code": code,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with pending_lock:
        with open(PENDING_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    print(f"saved transaction {code} ({bottles} bottles, {points} pts), waiting to sync to the server")


def sync_worker():
    """Background thread: sends saved transactions to the server and retries until they arrive."""
    was_down = False
    warned = set()
    while running:
        time.sleep(5)
        if not SYNC_ENABLED:
            continue
        with pending_lock:
            if not os.path.exists(PENDING_FILE):
                continue
            with open(PENDING_FILE, encoding="utf-8") as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
        sent_ids = set()
        for line in lines:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                r = requests.post(API_URL, json=rec, timeout=8,
                                  headers={"X-Machine-Key": MACHINE_KEY})
            except Exception as e:
                if not was_down:
                    print(f"server unreachable ({type(e).__name__}), will keep retrying")
                    was_down = True
                break
            was_down = False
            if r.status_code in (200, 201, 409):    # 409 = the server already has it
                sent_ids.add(rec["transaction_id"])
                print(f"server: saved {rec['code']} (HTTP {r.status_code})")
            elif rec["transaction_id"] not in warned:
                warned.add(rec["transaction_id"])
                print(f"server rejected {rec['code']}: HTTP {r.status_code} {r.text[:120]}")
        if sent_ids:
            with pending_lock:
                with open(PENDING_FILE, encoding="utf-8") as f:
                    keep = [ln for ln in f.read().splitlines()
                            if ln.strip() and json.loads(ln).get("transaction_id") not in sent_ids]
                with open(PENDING_FILE, "w", encoding="utf-8") as f:
                    f.write("".join(ln + "\n" for ln in keep))


def arduino_reader():
    """Background thread: reads every line the Arduino prints and tracks its state."""
    global arduino_ready, machine_active, last_got
    buf = b""
    while running:
        try:
            data = ser.read(64)          # waits up to the serial timeout
        except Exception:
            time.sleep(0.1)
            continue
        if not data:
            continue
        buf += data
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            line = raw.decode(errors="ignore").strip()
            if not line:
                continue
            print("arduino:", line)
            if line == "ARDUINO_READY":
                arduino_ready = True
                machine_active = False
            elif line == "START":
                machine_active = True
            elif line.startswith("TRANSACTION") or line == "COMPLETE":
                machine_active = False
                if line.startswith("TRANSACTION,"):
                    parts = line.split(",")
                    if len(parts) == 4 and parts[1].isdigit() and parts[2].isdigit():
                        queue_transaction(int(parts[1]), int(parts[2]), parts[3])
            elif line.startswith("GOT:"):
                last_got = (line[4:], time.time())


def grabber():
    """Background thread: keeps fetching the newest photo from the ESP32-CAM.
    ESP32-CAM only. There is deliberately NO laptop-webcam fallback: a photo from a
    different camera would look totally different from the empty reference and cause
    false 'NOT A BOTTLE' warnings."""
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
    """Share of pixels that really differ. Overall brightness shifts (camera auto-exposure) are cancelled."""
    a = a.astype(np.int16) - int(a.mean())
    b = b.astype(np.int16) - int(b.mean())
    return (np.abs(a - b) > PIXEL_DIFF).mean()


def check_ack(msg, sent_at):
    if last_got is None or last_got[1] < sent_at:
        print(f"!! Arduino did NOT confirm '{msg}' (no GOT line). "
              "Check the USB cable, COM port, and that the new sketch is uploaded.")


def send(msg):
    """Send ADD/WARN. Returns True only if it was really sent."""
    if REQUIRE_START and not machine_active:
        print(f"(not sent: press the physical START button first)  [{msg}]")
        return False
    sent_at = time.time()
    ser.write((msg + "\n").encode())
    print("-> Arduino:", msg)
    t = threading.Timer(ACK_WAIT, check_ack, args=(msg, sent_at))
    t.daemon = True
    t.start()
    return True


def log_photo(img, tag):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f"logs/{stamp}_{tag}.jpg", img)


threading.Thread(target=arduino_reader, daemon=True).start()
threading.Thread(target=sync_worker, daemon=True).start()
threading.Thread(target=grabber, daemon=True).start()

if SYNC_ENABLED:
    print(f"Server sync ON  -> {API_URL}  (machine {MACHINE_ID})")
else:
    print("Server sync OFF (no .env with API_URL and MACHINE_API_KEY). Transactions are saved to pending_transactions.jsonl")

# wait for the Arduino to say it is ready
t0 = time.time()
while not arduino_ready and time.time() - t0 < 15:
    time.sleep(0.1)
if arduino_ready:
    print("Arduino is ready.")
else:
    print("WARNING: no ARDUINO_READY received. Wrong COM port, or the new sketch is not uploaded.")

print("Keep the tunnel EMPTY. Capturing the empty reference in 5 seconds...")
time.sleep(5)
t0 = time.time()
while latest is None and time.time() - t0 < 30:
    time.sleep(0.2)
if latest is None:
    raise SystemExit(
        "No camera frame received in 30 s. Check the ESP32-CAM power supply, Wi-Fi, "
        f"and IP address ({ESP32_IP}), and make sure no browser is open on the camera."
    )
with lock:
    ref = small_gray(latest)
print("Reference saved.")
print("Press the physical START button on the machine, then bottles will count.")
print("Keys: ESC = quit | R = retake empty reference | S = save current frame | A = test ADD | W = test WARN")

armed = True
bottle_streak = other_streak = empty_streak = 0
count = 0
was_active = False
ref_pending = False
diff = 0.0
seen = -1
last = time.time()
last_out = None
stalled = False

try:
    while True:
        with lock:
            frame = None if latest is None or frame_id == seen else latest.copy()
            seen = frame_id
            age = time.time() - last_frame_time

        if frame is None:
            # no NEW photo yet: keep showing the last picture, and warn only if it is really stuck
            if last_out is not None:
                shown = last_out.copy()
                if age > STALE_SECONDS:
                    if not stalled:
                        stalled = True
                        bottle_streak = other_streak = empty_streak = 0   # don't mix old and new frames
                    cv2.putText(shown, "CAMERA STALLED", (5, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("Waste2Goods tunnel", shown)
            key = cv2.waitKey(20)
        else:
            stalled = False
            res = model(frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
            has_bottle = any(int(c) in bottle_ids for c in res.boxes.cls)
            gray = small_gray(frame)

            # when the user presses START the tunnel is empty: take a fresh reference
            if machine_active and not was_active:
                ref_pending = True
            was_active = machine_active
            if ref_pending and not has_bottle:
                ref = gray
                ref_pending = False
                print("Empty reference retaken for this session")

            diff = changed(gray, ref)
            present = has_bottle or diff > PRESENT_FRACTION

            if REQUIRE_START and not machine_active:
                # machine is idle: do not count anything
                bottle_streak = other_streak = empty_streak = 0
                armed = True
            elif armed:
                if has_bottle:
                    bottle_streak += 1
                    other_streak = 0
                else:
                    bottle_streak = 0
                    other_streak = other_streak + 1 if present else 0

                if bottle_streak >= NEED_BOTTLE_FRAMES:
                    if send("ADD"):
                        count += 1
                    log_photo(res.plot(), "ADD")
                    log_photo(frame, "ADD_raw")
                    armed = False
                    bottle_streak = other_streak = empty_streak = 0
                elif ENABLE_WARN and other_streak >= NEED_OTHER_FRAMES:
                    print(f"WARN reason: tunnel changed {diff:.0%} vs empty reference, no bottle found")
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
            if REQUIRE_START and not machine_active:
                status = "PRESS START BUTTON"
            else:
                status = "READY" if armed else "WAIT: clear the tunnel"
            cv2.putText(out, f"{status} | sent: {count} | {fps:.1f} FPS", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(out, f"change: {diff:.0%} (warn above {PRESENT_FRACTION:.0%})", (5, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
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