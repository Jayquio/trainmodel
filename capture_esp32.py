import cv2, os, time, requests, numpy as np

# ============== CONFIG ==============
ESP32_IP = "192.168.254.136"     # only the IP, no http:// and no slashes
TARGET = 150                     # photos per run
INTERVAL = 2.5                   # seconds between photos (rest for the board)
MAX_FAILS = 15                   # stop only after this many failures in a row
TIMEOUT = 10                     # seconds to wait for each photo
# ====================================

URL = f"http://{ESP32_IP}/capture"


def count(lbl):
    os.makedirs(f"raw/{lbl}", exist_ok=True)
    return len(os.listdir(f"raw/{lbl}"))


def get_photo():
    try:
        r = requests.get(URL, timeout=TIMEOUT, headers={"Connection": "close"})
        if r.status_code != 200:
            raise Exception(f"HTTP {r.status_code}")
        return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"  failed: {type(e).__name__}")
        return None


def make_screen(img, line1, line2):
    screen = img.copy() if img is not None else np.zeros((240, 320, 3), np.uint8)
    cv2.putText(screen, line1, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(screen, line2, (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return screen


label = None          # None = waiting, otherwise "bottle" or "background"
n = 0                 # file number for the current label
session = 0           # photos taken in this run
fails = 0
last_try = 0
last_img = None
last_saved = None

print("B = take 150 BOTTLE photos | G = take 150 BACKGROUND photos")
print("S = stop run | ESC = quit")

while True:
    if label is None:
        screen = make_screen(last_img, "B = bottle   G = background", "ESC = quit")
    else:
        screen = make_screen(last_img, f"{label.upper()}  {session}/{TARGET}",
                             f"fails in a row: {fails}   S = stop")
    cv2.imshow("capture", screen)
    key = cv2.waitKey(30) & 0xFF

    if key == 27:                                     # ESC
        break

    if label is None:
        if key in (ord('b'), ord('g')):
            label = "bottle" if key == ord('b') else "background"
            n = count(label)
            session, fails, last_try = 0, 0, 0
            print(f"Started {label}: taking {TARGET} photos. Move things around!")
        continue

    if key == ord('s'):
        print(f"Stopped. {session} {label} photos saved.")
        label = None
        continue

    if time.time() - last_try >= INTERVAL:
        last_try = time.time()
        img = get_photo()

        if img is None:
            fails += 1
            if fails >= MAX_FAILS:
                print("Board not responding. Close browser tabs, power-cycle it, "
                      "use a 5V 2A wall adapter, then try again.")
                label = None
            continue

        fails = 0

        if (last_saved is not None and img.shape == last_saved.shape
                and np.array_equal(img, last_saved)):
            print("  same frame as before, skipped")
            continue

        path = f"raw/{label}/{label}_{n:04d}.jpg"
        cv2.imwrite(path, img)
        n += 1
        session += 1
        last_img = img
        last_saved = img
        print(f"saved {path}  ({session}/{TARGET})")

        if session >= TARGET:
            print(f"Done! {TARGET} {label} photos saved.")
            label = None

cv2.destroyAllWindows()