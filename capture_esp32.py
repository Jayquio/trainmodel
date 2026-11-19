import cv2, os, time, requests, numpy as np

# ============== CONFIG ==============
ESP32_IP = "192.168.254.136"     # only the IP, no http:// and no slashes
INTERVAL = 2.5                   # seconds between photos (rest for the board, time to move things)
MAX_FAILS = 15                   # stop a run only after this many failures in a row
TIMEOUT = 10                     # seconds to wait for each photo

# key -> (folder name, photos per run, label shown on screen, Roboflow action)
MODES = {
    ord('b'): ("bottle",     150, "BOTTLE",       "draw boxes"),
    ord('g'): ("background",  40, "EMPTY TUNNEL", "mark Null"),
    ord('n'): ("other",      120, "NOT BOTTLE",   "mark Null"),
    ord('h'): ("hand",        30, "HAND",         "mark Null"),
}
# ====================================

URL = f"http://{ESP32_IP}/capture"


def count(folder):
    os.makedirs(f"raw/{folder}", exist_ok=True)
    return len(os.listdir(f"raw/{folder}"))


def get_photo():
    try:
        r = requests.get(URL, timeout=TIMEOUT, headers={"Connection": "close"})
        if r.status_code != 200:
            raise Exception(f"HTTP {r.status_code}")
        return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"  failed: {type(e).__name__}")
        return None


def make_screen(img, line1, line2, line3=""):
    screen = img.copy() if img is not None else np.zeros((240, 320, 3), np.uint8)
    cv2.putText(screen, line1, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(screen, line2, (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    if line3:
        cv2.putText(screen, line3, (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return screen


mode = None            # None = waiting, otherwise a key from MODES
folder = ""
n = 0                  # file number for the current folder
session = 0            # photos taken in this run
target = 0
fails = 0
last_try = 0
last_img = None
last_saved = None

print("B = bottle | G = empty tunnel | N = not-bottle items | H = hands")
print("S = stop run | ESC = quit")
print("In Roboflow: bottle photos get boxes, every other folder is marked Null.")

while True:
    if mode is None:
        counts = "  ".join(f"{MODES[k][0][:3]}:{count(MODES[k][0])}" for k in MODES)
        screen = make_screen(last_img, "B=bottle G=empty N=other H=hand", counts, "ESC = quit")
    else:
        screen = make_screen(last_img,
                             f"{MODES[mode][2]}  {session}/{target}",
                             f"fails in a row: {fails}",
                             "S = stop")
    cv2.imshow("capture", screen)
    key = cv2.waitKey(30) & 0xFF
    if key != 255:
        key = ord(chr(key).lower()) if 65 <= key <= 90 else key

    if key == 27:                                   # ESC
        break

    if mode is None:
        if key in MODES:
            mode = key
            folder, target, _, action = MODES[mode][0], MODES[mode][1], MODES[mode][2], MODES[mode][3]
            n = count(folder)
            session, fails, last_try = 0, 0, 0
            last_saved = None
            print(f"Started {folder}: up to {target} photos. Roboflow: {action}. Move things around!")
        continue

    if key == ord('s'):
        print(f"Stopped. {session} photos saved to raw/{folder}/")
        mode = None
        continue

    if time.time() - last_try >= INTERVAL:
        last_try = time.time()
        img = get_photo()

        if img is None:
            fails += 1
            if fails >= MAX_FAILS:
                print("Board not responding. Close browser tabs, power-cycle it, "
                      "use a 5V 2A supply, then try again.")
                mode = None
            continue

        fails = 0

        if (last_saved is not None and img.shape == last_saved.shape
                and np.array_equal(img, last_saved)):
            print("  same frame as before, skipped")
            continue

        path = f"raw/{folder}/{folder}_{n:04d}.jpg"
        cv2.imwrite(path, img)
        n += 1
        session += 1
        last_img = img
        last_saved = img
        print(f"saved {path}  ({session}/{target})")

        if session >= target:
            print(f"Done! {target} photos saved to raw/{folder}/")
            mode = None

cv2.destroyAllWindows()