import cv2
import face_recognition
import time
import threading

# =========================
# CONFIG
# =========================
REFERENCE_IMAGE_PATH = "marouane.jpg"
PERSON_NAME = "Marouane"

WIDTH, HEIGHT = 640, 480
TARGET_FPS = 30

# Vitesse / qualité reconnaissance
RESIZE_FACTOR = 0.5          # 0.5 => 320x240 pour le traitement. Mets 0.4 si besoin (plus rapide)
TOLERANCE = 0.50
RECO_EVERY_SECONDS = 0.25    # ex: 0.25s = 4 fois par seconde (augmente à 0.35/0.5 si encore lourd)

# Latence caméra
DROP_GRABS = 2               # augmente à 3-5 si tu sens un retard
BACKEND = cv2.CAP_DSHOW      # essaie cv2.CAP_MSMF si besoin


# =========================
# Camera thread (latest frame)
# =========================
class CameraStream:
    def __init__(self, src=0, width=640, height=480, fps=30, backend=cv2.CAP_DSHOW):
        self.cap = cv2.VideoCapture(src, backend)
        if not self.cap.isOpened():
            raise RuntimeError("Impossible d'ouvrir la webcam")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.frame = None
        self.ok = False
        self.ts = 0.0
        self.frame_id = 0
        self.lock = threading.Lock()
        self.stopped = False

        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            for _ in range(DROP_GRABS):
                self.cap.grab()

            ok, frame = self.cap.read()
            t = time.time()
            with self.lock:
                self.ok = ok
                if ok:
                    self.frame = frame
                    self.ts = t
                    self.frame_id += 1

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None, 0, 0.0
            # copie pour éviter conflit
            return self.ok, self.frame.copy(), self.frame_id, self.ts

    def release(self):
        self.stopped = True
        time.sleep(0.05)
        self.cap.release()


# =========================
# Chargement visage référence
# =========================
ref_img = face_recognition.load_image_file(REFERENCE_IMAGE_PATH)
ref_encs = face_recognition.face_encodings(ref_img)

if len(ref_encs) == 0:
    print(f"Aucun visage trouvé dans {REFERENCE_IMAGE_PATH}")
    raise SystemExit

known_encoding = ref_encs[0]


# =========================
# Thread reconnaissance (NE BLOQUE PAS l'affichage)
# =========================
result_lock = threading.Lock()
last_locations = []
last_names = []
last_reco_ms = 0.0

def recognition_worker(cam: CameraStream):
    global last_locations, last_names, last_reco_ms

    while True:
        time.sleep(RECO_EVERY_SECONDS)

        ok, frame_bgr, fid, ts = cam.read()
        if not ok or frame_bgr is None:
            continue

        t0 = time.time()

        # Downscale + BGR->RGB
        small = cv2.resize(frame_bgr, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # Détection rapide (upsample=0)
        locations = face_recognition.face_locations(
            rgb_small, number_of_times_to_upsample=0, model="hog"
        )

        names = []
        if locations:
            encs = face_recognition.face_encodings(rgb_small, locations)
            for e in encs:
                dist = face_recognition.face_distance([known_encoding], e)[0]
                if dist <= TOLERANCE:
                    names.append(f"{PERSON_NAME} ({dist:.2f})")
                else:
                    names.append(f"Inconnu ({dist:.2f})")

        reco_ms = (time.time() - t0) * 1000.0

        with result_lock:
            last_locations = locations
            last_names = names
            last_reco_ms = reco_ms


# =========================
# MAIN
# =========================
cam = CameraStream(0, WIDTH, HEIGHT, TARGET_FPS, BACKEND)
threading.Thread(target=recognition_worker, args=(cam,), daemon=True).start()

# Mesures FPS
loop_fps = 0.0
cam_fps = 0.0
prev_loop = time.time()
last_new_frame_time = time.time()
last_id = 0

print("Appuie sur 'q' pour quitter.")

while True:
    ok, frame_bgr, fid, ts = cam.read()
    if not ok or frame_bgr is None:
        continue

    # FPS boucle (peut être > 30)
    now = time.time()
    dt = now - prev_loop
    prev_loop = now
    if dt > 0:
        loop_fps = 0.9 * loop_fps + 0.1 * (1.0 / dt)

    # FPS caméra (quand nouvelle frame)
    if fid != last_id:
        dt_cam = now - last_new_frame_time
        last_new_frame_time = now
        if dt_cam > 0:
            cam_fps = 0.9 * cam_fps + 0.1 * (1.0 / dt_cam)
        last_id = fid

    lat_ms = (time.time() - ts) * 1000.0

    # Récupérer le dernier résultat de reconnaissance (sans bloquer)
    with result_lock:
        locs = list(last_locations)
        names = list(last_names)
        reco_ms = last_reco_ms

    # Dessiner
    for (top, right, bottom, left), name in zip(locs, names):
        top = int(top / RESIZE_FACTOR)
        right = int(right / RESIZE_FACTOR)
        bottom = int(bottom / RESIZE_FACTOR)
        left = int(left / RESIZE_FACTOR)

        cv2.rectangle(frame_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame_bgr, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame_bgr, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Overlay infos
    cv2.putText(frame_bgr, f"CamFPS: {cam_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame_bgr, f"Latency: {lat_ms:.0f} ms", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame_bgr, f"RecoTime: {reco_ms:.0f} ms", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame_bgr, f"LoopFPS: {loop_fps:.1f}", (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.imshow("Face Recognition (Smooth Display)", frame_bgr)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
