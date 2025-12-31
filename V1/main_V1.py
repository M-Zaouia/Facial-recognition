import cv2
import face_recognition

# --- Réglages ---
REFERENCE_IMAGE = "marouane.jpg"
PERSON_NAME = "Marouane"
SEUIL = 0.50  

# 1) Charger l'image de référence
image_ref = face_recognition.load_image_file(REFERENCE_IMAGE)
encodages_ref = face_recognition.face_encodings(image_ref)

if len(encodages_ref) == 0:
    print(f"Aucun visage trouvé dans {REFERENCE_IMAGE}")
    raise SystemExit

encodage_ref = encodages_ref[0]

# 2) Ouvrir la webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Impossible d'ouvrir la webcam")
    raise SystemExit

while True:
    ret, frame = cam.read()
    if not ret:
        print("Impossible de lire l'image de la webcam")
        break

    # 3) Réduire la taille pour accélérer
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # 4) BGR -> RGB
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # 5) Détection + encodage (méthode fiable)
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # 6) Comparaison + affichage
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distance = face_recognition.face_distance([encodage_ref], face_encoding)[0]
        name = PERSON_NAME if distance < SEUIL else "Inconnu"

        # Remettre à la taille originale (x4)
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Rectangle visage
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Bandeau nom
        cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 5, bottom - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


    cv2.imshow("Reconnaissance faciale simple (q pour quitter)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()  # fin
