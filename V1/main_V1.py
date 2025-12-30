import cv2
import face_recognition

# charger l'image de la personne qu'on veut reconnaitre
    image_marouane = face_recognition. load_image_file("marouane.jpg")

# trouver l'encodage du visage (vecteur de nombres qui représente le visage)
    encodages = face_recognition.face_encodings(image_marouane)
    if len(encodages) == 0:
        print("Aucun visage trouvé dans marouane.jpg")
        quit()
    encodage_marouane = encodages [0]
# ouvrir la webcam (0 = caméra par défaut)
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Impossible d'ouvrir la webcam")
        quit()
while True:
# lire une image de la webcam
    ret, frame = cam.read()
    if not ret:
        print("Impossible de lire l'image de la webcam")
        break
# réduire la taille pour aller plus vite
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
# convertir BGR (opencv) en RGB (face_recognition)
    rgb_small_frame = small_frame[:, :,::-1]
# trouver les visages dans l'image réduite
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = []
# on calcule l'encodage visage par visage
    for (top, right, bottom, left) in face_locations:
        face_image = rgb_small_frame[top:bottom, left:right]
#
        try:
            enc = face_recognition.face_encodings(face_image)
        except TypeError as e:
            print("Erreur dlib sur ce visage, on l'ignore :", e)
            enc = []
        if len(enc) > 0:
            face_encodings.append (enc [0])
        else:
            face_encodings.append(None)
#parcourir tous les visages trouvés
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        if face_encoding is None:
            name = "Inconnu"
        else:

# calculer la distance entre ton visage et la photo
            distance = face_recognition.face_distance(
                [encodage_marouane], face_encoding
            )[0]
            print("Distance :", distance)
            SEUIL = 0.3 # tu peux ajuster 0.7/ 0.8 si besoin
#
            if distance < SEUIL:
                name = "Marouane"
            else:
                name = "Inconnu"
# remettre les coordonnées à la taille originale (x4)
        top = top * 4
        right = right * 4
        bottom = bottom * 4
        left = left * 4
# dessiner un rectangle autour du visage
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
# dessiner le nom
        cv2.rectangle(frame, (left, bottom 20), (right, bottom), (0, 255, 0), cv2.FILLED
        cv2.putText(frame, name, (left + 5, bottom 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
#afficher l'image
    cv2.imshow("Reconnaissance faciale simple (q pour quitter)", frame)
# si on appuie sur q, on quitte
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# fermer la webcam et la fenêtre
cam.release()
cv2.destroyAllWindows()