# Facial Recognition (OpenCV + face_recognition) — V1 & V2

Projet de reconnaissance faciale en temps réel en Python avec OpenCV et face_recognition.

## Versions
- **V1** : version initiale (celle déposée sur Campus).
- **V2** : version améliorée (affichage plus fluide, latence réduite, reconnaissance optimisée).

## Structure
facial-recognition/
├─ v1/
│ └─ main_v1.py
├─ v2/
│ └─ main_v2.py
├─ requirements.txt
└─ README.md

## Prérequis
- Python 3.9+ (recommandé)
- Webcam (testée en 640×480 @ 30 FPS)

## Installation
Dans le dossier du projet :
```bash
pip install -r requirements.txt
Exécution
Lancer la V1
python v1/main_v1.py
Lancer la V2 (recommandée)
python v2/main_v2.py
-Appuyer sur q pour quitter.
-Vérifier que l’image de référence (ex : marouane.jpg) est accessible par le script (même dossier que le script ou chemin adapté dans le code).
V2 — Améliorations principales
-Affichage plus fluide (la vidéo reste stable même quand la reconnaissance tourne).
-Réduction de la latence (gestion du buffer + drop frames).
-Optimisations CPU (traitement sur image réduite, mise à jour périodique)
#Auteur
Marouane ZA