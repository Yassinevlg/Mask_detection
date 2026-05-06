# Implementation Plan - Mask Detection CNN

## 1. Objectif modelisation
- [x] Construire un CNN from scratch qui classe une image en `with_mask` ou `without_mask`.

## 2. Dataset cible et variables
- [x] Source: Kaggle Face Mask Detection
- [x] Classes: `with_mask`, `without_mask`
- [x] Entree modele: image RGB 128x128
- [x] Sortie: probabilites par classe

## 3. Pretraitement
- [x] Chargement images
- [x] Resize 128x128
- [x] Normalisation [0,1]
- [x] Split train/validation/test
- [x] Data augmentation

## 4. Modeles a tester
- [x] CNN baseline (3 blocs conv)
- [ ] CNN medium (4 blocs conv + batch norm)
- [ ] CNN final (dropout optimise)

## 5. Metriques d'evaluation
- [x] Accuracy
- [x] Precision
- [x] Recall
- [x] F1-score
- [x] Confusion matrix

## 6. Plan d'entrainement
- [x] Optimiseur: Adam
- [x] Loss: categorical_crossentropy
- [x] Epochs max: 50
- [x] Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

## 7. Plan de validation
- [ ] Evaluation sur split test
- [ ] Analyse des erreurs
- [ ] Visualisation des predictions incorrectes

## 8. Structure du projet (Notebook & Scripts)
- [x] 1. Setup environnement
- [x] 2. Chargement dataset
- [x] 3. Pretraitement
- [x] 4. Construction modele (src/train.py)
- [x] 5. Entrainement (src/train.py)
- [x] 6. Evaluation (Logique prête dans src/train.py)
- [x] 7. Inference (src/detect_realtime.py)
- [x] 8. Export modele (Logique prête dans src/train.py)

## 9. Resultats attendus
- [ ] Accuracy test >= 95% (à valider après entraînement)
- [x] Pipeline reproductible (src/train.py)
- [x] Modele exporte (Logique prête)

## 10. Prochaines ameliorations
- [ ] Calibration des seuils
- [ ] Amelioration robustesse lumiere/angles
- [x] Demo realtime webcam (src/detect_realtime.py)

## Integration portfolio
- id: `mask-detection-cnn`
- title: `Detection du port du masque - CNN Only`
- image: `/images/projects/mask-detection-cnn.png`
- reportUrl: `https://github.com/yassine-sekkour/mask-detection-cnn/blob/main/report/mask-detection-cnn_report.pdf`
- githubUrl: `https://github.com/yassine-sekkour/mask-detection-cnn`
- demoUrl: `/demo/mask-detection-cnn`
- status: `Termine`
