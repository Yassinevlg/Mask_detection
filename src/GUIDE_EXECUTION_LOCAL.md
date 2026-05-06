# 🚀 Guide d'Exécution - Mask Detection CNN (Mode Local)

Ce guide vous aide à exécuter le notebook `mask-detection-cnn_notebook.ipynb` en mode local.

## ✅ Pré-requis

### 1. Python et Environnement Virtuel
```bash
cd Portfolio
python -m venv .venv
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 2. Installer les dépendances

```bash
pip install tensorflow opencv-python matplotlib scikit-learn numpy pillow seaborn kagglehub
```

**Optionnel (pour la conversion TensorFlow.js):**
```bash
pip install tensorflowjs
```

## 📋 Structure du Projet

```
plugins/mask-detection-cnn/
├── notebooks/
│   └── mask-detection-cnn_notebook.ipynb  ← Notebook principal
├── data/                                   ← Créé automatiquement
│   ├── with_mask/
│   └── without_mask/
├── models/                                 ← Où le modèle se sauvegarde
│   ├── best_model.keras                   ← Checkpoint pendant entraînement
│   ├── mask_detection_model.keras         ← Modèle final (✓ À VÉRIFIER)
│   └── tfjs_model/                        ← (Optionnel) Conversion JS
├── src/
├── api/
└── demo/
```

## 🎯 Ordre d'Exécution des Cellules

### Phase 1 - Environnement (⚡ Rapide)
- **Cellule 6:** Vérification des dépendances locales
- **Cellule 7:** Vérification du GPU
- **Cellule 9:** Configuration environnement (crée dossiers `/data` et `/models`)

### Phase 2 - Préparation des Données (⏱️ 5-10 min)
- **Cellule 10:** Téléchargement dataset Kaggle (~500MB)
- **Cellule 11:** Exploration du dataset
- **Cellule 12:** Prétraitement des images
- **Cellule 13:** Split train/val/test (⚠️ **CRITIQUE** - crée `X_test`, `y_test`)

### Phase 3 - Entraînement (⏱️ 10-30 min selon GPU)
- **Cellule 15:** Construction du CNN
- **Cellule 16:** Compilation + configuration callbacks
- **Cellule 17:** Entraînement du modèle
- **Cellule 18:** ✅ **VÉRIFICATION SAUVEGARDE** (affiche si le modèle est bien sauvegardé)
- **Cellule 19:** Conversion TensorFlow.js (optionnel)

### Phase 4 - Évaluation (⚡ Rapide)
- **Cellule 20:** Évaluation sur test set
- **Cellule 21-23:** Visualisations (matrice de confusion, courbes)

## ⚠️ Points Critiques à Vérifier

### 1️⃣ Cellule 18 - Sauvegarde du Modèle
**Cette cellule affichera:**
```
============================================================
SAUVEGARDE DU MODÈLE
============================================================

✓ Dossier modèles: C:\...\plugins\mask-detection-cnn\models
  - Checkpoint: C:\...\mask-detection-cnn\models\best_model.keras
  - Modèle final: C:\...\mask-detection-cnn\models\mask_detection_model.keras

✓ Checkpoint exist: C:\...\mask_detection_model.keras
  Taille: XX.XX MB

✓ Checkpoint chargé avec succès

Sauvegarde du modèle final vers C:\...mask_detection_model.keras...

✓ Modèle sauvegardé

✓ VÉRIFICATION: Fichier existe
  Chemin: C:\...\mask_detection_model.keras
  Taille: XX.XX MB

✓ VÉRIFICATION: Modèle peut être chargé

✓ SAUVEGARDE RÉUSSIE

============================================================
```

### 2️⃣ Cellule de Vérification Finale (après cellule 18)
**Affichera le contenu complet du dossier models:**
```
============================================================
VÉRIFICATION FINALE - CONTENU DU DOSSIER MODÈLES
============================================================

Dossier: C:\...\mask-detection-cnn\models

Fichiers présents:
  ✓ best_model.keras (XX.XX MB)
  ✓ mask_detection_model.keras (XX.XX MB)
  ✓ tfjs_model/model.json (si conversion TF.js réussie)

============================================================
RÉCAPITULATIF
============================================================

✓ Modèle final sauvegardé: C:\...\mask_detection_model.keras
  Taille: XX.XX MB

✅ CONFIGURATION LOCAL COMPLÈTE ET VÉRIFIÉE

Prochaines étapes:
1. Exécuter les cellules de Phase 4 pour l'évaluation
2. Vérifier les visualisations et métriques
3. Le modèle est prêt pour le déploiement

============================================================
```

## 🐛 Dépannage

### Problème: "NameError: name 'X_test' is not defined"
**Solution:** Vous avez sauté la Phase 2. Exécutez les cellules 10-13 en ordre.

### Problème: "FileNotFoundError: ... best_model.keras"
**Solution:** L'entraînement n'a pas complété. Relancez la cellule 17 ou réduisez EPOCHS.

### Problème: Modèle non trouvé dans `/models`
**Solution:** 
1. Vérifiez que la cellule 18 affiche "✓ SAUVEGARDE RÉUSSIE"
2. Contrôlez manuellement: `ls plugins/mask-detection-cnn/models/`
3. Relancez la cellule 18 après entraînement

### Problème: GPU non détecté
**Solution:** 
- NVIDIA: Vérifiez CUDA et CuDNN
- CPU est okay mais plus lent
- Sortie de cellule 7 affichera "GPU disponible: False" (normal sur CPU)

## 🚀 Démarrage Rapide

```bash
# 1. Activez l'environnement
cd Portfolio
.\.venv\Scripts\Activate.ps1

# 2. Ouvrez le notebook
# Dans VS Code: File > Open -> mask-detection-cnn_notebook.ipynb

# 3. Exécutez les phases dans l'ordre:
# Phase 1: Cellules 6, 7, 9 (vérifications)
# Phase 2: Cellules 10, 11, 12, 13 (données)
# Phase 3: Cellules 15, 16, 17, 18, 19 (entraînement)
# Phase 4: Cellules 20, 21, 22, 23 (évaluation)

# 4. ✅ Vérifiez la cellule 18 + Cellule de vérification finale
```

## 📊 Artefacts Générés

Après exécution complète, vous trouverez:

```
plugins/mask-detection-cnn/
├── models/
│   ├── best_model.keras           ← Checkpoint du meilleur epoch
│   ├── mask_detection_model.keras  ← ✅ MODÈLE FINAL (le plus important)
│   └── tfjs_model/
│       ├── model.json
│       ├── model.weights.bin
│       └── ...
├── notebooks/
│   ├── mask-detection-cnn_notebook.ipynb
│   └── rapport_mask_detection.txt  ← Rapport d'évaluation (Phase 6)
└── data/
    └── [Dataset temporaire - ~500MB]
```

## 💡 Tips

- **Réduction dataset:** Pour test rapide, réduisez EPOCHS à 2-3 (cellule 17)
- **Limiter GPU:** Si OutOfMemory, réduisez BATCH_SIZE (cellule 17)
- **Nettoyer:** `rm -r plugins/mask-detection-cnn/data` pour relancer téléchargement

## 📞 Support

Si vous avez des problèmes:
1. Vérifiez la sortie complète de la cellule problématique
2. Vérifiez que les dépendances sont installées
3. Redémarrez le kernel du notebook (VS Code: Ctrl+Shift+P > "Restart")

---

**Dernière mise à jour:** 29/04/2026
