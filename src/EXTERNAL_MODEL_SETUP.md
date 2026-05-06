# Configuration du modèle externe mymodel.h5

## 🎯 Objectif
Cette configuration connecte l'interface de détection de masques au modèle externe `mymodel.h5` situé à :
```
C:\Users\yassi\OneDrive\Bureau\ProjetDeep\FaceMaskDetector\mymodel.h5
```

## ✅ Étapes de configuration

### 1. Vérifier la structure
L'API FastAPI a été modifiée pour charger automatiquement le modèle `.h5` externe en priorité.

**Modifications apportées à `api/main.py`:**
- ✓ Fonction `get_model_cache_token()` - Vérifie le modèle externe en premier
- ✓ Fonction `get_predict_fn()` - Charge le modèle `.h5` externe avec priorité
- ✓ Logging - Affiche le chemin du modèle chargé dans les logs

### 2. Démarrer l'API

**Option A: Script PowerShell (recommandé)**
```powershell
cd c:\Users\yassi\OneDrive\Bureau\Portfolio\plugins\mask-detection-cnn
.\START_API_EXTERNAL_MODEL.ps1
```

**Option B: Commande directe**
```powershell
cd c:\Users\yassi\OneDrive\Bureau\Portfolio\plugins\mask-detection-cnn
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Vérifier que l'API fonctionne
```bash
# Test de santé
curl http://127.0.0.1:8000/api/mask/health

# Réponse attendue:
# {"status":"ok","service":"Mask Detection API"}
```

### 4. Tester une prédiction
Ouvrez l'interface de démonstration dans le navigateur:
- URL: `http://localhost:5173/mask-detection-demo`
- L'interface se connectera automatiquement à l'API sur `http://127.0.0.1:8000`

## 📝 Logs de démarrage

Quand l'API démarre, vous verrez un message comme:
```
INFO:__main__:Chargement du modèle depuis: C:\Users\yassi\OneDrive\Bureau\ProjetDeep\FaceMaskDetector\mymodel.h5
INFO:__main__:Modèle Keras chargé avec succès depuis: C:\Users\yassi\OneDrive\Bureau\ProjetDeep\FaceMaskDetector\mymodel.h5
```

## 🔄 Flux de détection

1. **Frontend** (MaskDetectionDemo.jsx)
   - Récupère l'image de l'utilisateur
   - Envoie à l'API via POST `/api/mask/predict`

2. **API FastAPI** (api/main.py)
   - Reçoit l'image
   - Prétraite l'image (redimensionnement, normalisation)
   - Détecte le visage avec Haar Cascade
   - Lance la prédiction avec le modèle `.h5` externe
   - Retourne le résultat (label + confiance)

3. **Frontend**
   - Affiche les résultats (masque/sans masque + confiance)

## 📊 Sorties de prédiction

La réponse de l'API inclut:
```json
{
  "label": "with_mask",
  "confidence": 0.95,
  "predicted_class": "with_mask",
  "predicted_index": 0,
  "raw_prediction": [0.95, 0.05],
  "all_probabilities": {
    "with_mask": 0.95,
    "without_mask": 0.05
  },
  "preprocessed_image": "path/to/debug/image.png"
}
```

## ⚙️ Dépannage

### Le modèle ne se charge pas
1. Vérifiez que le fichier `.h5` existe:
   ```powershell
   Test-Path "C:/Users/yassi/OneDrive/Bureau/ProjetDeep/FaceMaskDetector/mymodel.h5"
   ```

2. Vérifiez les logs pour les erreurs de TensorFlow

3. Assurez-vous que TensorFlow est correctement installé dans le `.venv`

### Erreur CORS
L'API est configurée pour accepter les requêtes de:
- `http://localhost:5173`
- `http://127.0.0.1:5173`
- `http://localhost:4173`
- `http://127.0.0.1:4173`

### Taille d'image
- Limite: 8 MB
- Formats acceptés: JPG, PNG, WEBP
- Taille de prédiction: 128x128 pixels (après prétraitement)

## 🎨 Intégration avec le frontend

L'interface React est déjà configurée pour utiliser l'API:
- Endpoint API: `/api/mask/predict` (relatif à la racine)
- Variable d'environnement optionnelle: `VITE_MASK_API_URL`

Exemple d'utilisation dans le fichier vite.config.js:
```javascript
server: {
  proxy: {
    '/api/mask': 'http://127.0.0.1:8000'
  }
}
```

## ✨ Architecture complète

```
Frontend (React)
    ↓
.env / VITE_MASK_API_URL
    ↓
POST /api/mask/predict (FormData avec image)
    ↓
FastAPI (api/main.py)
    ↓
Charge mymodel.h5
    ↓
Prétraitement (détection visage + redimensionnement)
    ↓
Prédiction
    ↓
Retour JSON avec résultats
    ↓
Affichage dans l'interface
```

## 📚 Références

- **API:** `plugins/mask-detection-cnn/api/main.py`
- **Interface:** `src/pages/MaskDetectionDemo.jsx`
- **Configuration Vite:** `vite.config.js`
- **Modèle:** `C:/Users/yassi/OneDrive/Bureau/ProjetDeep/FaceMaskDetector/mymodel.h5`
