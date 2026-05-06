# Mask Detection API

## Endpoints

- `GET /health`: verifie que l'API est operationnelle
- `POST /api/mask/predict`: recoit une image et renvoie la prediction du modele

## Modele attendu

L'API charge par defaut `plugins/mask-detection-cnn/models/mask_detection_model.keras`.
Si besoin, defini `MASK_MODEL_PATH` vers un autre fichier `.keras` ou vers un SavedModel.

API FastAPI minimale pour servir le modele `mask-detection-cnn` au frontend du portfolio.

## Endpoint

- `GET /health`
- `POST /api/mask/predict`

## Lancer l'API

```bash
cd plugins/mask-detection-cnn
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

## Modele attendu

Place un fichier Keras dans `plugins/mask-detection-cnn/models/`:

- `mask_detection_model.keras`

Ou definis une variable d'environnement:

```bash
set MASK_MODEL_PATH=C:\chemin\vers\mask_detection_model.keras
```

## Retour JSON

```json
{
  "label": "with_mask",
  "confidence": 0.98,
  "predicted_class": "with_mask",
  "predicted_index": 0,
  "all_probabilities": {
    "with_mask": 0.98,
    "without_mask": 0.02
  }
}
```
