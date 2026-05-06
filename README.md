# Mask Detection CNN

Premier projet plugin du portfolio.

Repository GitHub: https://github.com/yassine-sekkour/mask-detection-cnn

## Objectif
Construire un modele CNN from scratch pour classer les images en `with_mask` ou `without_mask`.

## Structure
- `data/`: donnees brutes et traitees
- `model/`: modeles exportes
- `notebook/`: notebooks du projet
- `report/`: source LaTeX du rapport et espace reserve aux artefacts ignores
- `src/`: scripts d'entraînement et d'inférence temps-réel
- `assets/`: image de couverture du projet

## Fichiers clés
- `implementation_plan_mask-detection-cnn.md`
- `src/train.py`: script d'entraînement
- `notebook/mask-detection-cnn_notebook.ipynb`
- `requirements.txt`
- `api/main.py`: API FastAPI pour tester le modele dans le portfolio

## Liens utiles

- [Notebook principal](notebook/mask-detection-cnn_notebook.ipynb)
- [Rapport PDF](report/mask-detection-cnn_report.pdf)
- [Code source GitHub](https://github.com/yassine-sekkour/mask-detection-cnn)

## Demo web

Le projet est maintenant expose dans le portfolio avec une page de demo dediee. Elle appelle par defaut
`/api/mask/predict` via le proxy Vite local, ou l'URL configuree avec `VITE_MASK_API_URL`.

L'API expose aussi `GET /health` pour verifier rapidement que le modele est charge.

Les images de debug et de pretraitement ne sont plus stockees dans le depot. Elles sont ecrites dans le dossier temporaire systeme pour eviter d'alourdir `reports/`.


### Lancer la demo complete en local

1. Lancer le frontend du portfolio:

```bash
npm install
npm run dev
```

2. Lancer l'API de prediction dans un autre terminal:

```bash
cd plugins/mask-detection-cnn
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

3. Ouvrir la page:

- `http://localhost:5173/demo/mask-detection-cnn`

### Modele attendu

L'API charge un modele depuis `plugins/mask-detection-cnn/models/` ou depuis le dossier exporte par le notebook:

- `mask_detection_model.keras`
- `saved_model_mask_detection/`

Si besoin, definir `MASK_MODEL_PATH` vers un fichier `.keras` valide ou vers le dossier SavedModel.

## Statut
Termine

