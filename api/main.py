from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

try:
    import cv2
except Exception:  # pragma: no cover - cv2 may not be installed in all envs
    cv2 = None

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import logging
import tempfile
import time

# Lazy import of TensorFlow/Keras is performed inside `get_predict_fn`
# to allow the app to start even if TF is not installed in some environments.

APP_TITLE = "Mask Detection API"
DEFAULT_MODEL_NAMES = (
    "mask_detection_model.keras",
    "saved_model_mask_detection.keras",
    "saved_model_mask_detection",
)
IMAGE_SIZE = 136  # Taille du modèle external
CLASS_NAMES = ("with_mask", "without_mask")

app = FastAPI(title=APP_TITLE, version="1.0.0")

logging.basicConfig(level=logging.INFO)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_model_cache_token() -> str:
    """Retourne une cle de cache qui change quand le modele sur disque change."""
    # Vérifie d'abord le modèle externe reconstructed
    external_model_path = Path("C:/Users/yassi/OneDrive/Bureau/ProjetDeep/FaceMaskDetector/mymodel_reconstructed.keras")
    if external_model_path.exists():
        stat = external_model_path.stat()
        return f"{external_model_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
    
    env_path = os.getenv("MASK_MODEL_PATH")

    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            stat = candidate.stat()
            return f"{candidate.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"

    project_root = Path(__file__).resolve().parents[1]
    # Support both `models/` (legacy) and `model/` (repo actual) directories
    models_dir = project_root / "models"
    model_dir_alt = project_root / "model"
    for name in DEFAULT_MODEL_NAMES:
        candidate = models_dir / name
        if candidate.exists():
            stat = candidate.stat()
            return f"{candidate.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"

        # check alternative model dir
        candidate_alt = model_dir_alt / name
        if candidate_alt.exists():
            stat = candidate_alt.stat()
            return f"{candidate_alt.resolve()}:{stat.st_mtime_ns}:{stat_alt.st_size}"

    # fallback checks in both locations
    fallback = models_dir / "mask_detection_model"
    if fallback.exists():
        stat = fallback.stat()
        return f"{fallback.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"

    fallback_alt = model_dir_alt / "mask_detection_model"
    if fallback_alt.exists():
        stat = fallback_alt.stat()
        return f"{fallback_alt.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"

    return "missing-model"


@lru_cache(maxsize=8)
def get_predict_fn(cache_token: str) -> Callable[[np.ndarray], np.ndarray]:
    """Charge un modele Keras ou SavedModel et retourne une fonction de prediction."""
    env_path = os.getenv("MASK_MODEL_PATH")
    candidate_paths = []

    # external reconstructed path (if present)
    external_model_path = Path("C:/Users/yassi/OneDrive/Bureau/ProjetDeep/FaceMaskDetector/mymodel_reconstructed.keras")
    if external_model_path.exists():
        candidate_paths.insert(0, external_model_path)

    if env_path:
        candidate_paths.append(Path(env_path))

    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    model_dir_alt = project_root / "model"

    # prefer env, then explicit locations in both `models/` and `model/`
    for name in DEFAULT_MODEL_NAMES:
        candidate_paths.append(models_dir / name)
        candidate_paths.append(model_dir_alt / name)

    candidate_paths.append(models_dir / "mask_detection_model")
    candidate_paths.append(model_dir_alt / "mask_detection_model")
    candidate_paths.append(project_root / "saved_model_mask_detection.keras")
    candidate_paths.append(project_root / "saved_model_mask_detection")

    # Lazy-import heavy dependencies (TensorFlow/Keras) only when loading model
    for candidate in candidate_paths:
        try:
            if not candidate.exists():
                continue

            logging.info(f"Chargement du modèle depuis: {candidate.resolve()}")

            # import tensorflow and keras locally to avoid top-level import errors
            try:
                import tensorflow as tf  # type: ignore
            except Exception as exc:  # pragma: no cover - surfaced via HTTP error
                raise RuntimeError(f"Impossible d'importer tensorflow: {exc}") from exc

            try:
                from keras.models import load_model  # type: ignore
            except Exception:
                # Keras is sometimes under `tensorflow.keras`
                try:
                    from tensorflow.keras.models import load_model  # type: ignore
                except Exception as exc:  # pragma: no cover
                    raise RuntimeError(f"Impossible d'importer load_model: {exc}") from exc

            if candidate.is_dir():
                saved_model = tf.saved_model.load(str(candidate))
                signature = saved_model.signatures.get("serve")
                if signature is None:
                    signature = next(iter(saved_model.signatures.values()))

                def predict_fn(input_tensor: np.ndarray) -> np.ndarray:
                    outputs = signature(tf.constant(input_tensor))
                    if isinstance(outputs, dict):
                        outputs = next(iter(outputs.values()))
                    return outputs.numpy()

                return predict_fn

            keras_model = load_model(candidate)
            logging.info(f"Modèle Keras chargé avec succès depuis: {candidate.resolve()}")

            def predict_fn(input_tensor: np.ndarray) -> np.ndarray:
                return keras_model.predict(input_tensor, verbose=0)

            return predict_fn

        except Exception as exc:  # pragma: no cover - surfaced via HTTP error
            raise RuntimeError(f"Impossible de charger le modele depuis {candidate}: {exc}") from exc

    raise FileNotFoundError(
        "Aucun modele trouve. Placez `mask_detection_model.keras` dans plugins/mask-detection-cnn/model/ "
        "ou `plugins/mask-detection-cnn/models/`, ou defini\" MASK_MODEL_PATH."
    )


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convertit l'image en tenseur compatible avec le modele."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.asarray(image, dtype=np.uint8)

    # Les frames webcam contiennent souvent beaucoup d'arriere-plan.
    # On recadre le visage (si detecte), sinon centre-crop pour stabiliser l'inference.
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        margin = int(0.2 * max(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image_np.shape[1], x + w + margin)
        y2 = min(image_np.shape[0], y + h + margin)
        cropped = image_np[y1:y2, x1:x2]
    else:
        h, w, _ = image_np.shape
        side = min(h, w)
        x1 = (w - side) // 2
        y1 = (h - side) // 2
        cropped = image_np[y1:y1 + side, x1:x1 + side]

    resized = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    array = resized.astype(np.float32) / 255.0
    return np.expand_dims(array, axis=0)


@app.get("/api/mask/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": APP_TITLE}


@app.post("/api/mask/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Format non supporte. Utilise JPG, PNG ou WEBP.")

    try:
        predict_fn = get_predict_fn(get_model_cache_token())
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    image_bytes = await file.read()

    try:
        input_tensor = preprocess_image(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image invalide: {exc}") from exc

    # Sauvegarde de l'image prétraitée hors du dépôt pour debug
    try:
        pre_img = (input_tensor[0] * 255.0).astype('uint8')
        temp_debug_dir = Path(tempfile.gettempdir()) / "mask-detection-cnn" / "preprocessed"
        temp_debug_dir.mkdir(parents=True, exist_ok=True)
        pre_name = f"debug_preprocessed_{int(time.time()*1000)}.png"
        pre_path = temp_debug_dir / pre_name
        Image.fromarray(pre_img).save(pre_path)
        pre_path_str = str(pre_path)
    except Exception:
        logging.exception("Impossible de sauvegarder l'image prétraitée")
        pre_path_str = None

    try:
        prediction_raw = predict_fn(input_tensor)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction impossible: {exc}") from exc

    # Adapter à la sortie du modèle (single sigmoid neuron ou deux classes)
    if prediction_raw.shape[-1] == 1:
        # Modèle avec un seul neurone sigmoid
        # Valeur entre 0 et 1 où 0 = "with_mask", 1 = "without_mask" (convention inversée)
        prob_without_mask = float(prediction_raw[0, 0])
        prob_with_mask = 1.0 - prob_without_mask
        prediction = np.array([prob_with_mask, prob_without_mask])
    else:
        # Modèle avec deux neurones (softmax)
        prediction = prediction_raw[0]

    class_index = int(np.argmax(prediction))
    confidence = float(prediction[class_index])
    label = CLASS_NAMES[class_index]

    # Log pour debug
    logging.info("raw prediction: %s", prediction.tolist())
    logging.info("label -> %s (index %d) confidence=%.4f", label, class_index, confidence)

    return {
        "label": label,
        "confidence": confidence,
        "predicted_class": label,
        "predicted_index": class_index,
        "raw_prediction": [float(x) for x in prediction],
        "all_probabilities": {CLASS_NAMES[i]: float(prediction[i]) for i in range(len(CLASS_NAMES))},
        "preprocessed_image": pre_path_str,
    }
