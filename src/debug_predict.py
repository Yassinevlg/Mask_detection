"""
Debug script: preprocess an image (center-crop), run model prediction,
save preprocessed image and print JSON result.
Usage:
  .venv\Scripts\python.exe plugins\mask-detection-cnn\debug_predict.py "C:\path\to\image.jpg"
"""
import sys
from pathlib import Path
import json
import tempfile

try:
    import numpy as np
    from PIL import Image
    import tensorflow as tf
except Exception as exc:
    print(json.dumps({"error": f"Missing dependency: {exc}"}))
    raise

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: debug_predict.py <image_path>"}))
        sys.exit(2)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(json.dumps({"error": f"Image not found: {image_path}"}))
        sys.exit(2)

    ROOT = Path(__file__).resolve().parents[2]
    model_path = ROOT / 'plugins' / 'mask-detection-cnn' / 'models' / 'mask_detection_model.keras'
    if not model_path.exists():
        print(json.dumps({"error": f"Model not found: {model_path}"}))
        sys.exit(2)

    IMAGE_SIZE = 128
    CLASS_NAMES = ("with_mask","without_mask")

    # Load model
    model = tf.keras.models.load_model(str(model_path))

    # Preprocess (center-crop fallback similar to API when face detection not used)
    img = Image.open(str(image_path)).convert('RGB')
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    crop = img.crop((left, top, left+side, top+side))
    resized = crop.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    input_tensor = np.expand_dims(arr, axis=0)

    # Save preprocessed image outside the repository tree
    out_dir = Path(tempfile.gettempdir()) / 'mask-detection-cnn' / 'preprocessed'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'debug_preprocessed_{image_path.stem}.png'
    Image.fromarray((arr*255).astype('uint8')).save(out_path)

    # Predict
    pred = model.predict(input_tensor, verbose=0)[0].tolist()
    inverted = [pred[1], pred[0]]
    all_prob = {CLASS_NAMES[i]: float(inverted[i]) for i in range(len(CLASS_NAMES))}
    idx = int(np.argmax(inverted))
    label = CLASS_NAMES[idx]
    conf = float(inverted[idx])

    result = {
        "raw_prediction": pred,
        "all_probabilities": all_prob,
        "label": label,
        "confidence": conf,
        "preprocessed_image": str(out_path)
    }

    print(json.dumps(result, indent=2))
