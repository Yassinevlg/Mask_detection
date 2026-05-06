import os
import sys
import subprocess
from pathlib import Path
import shutil

# Fix Unicode issues on Windows terminal
os.environ["PYTHONIOENCODING"] = "utf-8"

try:
    import kagglehub
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kagglehub"])
    import kagglehub

import tensorflow as tf
import numpy as np
from PIL import Image

print("Downloading dataset...")
dataset_path = Path(kagglehub.dataset_download("omkargurav/face-mask-dataset"))
data_dir = dataset_path / "data"
if not data_dir.exists():
    data_dir = dataset_path

print(f"Dataset located at: {data_dir}")

IMAGE_SIZE = 128
CLASS_NAMES = ('with_mask', 'without_mask') # index 0 is with_mask, matches index.html and main.py

print("Loading images...")
X, y = [], []
for i, class_name in enumerate(CLASS_NAMES):
    class_dir = data_dir / class_name
    files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
    np.random.shuffle(files)
    for f in files[:800]: # 1600 total images
        img = Image.open(f).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
        arr = np.asarray(img, dtype=np.float32) / 255.0 # matching main.py preprocessing
        X.append(arr)
        y.append(i)

X = np.array(X)
y = tf.keras.utils.to_categorical(np.array(y), num_classes=2)

print("Building model...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.05),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax'),
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

print("Training model...")
model.fit(X, y, epochs=4, validation_split=0.2, batch_size=32)

print("Saving model...")
model.save("models/best_model.keras")

print("Exporting to TFJS...")
# Export to SavedModel
model.export("models/saved_model")
if Path("models/tfjs_model").exists():
    shutil.rmtree("models/tfjs_model")

# Convert using command line logic
subprocess.check_call([
    sys.executable, "-m", "tensorflowjs_converter",
    "--input_format=tf_saved_model",
    "--output_format=tfjs_graph_model",
    "models/saved_model",
    "models/tfjs_model"
])
print("Done!")
