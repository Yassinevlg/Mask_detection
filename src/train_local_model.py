from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import kagglehub
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 128
CLASS_NAMES = ("with_mask", "without_mask")


def find_dataset_root(download_path: str | Path) -> Path:
    download_path = Path(download_path)
    candidates = [download_path, download_path / "data"]

    for candidate in candidates:
        if (candidate / "with_mask").exists() and (candidate / "without_mask").exists():
            return candidate

    for candidate in candidates:
        if candidate.exists():
            for child in candidate.iterdir():
                if child.is_dir() and (child / "with_mask").exists() and (child / "without_mask").exists():
                    return child

    raise FileNotFoundError(
        "Impossible de trouver les dossiers with_mask et without_mask dans le dataset telecharge."
    )


def collect_image_paths(dataset_root: Path, max_images_per_class: int | None, seed: int) -> tuple[list[str], list[str]]:
    image_paths: list[str] = []
    labels: list[str] = []
    rng = random.Random(seed)

    for class_name in CLASS_NAMES:
        class_dir = dataset_root / class_name
        files = sorted(
            [
                *class_dir.glob("*.jpg"),
                *class_dir.glob("*.jpeg"),
                *class_dir.glob("*.png"),
                *class_dir.glob("*.webp"),
            ]
        )

        if max_images_per_class and len(files) > max_images_per_class:
            files = rng.sample(files, max_images_per_class)

        image_paths.extend(str(path) for path in files)
        labels.extend([class_name] * len(files))

    return image_paths, labels


def load_images(image_paths: list[str]) -> np.ndarray:
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        images.append(np.asarray(image, dtype=np.float32) / 255.0)
    return np.asarray(images, dtype=np.float32)


def build_model() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )


def preprocess_single_image(image_path: str | Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and save the local mask detection demo model.")
    parser.add_argument("--max-images-per-class", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Telechargement du dataset Kaggle...")
    dataset_path = kagglehub.dataset_download("omkargurav/face-mask-dataset")
    dataset_root = find_dataset_root(dataset_path)
    print(f"Dataset: {dataset_root}")

    image_paths, labels = collect_image_paths(dataset_root, args.max_images_per_class, args.seed)
    print(f"Images retenues: {len(image_paths)}")

    X = load_images(image_paths)
    label_to_index = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
    y = np.asarray([label_to_index[label] for label in labels], dtype=np.int32)
    y_cat = tf.keras.utils.to_categorical(y, num_classes=len(CLASS_NAMES))

    X_train, X_temp, y_train, y_temp, train_paths, temp_paths = train_test_split(
        X,
        y_cat,
        image_paths,
        test_size=0.3,
        random_state=args.seed,
        stratify=y,
    )
    y_temp_indices = np.asarray([int(np.argmax(row)) for row in y_temp], dtype=np.int32)
    X_val, X_test, y_val, y_test, val_paths, test_paths = train_test_split(
        X_temp,
        y_temp,
        temp_paths,
        test_size=0.5,
        random_state=args.seed,
        stratify=y_temp_indices,
    )

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_path = models_dir / "best_model.keras"
    final_model_path = models_dir / "mask_detection_model.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
    ]

    print("Entrainement du modele...")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print("Evaluation sur le jeu de test...")
    test_model = tf.keras.models.load_model(checkpoint_path)
    y_prob = test_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    test_model.save(final_model_path)
    print(f"Modele final sauvegarde vers: {final_model_path}")

    sample_path = test_paths[0]
    sample_prediction = test_model.predict(preprocess_single_image(sample_path), verbose=0)[0]
    predicted_index = int(np.argmax(sample_prediction))
    print(
        json.dumps(
            {
                "sample_path": sample_path,
                "predicted_label": CLASS_NAMES[predicted_index],
                "confidence": float(sample_prediction[predicted_index]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()