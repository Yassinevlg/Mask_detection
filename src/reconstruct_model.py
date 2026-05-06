#!/usr/bin/env python
"""
Recréer le modèle mymodel.h5 en chargeant les poids
"""

import h5py
import numpy as np
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

model_path = Path("C:/Users/yassi/OneDrive/Bureau/ProjetDeep/FaceMaskDetector/mymodel.h5")

if not model_path.exists():
    print(f"Modele non trouve: {model_path}")
    exit(1)

print("Reconstruction du modele a partir des poids...")
print("=" * 60)

# Lire les poids du fichier .h5
with h5py.File(model_path, 'r') as f:
    # Extraire les poids
    conv2d_7_kernel = np.array(f['model_weights/conv2d_7/conv2d_7/kernel:0'])
    conv2d_7_bias = np.array(f['model_weights/conv2d_7/conv2d_7/bias:0'])
    
    conv2d_8_kernel = np.array(f['model_weights/conv2d_8/conv2d_8/kernel:0'])
    conv2d_8_bias = np.array(f['model_weights/conv2d_8/conv2d_8/bias:0'])
    
    conv2d_9_kernel = np.array(f['model_weights/conv2d_9/conv2d_9/kernel:0'])
    conv2d_9_bias = np.array(f['model_weights/conv2d_9/conv2d_9/bias:0'])
    
    dense_5_kernel = np.array(f['model_weights/dense_5/dense_5/kernel:0'])
    dense_5_bias = np.array(f['model_weights/dense_5/dense_5/bias:0'])
    
    dense_6_kernel = np.array(f['model_weights/dense_6/dense_6/kernel:0'])
    dense_6_bias = np.array(f['model_weights/dense_6/dense_6/bias:0'])

print(f"Conv2D_7 kernel shape: {conv2d_7_kernel.shape}")
print(f"Conv2D_8 kernel shape: {conv2d_8_kernel.shape}")
print(f"Conv2D_9 kernel shape: {conv2d_9_kernel.shape}")
print(f"Dense_5 kernel shape: {dense_5_kernel.shape}")
print(f"Dense_6 kernel shape: {dense_6_kernel.shape}")

# Créer le modèle
print("\nConstruction du modele...")

model = keras.Sequential([
    layers.Input(shape=(136, 136, 3)),
    
    # Bloc 1
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Bloc 2  
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Bloc 3
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Classification
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

print(model.summary())

# Charger les poids
print("\nChargement des poids...")
model.layers[0].kernel.assign(conv2d_7_kernel)
model.layers[0].bias.assign(conv2d_7_bias)

model.layers[2].kernel.assign(conv2d_8_kernel)
model.layers[2].bias.assign(conv2d_8_bias)

model.layers[4].kernel.assign(conv2d_9_kernel)
model.layers[4].bias.assign(conv2d_9_bias)

model.layers[7].kernel.assign(dense_5_kernel)
model.layers[7].bias.assign(dense_5_bias)

model.layers[8].kernel.assign(dense_6_kernel)
model.layers[8].bias.assign(dense_6_bias)

print("Poids charges avec succes!")

# Sauvegarder en .keras
keras_path = model_path.parent / "mymodel_reconstructed.keras"
model.save(keras_path)
print(f"\nModele sauvegarde en: {keras_path}")

# Test de prediction
print("\nTest de prediction...")
test_input = np.random.random((1, 136, 136, 3)).astype(np.float32)
prediction = model.predict(test_input)
print(f"Prediction test: {prediction}")
print("Success!")
