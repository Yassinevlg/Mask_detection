#!/usr/bin/env python
"""
Utilitaire pour charger le modèle mymodel.h5 avec compatibilité améliorée
"""

import os
import sys
from pathlib import Path

# Désactiver les validations strictes de Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as img_preprocessing

def load_external_model():
    """Charge le modèle mymodel.h5 de manière compatible"""
    model_path = Path("C:/Users/yassi/OneDrive/Bureau/ProjetDeep/FaceMaskDetector/mymodel.h5")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
    
    print(f"Tentative de chargement du modèle depuis: {model_path}")
    
    try:
        # Première tentative: chargement normal
        print("  Essai 1: load_model() standard...")
        model = load_model(model_path)
        print(f"  ✓ Modèle chargé avec succès!")
        return model
    except Exception as e:
        print(f"  ✗ Erreur: {e}")
    
    # Deuxième tentative: charger avec custom_objects vide
    try:
        print("  Essai 2: load_model() avec custom_objects...")
        model = load_model(model_path, custom_objects={})
        print(f"  ✓ Modèle chargé avec succès!")
        return model
    except Exception as e:
        print(f"  ✗ Erreur: {e}")
    
    # Troisième tentative: utiliser tf.function pour contourner
    try:
        print("  Essai 3: Charger en tant que SavedModel...")
        
        # Convertir le .h5 en SavedModel
        temp_saved_model = Path(model_path.parent) / "temp_saved_model"
        
        # Charger avec un wrapper
        import h5py
        print("  Vérification de la structure du fichier .h5...")
        with h5py.File(model_path, 'r') as f:
            print(f"  Groupes h5: {list(f.keys())}")
            if 'model_weights' in f:
                print(f"  Poids trouvés: {list(f['model_weights'].keys())}")
        
        raise FileNotFoundError("Impossible de charger le modèle avec les méthodes standard")
        
    except Exception as e:
        print(f"  ✗ Erreur: {e}")
    
    raise RuntimeError("Impossible de charger le modèle mymodel.h5. Le fichier peut être corrompu ou incompatible avec TensorFlow/Keras.")

if __name__ == "__main__":
    try:
        model = load_external_model()
        print("\n✓ Modèle chargé avec succès!")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
    except Exception as e:
        print(f"\n✗ Impossible de charger le modèle: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
