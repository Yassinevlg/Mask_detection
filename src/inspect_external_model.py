#!/usr/bin/env python
"""
Inspecte la structure du modèle mymodel.h5 pour voir les entrées/sorties attendues
"""

import sys
from pathlib import Path

try:
    from tensorflow.keras.models import load_model
    
    model_path = Path("C:/Users/yassi/OneDrive/Bureau/ProjetDeep/FaceMaskDetector/mymodel.h5")
    
    if not model_path.exists():
        print(f"❌ Le modèle n'existe pas: {model_path}")
        sys.exit(1)
    
    print(f"Chargement du modèle depuis: {model_path}\n")
    
    try:
        model = load_model(model_path)
        
        print("=" * 60)
        print("📊 Informations du modèle")
        print("=" * 60)
        
        print(f"\nShape d'entrée (Input shape): {model.input_shape}")
        print(f"Shape de sortie (Output shape): {model.output_shape}")
        
        print("\n📋 Architecture du modèle:")
        model.summary()
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)
