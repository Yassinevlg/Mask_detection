#!/usr/bin/env python
"""
Analyser la structure détaillée du modèle .h5
"""

import h5py
from pathlib import Path
import json

model_path = Path("C:/Users/yassi/OneDrive/Bureau/ProjetDeep/FaceMaskDetector/mymodel.h5")

if not model_path.exists():
    print(f"Modele non trouve: {model_path}")
    exit(1)

print("Analyse detaillee du modele mymodel.h5")
print("=" * 60)

with h5py.File(model_path, 'r') as f:
    # Afficher la structure
    def print_attrs(name, obj):
        print(f"\n[{name}]")
        if isinstance(obj, h5py.Dataset):
            print(f"   Shape: {obj.shape}, Dtype: {obj.dtype}")
        else:
            print(f"   [Groupe]")
            
        # Afficher les attributs
        for key, val in obj.attrs.items():
            if isinstance(val, bytes):
                try:
                    val = val.decode('utf-8')
                except:
                    pass
            print(f"   @{key}: {val}")
    
    # Afficher la structure complète
    f.visititems(print_attrs)
    
    # Chercher la configuration du modèle
    print("\n" + "=" * 60)
    print("Configuration du modele:")
    print("=" * 60)
    
    if 'model_config' in f.attrs:
        config = f.attrs['model_config']
        if isinstance(config, bytes):
            config = config.decode('utf-8')
        try:
            config_dict = json.loads(config)
            print(json.dumps(config_dict, indent=2))
        except:
            print(config)
