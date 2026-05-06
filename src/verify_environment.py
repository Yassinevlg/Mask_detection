#!/usr/bin/env python
"""
Script de vérification de l'environnement pour mask-detection-cnn (mode LOCAL)
Usage: python verify_environment.py
"""

import sys
from pathlib import Path
import platform

print("=" * 70)
print("VÉRIFICATION ENVIRONNEMENT - MASK DETECTION CNN (MODE LOCAL)")
print("=" * 70)
print()

checks_passed = 0
checks_failed = 0

# 1. Python version
print(f"1️⃣  Python")
print(f"   Version: {sys.version.split()[0]}")
py_major, py_minor = sys.version_info[:2]
if (py_major, py_minor) < (3, 9) or (py_major, py_minor) > (3, 12):
    print("   ✗ FAIL (Required: Python >=3.9 and <=3.12 for TensorFlow)")
    print("   Conseil Windows: utilisez 'py -3.11 -m venv .venv'")
    checks_failed += 1
else:
    print("   ✓ PASS (Required: Python >=3.9 and <=3.12)")
    checks_passed += 1
print()

# 2. Système d'exploitation
print(f"2️⃣  Système d'exploitation")
print(f"   OS: {platform.system()} {platform.release()}")
print(f"   ✓ PASS")
checks_passed += 1
print()

# 3. Dossiers
print("3️⃣  Structure des dossiers")
notebook_dir = Path(__file__).parent  # On est déjà dans plugins/mask-detection-cnn/
if notebook_dir.exists():
    print(f"   ✓ Dossier plugin: {notebook_dir}")
    notebooks = notebook_dir / "notebooks"
    if notebooks.exists():
        print(f"   ✓ Dossier notebooks: {notebooks}")
        checks_passed += 1
    else:
        print(f"   ✗ MANQUANT: {notebooks}")
        checks_failed += 1
else:
    print(f"   ✗ MANQUANT: {notebook_dir}")
    checks_failed += 1
print()

# 4. Dépendances requises
print("4️⃣  Dépendances (REQUISES)")
required_packages = {
    "tensorflow": ("TensorFlow", "tensorflow"),
    "numpy": ("NumPy", "numpy"),
    "cv2": ("OpenCV", "opencv-python"),
    "sklearn": ("Scikit-learn", "scikit-learn"),
    "matplotlib": ("Matplotlib", "matplotlib"),
    "PIL": ("Pillow", "pillow"),
    "seaborn": ("Seaborn", "seaborn"),
}

for import_name, package_info in required_packages.items():
    display_name, pip_name = package_info
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown")
        print(f"   ✓ {display_name}: {version}")
        checks_passed += 1
    except ImportError:
        print(f"   ✗ MANQUANT: {display_name}")
        print(f"     Installer: pip install {pip_name}")
        checks_failed += 1
print()

# 5. Dépendances optionnelles
print("5️⃣  Dépendances (OPTIONNELLES)")
optional_packages = {
    "kagglehub": "KaggleHub (pour télécharger dataset)",
    "tensorflowjs": "TensorFlow.js (pour conversion web)",
}

for import_name, display_name in optional_packages.items():
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown")
        print(f"   ✓ {display_name}: {version}")
    except ImportError:
        print(f"   ⓘ {display_name}: NON INSTALLÉ (optionnel)")
        print(f"     Installer: pip install {import_name}")
print()

# 6. GPU
print("6️⃣  GPU (OPTIONNEL)")
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"   ✓ GPU DÉTECTÉ: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"     - {gpu}")
        checks_passed += 1
    else:
        print(f"   ⓘ AUCUN GPU: Entraînement sera sur CPU (plus lent)")
        checks_passed += 1
except Exception as e:
    print(f"   ⚠️  Erreur vérification GPU: {e}")
    checks_passed += 1
print()

# 7. Dossiers de travail
print("7️⃣  Dossiers de travail")
try:
    data_dir = notebook_dir / "data"
    models_dir = notebook_dir / "models"
    
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    print(f"   ✓ Dossier data: {data_dir}")
    
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
    print(f"   ✓ Dossier models: {models_dir}")
    
    checks_passed += 1
except Exception as e:
    print(f"   ✗ ERREUR création dossiers: {e}")
    checks_failed += 1
print()

# Résumé
print("=" * 70)
print("RÉSUMÉ")
print("=" * 70)
total = checks_passed + checks_failed
print(f"✓ Réussis: {checks_passed}/{total}")
if checks_failed > 0:
    print(f"✗ Échoués: {checks_failed}/{total}")
print()

if checks_failed == 0:
    print("✅ ENVIRONNEMENT PRÊT!")
    print()
    print("Prochaines étapes:")
    print("1. Ouvrir le notebook: mask-detection-cnn_notebook.ipynb")
    print("2. Exécuter les phases dans l'ordre (voir GUIDE_EXECUTION_LOCAL.md)")
    print("3. Vérifier la sauvegarde du modèle dans cellule 18")
    print()
    sys.exit(0)
else:
    print("⚠️  CORRECTIONS NÉCESSAIRES")
    print()
    print("Installez les dépendances manquantes avec:")
    print("  pip install -r requirements.txt")
    print()
    print("OU installez manuellement:")
    print("  pip install tensorflow opencv-python matplotlib scikit-learn numpy pillow seaborn kagglehub")
    print()
    sys.exit(1)
