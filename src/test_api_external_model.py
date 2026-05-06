#!/usr/bin/env python
"""
Script de test pour vérifier que l'API se connecte correctement au modèle mymodel.h5
"""

import requests
import sys
from pathlib import Path

API_URL = "http://127.0.0.1:8000/api/mask/predict"
HEALTH_URL = "http://127.0.0.1:8000/api/mask/health"

def test_health():
    """Teste l'endpoint de santé de l'API"""
    print("🔍 Test de santé de l'API...")
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        if response.status_code == 200:
            print(f"✅ API réactive: {response.json()}")
            return True
        else:
            print(f"❌ Erreur API: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Impossible de contacter l'API: {e}")
        return False

def test_prediction_with_sample():
    """Teste la prédiction avec une image d'exemple"""
    print("\n🔍 Test de prédiction...")
    
    # Créer une image de test simple
    from PIL import Image
    import io
    
    # Créer une image simple 100x100 en RGB
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    try:
        files = {'file': ('test.png', img_bytes, 'image/png')}
        response = requests.post(API_URL, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prédiction réussie!")
            print(f"   Label: {result.get('label')}")
            print(f"   Confiance: {result.get('confidence'):.2%}")
            print(f"   Probabilités: {result.get('all_probabilities')}")
            return True
        else:
            print(f"❌ Erreur prédiction: {response.status_code}")
            print(f"   Détail: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction: {e}")
        return False

def main():
    """Exécute tous les tests"""
    print("=" * 50)
    print("Tests de l'API Mask Detection")
    print("=" * 50)
    
    # Test 1: Santé
    if not test_health():
        print("\n❌ L'API n'est pas accessible")
        print("   Démarrez l'API avec: ./START_API_EXTERNAL_MODEL.ps1")
        return False
    
    # Test 2: Prédiction
    if not test_prediction_with_sample():
        print("\n❌ Le modèle ne peut pas faire de prédiction")
        return False
    
    print("\n" + "=" * 50)
    print("✅ Tous les tests sont passés!")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
