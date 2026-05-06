# Script de lancement de l'API avec le modèle externe mymodel.h5
# Ce script démarre l'API FastAPI configurée pour utiliser le modèle externe

Write-Host "========================================" -ForegroundColor Green
Write-Host "Démarrage de l'API Mask Detection" -ForegroundColor Green
Write-Host "Modèle: C:/Users/yassi/OneDrive/Bureau/ProjetDeep/FaceMaskDetector/mymodel.h5" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Green

# Vérifier que le fichier du modèle existe
$modelPath = "C:/Users/yassi/OneDrive/Bureau/ProjetDeep/FaceMaskDetector/mymodel.h5"
if (-not (Test-Path $modelPath)) {
    Write-Host "ERREUR: Le modèle n'existe pas à $modelPath" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Modèle trouvé" -ForegroundColor Green

# Activer l'environnement virtuel
Write-Host "Activation de l'environnement virtuel..." -ForegroundColor Yellow
& ".\.venv\Scripts\Activate.ps1"

# Démarrer l'API
Write-Host "Démarrage du serveur sur http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Appuyez sur Ctrl+C pour arrêter" -ForegroundColor Yellow
Write-Host ""

python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
