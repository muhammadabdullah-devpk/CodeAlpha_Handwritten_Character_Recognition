from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
