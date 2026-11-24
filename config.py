from pathlib import Path
import os
import secrets

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "database" / "security.db"
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"
FEATURE_METADATA_PATH = BASE_DIR / "models" / "feature_metadata.json"
# For persistent sessions, set FLASK_SECRET_KEY in the environment before launch.
# The generated fallback is private to this process and is never displayed.
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY") or secrets.token_urlsafe(48)
MAX_CONTENT_LENGTH = 8 * 1024
