"""
Centralized configuration for the Weather Anomaly Platform.
All settings are read from environment variables (.env file).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================
# Paths
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ============================================
# Database
# ============================================
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://weather_user:password@localhost:5432/weather_db")

# ============================================
# MLflow
# ============================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# ============================================
# ChromaDB
# ============================================
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8100"))

# ============================================
# Groq LLM (for RAG)
# ============================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ============================================
# Application
# ============================================
APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")

# ============================================
# Weather API (Open-Meteo — no key needed)
# ============================================
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
INGESTION_INTERVAL_HOURS = 6
BACKFILL_DAYS = 7

# ============================================
# Model Parameters
# ============================================
ANOMALY_CONTAMINATION = 0.05  # Expected 5% anomaly rate
FORECAST_HORIZON_HOURS = 48
RETRAIN_INTERVAL_DAYS = 7
