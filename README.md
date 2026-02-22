# Weather Anomaly Detection & Forecasting Platform

End-to-end MLOps pipeline for real-time weather anomaly detection with RAG-powered intelligent alerts.

## Architecture

- **Data Ingestion**: Open-Meteo API → PostgreSQL (Medallion Architecture)
- **ML Models**: Isolation Forest (anomaly detection) + XGBoost (forecasting)
- **Experiment Tracking**: MLflow
- **RAG Pipeline**: ChromaDB + NOAA Storm Events + Groq LLM
- **Deployment**: Docker Compose → AWS EC2
- **Monitoring**: Evidently AI (drift detection)
- **CI/CD**: GitHub Actions

## Quick Start
```bash
# Clone
git clone https://github.com/YOUR_USERNAME/weather-anomaly-platform.git
cd weather-anomaly-platform

# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Edit with your values

# Run with Docker
docker-compose up -d
```

## Tech Stack

Python | PostgreSQL | MLflow | Docker | FastAPI | Streamlit | ChromaDB | Evidently | AWS EC2 | GitHub Actions

## Status

🚧 Under active development
