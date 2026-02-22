"""
FastAPI Application — Weather Anomaly Platform API.

Endpoints:
    GET  /health              → System health check
    GET  /api/cities           → List all monitored cities
    GET  /api/weather/latest   → Latest weather for all cities
    GET  /api/anomalies        → Recent anomaly detections
    GET  /api/anomalies/{city} → Anomalies for a specific city
    GET  /api/forecast/{city}  → Temperature forecast for a city
    GET  /api/stats            → Platform statistics

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from src.utils.database import get_engine
from src.utils.logger import logger

app = FastAPI(
    title="Weather Anomaly Detection Platform",
    description="Real-time weather anomaly detection and forecasting for 25+ US cities",
    version="1.0.0",
)

# Allow Streamlit dashboard to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Health Check
# ============================================

@app.get("/health")
async def health_check():
    """System health check — used by monitoring and load balancers."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "database": db_status,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
    }


# ============================================
# City Endpoints
# ============================================

@app.get("/api/cities")
async def get_cities():
    """List all monitored cities with their coordinates."""
    from src.ingestion.city_config import get_all_cities
    cities = get_all_cities()
    return {"cities": cities, "count": len(cities)}


# ============================================
# Weather Data Endpoints
# ============================================

@app.get("/api/weather/latest")
async def get_latest_weather():
    """Get the most recent weather reading for each city."""
    engine = get_engine()
    query = text("""
        SELECT DISTINCT ON (city_name)
            city_name, latitude, longitude, timestamp,
            temperature_2m, relative_humidity, wind_speed,
            precipitation, pressure_msl, cloud_cover
        FROM silver_weather
        ORDER BY city_name, timestamp DESC
    """)
    try:
        df = pd.read_sql(query, engine)
        records = df.to_dict(orient="records")
        # Convert timestamps to strings for JSON
        for r in records:
            r["timestamp"] = str(r["timestamp"])
        return {"data": records, "count": len(records)}
    except Exception as e:
        logger.error(f"Error fetching latest weather: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/weather/{city}")
async def get_city_weather(city: str, hours: int = Query(default=48, le=768)):
    """Get recent weather data for a specific city."""
    engine = get_engine()
    query = text("""
        SELECT timestamp, temperature_2m, relative_humidity,
               wind_speed, precipitation, pressure_msl, cloud_cover
        FROM silver_weather
        WHERE city_name = :city
        ORDER BY timestamp DESC
        LIMIT :hours
    """)
    try:
        df = pd.read_sql(query, engine, params={"city": city, "hours": hours})
        if df.empty:
            raise HTTPException(status_code=404, detail=f"City '{city}' not found")
        records = df.to_dict(orient="records")
        for r in records:
            r["timestamp"] = str(r["timestamp"])
        return {"city": city, "data": records, "count": len(records)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching weather for {city}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Anomaly Endpoints
# ============================================

@app.get("/api/anomalies")
async def get_anomalies(
    limit: int = Query(default=50, le=200),
    min_zscore: float = Query(default=3.0),
):
    """Get recent weather anomalies across all cities."""
    engine = get_engine()
    query = text("""
        SELECT city_name, timestamp, temperature_2m,
               temp_zscore, temp_rolling_mean_24h, temp_change_1h
        FROM gold_weather_features
        WHERE ABS(temp_zscore) > :min_zscore
        ORDER BY ABS(temp_zscore) DESC
        LIMIT :limit
    """)
    try:
        df = pd.read_sql(query, engine, params={"min_zscore": min_zscore, "limit": limit})
        records = df.to_dict(orient="records")
        for r in records:
            r["timestamp"] = str(r["timestamp"])
            r["anomaly_type"] = "warm" if r["temp_zscore"] > 0 else "cold"
            r["severity"] = "high" if abs(r["temp_zscore"]) > 3.5 else "medium"
        return {
            "anomalies": records,
            "count": len(records),
            "threshold_zscore": min_zscore,
        }
    except Exception as e:
        logger.error(f"Error fetching anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/anomalies/{city}")
async def get_city_anomalies(city: str, min_zscore: float = Query(default=2.5)):
    """Get anomalies for a specific city."""
    engine = get_engine()
    query = text("""
        SELECT timestamp, temperature_2m, temp_zscore,
               temp_rolling_mean_24h, temp_change_1h, pressure_change_3h
        FROM gold_weather_features
        WHERE city_name = :city AND ABS(temp_zscore) > :min_zscore
        ORDER BY timestamp DESC
    """)
    try:
        df = pd.read_sql(query, engine, params={"city": city, "min_zscore": min_zscore})
        records = df.to_dict(orient="records")
        for r in records:
            r["timestamp"] = str(r["timestamp"])
        return {"city": city, "anomalies": records, "count": len(records)}
    except Exception as e:
        logger.error(f"Error fetching anomalies for {city}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Forecast Endpoint
# ============================================

@app.get("/api/forecast/{city}")
async def get_city_forecast(city: str):
    """Get temperature data and recent trend for a city (actual + forecast period)."""
    engine = get_engine()
    query = text("""
        SELECT timestamp, temperature_2m, temp_rolling_mean_24h,
               temp_zscore, temp_lag_24h
        FROM gold_weather_features
        WHERE city_name = :city
        ORDER BY timestamp DESC
        LIMIT 72
    """)
    try:
        df = pd.read_sql(query, engine, params={"city": city})
        if df.empty:
            raise HTTPException(status_code=404, detail=f"City '{city}' not found")
        records = df.to_dict(orient="records")
        for r in records:
            r["timestamp"] = str(r["timestamp"])
        return {"city": city, "forecast_data": records, "hours": len(records)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching forecast for {city}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Stats Endpoint
# ============================================

@app.get("/api/stats")
async def get_platform_stats():
    """Get overall platform statistics."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            bronze_count = conn.execute(text("SELECT COUNT(*) FROM bronze_weather")).scalar()
            silver_count = conn.execute(text("SELECT COUNT(*) FROM silver_weather")).scalar()
            gold_count = conn.execute(text("SELECT COUNT(*) FROM gold_weather_features")).scalar()
            city_count = conn.execute(
                text("SELECT COUNT(DISTINCT city_name) FROM silver_weather")
            ).scalar()
            anomaly_count = conn.execute(
                text("SELECT COUNT(*) FROM gold_weather_features WHERE ABS(temp_zscore) > 3")
            ).scalar()
            latest = conn.execute(
                text("SELECT MAX(timestamp) FROM silver_weather")
            ).scalar()

        return {
            "bronze_records": bronze_count,
            "silver_records": silver_count,
            "gold_records": gold_count,
            "cities_monitored": city_count,
            "anomalies_detected": anomaly_count,
            "latest_data": str(latest) if latest else None,
            "pipeline_status": "operational",
        }
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))