"""
Extended weather data fetcher — adds AQI, UV Index, and sunrise/sunset.

Open-Meteo provides these for FREE with no API key:
- Air Quality: PM2.5, PM10, US AQI index
- UV Index: Current and max daily
- Sunrise/Sunset times

This module adds new ingestion functions alongside the existing weather fetcher.

Usage:
    python -m src.ingestion.extended_fetcher           # Fetch extended data
    python -m src.ingestion.extended_fetcher --test     # Test one city
"""

import time
from datetime import datetime

import pandas as pd
import requests

from src.ingestion.city_config import get_all_cities, get_city_count
from src.utils.database import get_engine
from src.utils.logger import logger
from sqlalchemy import text


# ============================================
# Air Quality Fetch
# ============================================

def fetch_air_quality(city: dict, past_days: int = 7, forecast_days: int = 2) -> pd.DataFrame | None:
    """
    Fetch hourly air quality data from Open-Meteo Air Quality API.
    Free, no key needed.
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "hourly": "us_aqi,pm2_5,pm10",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "auto",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        hourly = data.get("hourly", {})

        if not hourly or "time" not in hourly:
            return None

        df = pd.DataFrame({
            "city_name": city["name"],
            "timestamp": pd.to_datetime(hourly["time"]),
            "us_aqi": hourly.get("us_aqi"),
            "pm2_5": hourly.get("pm2_5"),
            "pm10": hourly.get("pm10"),
        })
        return df

    except Exception as e:
        logger.error(f"AQI fetch failed for {city['name']}: {e}")
        return None


# ============================================
# UV Index + Sunrise/Sunset Fetch
# ============================================

def fetch_uv_and_sun(city: dict, past_days: int = 7, forecast_days: int = 2) -> pd.DataFrame | None:
    """
    Fetch daily UV index and sunrise/sunset from Open-Meteo.
    Uses the daily parameters of the main forecast API.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "daily": "uv_index_max,sunrise,sunset,temperature_2m_max,temperature_2m_min",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "auto",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        daily = data.get("daily", {})

        if not daily or "time" not in daily:
            return None

        df = pd.DataFrame({
            "city_name": city["name"],
            "date": pd.to_datetime(daily["time"]),
            "uv_index_max": daily.get("uv_index_max"),
            "sunrise": daily.get("sunrise"),
            "sunset": daily.get("sunset"),
            "temp_max": daily.get("temperature_2m_max"),
            "temp_min": daily.get("temperature_2m_min"),
        })
        return df

    except Exception as e:
        logger.error(f"UV/Sun fetch failed for {city['name']}: {e}")
        return None


# ============================================
# Fetch All Extended Data
# ============================================

def fetch_all_extended(past_days: int = 7, forecast_days: int = 2) -> dict:
    """
    Fetch AQI + UV/Sun data for all cities.
    Returns dict with 'aqi' and 'uv_sun' DataFrames.
    """
    cities = get_all_cities()
    aqi_data = []
    uv_sun_data = []

    logger.info(f"Fetching extended data for {get_city_count()} cities...")

    for i, city in enumerate(cities):
        aqi_df = fetch_air_quality(city, past_days, forecast_days)
        if aqi_df is not None:
            aqi_data.append(aqi_df)

        uv_df = fetch_uv_and_sun(city, past_days, forecast_days)
        if uv_df is not None:
            uv_sun_data.append(uv_df)

        if i < len(cities) - 1:
            time.sleep(0.3)

    result = {
        "aqi": pd.concat(aqi_data, ignore_index=True) if aqi_data else pd.DataFrame(),
        "uv_sun": pd.concat(uv_sun_data, ignore_index=True) if uv_sun_data else pd.DataFrame(),
    }

    logger.info(f"Extended fetch complete: {len(result['aqi'])} AQI records, {len(result['uv_sun'])} UV/Sun records")
    return result


# ============================================
# Create Tables and Load
# ============================================

def init_extended_tables():
    """Create tables for extended weather data."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS air_quality (
                id SERIAL PRIMARY KEY,
                city_name VARCHAR(100) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                us_aqi FLOAT,
                pm2_5 FLOAT,
                pm10 FLOAT,
                ingested_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(city_name, timestamp)
            );
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS daily_forecast (
                id SERIAL PRIMARY KEY,
                city_name VARCHAR(100) NOT NULL,
                date DATE NOT NULL,
                uv_index_max FLOAT,
                sunrise VARCHAR(30),
                sunset VARCHAR(30),
                temp_max FLOAT,
                temp_min FLOAT,
                ingested_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(city_name, date)
            );
        """))
        conn.commit()
    logger.info("Extended tables created successfully")


def load_extended_data(data: dict) -> dict:
    """Load AQI and UV/Sun data into PostgreSQL."""
    engine = get_engine()
    stats = {"aqi_loaded": 0, "uv_sun_loaded": 0}

    # Load AQI
    if not data["aqi"].empty:
        aqi_sql = text("""
            INSERT INTO air_quality (city_name, timestamp, us_aqi, pm2_5, pm10)
            VALUES (:city_name, :timestamp, :us_aqi, :pm2_5, :pm10)
            ON CONFLICT (city_name, timestamp) DO UPDATE SET
                us_aqi = EXCLUDED.us_aqi,
                pm2_5 = EXCLUDED.pm2_5,
                pm10 = EXCLUDED.pm10
        """)
        with engine.connect() as conn:
            for record in data["aqi"].to_dict(orient="records"):
                conn.execute(aqi_sql, record)
            conn.commit()
        stats["aqi_loaded"] = len(data["aqi"])

    # Load UV/Sun
    if not data["uv_sun"].empty:
        uv_sql = text("""
            INSERT INTO daily_forecast (city_name, date, uv_index_max, sunrise, sunset, temp_max, temp_min)
            VALUES (:city_name, :date, :uv_index_max, :sunrise, :sunset, :temp_max, :temp_min)
            ON CONFLICT (city_name, date) DO UPDATE SET
                uv_index_max = EXCLUDED.uv_index_max,
                sunrise = EXCLUDED.sunrise,
                sunset = EXCLUDED.sunset,
                temp_max = EXCLUDED.temp_max,
                temp_min = EXCLUDED.temp_min
        """)
        with engine.connect() as conn:
            for record in data["uv_sun"].to_dict(orient="records"):
                conn.execute(uv_sql, record)
            conn.commit()
        stats["uv_sun_loaded"] = len(data["uv_sun"])

    logger.info(f"Extended data loaded: {stats}")
    return stats


def run_extended_ingestion(past_days: int = 7, forecast_days: int = 2) -> dict:
    """Main extended ingestion pipeline."""
    logger.info("=" * 60)
    logger.info("STARTING EXTENDED DATA INGESTION (AQI + UV + Sun)")
    logger.info("=" * 60)

    init_extended_tables()
    data = fetch_all_extended(past_days, forecast_days)
    stats = load_extended_data(data)

    logger.info(f"EXTENDED INGESTION COMPLETE: {stats}")
    logger.info("=" * 60)
    return stats


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        from src.ingestion.city_config import CITIES
        city = CITIES[0]
        print(f"Testing extended fetch for {city['name']}...")

        aqi = fetch_air_quality(city, past_days=1, forecast_days=1)
        if aqi is not None:
            print(f"\nAQI: {len(aqi)} records")
            print(aqi.head(5).to_string(index=False))

        uv = fetch_uv_and_sun(city, past_days=1, forecast_days=1)
        if uv is not None:
            print(f"\nUV/Sun: {len(uv)} records")
            print(uv.to_string(index=False))
    else:
        backfill = 7
        for i, arg in enumerate(sys.argv):
            if arg == "--backfill" and i + 1 < len(sys.argv):
                backfill = int(sys.argv[i + 1])
        run_extended_ingestion(past_days=backfill)