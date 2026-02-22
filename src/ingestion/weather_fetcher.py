"""
Weather data fetcher for Open-Meteo API.

This module handles:
1. Fetching hourly weather data from the Open-Meteo API (free, no key needed)
2. Parsing the JSON response into a pandas DataFrame
3. Loading raw data into the PostgreSQL Bronze layer

WHY Open-Meteo?
- Completely free, no API key required
- Reliable uptime, well-documented
- Supports historical + forecast data in one call
- Returns hourly granularity (perfect for anomaly detection)

Usage:
    python -m src.ingestion.weather_fetcher              # Fetch current data for all cities
    python -m src.ingestion.weather_fetcher --backfill 7  # Fetch last 7 days
"""

import time
from datetime import datetime

import pandas as pd
import requests
from sqlalchemy import text

from src.config import OPEN_METEO_BASE_URL, BACKFILL_DAYS
from src.ingestion.city_config import get_all_cities, get_city_count
from src.utils.database import get_engine
from src.utils.logger import logger


# Weather variables we request from Open-Meteo
# These are the features our ML models will use
HOURLY_VARIABLES = [
    "temperature_2m",          # Air temperature at 2m height (°C)
    "relative_humidity_2m",    # Relative humidity at 2m height (%)
    "wind_speed_10m",          # Wind speed at 10m height (km/h)
    "precipitation",           # Total precipitation (mm)
    "pressure_msl",            # Mean sea level pressure (hPa)
    "cloud_cover",             # Total cloud cover (%)
]


def fetch_weather_for_city(city: dict, past_days: int = 7, forecast_days: int = 2) -> pd.DataFrame | None:
    """
    Fetch hourly weather data for a single city from Open-Meteo API.

    Args:
        city: Dict with 'name', 'lat', 'lon', 'state' keys
        past_days: Number of historical days to fetch (max 92)
        forecast_days: Number of forecast days to fetch (max 16)

    Returns:
        DataFrame with hourly weather data, or None if request fails

    API docs: https://open-meteo.com/en/docs
    """
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "hourly": ",".join(HOURLY_VARIABLES),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "auto",
    }

    try:
        logger.debug(f"Fetching weather for {city['name']}, {city['state']}")
        response = requests.get(OPEN_METEO_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching weather for {city['name']}")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error for {city['name']}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {city['name']}: {e}")
        return None

    # Parse the response into a DataFrame
    try:
        hourly_data = data.get("hourly", {})
        if not hourly_data or "time" not in hourly_data:
            logger.warning(f"No hourly data returned for {city['name']}")
            return None

        df = pd.DataFrame({
            "city_name": city["name"],
            "latitude": city["lat"],
            "longitude": city["lon"],
            "timestamp": pd.to_datetime(hourly_data["time"]),
            "temperature_2m": hourly_data.get("temperature_2m"),
            "relative_humidity": hourly_data.get("relative_humidity_2m"),
            "wind_speed": hourly_data.get("wind_speed_10m"),
            "precipitation": hourly_data.get("precipitation"),
            "pressure_msl": hourly_data.get("pressure_msl"),
            "cloud_cover": hourly_data.get("cloud_cover"),
        })

        logger.debug(f"Fetched {len(df)} records for {city['name']}")
        return df

    except (KeyError, ValueError) as e:
        logger.error(f"Error parsing data for {city['name']}: {e}")
        return None


def fetch_all_cities(past_days: int = 7, forecast_days: int = 2) -> pd.DataFrame:
    """
    Fetch weather data for ALL configured cities.

    Includes rate limiting (0.5s between requests) to be respectful
    to the free API. Open-Meteo allows 10,000 requests/day.

    Args:
        past_days: Number of historical days to fetch
        forecast_days: Number of forecast days to fetch

    Returns:
        Combined DataFrame for all cities
    """
    cities = get_all_cities()
    all_data = []
    success_count = 0
    fail_count = 0

    logger.info(f"Starting weather fetch for {get_city_count()} cities "
                f"(past_days={past_days}, forecast_days={forecast_days})")

    for i, city in enumerate(cities):
        df = fetch_weather_for_city(city, past_days, forecast_days)

        if df is not None and not df.empty:
            all_data.append(df)
            success_count += 1
        else:
            fail_count += 1

        # Rate limiting: 0.5 second between requests (respectful to free API)
        if i < len(cities) - 1:
            time.sleep(0.5)

    if not all_data:
        logger.error("No data fetched for any city!")
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(
        f"Fetch complete: {success_count} cities succeeded, {fail_count} failed, "
        f"{len(combined_df)} total records"
    )
    return combined_df


def load_to_bronze(df: pd.DataFrame) -> int:
    """
    Load raw weather data into the PostgreSQL Bronze layer.

    Uses INSERT ... ON CONFLICT DO NOTHING to handle duplicates
    (same city + timestamp won't be inserted twice).

    Args:
        df: DataFrame from fetch_all_cities()

    Returns:
        Number of new rows inserted
    """
    if df.empty:
        logger.warning("Empty DataFrame — nothing to load")
        return 0

    engine = get_engine()

    # Add ingestion metadata
    df["ingested_at"] = datetime.utcnow()
    df["source"] = "open-meteo"

    insert_sql = text("""
        INSERT INTO bronze_weather
            (city_name, latitude, longitude, timestamp,
             temperature_2m, relative_humidity, wind_speed,
             precipitation, pressure_msl, cloud_cover,
             ingested_at, source)
        VALUES
            (:city_name, :latitude, :longitude, :timestamp,
             :temperature_2m, :relative_humidity, :wind_speed,
             :precipitation, :pressure_msl, :cloud_cover,
             :ingested_at, :source)
        ON CONFLICT (city_name, timestamp) DO NOTHING
    """)

    rows_before = 0
    rows_after = 0

    try:
        with engine.connect() as conn:
            # Count existing rows
            result = conn.execute(text("SELECT COUNT(*) FROM bronze_weather"))
            rows_before = result.scalar()

            # Insert records
            records = df.to_dict(orient="records")
            for record in records:
                conn.execute(insert_sql, record)

            conn.commit()

            # Count after insert
            result = conn.execute(text("SELECT COUNT(*) FROM bronze_weather"))
            rows_after = result.scalar()

    except Exception as e:
        logger.error(f"Failed to load data to Bronze layer: {e}")
        raise

    new_rows = rows_after - rows_before
    logger.info(
        f"Bronze layer load complete: {new_rows} new rows inserted "
        f"(total: {rows_after}, duplicates skipped: {len(records) - new_rows})"
    )
    return new_rows


def run_ingestion(past_days: int = None, forecast_days: int = 2) -> dict:
    """
    Main ingestion pipeline: fetch from API → load to Bronze.

    This is the function called by the scheduler and CLI.

    Args:
        past_days: Override for historical days (default from config)
        forecast_days: Number of forecast days

    Returns:
        Dict with ingestion stats
    """
    if past_days is None:
        past_days = BACKFILL_DAYS

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("STARTING WEATHER DATA INGESTION")
    logger.info("=" * 60)

    # Step 1: Fetch from API
    df = fetch_all_cities(past_days=past_days, forecast_days=forecast_days)

    if df.empty:
        logger.error("Ingestion failed: no data fetched")
        return {"status": "failed", "records_fetched": 0, "records_inserted": 0}

    # Step 2: Load to Bronze layer
    new_rows = load_to_bronze(df)

    elapsed = round(time.time() - start_time, 2)
    stats = {
        "status": "success",
        "records_fetched": len(df),
        "records_inserted": new_rows,
        "cities_count": df["city_name"].nunique(),
        "time_elapsed_seconds": elapsed,
        "timestamp": datetime.utcnow().isoformat(),
    }

    logger.info(f"INGESTION COMPLETE in {elapsed}s: {stats}")
    logger.info("=" * 60)
    return stats


if __name__ == "__main__":
    """
    CLI usage:
        python -m src.ingestion.weather_fetcher              # Default (7 days backfill)
        python -m src.ingestion.weather_fetcher --backfill 14 # 14 days backfill
        python -m src.ingestion.weather_fetcher --test        # Test with 1 city only
    """
    import sys

    if "--test" in sys.argv:
        # Quick test: fetch one city, print results (no DB needed)
        from src.ingestion.city_config import CITIES
        test_city = CITIES[0]
        logger.info(f"Testing fetch for {test_city['name']}...")
        df = fetch_weather_for_city(test_city, past_days=2, forecast_days=1)
        if df is not None:
            print(f"\nFetched {len(df)} records for {test_city['name']}")
            print(f"\nColumns: {list(df.columns)}")
            print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"\nSample data:")
            print(df.head(10).to_string(index=False))
            print(f"\nBasic stats:")
            print(df.describe().round(2).to_string())
        else:
            print("Fetch failed!")
    else:
        # Full ingestion pipeline
        backfill = BACKFILL_DAYS
        for i, arg in enumerate(sys.argv):
            if arg == "--backfill" and i + 1 < len(sys.argv):
                backfill = int(sys.argv[i + 1])

        run_ingestion(past_days=backfill)