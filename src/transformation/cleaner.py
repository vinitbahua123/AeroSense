"""
Silver Layer: Data Cleaning and Validation.

Transforms raw Bronze data into clean, validated Silver data.

What this does:
1. Reads raw data from bronze_weather table
2. Handles null/missing values (forward fill — makes sense for weather)
3. Removes duplicates (same city + timestamp)
4. Validates data ranges (temperature between -60 and 60°C, etc.)
5. Flags bad records with quality scores
6. Writes clean data to silver_weather table

WHY forward fill for weather?
- Weather changes gradually (temperature doesn't jump from 20°C to null to 25°C)
- Forward fill assumes "last known value is still valid" which is reasonable
- This is an interview-ready answer for "how do you handle missing data?"

Usage:
    python -m src.transformation.cleaner           # Clean latest Bronze data
    python -m src.transformation.cleaner --full     # Reclean all Bronze data
"""

import pandas as pd
from sqlalchemy import text

from src.utils.database import get_engine
from src.utils.logger import logger


# Validation rules: column → (min, max)
# Values outside these ranges are flagged as bad
VALIDATION_RULES = {
    "temperature_2m": (-60.0, 60.0),       # °C — coldest/hottest on Earth
    "relative_humidity": (0.0, 100.0),      # Percentage
    "wind_speed": (0.0, 250.0),             # km/h — hurricane force ~250
    "precipitation": (0.0, 500.0),          # mm — extreme rainfall
    "pressure_msl": (870.0, 1084.0),        # hPa — recorded extremes
    "cloud_cover": (0.0, 100.0),            # Percentage
}

# Columns that must not be null after cleaning
REQUIRED_COLUMNS = [
    "temperature_2m", "relative_humidity", "wind_speed",
    "precipitation", "pressure_msl", "cloud_cover"
]


def extract_bronze_data(since: str = None) -> pd.DataFrame:
    """
    Read raw data from Bronze layer.

    Args:
        since: Optional ISO timestamp — only fetch records ingested after this time.
               If None, fetches all records.

    Returns:
        DataFrame with raw weather data
    """
    engine = get_engine()

    if since:
        query = text("""
            SELECT city_name, latitude, longitude, timestamp,
                   temperature_2m, relative_humidity, wind_speed,
                   precipitation, pressure_msl, cloud_cover
            FROM bronze_weather
            WHERE ingested_at > :since
            ORDER BY city_name, timestamp
        """)
        df = pd.read_sql(query, engine, params={"since": since})
    else:
        query = text("""
            SELECT city_name, latitude, longitude, timestamp,
                   temperature_2m, relative_humidity, wind_speed,
                   precipitation, pressure_msl, cloud_cover
            FROM bronze_weather
            ORDER BY city_name, timestamp
        """)
        df = pd.read_sql(query, engine)

    logger.info(f"Extracted {len(df)} records from Bronze layer")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle null/missing values using forward fill within each city.

    Strategy:
    - Forward fill: use last known value (weather is continuous)
    - Backward fill: catch any leading nulls
    - Fill remaining with column median (last resort)

    Returns:
        DataFrame with no null values in weather columns
    """
    initial_nulls = df[REQUIRED_COLUMNS].isnull().sum().sum()

    if initial_nulls == 0:
        logger.info("No missing values found")
        return df

    # Forward fill within each city (weather is continuous per location)
    df[REQUIRED_COLUMNS] = df.groupby("city_name")[REQUIRED_COLUMNS].transform(
        lambda group: group.ffill().bfill()
    )

    # If still any nulls (entire city column was null), fill with global median
    for col in REQUIRED_COLUMNS:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.warning(f"Filled remaining nulls in {col} with median: {median_val}")

    remaining_nulls = df[REQUIRED_COLUMNS].isnull().sum().sum()
    logger.info(f"Missing values handled: {initial_nulls} found → {remaining_nulls} remaining")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate records (same city + timestamp).
    Keeps the first occurrence.
    """
    before = len(df)
    df = df.drop_duplicates(subset=["city_name", "timestamp"], keep="first")
    removed = before - len(df)

    if removed > 0:
        logger.info(f"Removed {removed} duplicate records")
    else:
        logger.info("No duplicates found")

    return df.reset_index(drop=True)


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that all values fall within expected ranges.
    Records with out-of-range values get a lower quality score.

    Returns:
        DataFrame with 'quality_score' column added (0.0 to 1.0)
    """
    # Start with perfect quality
    df["quality_score"] = 1.0
    total_violations = 0

    for col, (min_val, max_val) in VALIDATION_RULES.items():
        if col not in df.columns:
            continue

        # Find out-of-range values
        violations = (df[col] < min_val) | (df[col] > max_val)
        violation_count = violations.sum()

        if violation_count > 0:
            # Reduce quality score for bad records
            df.loc[violations, "quality_score"] -= (1.0 / len(VALIDATION_RULES))
            # Clip values to valid range (don't discard, just constrain)
            df[col] = df[col].clip(lower=min_val, upper=max_val)
            logger.warning(
                f"Column '{col}': {violation_count} values out of range "
                f"[{min_val}, {max_val}] — clipped to bounds"
            )
            total_violations += violation_count

    # Ensure quality score is between 0 and 1
    df["quality_score"] = df["quality_score"].clip(0.0, 1.0)

    good_records = (df["quality_score"] == 1.0).sum()
    logger.info(
        f"Validation complete: {good_records}/{len(df)} records passed all checks "
        f"({total_violations} total violations found)"
    )
    return df


def load_to_silver(df: pd.DataFrame) -> int:
    """
    Load cleaned data into the Silver layer.
    Uses ON CONFLICT to handle re-runs without duplicating data.

    Returns:
        Number of new rows inserted
    """
    if df.empty:
        logger.warning("Empty DataFrame — nothing to load to Silver")
        return 0

    engine = get_engine()

    insert_sql = text("""
        INSERT INTO silver_weather
            (city_name, latitude, longitude, timestamp,
             temperature_2m, relative_humidity, wind_speed,
             precipitation, pressure_msl, cloud_cover,
             quality_score)
        VALUES
            (:city_name, :latitude, :longitude, :timestamp,
             :temperature_2m, :relative_humidity, :wind_speed,
             :precipitation, :pressure_msl, :cloud_cover,
             :quality_score)
        ON CONFLICT (city_name, timestamp) DO UPDATE SET
            temperature_2m = EXCLUDED.temperature_2m,
            relative_humidity = EXCLUDED.relative_humidity,
            wind_speed = EXCLUDED.wind_speed,
            precipitation = EXCLUDED.precipitation,
            pressure_msl = EXCLUDED.pressure_msl,
            cloud_cover = EXCLUDED.cloud_cover,
            quality_score = EXCLUDED.quality_score,
            validated_at = NOW()
    """)

    try:
        with engine.connect() as conn:
            records = df.to_dict(orient="records")
            for record in records:
                conn.execute(insert_sql, record)
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to load to Silver layer: {e}")
        raise

    logger.info(f"Silver layer load complete: {len(df)} records upserted")
    return len(df)


def run_cleaning(full_refresh: bool = False) -> dict:
    """
    Main cleaning pipeline: Bronze → clean → validate → Silver.

    Args:
        full_refresh: If True, reclean ALL Bronze data.
                      If False, only clean new data since last run.

    Returns:
        Dict with cleaning stats
    """
    logger.info("=" * 60)
    logger.info("STARTING SILVER LAYER CLEANING")
    logger.info("=" * 60)

    # Step 1: Extract from Bronze
    df = extract_bronze_data(since=None if full_refresh else None)
    # NOTE: For incremental loads, we'd track last_cleaned_at
    # and pass it as 'since'. For now, full refresh is fine.

    if df.empty:
        logger.warning("No data in Bronze layer to clean")
        return {"status": "no_data", "records_cleaned": 0}

    initial_count = len(df)

    # Step 2: Remove duplicates
    df = remove_duplicates(df)

    # Step 3: Handle missing values
    df = handle_missing_values(df)

    # Step 4: Validate ranges
    df = validate_ranges(df)

    # Step 5: Load to Silver
    loaded = load_to_silver(df)

    stats = {
        "status": "success",
        "records_from_bronze": initial_count,
        "records_after_dedup": len(df),
        "records_loaded_to_silver": loaded,
        "quality_score_mean": round(df["quality_score"].mean(), 4),
        "quality_score_min": round(df["quality_score"].min(), 4),
    }

    logger.info(f"CLEANING COMPLETE: {stats}")
    logger.info("=" * 60)
    return stats


if __name__ == "__main__":
    import sys
    full = "--full" in sys.argv
    stats = run_cleaning(full_refresh=full)
    print(f"\nCleaning stats: {stats}")