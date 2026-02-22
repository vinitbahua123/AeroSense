"""
Gold Layer: Feature Engineering.

Transforms clean Silver data into ML-ready features in the Gold layer.

Features created:
1. Rolling statistics (24h mean, std) — captures trends
2. Z-scores — how abnormal is this reading?
3. Lag features (1h, 6h, 12h, 24h) — for time-series forecasting
4. Rate of change — how fast is weather changing?
5. Interaction features — humidity × wind speed
6. Temporal features — hour, day of week, month, is_night

WHY these features?
- Rolling stats: anomaly detection needs "what's normal for this city?"
- Z-scores: >3 standard deviations = likely anomaly
- Lag features: XGBoost needs past values to predict future
- Rate of change: rapid pressure drops = storms incoming
- Temporal: weather follows daily/seasonal patterns

Usage:
    python -m src.transformation.feature_engineer           # Engineer features
    python -m src.transformation.feature_engineer --full     # Full refresh
"""

import pandas as pd
from sqlalchemy import text

from src.utils.database import get_engine
from src.utils.logger import logger


def extract_silver_data() -> pd.DataFrame:
    """Read clean data from Silver layer."""
    engine = get_engine()

    query = text("""
        SELECT city_name, latitude, longitude, timestamp,
               temperature_2m, relative_humidity, wind_speed,
               precipitation, pressure_msl, cloud_cover
        FROM silver_weather
        ORDER BY city_name, timestamp
    """)

    df = pd.read_sql(query, engine)
    logger.info(f"Extracted {len(df)} records from Silver layer")
    return df


def add_rolling_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 24-hour rolling mean and standard deviation for temperature.

    WHY 24 hours? Weather has a strong daily cycle (warm days, cool nights).
    A 24h window captures one full cycle, making deviations meaningful.
    """
    logger.info("Computing rolling statistics (24h window)...")

    df = df.sort_values(["city_name", "timestamp"]).reset_index(drop=True)

    # Rolling mean and std, computed per city
    df["temp_rolling_mean_24h"] = df.groupby("city_name")["temperature_2m"].transform(
        lambda x: x.rolling(window=24, min_periods=1).mean()
    )
    df["temp_rolling_std_24h"] = df.groupby("city_name")["temperature_2m"].transform(
        lambda x: x.rolling(window=24, min_periods=1).std()
    )

    # Fill any NaN std (happens when window has only 1 value)
    df["temp_rolling_std_24h"] = df["temp_rolling_std_24h"].fillna(1.0)

    return df


def add_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add z-score: how many standard deviations from the rolling mean.

    Z-score interpretation:
    - |z| < 2: normal weather
    - 2 < |z| < 3: unusual weather
    - |z| > 3: potential anomaly (investigate)

    This is the PRIMARY feature for anomaly detection.
    """
    logger.info("Computing z-scores...")

    # Avoid division by zero (std of 0 means constant temperature)
    safe_std = df["temp_rolling_std_24h"].replace(0, 1.0)
    df["temp_zscore"] = (df["temperature_2m"] - df["temp_rolling_mean_24h"]) / safe_std

    # Log extreme z-scores
    extreme = (df["temp_zscore"].abs() > 3).sum()
    if extreme > 0:
        logger.info(f"Found {extreme} records with |z-score| > 3 (potential anomalies)")

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features: temperature at previous time steps.

    WHY lags? XGBoost can't "look back in time" natively.
    By adding past values as features, we give it temporal context.

    Lags chosen:
    - 1h: immediate trend
    - 6h: quarter-day pattern
    - 12h: half-day (day vs night)
    - 24h: same hour yesterday (strongest weather predictor)
    """
    logger.info("Computing lag features...")

    for lag in [1, 6, 12, 24]:
        df[f"temp_lag_{lag}h"] = df.groupby("city_name")["temperature_2m"].shift(lag)

    return df


def add_rate_of_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rate of change features.

    - Temperature change over 1 hour: detects rapid warming/cooling
    - Pressure change over 3 hours: rapid drops = storm approaching
      (this is what meteorologists actually use!)
    """
    logger.info("Computing rate of change features...")

    df["temp_change_1h"] = df.groupby("city_name")["temperature_2m"].diff(1)
    df["pressure_change_3h"] = df.groupby("city_name")["pressure_msl"].diff(3)

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction features that capture combined weather effects.

    Humidity × Wind Speed: high values = wind chill or heat index danger
    This is a simple but effective feature that models miss without it.
    """
    logger.info("Computing interaction features...")

    df["humidity_wind_interaction"] = df["relative_humidity"] * df["wind_speed"] / 100.0

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features.

    WHY? Weather follows strong temporal patterns:
    - Hour: temperature peaks ~2pm, lowest ~5am
    - Day of week: slight urban heat island differences on weekdays
    - Month: seasonal patterns
    - is_night: different weather behavior at night
    """
    logger.info("Computing temporal features...")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if h < 6 or h > 20 else 0)

    return df


def handle_feature_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle nulls created by lag and rolling features.

    Lag features create NaN for the first N rows (no past data).
    We fill these with 0 for rate-of-change and column mean for lags.
    """
    logger.info("Handling feature nulls...")

    # Rate of change: 0 means "no change" (safe default)
    df["temp_change_1h"] = df["temp_change_1h"].fillna(0.0)
    df["pressure_change_3h"] = df["pressure_change_3h"].fillna(0.0)

    # Lag features: fill with current value (conservative estimate)
    for lag in [1, 6, 12, 24]:
        col = f"temp_lag_{lag}h"
        df[col] = df[col].fillna(df["temperature_2m"])

    return df


def load_to_gold(df: pd.DataFrame) -> int:
    """
    Load feature-engineered data into the Gold layer.

    Only stores the features needed for ML — not raw weather values
    (those stay in Silver for reference).
    """
    if df.empty:
        logger.warning("Empty DataFrame — nothing to load to Gold")
        return 0

    engine = get_engine()

    # Select only the columns we need for Gold
    gold_columns = [
        "city_name", "timestamp", "temperature_2m",
        "temp_rolling_mean_24h", "temp_rolling_std_24h", "temp_zscore",
        "temp_lag_1h", "temp_lag_6h", "temp_lag_12h", "temp_lag_24h",
        "temp_change_1h", "pressure_change_3h",
        "humidity_wind_interaction",
        "hour_of_day", "day_of_week", "month", "is_night",
    ]

    gold_df = df[gold_columns].copy()

    insert_sql = text("""
        INSERT INTO gold_weather_features
            (city_name, timestamp, temperature_2m,
             temp_rolling_mean_24h, temp_rolling_std_24h, temp_zscore,
             temp_lag_1h, temp_lag_6h, temp_lag_12h, temp_lag_24h,
             temp_change_1h, pressure_change_3h,
             humidity_wind_interaction,
             hour_of_day, day_of_week, month, is_night)
        VALUES
            (:city_name, :timestamp, :temperature_2m,
             :temp_rolling_mean_24h, :temp_rolling_std_24h, :temp_zscore,
             :temp_lag_1h, :temp_lag_6h, :temp_lag_12h, :temp_lag_24h,
             :temp_change_1h, :pressure_change_3h,
             :humidity_wind_interaction,
             :hour_of_day, :day_of_week, :month, :is_night)
        ON CONFLICT (city_name, timestamp) DO UPDATE SET
            temperature_2m = EXCLUDED.temperature_2m,
            temp_rolling_mean_24h = EXCLUDED.temp_rolling_mean_24h,
            temp_rolling_std_24h = EXCLUDED.temp_rolling_std_24h,
            temp_zscore = EXCLUDED.temp_zscore,
            temp_lag_1h = EXCLUDED.temp_lag_1h,
            temp_lag_6h = EXCLUDED.temp_lag_6h,
            temp_lag_12h = EXCLUDED.temp_lag_12h,
            temp_lag_24h = EXCLUDED.temp_lag_24h,
            temp_change_1h = EXCLUDED.temp_change_1h,
            pressure_change_3h = EXCLUDED.pressure_change_3h,
            humidity_wind_interaction = EXCLUDED.humidity_wind_interaction,
            hour_of_day = EXCLUDED.hour_of_day,
            day_of_week = EXCLUDED.day_of_week,
            month = EXCLUDED.month,
            is_night = EXCLUDED.is_night,
            created_at = NOW()
    """)

    try:
        with engine.connect() as conn:
            records = gold_df.to_dict(orient="records")
            for record in records:
                # Convert numpy types to Python native (PostgreSQL compatibility)
                clean_record = {}
                for key, value in record.items():
                    if pd.isna(value):
                        clean_record[key] = None
                    elif hasattr(value, "item"):
                        clean_record[key] = value.item()
                    else:
                        clean_record[key] = value
                conn.execute(insert_sql, clean_record)
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to load to Gold layer: {e}")
        raise

    logger.info(f"Gold layer load complete: {len(gold_df)} records upserted")
    return len(gold_df)


def run_feature_engineering(full_refresh: bool = False) -> dict:
    """
    Main feature engineering pipeline: Silver → features → Gold.

    Returns:
        Dict with feature engineering stats
    """
    logger.info("=" * 60)
    logger.info("STARTING GOLD LAYER FEATURE ENGINEERING")
    logger.info("=" * 60)

    # Step 1: Extract from Silver
    df = extract_silver_data()

    if df.empty:
        logger.warning("No data in Silver layer")
        return {"status": "no_data", "records_engineered": 0}

    # Step 2: Engineer features (order matters!)
    df = add_rolling_statistics(df)
    df = add_zscore(df)
    df = add_lag_features(df)
    df = add_rate_of_change(df)
    df = add_interaction_features(df)
    df = add_temporal_features(df)
    df = handle_feature_nulls(df)

    # Step 3: Load to Gold
    loaded = load_to_gold(df)

    # Compute stats
    anomaly_candidates = (df["temp_zscore"].abs() > 3).sum()

    stats = {
        "status": "success",
        "records_from_silver": len(df),
        "records_loaded_to_gold": loaded,
        "features_created": 13,
        "anomaly_candidates_zscore_gt_3": int(anomaly_candidates),
        "temp_zscore_mean": round(df["temp_zscore"].mean(), 4),
        "temp_zscore_max": round(df["temp_zscore"].abs().max(), 4),
    }

    logger.info(f"FEATURE ENGINEERING COMPLETE: {stats}")
    logger.info("=" * 60)
    return stats


if __name__ == "__main__":
    import sys
    full = "--full" in sys.argv
    stats = run_feature_engineering(full_refresh=full)
    print(f"\nFeature engineering stats: {stats}")