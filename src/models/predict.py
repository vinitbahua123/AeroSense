"""
Prediction Service — Serves trained ML models for real-time inference.

Loads the best XGBoost model from MLflow and makes temperature predictions.
Also generates weather alerts based on anomaly detection + thresholds.

This is what makes the project go from "I trained a model" to "I serve predictions."

Usage:
    python -m src.models.predict --city "New York"
    python -m src.models.predict --alerts
"""

import os
import glob

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.utils.database import get_engine
from src.utils.logger import logger


# ============================================
# Model Loading
# ============================================

def find_latest_model_path(experiment_name: str, model_subdir: str) -> str | None:
    """
    Find the latest model artifact path from MLflow local storage.
    Searches mlruns/ for the most recent run of a given experiment.
    """
    mlruns_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "mlruns")

    # Find experiment directories
    for exp_dir in glob.glob(os.path.join(mlruns_dir, "*")):
        meta_file = os.path.join(exp_dir, "meta.yaml")
        if os.path.exists(meta_file):
            with open(meta_file) as f:
                content = f.read()
                if experiment_name in content:
                    # Find the latest run
                    runs = []
                    for run_dir in glob.glob(os.path.join(exp_dir, "*")):
                        if os.path.isdir(run_dir) and run_dir != os.path.join(exp_dir, "meta.yaml"):
                            model_path = os.path.join(run_dir, "artifacts", model_subdir)
                            if os.path.exists(model_path):
                                runs.append(run_dir)
                    if runs:
                        # Return the latest run (sorted by name, which includes timestamp)
                        latest = sorted(runs)[-1]
                        return os.path.join(latest, "artifacts", model_subdir)
    return None


def load_forecast_model():
    """Load the trained XGBoost forecast model."""
    try:
        import mlflow.sklearn
        model_path = find_latest_model_path("temperature-forecasting", "forecast_model")
        if model_path:
            model = mlflow.sklearn.load_model(model_path)
            logger.info("Forecast model loaded successfully")
            return model
    except Exception as e:
        logger.warning(f"Could not load forecast model from MLflow: {e}")

    # Fallback: train a quick model on the fly
    logger.info("Training a quick forecast model...")
    return train_quick_forecast_model()


def train_quick_forecast_model():
    """Train a quick XGBoost model if MLflow model is not available."""
    from xgboost import XGBRegressor

    engine = get_engine()
    query = text("""
        SELECT temperature_2m, temp_lag_1h, temp_lag_6h, temp_lag_12h, temp_lag_24h,
               temp_rolling_mean_24h, temp_rolling_std_24h,
               temp_change_1h, pressure_change_3h,
               humidity_wind_interaction,
               hour_of_day, day_of_week, month, is_night
        FROM gold_weather_features
        ORDER BY timestamp
    """)
    df = pd.read_sql(query, engine)

    if df.empty:
        return None

    features = ["temp_lag_1h", "temp_lag_6h", "temp_lag_12h", "temp_lag_24h",
                "temp_rolling_mean_24h", "temp_rolling_std_24h",
                "temp_change_1h", "pressure_change_3h",
                "humidity_wind_interaction",
                "hour_of_day", "day_of_week", "month", "is_night"]

    X = df[features]
    y = df["temperature_2m"]

    model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0)
    model.fit(X, y)
    logger.info("Quick forecast model trained")
    return model


# ============================================
# Predictions
# ============================================

FORECAST_FEATURES = [
    "temp_lag_1h", "temp_lag_6h", "temp_lag_12h", "temp_lag_24h",
    "temp_rolling_mean_24h", "temp_rolling_std_24h",
    "temp_change_1h", "pressure_change_3h",
    "humidity_wind_interaction",
    "hour_of_day", "day_of_week", "month", "is_night",
]


def predict_for_city(city_name: str, model=None) -> pd.DataFrame:
    """
    Generate temperature predictions for a city using the trained model.
    Compares model predictions vs actual values.
    """
    if model is None:
        model = load_forecast_model()
        if model is None:
            logger.error("No forecast model available")
            return pd.DataFrame()

    engine = get_engine()
    query = text("""
        SELECT city_name, timestamp, temperature_2m,
               temp_lag_1h, temp_lag_6h, temp_lag_12h, temp_lag_24h,
               temp_rolling_mean_24h, temp_rolling_std_24h,
               temp_change_1h, pressure_change_3h,
               humidity_wind_interaction,
               hour_of_day, day_of_week, month, is_night
        FROM gold_weather_features
        WHERE city_name = :city
        ORDER BY timestamp DESC
        LIMIT 72
    """)

    df = pd.read_sql(query, engine, params={"city": city_name})
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Make predictions
    X = df[FORECAST_FEATURES]
    df["predicted_temp"] = model.predict(X)
    df["prediction_error"] = df["temperature_2m"] - df["predicted_temp"]
    df["abs_error"] = df["prediction_error"].abs()

    return df


# ============================================
# Weather Alerts
# ============================================

def generate_alerts() -> list[dict]:
    """
    Generate weather alerts based on multiple criteria:
    1. Temperature anomalies (z-score > 3)
    2. Rapid temperature changes (> 5°C in 1 hour)
    3. Rapid pressure drops (> 5 hPa in 3 hours — storm indicator)
    4. High AQI (> 100 — unhealthy)
    5. High UV index (> 8 — very high)
    """
    engine = get_engine()
    alerts = []

    # Alert 1: Temperature anomalies
    try:
        query = text("""
            SELECT city_name, timestamp, temperature_2m, temp_zscore,
                   temp_rolling_mean_24h
            FROM gold_weather_features
            WHERE ABS(temp_zscore) > 3
            ORDER BY timestamp DESC
            LIMIT 20
        """)
        anomalies = pd.read_sql(query, engine)
        for _, row in anomalies.iterrows():
            alert_type = "Unusual Warming" if row["temp_zscore"] > 0 else "Unusual Cooling"
            severity = "high" if abs(row["temp_zscore"]) > 3.5 else "medium"
            alerts.append({
                "city": row["city_name"],
                "timestamp": str(row["timestamp"]),
                "type": alert_type,
                "severity": severity,
                "message": f"{row['city_name']}: {row['temperature_2m']:.1f}°C vs expected {row['temp_rolling_mean_24h']:.1f}°C (z-score: {row['temp_zscore']:.2f})",
                "category": "temperature_anomaly",
            })
    except Exception as e:
        logger.error(f"Error generating temperature alerts: {e}")

    # Alert 2: Rapid temperature changes
    try:
        query = text("""
            SELECT city_name, timestamp, temperature_2m, temp_change_1h
            FROM gold_weather_features
            WHERE ABS(temp_change_1h) > 5
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        rapid_changes = pd.read_sql(query, engine)
        for _, row in rapid_changes.iterrows():
            direction = "rise" if row["temp_change_1h"] > 0 else "drop"
            alerts.append({
                "city": row["city_name"],
                "timestamp": str(row["timestamp"]),
                "type": f"Rapid Temperature {direction.title()}",
                "severity": "medium",
                "message": f"{row['city_name']}: {abs(row['temp_change_1h']):.1f}°C {direction} in 1 hour",
                "category": "rapid_change",
            })
    except Exception as e:
        logger.error(f"Error generating rapid change alerts: {e}")

    # Alert 3: Rapid pressure drops (storm indicator)
    try:
        query = text("""
            SELECT city_name, timestamp, pressure_change_3h
            FROM gold_weather_features
            WHERE pressure_change_3h < -5
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        pressure_drops = pd.read_sql(query, engine)
        for _, row in pressure_drops.iterrows():
            alerts.append({
                "city": row["city_name"],
                "timestamp": str(row["timestamp"]),
                "type": "Storm Warning",
                "severity": "high",
                "message": f"{row['city_name']}: Pressure dropped {abs(row['pressure_change_3h']):.1f} hPa in 3 hours — potential storm approaching",
                "category": "pressure_drop",
            })
    except Exception as e:
        logger.error(f"Error generating pressure alerts: {e}")

    # Alert 4: High AQI
    try:
        query = text("""
            SELECT city_name, timestamp, us_aqi
            FROM air_quality
            WHERE us_aqi > 100
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        high_aqi = pd.read_sql(query, engine)
        for _, row in high_aqi.iterrows():
            severity = "high" if row["us_aqi"] > 150 else "medium"
            label = "Unhealthy" if row["us_aqi"] > 150 else "Unhealthy for Sensitive Groups"
            alerts.append({
                "city": row["city_name"],
                "timestamp": str(row["timestamp"]),
                "type": f"Air Quality: {label}",
                "severity": severity,
                "message": f"{row['city_name']}: AQI {row['us_aqi']:.0f} — {label}",
                "category": "air_quality",
            })
    except Exception as e:
        logger.debug(f"Air quality table may not exist yet: {e}")

    # Alert 5: High UV
    try:
        query = text("""
            SELECT city_name, date, uv_index_max
            FROM daily_forecast
            WHERE uv_index_max > 8
            ORDER BY date DESC
            LIMIT 10
        """)
        high_uv = pd.read_sql(query, engine)
        for _, row in high_uv.iterrows():
            alerts.append({
                "city": row["city_name"],
                "timestamp": str(row["date"]),
                "type": "Very High UV Index",
                "severity": "medium",
                "message": f"{row['city_name']}: UV index {row['uv_index_max']:.1f} — sun protection essential",
                "category": "uv_index",
            })
    except Exception as e:
        logger.debug(f"Daily forecast table may not exist yet: {e}")

    # Sort by severity then time
    severity_order = {"high": 0, "medium": 1, "low": 2}
    alerts.sort(key=lambda a: (severity_order.get(a["severity"], 2), a["timestamp"]), reverse=False)

    logger.info(f"Generated {len(alerts)} weather alerts")
    return alerts


if __name__ == "__main__":
    import sys

    if "--alerts" in sys.argv:
        alerts = generate_alerts()
        print(f"\n{'='*60}")
        print(f"WEATHER ALERTS ({len(alerts)} total)")
        print(f"{'='*60}")
        for alert in alerts:
            icon = "🔴" if alert["severity"] == "high" else "🟡"
            print(f"{icon} [{alert['type']}] {alert['message']}")
    else:
        city = "New York"
        for i, arg in enumerate(sys.argv):
            if arg == "--city" and i + 1 < len(sys.argv):
                city = sys.argv[i + 1]

        df = predict_for_city(city)
        if not df.empty:
            print(f"\nPredictions for {city} (last 72 hours):")
            print(f"Mean Absolute Error: {df['abs_error'].mean():.3f}°C")
            print(f"Max Error: {df['abs_error'].max():.3f}°C")
            print(f"\nSample predictions:")
            display = df[["timestamp", "temperature_2m", "predicted_temp", "prediction_error"]].tail(10)
            display.columns = ["Time", "Actual", "Predicted", "Error"]
            print(display.to_string(index=False))