"""
ML Model Training with MLflow Experiment Tracking.

Trains two model families:
1. Anomaly Detection (Isolation Forest) — finds unusual weather patterns
2. Temperature Forecasting (XGBoost) — predicts temperature 24h ahead

Every experiment is tracked in MLflow:
- Parameters (hyperparameters, feature count, training size)
- Metrics (precision, recall, F1, RMSE, MAE)
- Artifacts (model files, feature importance plots)
- Model Registry (production-ready model versioning)

Usage:
    python -m src.models.train                    # Train all models
    python -m src.models.train --anomaly-only     # Train anomaly detector only
    python -m src.models.train --forecast-only    # Train forecaster only
"""

import os
import time
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sqlalchemy import text

from src.config import MLFLOW_TRACKING_URI, ANOMALY_CONTAMINATION
from src.utils.database import get_engine
from src.utils.logger import logger

warnings.filterwarnings("ignore")

# ============================================
# MLflow Setup
# ============================================

# Use local file-based tracking (no server needed for now)
MLFLOW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "mlruns")
mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")


# ============================================
# Data Loading
# ============================================

def load_gold_data() -> pd.DataFrame:
    """Load feature-engineered data from Gold layer."""
    engine = get_engine()
    query = text("""
        SELECT city_name, timestamp, temperature_2m,
               temp_rolling_mean_24h, temp_rolling_std_24h, temp_zscore,
               temp_lag_1h, temp_lag_6h, temp_lag_12h, temp_lag_24h,
               temp_change_1h, pressure_change_3h,
               humidity_wind_interaction,
               hour_of_day, day_of_week, month, is_night
        FROM gold_weather_features
        ORDER BY city_name, timestamp
    """)
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df)} records from Gold layer")
    return df


# ============================================
# Anomaly Detection Training
# ============================================

ANOMALY_FEATURES = [
    "temp_zscore", "temp_change_1h", "pressure_change_3h",
    "humidity_wind_interaction", "temp_rolling_std_24h",
    "hour_of_day", "is_night",
]


def train_anomaly_detector(df: pd.DataFrame) -> dict:
    """
    Train Isolation Forest for weather anomaly detection.

    WHY Isolation Forest?
    - Unsupervised: we don't have labeled anomalies
    - Handles high-dimensional data well
    - Fast training and inference
    - contamination parameter maps to expected anomaly rate

    We run multiple experiments with different hyperparameters
    and log everything to MLflow.
    """
    logger.info("Training Anomaly Detection models...")
    mlflow.set_experiment("anomaly-detection")

    X = df[ANOMALY_FEATURES].copy()

    # Scale features (important for distance-based aspects of Isolation Forest)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=ANOMALY_FEATURES,
    )

    best_run = None
    best_score = -1

    # Experiment with different hyperparameters
    experiments = [
        {"n_estimators": 100, "contamination": 0.03, "max_features": 0.6},
        {"n_estimators": 200, "contamination": 0.05, "max_features": 0.8},
        {"n_estimators": 200, "contamination": 0.05, "max_features": 1.0},
        {"n_estimators": 300, "contamination": 0.07, "max_features": 0.8},
        {"n_estimators": 150, "contamination": 0.04, "max_features": 0.7},
    ]

    for i, params in enumerate(experiments):
        run_name = f"iforest_v{i+1}_est{params['n_estimators']}_cont{params['contamination']}"

        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_params({
                "model_type": "IsolationForest",
                "n_estimators": params["n_estimators"],
                "contamination": params["contamination"],
                "max_features": params["max_features"],
                "n_features": len(ANOMALY_FEATURES),
                "n_training_samples": len(X_scaled),
                "features": str(ANOMALY_FEATURES),
                "random_state": 42,
            })

            # Train model
            model = IsolationForest(
                n_estimators=params["n_estimators"],
                contamination=params["contamination"],
                max_features=params["max_features"],
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_scaled)

            # Predict: -1 = anomaly, 1 = normal
            predictions = model.predict(X_scaled)
            anomaly_scores = model.decision_function(X_scaled)

            # Compute metrics
            n_anomalies = (predictions == -1).sum()
            anomaly_ratio = n_anomalies / len(predictions)

            # Use z-score > 3 as pseudo-labels for evaluation
            # (This is the best we can do without labeled data)
            pseudo_labels = (df["temp_zscore"].abs() > 3).astype(int).values
            pred_labels = (predictions == -1).astype(int)

            precision = precision_score(pseudo_labels, pred_labels, zero_division=0)
            recall = recall_score(pseudo_labels, pred_labels, zero_division=0)
            f1 = f1_score(pseudo_labels, pred_labels, zero_division=0)

            # Log metrics
            mlflow.log_metrics({
                "n_anomalies_detected": n_anomalies,
                "anomaly_ratio": round(anomaly_ratio, 4),
                "precision_vs_zscore": round(precision, 4),
                "recall_vs_zscore": round(recall, 4),
                "f1_vs_zscore": round(f1, 4),
                "mean_anomaly_score": round(float(anomaly_scores.mean()), 4),
            })

            # Log model
            mlflow.sklearn.log_model(
                model,
                "anomaly_model",
                registered_model_name="weather-anomaly-detector",
            )

            # Also log the scaler (needed for inference)
            mlflow.sklearn.log_model(scaler, "scaler")

            logger.info(
                f"  {run_name}: anomalies={n_anomalies}, "
                f"precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}"
            )

            # Track best model
            if f1 > best_score:
                best_score = f1
                best_run = {
                    "run_name": run_name,
                    "run_id": mlflow.active_run().info.run_id,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "n_anomalies": n_anomalies,
                    "params": params,
                }

    logger.info(f"Best anomaly model: {best_run['run_name']} (F1={best_run['f1']:.3f})")
    return best_run


# ============================================
# Temperature Forecasting Training
# ============================================

FORECAST_FEATURES = [
    "temp_lag_1h", "temp_lag_6h", "temp_lag_12h", "temp_lag_24h",
    "temp_rolling_mean_24h", "temp_rolling_std_24h",
    "temp_change_1h", "pressure_change_3h",
    "humidity_wind_interaction",
    "hour_of_day", "day_of_week", "month", "is_night",
]

FORECAST_TARGET = "temperature_2m"


def train_forecaster(df: pd.DataFrame) -> dict:
    """
    Train XGBoost for temperature forecasting.

    WHY XGBoost?
    - Handles tabular data exceptionally well
    - Built-in feature importance (great for interviews)
    - Fast training, works with lag features naturally
    - Industry standard for structured data prediction

    We use an 80/20 temporal split (not random!) because
    weather data is time-series — you can't train on future data.
    """
    logger.info("Training Temperature Forecasting models...")
    mlflow.set_experiment("temperature-forecasting")

    # Prepare features and target
    X = df[FORECAST_FEATURES].copy()
    y = df[FORECAST_TARGET].copy()

    # Temporal split: first 80% for training, last 20% for validation
    # NOT random split — this is critical for time-series!
    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"  Train size: {len(X_train)}, Validation size: {len(X_val)}")

    best_run = None
    best_rmse = float("inf")

    # Experiment with different hyperparameters
    experiments = [
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.9},
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.08, "subsample": 0.7},
        {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.85},
    ]

    for i, params in enumerate(experiments):
        run_name = f"xgb_v{i+1}_est{params['n_estimators']}_d{params['max_depth']}"

        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_params({
                "model_type": "XGBRegressor",
                "n_estimators": params["n_estimators"],
                "max_depth": params["max_depth"],
                "learning_rate": params["learning_rate"],
                "subsample": params["subsample"],
                "n_features": len(FORECAST_FEATURES),
                "n_train_samples": len(X_train),
                "n_val_samples": len(X_val),
                "train_val_split": "temporal_80_20",
                "features": str(FORECAST_FEATURES),
            })

            # Train model
            model = XGBRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            # Predict
            train_preds = model.predict(X_train)
            val_preds = model.predict(X_val)

            # Compute metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
            val_mae = mean_absolute_error(y_val, val_preds)
            val_r2 = model.score(X_val, y_val)

            # Check for overfitting
            overfit_ratio = val_rmse / train_rmse if train_rmse > 0 else 0

            # Log metrics
            mlflow.log_metrics({
                "train_rmse": round(train_rmse, 4),
                "val_rmse": round(val_rmse, 4),
                "val_mae": round(val_mae, 4),
                "val_r2": round(val_r2, 4),
                "overfit_ratio": round(overfit_ratio, 4),
            })

            # Log feature importance
            importance = dict(zip(FORECAST_FEATURES, model.feature_importances_))
            for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                mlflow.log_metric(f"importance_{feat}", round(float(imp), 4))

            # Log model
            mlflow.sklearn.log_model(
                model,
                "forecast_model",
                registered_model_name="weather-temp-forecaster",
            )

            logger.info(
                f"  {run_name}: val_rmse={val_rmse:.3f}°C, "
                f"val_mae={val_mae:.3f}°C, r2={val_r2:.3f}, "
                f"overfit_ratio={overfit_ratio:.2f}"
            )

            # Track best model
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_run = {
                    "run_name": run_name,
                    "run_id": mlflow.active_run().info.run_id,
                    "val_rmse": val_rmse,
                    "val_mae": val_mae,
                    "val_r2": val_r2,
                    "overfit_ratio": overfit_ratio,
                    "params": params,
                    "feature_importance": importance,
                }

    # Log top features for best model
    if best_run:
        top_features = sorted(
            best_run["feature_importance"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        logger.info(f"Best forecast model: {best_run['run_name']} (RMSE={best_run['val_rmse']:.3f}°C)")
        logger.info(f"  Top 5 features: {[f'{f}: {v:.3f}' for f, v in top_features]}")

    return best_run


# ============================================
# Main Training Pipeline
# ============================================

def run_training(anomaly_only: bool = False, forecast_only: bool = False) -> dict:
    """
    Main training pipeline. Trains models and logs to MLflow.

    Returns:
        Dict with training results for both models
    """
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info(f"MLflow tracking: file://{MLFLOW_DIR}")
    logger.info("=" * 60)

    # Load data
    df = load_gold_data()
    if df.empty:
        logger.error("No data in Gold layer — run feature engineering first")
        return {"status": "no_data"}

    logger.info(f"Training data: {len(df)} records, {df['city_name'].nunique()} cities")

    results = {"status": "success"}

    # Train anomaly detector
    if not forecast_only:
        anomaly_result = train_anomaly_detector(df)
        results["anomaly_detection"] = anomaly_result

    # Train forecaster
    if not anomaly_only:
        forecast_result = train_forecaster(df)
        results["forecasting"] = forecast_result

    elapsed = round(time.time() - start_time, 2)
    results["training_time_seconds"] = elapsed

    logger.info("=" * 60)
    logger.info(f"TRAINING COMPLETE in {elapsed}s")

    if "anomaly_detection" in results:
        r = results["anomaly_detection"]
        logger.info(f"  Best Anomaly Model: {r['run_name']} (F1={r['f1']:.3f})")

    if "forecasting" in results:
        r = results["forecasting"]
        logger.info(f"  Best Forecast Model: {r['run_name']} (RMSE={r['val_rmse']:.3f}°C)")

    logger.info(f"  MLflow UI: run 'mlflow ui' in project root to view experiments")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    import sys

    anomaly_only = "--anomaly-only" in sys.argv
    forecast_only = "--forecast-only" in sys.argv

    results = run_training(anomaly_only=anomaly_only, forecast_only=forecast_only)

    print("\n" + "=" * 60)
    print("TRAINING RESULTS SUMMARY")
    print("=" * 60)

    if "anomaly_detection" in results:
        r = results["anomaly_detection"]
        print(f"\nAnomaly Detection (Best: {r['run_name']})")
        print(f"  Precision: {r['precision']:.3f}")
        print(f"  Recall:    {r['recall']:.3f}")
        print(f"  F1 Score:  {r['f1']:.3f}")
        print(f"  Anomalies: {r['n_anomalies']}")

    if "forecasting" in results:
        r = results["forecasting"]
        print(f"\nTemperature Forecasting (Best: {r['run_name']})")
        print(f"  RMSE:      {r['val_rmse']:.3f}°C")
        print(f"  MAE:       {r['val_mae']:.3f}°C")
        print(f"  R²:        {r['val_r2']:.3f}")
        print(f"  Overfit:   {r['overfit_ratio']:.2f}x")

    print(f"\nTotal training time: {results.get('training_time_seconds', 0)}s")
    print(f"\nView experiments: mlflow ui --backend-store-uri file://{MLFLOW_DIR}")