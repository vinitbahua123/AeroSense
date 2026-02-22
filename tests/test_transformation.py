"""
Tests for the transformation module (Silver + Gold layers).

Run with: pytest tests/test_transformation.py -v
"""

import pandas as pd
import numpy as np
import pytest

from src.transformation.cleaner import (
    handle_missing_values,
    remove_duplicates,
    validate_ranges,
    VALIDATION_RULES,
)
from src.transformation.feature_engineer import (
    add_rolling_statistics,
    add_zscore,
    add_lag_features,
    add_rate_of_change,
    add_interaction_features,
    add_temporal_features,
    handle_feature_nulls,
)


# ============================================
# Helper: Create sample data
# ============================================

def make_sample_df(n_hours=48, n_cities=2) -> pd.DataFrame:
    """Create a sample weather DataFrame for testing."""
    rows = []
    for city_idx in range(n_cities):
        city_name = f"City_{city_idx}"
        for hour in range(n_hours):
            rows.append({
                "city_name": city_name,
                "latitude": 40.0 + city_idx,
                "longitude": -74.0 - city_idx,
                "timestamp": pd.Timestamp("2026-02-14") + pd.Timedelta(hours=hour),
                "temperature_2m": 5.0 + 3.0 * np.sin(hour * np.pi / 12) + np.random.normal(0, 0.5),
                "relative_humidity": 70.0 + np.random.normal(0, 5),
                "wind_speed": 10.0 + np.random.normal(0, 2),
                "precipitation": max(0, np.random.normal(0, 0.5)),
                "pressure_msl": 1013.0 + np.random.normal(0, 3),
                "cloud_cover": 50.0 + np.random.normal(0, 15),
            })
    df = pd.DataFrame(rows)
    # Clip to valid ranges
    df["relative_humidity"] = df["relative_humidity"].clip(0, 100)
    df["wind_speed"] = df["wind_speed"].clip(0, 250)
    df["precipitation"] = df["precipitation"].clip(0, 500)
    df["cloud_cover"] = df["cloud_cover"].clip(0, 100)
    return df


# ============================================
# Cleaner Tests (Silver Layer)
# ============================================

class TestCleaner:
    """Tests for Silver layer cleaning functions."""

    def test_handle_missing_values_fills_nulls(self):
        """Forward fill should eliminate nulls."""
        df = make_sample_df(n_hours=24, n_cities=1)
        # Inject some nulls
        df.loc[5, "temperature_2m"] = None
        df.loc[10, "wind_speed"] = None

        result = handle_missing_values(df)

        assert result["temperature_2m"].isnull().sum() == 0
        assert result["wind_speed"].isnull().sum() == 0

    def test_handle_missing_values_preserves_valid_data(self):
        """Non-null values should not be modified."""
        df = make_sample_df(n_hours=24, n_cities=1)
        original_temps = df["temperature_2m"].copy()

        result = handle_missing_values(df)

        pd.testing.assert_series_equal(result["temperature_2m"], original_temps)

    def test_remove_duplicates(self):
        """Duplicate city+timestamp rows should be removed."""
        df = make_sample_df(n_hours=10, n_cities=1)
        # Add a duplicate row
        dup = df.iloc[0:1].copy()
        df = pd.concat([df, dup], ignore_index=True)

        assert len(df) == 11  # 10 + 1 duplicate
        result = remove_duplicates(df)
        assert len(result) == 10

    def test_remove_duplicates_no_dups(self):
        """If no duplicates, length should stay the same."""
        df = make_sample_df(n_hours=10, n_cities=1)
        result = remove_duplicates(df)
        assert len(result) == 10

    def test_validate_ranges_perfect_data(self):
        """Clean data should get quality score of 1.0."""
        df = make_sample_df(n_hours=24, n_cities=1)
        result = validate_ranges(df)

        assert "quality_score" in result.columns
        assert (result["quality_score"] == 1.0).all()

    def test_validate_ranges_flags_bad_data(self):
        """Out-of-range values should lower quality score."""
        df = make_sample_df(n_hours=24, n_cities=1)
        # Inject an impossible temperature
        df.loc[0, "temperature_2m"] = 100.0  # Way above max

        result = validate_ranges(df)

        # First row should have lower quality
        assert result.loc[0, "quality_score"] < 1.0
        # Temperature should be clipped to max
        assert result.loc[0, "temperature_2m"] <= 60.0

    def test_validate_ranges_clips_values(self):
        """Values outside range should be clipped, not removed."""
        df = make_sample_df(n_hours=10, n_cities=1)
        df.loc[0, "relative_humidity"] = 150.0  # Above 100%

        result = validate_ranges(df)

        assert result.loc[0, "relative_humidity"] == 100.0
        assert len(result) == 10  # No rows removed


# ============================================
# Feature Engineer Tests (Gold Layer)
# ============================================

class TestFeatureEngineer:
    """Tests for Gold layer feature engineering functions."""

    def test_rolling_statistics_columns_added(self):
        """Rolling stats should add mean and std columns."""
        df = make_sample_df(n_hours=48, n_cities=1)
        result = add_rolling_statistics(df)

        assert "temp_rolling_mean_24h" in result.columns
        assert "temp_rolling_std_24h" in result.columns

    def test_rolling_statistics_no_nulls(self):
        """Rolling stats should not have null values (min_periods=1)."""
        df = make_sample_df(n_hours=48, n_cities=1)
        result = add_rolling_statistics(df)

        assert result["temp_rolling_mean_24h"].isnull().sum() == 0
        assert result["temp_rolling_std_24h"].isnull().sum() == 0

    def test_zscore_column_added(self):
        """Z-score should be computed after rolling stats."""
        df = make_sample_df(n_hours=48, n_cities=1)
        df = add_rolling_statistics(df)
        result = add_zscore(df)

        assert "temp_zscore" in result.columns

    def test_zscore_centered_near_zero(self):
        """Mean z-score should be approximately 0."""
        df = make_sample_df(n_hours=96, n_cities=1)
        df = add_rolling_statistics(df)
        result = add_zscore(df)

        # After enough data points, mean z-score should be near 0
        assert abs(result["temp_zscore"].mean()) < 1.0

    def test_lag_features_created(self):
        """Lag features should be added for 1h, 6h, 12h, 24h."""
        df = make_sample_df(n_hours=48, n_cities=1)
        result = add_lag_features(df)

        for lag in [1, 6, 12, 24]:
            assert f"temp_lag_{lag}h" in result.columns

    def test_lag_features_values_correct(self):
        """1h lag should equal previous row's temperature."""
        df = make_sample_df(n_hours=48, n_cities=1)
        result = add_lag_features(df)

        # Row 5's lag_1h should equal row 4's temperature
        assert result.loc[5, "temp_lag_1h"] == result.loc[4, "temperature_2m"]

    def test_rate_of_change_columns(self):
        """Rate of change should add temp_change and pressure_change."""
        df = make_sample_df(n_hours=48, n_cities=1)
        result = add_rate_of_change(df)

        assert "temp_change_1h" in result.columns
        assert "pressure_change_3h" in result.columns

    def test_interaction_features(self):
        """Humidity × wind interaction should be computed."""
        df = make_sample_df(n_hours=10, n_cities=1)
        result = add_interaction_features(df)

        assert "humidity_wind_interaction" in result.columns
        # Check formula: humidity * wind_speed / 100
        expected = df["relative_humidity"].iloc[0] * df["wind_speed"].iloc[0] / 100.0
        assert abs(result["humidity_wind_interaction"].iloc[0] - expected) < 0.01

    def test_temporal_features(self):
        """Temporal features should extract hour, day, month, is_night."""
        df = make_sample_df(n_hours=48, n_cities=1)
        result = add_temporal_features(df)

        assert "hour_of_day" in result.columns
        assert "day_of_week" in result.columns
        assert "month" in result.columns
        assert "is_night" in result.columns

    def test_is_night_correct(self):
        """Hours before 6am and after 8pm should be night."""
        df = make_sample_df(n_hours=48, n_cities=1)
        result = add_temporal_features(df)

        for _, row in result.iterrows():
            if row["hour_of_day"] < 6 or row["hour_of_day"] > 20:
                assert row["is_night"] == 1
            else:
                assert row["is_night"] == 0

    def test_handle_feature_nulls(self):
        """After handling nulls, no NaN should remain in feature columns."""
        df = make_sample_df(n_hours=48, n_cities=1)
        df = add_rolling_statistics(df)
        df = add_zscore(df)
        df = add_lag_features(df)
        df = add_rate_of_change(df)
        df = add_interaction_features(df)
        df = add_temporal_features(df)
        result = handle_feature_nulls(df)

        feature_cols = [
            "temp_change_1h", "pressure_change_3h",
            "temp_lag_1h", "temp_lag_6h", "temp_lag_12h", "temp_lag_24h"
        ]
        for col in feature_cols:
            assert result[col].isnull().sum() == 0, f"Null found in {col}"

    def test_full_pipeline_no_nulls(self):
        """Complete feature pipeline should produce zero nulls."""
        df = make_sample_df(n_hours=72, n_cities=2)
        df = add_rolling_statistics(df)
        df = add_zscore(df)
        df = add_lag_features(df)
        df = add_rate_of_change(df)
        df = add_interaction_features(df)
        df = add_temporal_features(df)
        df = handle_feature_nulls(df)

        all_feature_cols = [
            "temp_rolling_mean_24h", "temp_rolling_std_24h", "temp_zscore",
            "temp_lag_1h", "temp_lag_6h", "temp_lag_12h", "temp_lag_24h",
            "temp_change_1h", "pressure_change_3h",
            "humidity_wind_interaction",
            "hour_of_day", "day_of_week", "month", "is_night",
        ]
        for col in all_feature_cols:
            assert df[col].isnull().sum() == 0, f"Null in {col} after full pipeline"