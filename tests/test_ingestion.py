"""
Tests for the weather data ingestion module.

Run with: pytest tests/test_ingestion.py -v
"""

import pandas as pd
import pytest

from src.ingestion.city_config import get_all_cities, get_city_by_name, get_city_count
from src.ingestion.weather_fetcher import fetch_weather_for_city, HOURLY_VARIABLES


# ============================================
# City Config Tests
# ============================================

class TestCityConfig:
    """Tests for city configuration."""

    def test_cities_not_empty(self):
        """We should have at least 20 cities configured."""
        assert get_city_count() >= 20

    def test_city_has_required_fields(self):
        """Every city must have name, lat, lon, state, timezone."""
        for city in get_all_cities():
            assert "name" in city, f"City missing 'name': {city}"
            assert "lat" in city, f"City missing 'lat': {city}"
            assert "lon" in city, f"City missing 'lon': {city}"
            assert "state" in city, f"City missing 'state': {city}"
            assert "timezone" in city, f"City missing 'timezone': {city}"

    def test_city_coordinates_valid(self):
        """Coordinates should be within US bounds."""
        for city in get_all_cities():
            assert 24.0 <= city["lat"] <= 50.0, f"{city['name']} lat out of US range"
            assert -125.0 <= city["lon"] <= -66.0, f"{city['name']} lon out of US range"

    def test_get_city_by_name(self):
        """Should find cities by name (case-insensitive)."""
        city = get_city_by_name("new york")
        assert city is not None
        assert city["state"] == "NY"

    def test_get_city_by_name_not_found(self):
        """Should return None for unknown cities."""
        assert get_city_by_name("Atlantis") is None

    def test_no_duplicate_cities(self):
        """City names should be unique."""
        names = [c["name"] for c in get_all_cities()]
        assert len(names) == len(set(names)), "Duplicate city names found!"


# ============================================
# Weather Fetcher Tests (hits real API)
# ============================================

class TestWeatherFetcher:
    """Tests for the Open-Meteo API fetcher.

    NOTE: These tests hit the real Open-Meteo API.
    They're fast (< 2s each) and the API is free with no key.
    """

    @pytest.fixture
    def sample_city(self):
        """Return a known city for testing."""
        return {"name": "Boston", "lat": 42.3601, "lon": -71.0589, "state": "MA", "timezone": "America/New_York"}

    def test_fetch_returns_dataframe(self, sample_city):
        """API should return a non-empty DataFrame."""
        df = fetch_weather_for_city(sample_city, past_days=1, forecast_days=1)
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_fetch_has_correct_columns(self, sample_city):
        """DataFrame should have all expected columns."""
        df = fetch_weather_for_city(sample_city, past_days=1, forecast_days=1)
        expected_columns = [
            "city_name", "latitude", "longitude", "timestamp",
            "temperature_2m", "relative_humidity", "wind_speed",
            "precipitation", "pressure_msl", "cloud_cover"
        ]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_fetch_city_name_matches(self, sample_city):
        """All rows should have the correct city name."""
        df = fetch_weather_for_city(sample_city, past_days=1, forecast_days=1)
        assert (df["city_name"] == "Boston").all()

    def test_fetch_timestamps_are_datetime(self, sample_city):
        """Timestamps should be proper datetime objects."""
        df = fetch_weather_for_city(sample_city, past_days=1, forecast_days=1)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_fetch_temperature_in_range(self, sample_city):
        """Temperatures should be within reasonable range (°C)."""
        df = fetch_weather_for_city(sample_city, past_days=1, forecast_days=1)
        # Drop NaN values before checking
        temps = df["temperature_2m"].dropna()
        assert temps.min() > -60, "Temperature below -60°C is unreasonable"
        assert temps.max() < 60, "Temperature above 60°C is unreasonable"

    def test_fetch_record_count(self, sample_city):
        """1 past day + 1 forecast day = ~48 hourly records."""
        df = fetch_weather_for_city(sample_city, past_days=1, forecast_days=1)
        assert len(df) >= 24, f"Expected at least 24 records, got {len(df)}"
        assert len(df) <= 72, f"Expected at most 72 records, got {len(df)}"

    def test_fetch_invalid_coordinates(self):
        """Invalid coordinates should return None, not crash."""
        bad_city = {"name": "Nowhere", "lat": 999, "lon": 999, "state": "XX", "timezone": "UTC"}
        df = fetch_weather_for_city(bad_city, past_days=1, forecast_days=1)
        # Open-Meteo may return data or error for invalid coords
        # The important thing is it doesn't crash
        assert df is None or isinstance(df, pd.DataFrame)