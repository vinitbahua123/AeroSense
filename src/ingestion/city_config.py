"""
City configuration for weather data ingestion.
Each city has a name, latitude, and longitude for the Open-Meteo API.

WHY this is a separate file:
- Easy to add/remove cities without touching ingestion logic
- Can be swapped for a database lookup in production
- Single source of truth for all city coordinates
"""

# Major US cities with diverse climate zones
# This gives us variety in anomaly detection (desert, coastal, midwest, etc.)
CITIES = [
    # Northeast
    {"name": "New York", "lat": 40.7128, "lon": -74.0060, "state": "NY", "timezone": "America/New_York"},
    {"name": "Boston", "lat": 42.3601, "lon": -71.0589, "state": "MA", "timezone": "America/New_York"},
    {"name": "Philadelphia", "lat": 39.9526, "lon": -75.1652, "state": "PA", "timezone": "America/New_York"},
    {"name": "Pittsburgh", "lat": 40.4406, "lon": -79.9959, "state": "PA", "timezone": "America/New_York"},

    # Southeast
    {"name": "Miami", "lat": 25.7617, "lon": -80.1918, "state": "FL", "timezone": "America/New_York"},
    {"name": "Atlanta", "lat": 33.7490, "lon": -84.3880, "state": "GA", "timezone": "America/New_York"},
    {"name": "Charlotte", "lat": 35.2271, "lon": -80.8431, "state": "NC", "timezone": "America/New_York"},
    {"name": "Nashville", "lat": 36.1627, "lon": -86.7816, "state": "TN", "timezone": "America/Chicago"},

    # Midwest
    {"name": "Chicago", "lat": 41.8781, "lon": -87.6298, "state": "IL", "timezone": "America/Chicago"},
    {"name": "Detroit", "lat": 42.3314, "lon": -83.0458, "state": "MI", "timezone": "America/Detroit"},
    {"name": "Minneapolis", "lat": 44.9778, "lon": -93.2650, "state": "MN", "timezone": "America/Chicago"},
    {"name": "St Louis", "lat": 38.6270, "lon": -90.1994, "state": "MO", "timezone": "America/Chicago"},

    # South / Southwest
    {"name": "Dallas", "lat": 32.7767, "lon": -96.7970, "state": "TX", "timezone": "America/Chicago"},
    {"name": "Houston", "lat": 29.7604, "lon": -95.3698, "state": "TX", "timezone": "America/Chicago"},
    {"name": "Phoenix", "lat": 33.4484, "lon": -112.0740, "state": "AZ", "timezone": "America/Phoenix"},
    {"name": "San Antonio", "lat": 29.4241, "lon": -98.4936, "state": "TX", "timezone": "America/Chicago"},

    # West
    {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437, "state": "CA", "timezone": "America/Los_Angeles"},
    {"name": "San Francisco", "lat": 37.7749, "lon": -122.4194, "state": "CA", "timezone": "America/Los_Angeles"},
    {"name": "Seattle", "lat": 47.6062, "lon": -122.3321, "state": "WA", "timezone": "America/Los_Angeles"},
    {"name": "Portland", "lat": 45.5152, "lon": -122.6784, "state": "OR", "timezone": "America/Los_Angeles"},
    {"name": "Denver", "lat": 39.7392, "lon": -104.9903, "state": "CO", "timezone": "America/Denver"},
    {"name": "Las Vegas", "lat": 36.1699, "lon": -115.1398, "state": "NV", "timezone": "America/Los_Angeles"},
    {"name": "Salt Lake City", "lat": 40.7608, "lon": -111.8910, "state": "UT", "timezone": "America/Denver"},

    # Mountain / Plains
    {"name": "Kansas City", "lat": 39.0997, "lon": -94.5786, "state": "MO", "timezone": "America/Chicago"},
    {"name": "Oklahoma City", "lat": 35.4676, "lon": -97.5164, "state": "OK", "timezone": "America/Chicago"},
]


def get_all_cities() -> list[dict]:
    """Return all configured cities."""
    return CITIES


def get_city_by_name(name: str) -> dict | None:
    """Look up a city by name (case-insensitive)."""
    for city in CITIES:
        if city["name"].lower() == name.lower():
            return city
    return None


def get_city_count() -> int:
    """Return total number of configured cities."""
    return len(CITIES)


if __name__ == "__main__":
    print(f"Total cities configured: {get_city_count()}")
    for city in CITIES:
        print(f"  {city['name']}, {city['state']} ({city['lat']}, {city['lon']})")
        