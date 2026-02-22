"""
Database connection management using SQLAlchemy.
Provides engine and session factory for PostgreSQL.
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from src.config import DATABASE_URL
from src.utils.logger import logger


def get_engine():
    """Create and return a SQLAlchemy engine."""
    engine = create_engine(
        DATABASE_URL,
        pool_size=5,            # Max 5 simultaneous connections
        max_overflow=10,        # 10 extra connections if pool is full
        pool_pre_ping=True,     # Check connection is alive before using
        echo=False,             # Set True to see SQL queries in logs
    )
    return engine


def get_session():
    """Create and return a new database session."""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def test_connection():
    """Test database connectivity. Returns True if successful."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def init_tables():
    """Create all required tables if they don't exist."""
    engine = get_engine()
    with engine.connect() as conn:
        # Bronze layer — raw API data
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS bronze_weather (
                id SERIAL PRIMARY KEY,
                city_name VARCHAR(100) NOT NULL,
                latitude FLOAT NOT NULL,
                longitude FLOAT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                temperature_2m FLOAT,
                relative_humidity FLOAT,
                wind_speed FLOAT,
                precipitation FLOAT,
                pressure_msl FLOAT,
                cloud_cover FLOAT,
                ingested_at TIMESTAMPTZ DEFAULT NOW(),
                source VARCHAR(50) DEFAULT 'open-meteo',
                UNIQUE(city_name, timestamp)
            );
        """))

        # Silver layer — cleaned and validated data
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS silver_weather (
                id SERIAL PRIMARY KEY,
                city_name VARCHAR(100) NOT NULL,
                latitude FLOAT NOT NULL,
                longitude FLOAT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                temperature_2m FLOAT NOT NULL,
                relative_humidity FLOAT NOT NULL,
                wind_speed FLOAT NOT NULL,
                precipitation FLOAT NOT NULL,
                pressure_msl FLOAT NOT NULL,
                cloud_cover FLOAT NOT NULL,
                validated_at TIMESTAMPTZ DEFAULT NOW(),
                quality_score FLOAT DEFAULT 1.0,
                UNIQUE(city_name, timestamp)
            );
        """))

        # Gold layer — feature-engineered data
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS gold_weather_features (
                id SERIAL PRIMARY KEY,
                city_name VARCHAR(100) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                temperature_2m FLOAT NOT NULL,
                temp_rolling_mean_24h FLOAT,
                temp_rolling_std_24h FLOAT,
                temp_zscore FLOAT,
                temp_lag_1h FLOAT,
                temp_lag_6h FLOAT,
                temp_lag_12h FLOAT,
                temp_lag_24h FLOAT,
                temp_change_1h FLOAT,
                pressure_change_3h FLOAT,
                humidity_wind_interaction FLOAT,
                hour_of_day INT,
                day_of_week INT,
                month INT,
                is_night INT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(city_name, timestamp)
            );
        """))

        # Anomaly detections log
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS anomaly_detections (
                id SERIAL PRIMARY KEY,
                city_name VARCHAR(100) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                anomaly_type VARCHAR(50),
                severity_score FLOAT,
                model_version VARCHAR(50),
                alert_text TEXT,
                detected_at TIMESTAMPTZ DEFAULT NOW()
            );
        """))

        conn.commit()
        logger.info("All tables created successfully")


if __name__ == "__main__":
    """Allow running directly: python -m src.utils.database --init"""
    import sys
    if "--init" in sys.argv:
        init_tables()
    else:
        test_connection()
