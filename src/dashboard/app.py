"""
Streamlit Dashboard — Weather Anomaly Detection Platform.

This is what recruiters see. Four pages:
1. Live Weather Map — current conditions for all 25 cities
2. Anomaly Feed — detected anomalies with severity scores
3. City Deep Dive — time-series charts for any city
4. Platform Stats — pipeline health and data counts

Run with:
    streamlit run src/dashboard/app.py --server.port 8501
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import text
from datetime import datetime, timedelta

from src.utils.database import get_engine
from src.ingestion.city_config import get_all_cities

# ============================================
# Page Config
# ============================================
st.set_page_config(
    page_title="Weather Anomaly Platform",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# Sidebar Navigation
# ============================================
st.sidebar.title("🌦️ Weather Anomaly Platform")
st.sidebar.markdown("*Real-time anomaly detection for 25 US cities*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🗺️ Live Weather Map", "🚨 Anomaly Feed", "📊 City Deep Dive", "⚙️ Platform Stats"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Built by Vinit Bahua**")
st.sidebar.markdown("[GitHub](https://github.com/vinitbahua123/weather-anomaly-platform)")
st.sidebar.markdown("MLOps Portfolio Project")


# ============================================
# Helper Functions
# ============================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_latest_weather():
    """Load most recent weather for each city."""
    engine = get_engine()
    query = text("""
        SELECT DISTINCT ON (city_name)
            city_name, latitude, longitude, timestamp,
            temperature_2m, relative_humidity, wind_speed,
            precipitation, pressure_msl, cloud_cover
        FROM silver_weather
        ORDER BY city_name, timestamp DESC
    """)
    return pd.read_sql(query, engine)


@st.cache_data(ttl=300)
def load_anomalies(min_zscore=3.0):
    """Load anomaly data."""
    engine = get_engine()
    query = text("""
        SELECT city_name, timestamp, temperature_2m,
               temp_zscore, temp_rolling_mean_24h, temp_change_1h
        FROM gold_weather_features
        WHERE ABS(temp_zscore) > :min_zscore
        ORDER BY ABS(temp_zscore) DESC
    """)
    return pd.read_sql(query, engine, params={"min_zscore": min_zscore})


@st.cache_data(ttl=300)
def load_city_timeseries(city_name, hours=168):
    """Load time-series data for a specific city."""
    engine = get_engine()
    query = text("""
        SELECT g.timestamp, g.temperature_2m, g.temp_zscore,
               g.temp_rolling_mean_24h, g.temp_rolling_std_24h,
               g.temp_change_1h, g.pressure_change_3h,
               s.relative_humidity, s.wind_speed, s.precipitation
        FROM gold_weather_features g
        JOIN silver_weather s ON g.city_name = s.city_name AND g.timestamp = s.timestamp
        WHERE g.city_name = :city
        ORDER BY g.timestamp DESC
        LIMIT :hours
    """)
    return pd.read_sql(query, engine, params={"city": city_name, "hours": hours})


@st.cache_data(ttl=300)
def load_platform_stats():
    """Load platform-wide statistics."""
    engine = get_engine()
    with engine.connect() as conn:
        stats = {
            "bronze": conn.execute(text("SELECT COUNT(*) FROM bronze_weather")).scalar(),
            "silver": conn.execute(text("SELECT COUNT(*) FROM silver_weather")).scalar(),
            "gold": conn.execute(text("SELECT COUNT(*) FROM gold_weather_features")).scalar(),
            "cities": conn.execute(text("SELECT COUNT(DISTINCT city_name) FROM silver_weather")).scalar(),
            "anomalies": conn.execute(text("SELECT COUNT(*) FROM gold_weather_features WHERE ABS(temp_zscore) > 3")).scalar(),
            "latest": conn.execute(text("SELECT MAX(timestamp) FROM silver_weather")).scalar(),
        }
    return stats


# ============================================
# Page 1: Live Weather Map
# ============================================

def render_live_map():
    st.title("🗺️ Live Weather Map")
    st.markdown("Current conditions across 25 US cities. Color = temperature, Size = wind speed.")

    df = load_latest_weather()
    if df.empty:
        st.warning("No weather data available. Run the ingestion pipeline first.")
        return

    # Temperature color map
    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="temperature_2m",
        size="wind_speed",
        hover_name="city_name",
        hover_data={
            "temperature_2m": ":.1f",
            "relative_humidity": ":.0f",
            "wind_speed": ":.1f",
            "precipitation": ":.1f",
            "latitude": False,
            "longitude": False,
        },
        color_continuous_scale="RdYlBu_r",
        size_max=20,
        zoom=3,
        center={"lat": 39.0, "lon": -98.0},
        mapbox_style="carto-positron",
        title="Current Temperature (°C) & Wind Speed",
    )
    fig.update_layout(height=500, margin={"r": 0, "t": 40, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌡️ Hottest", f"{df['temperature_2m'].max():.1f}°C",
                df.loc[df['temperature_2m'].idxmax(), 'city_name'])
    col2.metric("❄️ Coldest", f"{df['temperature_2m'].min():.1f}°C",
                df.loc[df['temperature_2m'].idxmin(), 'city_name'])
    col3.metric("💨 Windiest", f"{df['wind_speed'].max():.1f} km/h",
                df.loc[df['wind_speed'].idxmax(), 'city_name'])
    col4.metric("🌧️ Most Rain", f"{df['precipitation'].max():.1f} mm",
                df.loc[df['precipitation'].idxmax(), 'city_name'])

    # Data table
    st.markdown("### Current Readings")
    display_df = df[["city_name", "temperature_2m", "relative_humidity",
                     "wind_speed", "precipitation", "pressure_msl", "cloud_cover"]].copy()
    display_df.columns = ["City", "Temp (°C)", "Humidity (%)", "Wind (km/h)",
                          "Rain (mm)", "Pressure (hPa)", "Cloud (%)"]
    display_df = display_df.sort_values("Temp (°C)", ascending=False).reset_index(drop=True)
    st.dataframe(display_df, use_container_width=True, height=400)


# ============================================
# Page 2: Anomaly Feed
# ============================================

def render_anomaly_feed():
    st.title("🚨 Anomaly Feed")
    st.markdown("Weather events that deviate significantly from normal patterns.")

    # Threshold slider
    threshold = st.slider("Z-Score Threshold", 2.0, 5.0, 3.0, 0.1,
                          help="Higher = fewer but more extreme anomalies")

    df = load_anomalies(min_zscore=threshold)

    if df.empty:
        st.success(f"No anomalies detected above z-score threshold of {threshold}")
        return

    st.metric("Total Anomalies Detected", len(df))

    # Anomaly type breakdown
    df["anomaly_type"] = df["temp_zscore"].apply(lambda z: "🔴 Unusually Warm" if z > 0 else "🔵 Unusually Cold")
    df["severity"] = df["temp_zscore"].abs().apply(
        lambda z: "🟥 High" if z > 3.5 else "🟧 Medium" if z > 3.0 else "🟨 Low"
    )

    col1, col2 = st.columns(2)

    # Anomalies by city
    city_counts = df["city_name"].value_counts().reset_index()
    city_counts.columns = ["City", "Anomalies"]
    fig1 = px.bar(city_counts, x="City", y="Anomalies",
                  title="Anomalies by City", color="Anomalies",
                  color_continuous_scale="Reds")
    fig1.update_layout(height=400)
    col1.plotly_chart(fig1, use_container_width=True)

    # Z-score distribution
    fig2 = px.histogram(df, x="temp_zscore", nbins=20,
                        title="Z-Score Distribution of Anomalies",
                        color_discrete_sequence=["#FF6B6B"])
    fig2.update_layout(height=400)
    col2.plotly_chart(fig2, use_container_width=True)

    # Anomaly timeline
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    fig3 = px.scatter(df, x="timestamp", y="temp_zscore",
                      color="city_name", size=df["temp_zscore"].abs(),
                      title="Anomaly Timeline",
                      hover_data=["temperature_2m", "temp_rolling_mean_24h"])
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

    # Detailed table
    st.markdown("### Anomaly Details")
    display_df = df[["city_name", "timestamp", "temperature_2m", "temp_zscore",
                     "temp_rolling_mean_24h", "anomaly_type", "severity"]].copy()
    display_df.columns = ["City", "Time", "Actual Temp", "Z-Score",
                          "Expected Temp", "Type", "Severity"]
    display_df["Z-Score"] = display_df["Z-Score"].round(2)
    display_df["Actual Temp"] = display_df["Actual Temp"].apply(lambda x: f"{x:.1f}°C")
    display_df["Expected Temp"] = display_df["Expected Temp"].apply(lambda x: f"{x:.1f}°C")
    st.dataframe(display_df, use_container_width=True, height=400)


# ============================================
# Page 3: City Deep Dive
# ============================================

def render_city_deep_dive():
    st.title("📊 City Deep Dive")

    cities = [c["name"] for c in get_all_cities()]
    selected_city = st.selectbox("Select a City", cities, index=cities.index("New York") if "New York" in cities else 0)

    hours = st.slider("Hours of History", 24, 720, 168, 24,
                      help="How many hours of data to show")

    df = load_city_timeseries(selected_city, hours)

    if df.empty:
        st.warning(f"No data available for {selected_city}")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Temperature chart with rolling mean and anomaly bands
    fig = go.Figure()

    # Actual temperature
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["temperature_2m"],
        name="Actual Temperature",
        line=dict(color="#FF6B6B", width=2),
    ))

    # Rolling mean
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["temp_rolling_mean_24h"],
        name="24h Rolling Mean",
        line=dict(color="#4ECDC4", width=2, dash="dash"),
    ))

    # Anomaly bands (±3 std)
    upper = df["temp_rolling_mean_24h"] + 3 * df["temp_rolling_std_24h"]
    lower = df["temp_rolling_mean_24h"] - 3 * df["temp_rolling_std_24h"]

    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=upper,
        name="Upper Anomaly Bound",
        line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=lower,
        name="Anomaly Band (±3σ)",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(78, 205, 196, 0.1)",
    ))

    # Mark anomalies
    anomalies = df[df["temp_zscore"].abs() > 3]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies["timestamp"], y=anomalies["temperature_2m"],
            name="Anomalies",
            mode="markers",
            marker=dict(color="red", size=12, symbol="x"),
        ))

    fig.update_layout(
        title=f"Temperature Timeline — {selected_city}",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        height=450,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Z-Score chart
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["timestamp"], y=df["temp_zscore"],
        name="Z-Score",
        line=dict(color="#6C5CE7", width=2),
        fill="tozeroy",
        fillcolor="rgba(108, 92, 231, 0.1)",
    ))
    fig2.add_hline(y=3, line_dash="dash", line_color="red", annotation_text="Anomaly Threshold (+3)")
    fig2.add_hline(y=-3, line_dash="dash", line_color="red", annotation_text="Anomaly Threshold (-3)")

    fig2.update_layout(
        title=f"Z-Score Timeline — {selected_city}",
        xaxis_title="Time",
        yaxis_title="Z-Score",
        height=350,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Additional weather metrics
    col1, col2, col3 = st.columns(3)

    fig3 = px.line(df, x="timestamp", y="relative_humidity",
                   title="Humidity (%)", color_discrete_sequence=["#00B894"])
    fig3.update_layout(height=250, showlegend=False)
    col1.plotly_chart(fig3, use_container_width=True)

    fig4 = px.line(df, x="timestamp", y="wind_speed",
                   title="Wind Speed (km/h)", color_discrete_sequence=["#FDCB6E"])
    fig4.update_layout(height=250, showlegend=False)
    col2.plotly_chart(fig4, use_container_width=True)

    fig5 = px.bar(df, x="timestamp", y="precipitation",
                  title="Precipitation (mm)", color_discrete_sequence=["#74B9FF"])
    fig5.update_layout(height=250, showlegend=False)
    col3.plotly_chart(fig5, use_container_width=True)

    # City stats
    st.markdown(f"### {selected_city} Summary")
    scol1, scol2, scol3, scol4 = st.columns(4)
    scol1.metric("Avg Temp", f"{df['temperature_2m'].mean():.1f}°C")
    scol2.metric("Max Temp", f"{df['temperature_2m'].max():.1f}°C")
    scol3.metric("Min Temp", f"{df['temperature_2m'].min():.1f}°C")
    scol4.metric("Anomalies", f"{len(anomalies)}")


# ============================================
# Page 4: Platform Stats
# ============================================

def render_platform_stats():
    st.title("⚙️ Platform Stats")
    st.markdown("Pipeline health and data overview.")

    stats = load_platform_stats()

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏙️ Cities Monitored", stats["cities"])
    col2.metric("📊 Total Records", f"{stats['gold']:,}")
    col3.metric("🚨 Anomalies Found", stats["anomalies"])
    col4.metric("🕐 Latest Data", str(stats["latest"])[:16] if stats["latest"] else "N/A")

    st.markdown("---")

    # Medallion architecture stats
    st.markdown("### Data Pipeline (Medallion Architecture)")
    pipeline_df = pd.DataFrame({
        "Layer": ["🥉 Bronze (Raw)", "🥈 Silver (Clean)", "🥇 Gold (Features)"],
        "Records": [stats["bronze"], stats["silver"], stats["gold"]],
        "Description": [
            "Raw API data, append-only",
            "Validated, deduplicated, quality-scored",
            "13 engineered features, ML-ready",
        ]
    })
    st.dataframe(pipeline_df, use_container_width=True, hide_index=True)

    # Pipeline flow visual
    st.markdown("### Pipeline Flow")
    st.markdown("""
    ```
    Open-Meteo API → Bronze (Raw) → Silver (Clean) → Gold (Features) → ML Models → Dashboard
         ↑                                                                    ↓
    Every 6 hours                                              Anomaly Detection + Forecasting
    ```
    """)

    # Tech stack
    st.markdown("### Tech Stack")
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    tech_col1.markdown("""
    **Data Engineering**
    - Python + Pandas
    - PostgreSQL
    - Open-Meteo API
    - Medallion Architecture
    """)
    tech_col2.markdown("""
    **Machine Learning**
    - Isolation Forest
    - XGBoost
    - MLflow Tracking
    - 10 Experiments
    """)
    tech_col3.markdown("""
    **Deployment**
    - FastAPI
    - Streamlit
    - Docker
    - AWS EC2
    """)


# ============================================
# Page Router
# ============================================

if page == "🗺️ Live Weather Map":
    render_live_map()
elif page == "🚨 Anomaly Feed":
    render_anomaly_feed()
elif page == "📊 City Deep Dive":
    render_city_deep_dive()
elif page == "⚙️ Platform Stats":
    render_platform_stats()