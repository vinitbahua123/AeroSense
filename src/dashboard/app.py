"""
Streamlit Dashboard — Weather Anomaly Detection Platform (v2 - Polished).

Pages:
1. Live Weather Map — current conditions + AQI + UV for all 25 cities
2. Anomaly Feed — detected anomalies with severity scores
3. City Deep Dive — time-series charts with AQI history
4. Weather Alerts — redesigned alert cards with grouping
5. Hourly Forecast — Apple Weather-style hourly/daily view (fixed)
6. City Comparison — compare 2-3 cities side by side
7. Platform Stats — pipeline health and data counts

Run with:
    streamlit run src/dashboard/app.py --server.port 8501
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import text
from datetime import datetime, date

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
# Sidebar
# ============================================
st.sidebar.title("🌦️ Weather Anomaly Platform")
st.sidebar.markdown("*Real-time anomaly detection for 25 US cities*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🗺️ Live Weather Map", "🚨 Anomaly Feed", "📊 City Deep Dive",
     "⚠️ Weather Alerts", "🕐 Hourly Forecast", "🔀 City Comparison", "⚙️ Platform Stats"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Built by Vinit Bahua**")
st.sidebar.markdown("[GitHub](https://github.com/vinitbahua123/weather-anomaly-platform)")


# ============================================
# Helpers
# ============================================

def calc_feels_like(temp, humidity, wind_speed):
    """Calculate feels-like temperature using wind chill and heat index."""
    if pd.isna(temp):
        return temp
    # Wind chill (cold weather)
    if temp <= 10 and wind_speed > 4.8:
        return 13.12 + 0.6215 * temp - 11.37 * (wind_speed ** 0.16) + 0.3965 * temp * (wind_speed ** 0.16)
    # Heat index (warm weather)
    elif temp >= 27 and humidity >= 40:
        hi = -8.785 + 1.611 * temp + 2.339 * humidity - 0.146 * temp * humidity
        return hi
    return temp


def get_aqi_label(aqi):
    if pd.isna(aqi) or aqi is None:
        return "N/A", "#gray"
    aqi = float(aqi)
    if aqi <= 50: return "Good", "#00E400"
    elif aqi <= 100: return "Moderate", "#FFFF00"
    elif aqi <= 150: return "Unhealthy (Sensitive)", "#FF7E00"
    elif aqi <= 200: return "Unhealthy", "#FF0000"
    elif aqi <= 300: return "Very Unhealthy", "#8F3F97"
    else: return "Hazardous", "#7E0023"


def get_uv_label(uv):
    if pd.isna(uv) or uv is None: return "N/A"
    uv = float(uv)
    if uv <= 2: return "Low"
    elif uv <= 5: return "Moderate"
    elif uv <= 7: return "High"
    elif uv <= 10: return "Very High"
    else: return "Extreme"


# ============================================
# Data Loaders (cached)
# ============================================

@st.cache_data(ttl=300)
def load_latest_weather():
    engine = get_engine()
    query = text("""
        SELECT DISTINCT ON (city_name)
            city_name, latitude, longitude, timestamp,
            temperature_2m, relative_humidity, wind_speed,
            precipitation, pressure_msl, cloud_cover
        FROM silver_weather
        ORDER BY city_name, timestamp DESC
    """)
    df = pd.read_sql(query, engine)
    # Add feels-like
    df["feels_like"] = df.apply(
        lambda r: calc_feels_like(r["temperature_2m"], r["relative_humidity"], r["wind_speed"]), axis=1
    )
    return df


@st.cache_data(ttl=300)
def load_latest_aqi():
    engine = get_engine()
    try:
        query = text("""
            SELECT DISTINCT ON (city_name)
                city_name, timestamp, us_aqi, pm2_5, pm10
            FROM air_quality ORDER BY city_name, timestamp DESC
        """)
        return pd.read_sql(query, engine)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_daily_forecast(city_name=None):
    engine = get_engine()
    try:
        if city_name:
            query = text("""
                SELECT city_name, date, uv_index_max, sunrise, sunset, temp_max, temp_min
                FROM daily_forecast WHERE city_name = :city ORDER BY date DESC
            """)
            return pd.read_sql(query, engine, params={"city": city_name})
        else:
            query = text("""
                SELECT city_name, date, uv_index_max, sunrise, sunset, temp_max, temp_min
                FROM daily_forecast ORDER BY city_name, date DESC
            """)
            return pd.read_sql(query, engine)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_anomalies(min_zscore=3.0):
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
    engine = get_engine()
    query = text("""
        SELECT g.timestamp, g.temperature_2m, g.temp_zscore,
               g.temp_rolling_mean_24h, g.temp_rolling_std_24h,
               g.temp_change_1h, g.pressure_change_3h,
               s.relative_humidity, s.wind_speed, s.precipitation
        FROM gold_weather_features g
        JOIN silver_weather s ON g.city_name = s.city_name AND g.timestamp = s.timestamp
        WHERE g.city_name = :city
        ORDER BY g.timestamp DESC LIMIT :hours
    """)
    return pd.read_sql(query, engine, params={"city": city_name, "hours": hours})


@st.cache_data(ttl=300)
def load_city_aqi_history(city_name, hours=168):
    engine = get_engine()
    try:
        query = text("""
            SELECT timestamp, us_aqi, pm2_5, pm10
            FROM air_quality WHERE city_name = :city
            ORDER BY timestamp DESC LIMIT :hours
        """)
        return pd.read_sql(query, engine, params={"city": city_name, "hours": hours})
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_hourly_forecast(city_name):
    engine = get_engine()
    query = text("""
        SELECT timestamp, temperature_2m, relative_humidity, wind_speed,
               precipitation, pressure_msl, cloud_cover
        FROM silver_weather WHERE city_name = :city
        ORDER BY timestamp DESC LIMIT 48
    """)
    return pd.read_sql(query, engine, params={"city": city_name})


@st.cache_data(ttl=300)
def load_platform_stats():
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
        try:
            stats["aqi_records"] = conn.execute(text("SELECT COUNT(*) FROM air_quality")).scalar()
        except Exception:
            stats["aqi_records"] = 0
        try:
            stats["forecast_records"] = conn.execute(text("SELECT COUNT(*) FROM daily_forecast")).scalar()
        except Exception:
            stats["forecast_records"] = 0
    return stats


# ============================================
# Page 1: Live Weather Map
# ============================================

def render_live_map():
    st.title("🗺️ Live Weather Map")
    st.markdown("Current conditions across 25 US cities. Color = temperature, Size = wind speed.")

    df = load_latest_weather()
    if df.empty:
        st.warning("No weather data available.")
        return

    aqi_df = load_latest_aqi()
    if not aqi_df.empty:
        df = df.merge(aqi_df[["city_name", "us_aqi"]], on="city_name", how="left")
    else:
        df["us_aqi"] = None

    daily_df = load_daily_forecast()
    if not daily_df.empty:
        latest_daily = daily_df.drop_duplicates(subset="city_name", keep="first")
        df = df.merge(latest_daily[["city_name", "uv_index_max", "sunrise", "sunset"]], on="city_name", how="left")
    else:
        df["uv_index_max"] = None
        df["sunrise"] = None
        df["sunset"] = None

    fig = px.scatter_mapbox(
        df, lat="latitude", lon="longitude",
        color="temperature_2m", size="wind_speed",
        hover_name="city_name",
        hover_data={"temperature_2m": ":.1f", "feels_like": ":.1f",
                    "relative_humidity": ":.0f", "wind_speed": ":.1f",
                    "latitude": False, "longitude": False},
        color_continuous_scale="RdYlBu_r", size_max=20,
        zoom=3, center={"lat": 39.0, "lon": -98.0},
        mapbox_style="carto-positron",
    )
    fig.update_layout(height=500, margin={"r": 0, "t": 10, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🌡️ Hottest", f"{df['temperature_2m'].max():.1f}°C",
                df.loc[df['temperature_2m'].idxmax(), 'city_name'])
    col2.metric("❄️ Coldest", f"{df['temperature_2m'].min():.1f}°C",
                df.loc[df['temperature_2m'].idxmin(), 'city_name'])
    col3.metric("🤔 Feels Like Range",
                f"{df['feels_like'].min():.0f}° to {df['feels_like'].max():.0f}°C")
    col4.metric("💨 Windiest", f"{df['wind_speed'].max():.1f} km/h",
                df.loc[df['wind_speed'].idxmax(), 'city_name'])
    col5.metric("🌧️ Most Rain", f"{df['precipitation'].max():.1f} mm",
                df.loc[df['precipitation'].idxmax(), 'city_name'])

    if "us_aqi" in df.columns and df["us_aqi"].notna().any():
        st.markdown("### Air Quality & UV Index")
        aq1, aq2, aq3 = st.columns(3)
        worst = df.loc[df["us_aqi"].idxmax()]
        best = df.loc[df["us_aqi"].idxmin()]
        label_w, _ = get_aqi_label(worst["us_aqi"])
        label_b, _ = get_aqi_label(best["us_aqi"])
        aq1.metric("😷 Worst AQI", f"{worst['us_aqi']:.0f} ({label_w})", worst["city_name"])
        aq2.metric("🌿 Best AQI", f"{best['us_aqi']:.0f} ({label_b})", best["city_name"])
        if "uv_index_max" in df.columns and df["uv_index_max"].notna().any():
            uv_row = df.loc[df["uv_index_max"].idxmax()]
            aq3.metric("☀️ Highest UV", f"{uv_row['uv_index_max']:.1f} ({get_uv_label(uv_row['uv_index_max'])})",
                       uv_row["city_name"])

    st.markdown("### Current Readings")
    display_cols = ["city_name", "temperature_2m", "feels_like", "relative_humidity",
                    "wind_speed", "precipitation", "pressure_msl", "cloud_cover"]
    display_names = ["City", "Temp (°C)", "Feels Like (°C)", "Humidity (%)",
                     "Wind (km/h)", "Rain (mm)", "Pressure (hPa)", "Cloud (%)"]
    if "us_aqi" in df.columns:
        display_cols.append("us_aqi")
        display_names.append("AQI")
    display_df = df[display_cols].copy()
    display_df.columns = display_names
    display_df["Feels Like (°C)"] = display_df["Feels Like (°C)"].round(1)
    display_df = display_df.sort_values("Temp (°C)", ascending=False).reset_index(drop=True)
    st.dataframe(display_df, use_container_width=True, height=400)


# ============================================
# Page 2: Anomaly Feed
# ============================================

def render_anomaly_feed():
    st.title("🚨 Anomaly Feed")
    st.markdown("Weather events that deviate significantly from normal patterns.")

    threshold = st.slider("Z-Score Threshold", 2.0, 5.0, 3.0, 0.1)
    df = load_anomalies(min_zscore=threshold)

    if df.empty:
        st.success(f"No anomalies above z-score {threshold}")
        return

    st.metric("Total Anomalies Detected", len(df))

    df["anomaly_type"] = df["temp_zscore"].apply(lambda z: "🔴 Warm" if z > 0 else "🔵 Cold")
    df["severity"] = df["temp_zscore"].abs().apply(
        lambda z: "🟥 High" if z > 3.5 else "🟧 Medium" if z > 3.0 else "🟨 Low")

    col1, col2 = st.columns(2)
    city_counts = df["city_name"].value_counts().reset_index()
    city_counts.columns = ["City", "Anomalies"]
    fig1 = px.bar(city_counts, x="City", y="Anomalies", title="Anomalies by City",
                  color="Anomalies", color_continuous_scale="Reds")
    fig1.update_layout(height=400)
    col1.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(df, x="temp_zscore", nbins=20, title="Z-Score Distribution",
                        color_discrete_sequence=["#FF6B6B"])
    fig2.update_layout(height=400)
    col2.plotly_chart(fig2, use_container_width=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    fig3 = px.scatter(df, x="timestamp", y="temp_zscore", color="city_name",
                      size=df["temp_zscore"].abs(), title="Anomaly Timeline")
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Anomaly Details")
    table = df[["city_name", "timestamp", "temperature_2m", "temp_zscore",
                "temp_rolling_mean_24h", "anomaly_type", "severity"]].copy()
    table.columns = ["City", "Time", "Actual", "Z-Score", "Expected", "Type", "Severity"]
    table["Z-Score"] = table["Z-Score"].round(2)
    table["Actual"] = table["Actual"].apply(lambda x: f"{x:.1f}°C")
    table["Expected"] = table["Expected"].apply(lambda x: f"{x:.1f}°C")
    st.dataframe(table, use_container_width=True, height=400)


# ============================================
# Page 3: City Deep Dive
# ============================================

def render_city_deep_dive():
    st.title("📊 City Deep Dive")

    cities = [c["name"] for c in get_all_cities()]
    selected = st.selectbox("Select a City", cities,
                            index=cities.index("New York") if "New York" in cities else 0)
    hours = st.slider("Hours of History", 24, 720, 168, 24)

    df = load_city_timeseries(selected, hours)
    if df.empty:
        st.warning(f"No data for {selected}")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["feels_like"] = df.apply(
        lambda r: calc_feels_like(r["temperature_2m"], r["relative_humidity"], r["wind_speed"]), axis=1)

    # Temperature chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["temperature_2m"],
                             name="Actual", line=dict(color="#FF6B6B", width=2)))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["feels_like"],
                             name="Feels Like", line=dict(color="#FFA07A", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["temp_rolling_mean_24h"],
                             name="24h Mean", line=dict(color="#4ECDC4", width=2, dash="dash")))

    upper = df["temp_rolling_mean_24h"] + 3 * df["temp_rolling_std_24h"]
    lower = df["temp_rolling_mean_24h"] - 3 * df["temp_rolling_std_24h"]
    fig.add_trace(go.Scatter(x=df["timestamp"], y=upper, line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=lower, name="±3σ Band",
                             line=dict(width=0), fill="tonexty", fillcolor="rgba(78,205,196,0.1)"))

    anomalies = df[df["temp_zscore"].abs() > 3]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(x=anomalies["timestamp"], y=anomalies["temperature_2m"],
                                 name="Anomalies", mode="markers",
                                 marker=dict(color="red", size=12, symbol="x")))

    fig.update_layout(title=f"Temperature — {selected}", height=450, hovermode="x unified",
                      xaxis_title="Time", yaxis_title="°C")
    st.plotly_chart(fig, use_container_width=True)

    # Z-Score
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["timestamp"], y=df["temp_zscore"], name="Z-Score",
                              line=dict(color="#6C5CE7", width=2),
                              fill="tozeroy", fillcolor="rgba(108,92,231,0.1)"))
    fig2.add_hline(y=3, line_dash="dash", line_color="red", annotation_text="+3σ")
    fig2.add_hline(y=-3, line_dash="dash", line_color="red", annotation_text="-3σ")
    fig2.update_layout(title=f"Z-Score — {selected}", height=300)
    st.plotly_chart(fig2, use_container_width=True)

    # Weather metrics
    c1, c2, c3 = st.columns(3)
    fig_h = px.line(df, x="timestamp", y="relative_humidity", title="Humidity (%)",
                    color_discrete_sequence=["#00B894"])
    fig_h.update_layout(height=250, showlegend=False)
    c1.plotly_chart(fig_h, use_container_width=True)

    fig_w = px.line(df, x="timestamp", y="wind_speed", title="Wind (km/h)",
                    color_discrete_sequence=["#FDCB6E"])
    fig_w.update_layout(height=250, showlegend=False)
    c2.plotly_chart(fig_w, use_container_width=True)

    fig_p = px.bar(df, x="timestamp", y="precipitation", title="Rain (mm)",
                   color_discrete_sequence=["#74B9FF"])
    fig_p.update_layout(height=250, showlegend=False)
    c3.plotly_chart(fig_p, use_container_width=True)

    # AQI
    aqi_df = load_city_aqi_history(selected, hours)
    if not aqi_df.empty:
        aqi_df["timestamp"] = pd.to_datetime(aqi_df["timestamp"])
        aqi_df = aqi_df.sort_values("timestamp")
        st.markdown(f"### Air Quality — {selected}")
        fig_aqi = go.Figure()
        fig_aqi.add_trace(go.Scatter(x=aqi_df["timestamp"], y=aqi_df["us_aqi"],
                                     name="AQI", fill="tozeroy", line=dict(color="#6C5CE7")))
        fig_aqi.add_hline(y=50, line_dash="dot", line_color="green", annotation_text="Good")
        fig_aqi.add_hline(y=100, line_dash="dot", line_color="orange", annotation_text="Moderate")
        fig_aqi.add_hline(y=150, line_dash="dot", line_color="red", annotation_text="Unhealthy")
        fig_aqi.update_layout(height=300, yaxis_title="AQI")
        st.plotly_chart(fig_aqi, use_container_width=True)

    # Summary
    st.markdown(f"### {selected} Summary")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Avg Temp", f"{df['temperature_2m'].mean():.1f}°C")
    s2.metric("Max Temp", f"{df['temperature_2m'].max():.1f}°C")
    s3.metric("Min Temp", f"{df['temperature_2m'].min():.1f}°C")
    s4.metric("Avg Feels Like", f"{df['feels_like'].mean():.1f}°C")
    s5.metric("Anomalies", f"{len(anomalies)}")


# ============================================
# Page 4: Weather Alerts (Redesigned)
# ============================================

def render_weather_alerts():
    st.title("⚠️ Weather Alerts")
    st.markdown("Active alerts based on anomalies, rapid changes, pressure drops, air quality, and UV index.")

    from src.models.predict import generate_alerts
    alerts = generate_alerts()

    if not alerts:
        st.success("✅ No active weather alerts. All conditions normal across all 25 cities.")
        return

    high = [a for a in alerts if a["severity"] == "high"]
    medium = [a for a in alerts if a["severity"] == "medium"]

    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 High Severity", len(high))
    col2.metric("🟡 Medium Severity", len(medium))
    col3.metric("📊 Total Alerts", len(alerts))

    st.markdown("---")

    # Group alerts by category
    categories = {}
    for alert in alerts:
        cat = alert["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(alert)

    category_icons = {
        "temperature_anomaly": "🌡️ Temperature Anomalies",
        "rapid_change": "⚡ Rapid Temperature Changes",
        "pressure_drop": "🌪️ Storm Warnings (Pressure Drops)",
        "air_quality": "😷 Air Quality Alerts",
        "uv_index": "☀️ UV Index Warnings",
    }

    for cat, cat_alerts in categories.items():
        cat_name = category_icons.get(cat, cat)
        high_count = sum(1 for a in cat_alerts if a["severity"] == "high")

        with st.expander(f"{cat_name} ({len(cat_alerts)} alerts, {high_count} high severity)", expanded=(high_count > 0)):
            for alert in cat_alerts:
                # Build alert card
                city = alert["city"]
                time_str = alert["timestamp"][:16] if len(alert["timestamp"]) > 16 else alert["timestamp"]
                msg = alert["message"]

                if alert["severity"] == "high":
                    st.markdown(
                        f"""<div style="background-color:#FFE0E0; padding:12px; border-radius:8px; 
                        border-left:4px solid #FF0000; margin-bottom:8px;">
                        <strong>🔴 {city}</strong> · <span style="color:#666">{time_str}</span><br>
                        {msg}</div>""",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""<div style="background-color:#FFF8E1; padding:12px; border-radius:8px; 
                        border-left:4px solid #FFA000; margin-bottom:8px;">
                        <strong>🟡 {city}</strong> · <span style="color:#666">{time_str}</span><br>
                        {msg}</div>""",
                        unsafe_allow_html=True
                    )

    # Alert distribution chart
    st.markdown("### Alert Distribution")
    alert_df = pd.DataFrame(alerts)

    c1, c2 = st.columns(2)

    # By category
    cat_counts = alert_df["category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    cat_counts["Category"] = cat_counts["Category"].map({
        "temperature_anomaly": "🌡️ Temperature",
        "rapid_change": "⚡ Rapid Change",
        "pressure_drop": "🌪️ Storm",
        "air_quality": "😷 AQI",
        "uv_index": "☀️ UV",
    }).fillna(cat_counts["Category"])
    fig1 = px.pie(cat_counts, names="Category", values="Count", title="By Category",
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig1.update_layout(height=350)
    c1.plotly_chart(fig1, use_container_width=True)

    # By city
    city_counts = alert_df["city"].value_counts().head(10).reset_index()
    city_counts.columns = ["City", "Alerts"]
    fig2 = px.bar(city_counts, x="City", y="Alerts", title="Top 10 Cities by Alerts",
                  color="Alerts", color_continuous_scale="OrRd")
    fig2.update_layout(height=350)
    c2.plotly_chart(fig2, use_container_width=True)


# ============================================
# Page 5: Hourly Forecast (Fixed)
# ============================================

def render_hourly_forecast():
    st.title("🕐 Hourly Forecast")

    cities = [c["name"] for c in get_all_cities()]
    selected = st.selectbox("Select a City", cities,
                            index=cities.index("New York") if "New York" in cities else 0)

    df = load_hourly_forecast(selected)
    if df.empty:
        st.warning(f"No forecast data for {selected}")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["feels_like"] = df.apply(
        lambda r: calc_feels_like(r["temperature_2m"], r["relative_humidity"], r["wind_speed"]), axis=1)

    # 48-hour temperature
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["temperature_2m"], mode="lines+markers",
                             name="Temperature", line=dict(color="#FF6B6B", width=3),
                             marker=dict(size=5), fill="tozeroy", fillcolor="rgba(255,107,107,0.1)"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["feels_like"],
                             name="Feels Like", line=dict(color="#FFA07A", width=2, dash="dot")))
    fig.update_layout(title=f"48-Hour Temperature — {selected}", height=400,
                      hovermode="x unified", xaxis_title="Time", yaxis_title="°C")
    st.plotly_chart(fig, use_container_width=True)

    # Weather details
    c1, c2 = st.columns(2)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df["timestamp"], y=df["precipitation"], name="Rain", marker_color="#74B9FF"))
    fig2.update_layout(title="Precipitation (mm)", height=300)
    c1.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df["timestamp"], y=df["wind_speed"], name="Wind",
                              fill="tozeroy", line=dict(color="#FDCB6E"), fillcolor="rgba(253,203,110,0.2)"))
    fig3.update_layout(title="Wind Speed (km/h)", height=300)
    c2.plotly_chart(fig3, use_container_width=True)

    c3, c4 = st.columns(2)
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df["timestamp"], y=df["relative_humidity"], name="Humidity",
                              fill="tozeroy", line=dict(color="#00B894"), fillcolor="rgba(0,184,148,0.2)"))
    fig4.update_layout(title="Humidity (%)", height=300)
    c3.plotly_chart(fig4, use_container_width=True)

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=df["timestamp"], y=df["cloud_cover"], name="Clouds",
                              fill="tozeroy", line=dict(color="#636E72"), fillcolor="rgba(99,110,114,0.2)"))
    fig5.update_layout(title="Cloud Cover (%)", height=300)
    c4.plotly_chart(fig5, use_container_width=True)

    # Daily overview — FIXED: show only today's data correctly
    daily_df = load_daily_forecast(selected)
    if not daily_df.empty:
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        daily_df = daily_df.sort_values("date")

        # Get today's row
        today = date.today()
        today_row = daily_df[daily_df["date"].dt.date == today]
        if today_row.empty:
            today_row = daily_df.iloc[[-1]]  # Use latest available
        today_data = today_row.iloc[0]

        st.markdown(f"### Today's Overview — {selected}")
        d1, d2, d3 = st.columns(3)
        d1.metric("🌡️ High / Low", f"{today_data['temp_max']:.0f}°C / {today_data['temp_min']:.0f}°C")
        if today_data["uv_index_max"] is not None and not pd.isna(today_data["uv_index_max"]):
            d2.metric("☀️ UV Index", f"{today_data['uv_index_max']:.1f} ({get_uv_label(today_data['uv_index_max'])})")
        if today_data["sunrise"] is not None:
            sr = str(today_data["sunrise"]).split("T")[-1][:5] if "T" in str(today_data["sunrise"]) else str(today_data["sunrise"])
            ss = str(today_data["sunset"]).split("T")[-1][:5] if "T" in str(today_data["sunset"]) else str(today_data["sunset"])
            d3.metric("🌅 Sunrise / Sunset", f"{sr} / {ss}")

        # Daily high/low chart — only show last 7 days for clarity
        recent = daily_df.tail(9)  # ~7 days of actual + 2 forecast
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(x=recent["date"], y=recent["temp_max"],
                                       name="High", line=dict(color="#FF6B6B", width=2), mode="lines+markers"))
        fig_daily.add_trace(go.Scatter(x=recent["date"], y=recent["temp_min"],
                                       name="Low", line=dict(color="#74B9FF", width=2), mode="lines+markers",
                                       fill="tonexty", fillcolor="rgba(116,185,255,0.15)"))
        fig_daily.update_layout(title="Daily High / Low (Last 7 Days)", height=300, yaxis_title="°C")
        st.plotly_chart(fig_daily, use_container_width=True)


# ============================================
# Page 6: City Comparison (NEW)
# ============================================

def render_city_comparison():
    st.title("🔀 City Comparison")
    st.markdown("Compare weather patterns across cities side by side.")

    cities = [c["name"] for c in get_all_cities()]
    selected = st.multiselect("Select 2-4 Cities", cities,
                              default=["New York", "Miami", "Chicago"],
                              max_selections=4)

    if len(selected) < 2:
        st.info("Select at least 2 cities to compare.")
        return

    hours = st.slider("Hours of History", 24, 168, 72, 24)

    # Load data for all selected cities
    all_data = []
    for city in selected:
        df = load_city_timeseries(city, hours)
        if not df.empty:
            df["city_name"] = city
            all_data.append(df)

    if not all_data:
        st.warning("No data available for selected cities.")
        return

    combined = pd.concat(all_data, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.sort_values("timestamp")

    # Temperature comparison
    fig1 = px.line(combined, x="timestamp", y="temperature_2m", color="city_name",
                   title="Temperature Comparison (°C)")
    fig1.update_layout(height=400, hovermode="x unified")
    st.plotly_chart(fig1, use_container_width=True)

    # Z-Score comparison
    fig2 = px.line(combined, x="timestamp", y="temp_zscore", color="city_name",
                   title="Z-Score Comparison (Anomaly Indicator)")
    fig2.add_hline(y=3, line_dash="dash", line_color="red", annotation_text="+3σ")
    fig2.add_hline(y=-3, line_dash="dash", line_color="red", annotation_text="-3σ")
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)

    c1, c2 = st.columns(2)

    # Wind comparison
    fig3 = px.line(combined, x="timestamp", y="wind_speed", color="city_name",
                   title="Wind Speed (km/h)")
    fig3.update_layout(height=300)
    c1.plotly_chart(fig3, use_container_width=True)

    # Humidity comparison
    fig4 = px.line(combined, x="timestamp", y="relative_humidity", color="city_name",
                   title="Humidity (%)")
    fig4.update_layout(height=300)
    c2.plotly_chart(fig4, use_container_width=True)

    # Summary table
    st.markdown("### City Comparison Summary")
    summary_rows = []
    for city in selected:
        city_df = combined[combined["city_name"] == city]
        if not city_df.empty:
            anomaly_count = (city_df["temp_zscore"].abs() > 3).sum()
            summary_rows.append({
                "City": city,
                "Avg Temp (°C)": round(city_df["temperature_2m"].mean(), 1),
                "Max Temp (°C)": round(city_df["temperature_2m"].max(), 1),
                "Min Temp (°C)": round(city_df["temperature_2m"].min(), 1),
                "Avg Wind (km/h)": round(city_df["wind_speed"].mean(), 1),
                "Avg Humidity (%)": round(city_df["relative_humidity"].mean(), 0),
                "Anomalies": int(anomaly_count),
            })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ============================================
# Page 7: Platform Stats
# ============================================

def render_platform_stats():
    st.title("⚙️ Platform Stats")
    stats = load_platform_stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏙️ Cities", stats["cities"])
    c2.metric("📊 Records", f"{stats['gold']:,}")
    c3.metric("🚨 Anomalies", stats["anomalies"])
    c4.metric("🕐 Latest Data", str(stats["latest"])[:16] if stats["latest"] else "N/A")

    if stats["aqi_records"] > 0 or stats["forecast_records"] > 0:
        c5, c6 = st.columns(2)
        c5.metric("😷 AQI Records", f"{stats['aqi_records']:,}")
        c6.metric("📅 Daily Forecasts", f"{stats['forecast_records']:,}")

    st.markdown("---")
    st.markdown("### Data Pipeline (Medallion Architecture)")
    pipeline_df = pd.DataFrame({
        "Layer": ["🥉 Bronze (Raw)", "🥈 Silver (Clean)", "🥇 Gold (Features)"],
        "Records": [stats["bronze"], stats["silver"], stats["gold"]],
        "Description": ["Raw API data, append-only", "Validated, deduplicated, quality-scored",
                        "13 engineered features, ML-ready"],
    })
    st.dataframe(pipeline_df, use_container_width=True, hide_index=True)

    st.markdown("### Pipeline Flow")
    st.code("Open-Meteo API → Bronze (Raw) → Silver (Clean) → Gold (Features) → ML Models → Dashboard\n"
            "     ↑                                                                    ↓\n"
            "Every 6 hours                                        Anomaly Detection + Forecasting")

    st.markdown("### Tech Stack")
    t1, t2, t3 = st.columns(3)
    t1.markdown("**Data Engineering**\n- Python + Pandas\n- PostgreSQL\n- Open-Meteo API\n- Medallion Architecture")
    t2.markdown("**Machine Learning**\n- Isolation Forest\n- XGBoost\n- MLflow Tracking\n- 10 Experiments")
    t3.markdown("**Deployment**\n- FastAPI\n- Streamlit\n- Docker\n- AWS EC2")


# ============================================
# Router
# ============================================

if page == "🗺️ Live Weather Map": render_live_map()
elif page == "🚨 Anomaly Feed": render_anomaly_feed()
elif page == "📊 City Deep Dive": render_city_deep_dive()
elif page == "⚠️ Weather Alerts": render_weather_alerts()
elif page == "🕐 Hourly Forecast": render_hourly_forecast()
elif page == "🔀 City Comparison": render_city_comparison()
elif page == "⚙️ Platform Stats": render_platform_stats() 