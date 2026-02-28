"""
Streamlit Dashboard — Weather Anomaly Detection Platform (v3 - Polished).

Pages:
1. Live Weather Map — current conditions + AQI + UV (no table)
2. Anomaly Feed — detected anomalies with severity scores
3. City Deep Dive — current conditions card + time-series + AQI
4. Weather Alerts — grouped alert cards
5. Hourly Forecast — Apple Weather-style hourly/daily view
6. City Comparison — compare 2-4 cities side by side
7. Platform Stats — pipeline health and data counts
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import text
from datetime import datetime, date

from src.utils.database import get_engine
from src.ingestion.city_config import get_all_cities

st.set_page_config(page_title="Weather Anomaly Platform", page_icon="🌦️",
                   layout="wide", initial_sidebar_state="expanded")

# Sidebar
st.sidebar.title("🌦️ Weather Anomaly Platform")
st.sidebar.markdown("*Real-time anomaly detection for 25 US cities*")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate",
    ["🗺️ Live Weather Map", "🚨 Anomaly Feed", "📊 City Deep Dive",
     "⚠️ Weather Alerts", "🕐 Hourly Forecast", "🔀 City Comparison", "⚙️ Platform Stats"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Built by Vinit Bahua**")
st.sidebar.markdown("[GitHub](https://github.com/vinitbahua123/weather-anomaly-platform)")


# ============================================
# Helpers
# ============================================

def calc_feels_like(temp, humidity, wind_speed):
    if pd.isna(temp): return temp
    if temp <= 10 and wind_speed > 4.8:
        return 13.12 + 0.6215 * temp - 11.37 * (wind_speed ** 0.16) + 0.3965 * temp * (wind_speed ** 0.16)
    elif temp >= 27 and humidity >= 40:
        return -8.785 + 1.611 * temp + 2.339 * humidity - 0.146 * temp * humidity
    return temp

def get_aqi_label(aqi):
    if pd.isna(aqi) or aqi is None: return "N/A", "#gray"
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
# Data Loaders
# ============================================

@st.cache_data(ttl=300)
def load_latest_weather():
    engine = get_engine()
    df = pd.read_sql(text("""
        SELECT DISTINCT ON (city_name)
            city_name, latitude, longitude, timestamp,
            temperature_2m, relative_humidity, wind_speed,
            precipitation, pressure_msl, cloud_cover
        FROM silver_weather
        WHERE timestamp <= NOW()
        ORDER BY city_name, timestamp DESC
    """), engine)
    df["feels_like"] = df.apply(
        lambda r: calc_feels_like(r["temperature_2m"], r["relative_humidity"], r["wind_speed"]), axis=1)
    return df

@st.cache_data(ttl=300)
def load_latest_aqi():
    engine = get_engine()
    try:
        return pd.read_sql(text("""
            SELECT DISTINCT ON (city_name) city_name, timestamp, us_aqi, pm2_5, pm10
            FROM air_quality ORDER BY city_name, timestamp DESC
        """), engine)
    except: return pd.DataFrame()

@st.cache_data(ttl=300)
def load_daily_forecast(city_name=None):
    engine = get_engine()
    try:
        if city_name:
            return pd.read_sql(text("""
                SELECT city_name, date, uv_index_max, sunrise, sunset, temp_max, temp_min
                FROM daily_forecast WHERE city_name = :city ORDER BY date DESC
            """), engine, params={"city": city_name})
        else:
            return pd.read_sql(text("""
                SELECT city_name, date, uv_index_max, sunrise, sunset, temp_max, temp_min
                FROM daily_forecast ORDER BY city_name, date DESC
            """), engine)
    except: return pd.DataFrame()

@st.cache_data(ttl=300)
def load_anomalies(min_zscore=3.0):
    engine = get_engine()
    return pd.read_sql(text("""
        SELECT city_name, timestamp, temperature_2m,
               temp_zscore, temp_rolling_mean_24h, temp_change_1h
        FROM gold_weather_features WHERE ABS(temp_zscore) > :min_zscore
        ORDER BY ABS(temp_zscore) DESC
    """), engine, params={"min_zscore": min_zscore})

@st.cache_data(ttl=300)
def load_city_timeseries(city_name, hours=168):
    engine = get_engine()
    return pd.read_sql(text("""
        SELECT g.timestamp, g.temperature_2m, g.temp_zscore,
               g.temp_rolling_mean_24h, g.temp_rolling_std_24h,
               g.temp_change_1h, g.pressure_change_3h,
               s.relative_humidity, s.wind_speed, s.precipitation, s.cloud_cover
        FROM gold_weather_features g
        JOIN silver_weather s ON g.city_name = s.city_name AND g.timestamp = s.timestamp
        WHERE g.city_name = :city ORDER BY g.timestamp DESC LIMIT :hours
    """), engine, params={"city": city_name, "hours": hours})

@st.cache_data(ttl=300)
def load_city_aqi_history(city_name, hours=168):
    engine = get_engine()
    try:
        return pd.read_sql(text("""
            SELECT timestamp, us_aqi, pm2_5, pm10 FROM air_quality
            WHERE city_name = :city ORDER BY timestamp DESC LIMIT :hours
        """), engine, params={"city": city_name, "hours": hours})
    except: return pd.DataFrame()

@st.cache_data(ttl=300)
def load_hourly_forecast(city_name):
    engine = get_engine()
    return pd.read_sql(text("""
        SELECT timestamp, temperature_2m, relative_humidity, wind_speed,
               precipitation, pressure_msl, cloud_cover
        FROM silver_weather WHERE city_name = :city
        ORDER BY timestamp DESC LIMIT 48
    """), engine, params={"city": city_name})

@st.cache_data(ttl=300)
def load_platform_stats():
    engine = get_engine()
    with engine.connect() as c:
        stats = {
            "bronze": c.execute(text("SELECT COUNT(*) FROM bronze_weather")).scalar(),
            "silver": c.execute(text("SELECT COUNT(*) FROM silver_weather")).scalar(),
            "gold": c.execute(text("SELECT COUNT(*) FROM gold_weather_features")).scalar(),
            "cities": c.execute(text("SELECT COUNT(DISTINCT city_name) FROM silver_weather")).scalar(),
            "anomalies": c.execute(text("SELECT COUNT(*) FROM gold_weather_features WHERE ABS(temp_zscore) > 3")).scalar(),
            "latest": c.execute(text("SELECT MAX(timestamp) FROM silver_weather")).scalar(),
        }
        try: stats["aqi_records"] = c.execute(text("SELECT COUNT(*) FROM air_quality")).scalar()
        except: stats["aqi_records"] = 0
        try: stats["forecast_records"] = c.execute(text("SELECT COUNT(*) FROM daily_forecast")).scalar()
        except: stats["forecast_records"] = 0
    return stats


# ============================================
# Page 1: Live Weather Map (removed table)
# ============================================

def render_live_map():
    st.title("🗺️ Live Weather Map")
    st.markdown("Current conditions across 25 US cities. Color = temperature, Size = wind speed. *Click any city on the map for details.*")

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
        df = df.merge(latest_daily[["city_name", "uv_index_max"]], on="city_name", how="left")
    else:
        df["uv_index_max"] = None

    fig = px.scatter_mapbox(
        df, lat="latitude", lon="longitude",
        color="temperature_2m", size="wind_speed",
        hover_name="city_name",
        hover_data={"temperature_2m": ":.1f", "feels_like": ":.1f",
                    "relative_humidity": ":.0f", "wind_speed": ":.1f",
                    "precipitation": ":.1f", "latitude": False, "longitude": False},
        color_continuous_scale="RdYlBu_r", size_max=20,
        zoom=3, center={"lat": 39.0, "lon": -98.0},
        mapbox_style="carto-positron",
    )
    fig.update_layout(height=550, margin={"r": 0, "t": 10, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

    # Key metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🌡️ Hottest", f"{df['temperature_2m'].max():.1f}°C",
              df.loc[df['temperature_2m'].idxmax(), 'city_name'])
    c2.metric("❄️ Coldest", f"{df['temperature_2m'].min():.1f}°C",
              df.loc[df['temperature_2m'].idxmin(), 'city_name'])
    c3.metric("🤔 Feels Like", f"{df['feels_like'].min():.0f}° to {df['feels_like'].max():.0f}°C")
    c4.metric("💨 Windiest", f"{df['wind_speed'].max():.1f} km/h",
              df.loc[df['wind_speed'].idxmax(), 'city_name'])
    c5.metric("🌧️ Most Rain", f"{df['precipitation'].max():.1f} mm",
              df.loc[df['precipitation'].idxmax(), 'city_name'])

    # AQI + UV row
    if df["us_aqi"].notna().any():
        st.markdown("### Air Quality & UV Index")
        a1, a2, a3 = st.columns(3)
        worst = df.loc[df["us_aqi"].idxmax()]
        best = df.loc[df["us_aqi"].idxmin()]
        lw, _ = get_aqi_label(worst["us_aqi"])
        lb, _ = get_aqi_label(best["us_aqi"])
        a1.metric("😷 Worst AQI", f"{worst['us_aqi']:.0f} ({lw})", worst["city_name"])
        a2.metric("🌿 Best AQI", f"{best['us_aqi']:.0f} ({lb})", best["city_name"])
        if "uv_index_max" in df.columns and df["uv_index_max"].notna().any():
            uv_row = df.loc[df["uv_index_max"].idxmax()]
            a3.metric("☀️ Highest UV", f"{uv_row['uv_index_max']:.1f} ({get_uv_label(uv_row['uv_index_max'])})",
                       uv_row["city_name"])

    st.markdown("---")
    st.info("💡 **Tip:** Select a city from **📊 City Deep Dive** in the sidebar to see detailed current conditions, forecasts, and anomaly history.")


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

    c1, c2 = st.columns(2)
    city_counts = df["city_name"].value_counts().reset_index()
    city_counts.columns = ["City", "Anomalies"]
    fig1 = px.bar(city_counts, x="City", y="Anomalies", title="Anomalies by City",
                  color="Anomalies", color_continuous_scale="Reds")
    fig1.update_layout(height=400)
    c1.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(df, x="temp_zscore", nbins=20, title="Z-Score Distribution",
                        color_discrete_sequence=["#FF6B6B"])
    fig2.update_layout(height=400)
    c2.plotly_chart(fig2, use_container_width=True)

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
# Page 3: City Deep Dive (with current conditions card)
# ============================================

def render_city_deep_dive():
    st.title("📊 City Deep Dive")

    cities = [c["name"] for c in get_all_cities()]
    selected = st.selectbox("Select a City", cities,
                            index=cities.index("New York") if "New York" in cities else 0)

    # ---- CURRENT CONDITIONS CARD ----
    latest = load_latest_weather()
    city_latest = latest[latest["city_name"] == selected]

    if not city_latest.empty:
        row = city_latest.iloc[0]
        fl = calc_feels_like(row["temperature_2m"], row["relative_humidity"], row["wind_speed"])
        utc_time = pd.to_datetime(row["timestamp"])
        local_time = utc_time - pd.Timedelta(hours=5)  # EST offset
        data_time = local_time.strftime("%b %d, %I:%M %p") + " EST"


        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
             padding: 24px; border-radius: 16px; color: white; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <h2 style="margin:0; color:white; font-size:28px;">{selected}</h2>
                    <p style="margin:4px 0 0 0; opacity:0.8; font-size:14px;">As of {data_time}</p>
                </div>
                <div style="text-align:right;">
                    <span style="font-size:56px; font-weight:bold;">{row['temperature_2m']:.1f}°C</span>
                </div>
            </div>
            <div style="display: flex; gap: 32px; margin-top: 16px; flex-wrap: wrap;">
                <div>
                    <span style="opacity:0.7; font-size:13px;">Feels Like</span><br>
                    <span style="font-size:20px; font-weight:600;">{fl:.1f}°C</span>
                </div>
                <div>
                    <span style="opacity:0.7; font-size:13px;">Humidity</span><br>
                    <span style="font-size:20px; font-weight:600;">{row['relative_humidity']:.0f}%</span>
                </div>
                <div>
                    <span style="opacity:0.7; font-size:13px;">Wind</span><br>
                    <span style="font-size:20px; font-weight:600;">{row['wind_speed']:.1f} km/h</span>
                </div>
                <div>
                    <span style="opacity:0.7; font-size:13px;">Pressure</span><br>
                    <span style="font-size:20px; font-weight:600;">{row['pressure_msl']:.0f} hPa</span>
                </div>
                <div>
                    <span style="opacity:0.7; font-size:13px;">Cloud Cover</span><br>
                    <span style="font-size:20px; font-weight:600;">{row['cloud_cover']:.0f}%</span>
                </div>
                <div>
                    <span style="opacity:0.7; font-size:13px;">Rain</span><br>
                    <span style="font-size:20px; font-weight:600;">{row['precipitation']:.1f} mm</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # AQI card if available
        aqi_df = load_latest_aqi()
        if not aqi_df.empty:
            city_aqi = aqi_df[aqi_df["city_name"] == selected]
            if not city_aqi.empty:
                aqi_val = city_aqi.iloc[0]["us_aqi"]
                aqi_label, aqi_color = get_aqi_label(aqi_val)
                st.markdown(f"""
                <div style="background-color: {aqi_color}20; border-left: 4px solid {aqi_color};
                     padding: 12px 16px; border-radius: 8px; margin-bottom: 16px;">
                    <strong>😷 Air Quality Index: {aqi_val:.0f}</strong> — {aqi_label}
                    &nbsp;&nbsp;|&nbsp;&nbsp; PM2.5: {city_aqi.iloc[0]['pm2_5']:.1f} µg/m³
                    &nbsp;&nbsp;|&nbsp;&nbsp; PM10: {city_aqi.iloc[0]['pm10']:.1f} µg/m³
                </div>
                """, unsafe_allow_html=True)

        # UV + Sunrise/Sunset
        daily = load_daily_forecast(selected)
        if not daily.empty:
            daily["date"] = pd.to_datetime(daily["date"])
            today_rows = daily[daily["date"].dt.date == date.today()]
            if today_rows.empty:
                today_rows = daily.head(1)
            today_data = today_rows.iloc[0]

            uv_parts = []
            if today_data["uv_index_max"] is not None and not pd.isna(today_data["uv_index_max"]):
                uv_parts.append(f"☀️ UV Index: {today_data['uv_index_max']:.1f} ({get_uv_label(today_data['uv_index_max'])})")
            if today_data["sunrise"] is not None:
                sr = str(today_data["sunrise"]).split("T")[-1][:5] if "T" in str(today_data["sunrise"]) else ""
                ss = str(today_data["sunset"]).split("T")[-1][:5] if "T" in str(today_data["sunset"]) else ""
                if sr and ss:
                    uv_parts.append(f"🌅 Sunrise: {sr} &nbsp;|&nbsp; 🌇 Sunset: {ss}")
            if today_data["temp_max"] is not None:
                uv_parts.append(f"📈 High: {today_data['temp_max']:.0f}°C &nbsp;|&nbsp; 📉 Low: {today_data['temp_min']:.0f}°C")

            if uv_parts:
                st.markdown(f"""
                <div style="background-color:#f0f2f6; padding:12px 16px; border-radius:8px; margin-bottom:16px;">
                    {"&nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;&nbsp;".join(uv_parts)}
                </div>
                """, unsafe_allow_html=True)

    # ---- TIME SERIES CHARTS ----
    st.markdown("---")
    hours = st.slider("Hours of History", 24, 720, 168, 24)

    df = load_city_timeseries(selected, hours)
    if df.empty:
        st.warning(f"No historical data for {selected}")
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

    fig.update_layout(title=f"Temperature History — {selected}", height=450,
                      hovermode="x unified", xaxis_title="Time", yaxis_title="°C")
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

    # AQI history
    aqi_hist = load_city_aqi_history(selected, hours)
    if not aqi_hist.empty:
        aqi_hist["timestamp"] = pd.to_datetime(aqi_hist["timestamp"])
        aqi_hist = aqi_hist.sort_values("timestamp")
        st.markdown(f"### Air Quality History — {selected}")
        fig_aqi = go.Figure()
        fig_aqi.add_trace(go.Scatter(x=aqi_hist["timestamp"], y=aqi_hist["us_aqi"],
                                     name="AQI", fill="tozeroy", line=dict(color="#6C5CE7")))
        fig_aqi.add_hline(y=50, line_dash="dot", line_color="green", annotation_text="Good")
        fig_aqi.add_hline(y=100, line_dash="dot", line_color="orange", annotation_text="Moderate")
        fig_aqi.add_hline(y=150, line_dash="dot", line_color="red", annotation_text="Unhealthy")
        fig_aqi.update_layout(height=300, yaxis_title="AQI")
        st.plotly_chart(fig_aqi, use_container_width=True)


# ============================================
# Page 4: Weather Alerts (Redesigned)
# ============================================

def render_weather_alerts():
    st.title("⚠️ Weather Alerts")
    st.markdown("Active alerts based on anomalies, rapid changes, pressure drops, air quality, and UV index.")

    from src.models.predict import generate_alerts
    alerts = generate_alerts()

    if not alerts:
        st.success("✅ No active weather alerts. All conditions normal.")
        return

    high = [a for a in alerts if a["severity"] == "high"]
    medium = [a for a in alerts if a["severity"] == "medium"]

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 High Severity", len(high))
    c2.metric("🟡 Medium Severity", len(medium))
    c3.metric("📊 Total Alerts", len(alerts))
    st.markdown("---")

    # Group by category
    categories = {}
    for alert in alerts:
        cat = alert["category"]
        if cat not in categories: categories[cat] = []
        categories[cat].append(alert)

    cat_icons = {
        "temperature_anomaly": "🌡️ Temperature Anomalies",
        "rapid_change": "⚡ Rapid Temperature Changes",
        "pressure_drop": "🌪️ Storm Warnings",
        "air_quality": "😷 Air Quality Alerts",
        "uv_index": "☀️ UV Index Warnings",
    }

    for cat, cat_alerts in categories.items():
        cat_name = cat_icons.get(cat, cat)
        high_count = sum(1 for a in cat_alerts if a["severity"] == "high")

        with st.expander(f"{cat_name} ({len(cat_alerts)} alerts, {high_count} high)", expanded=(high_count > 0)):
            for alert in cat_alerts:
                time_str = alert["timestamp"][:16]
                if alert["severity"] == "high":
                    st.markdown(f"""<div style="background:#FFE0E0; padding:12px; border-radius:8px;
                        border-left:4px solid #FF0000; margin-bottom:8px;">
                        <strong>🔴 {alert['city']}</strong> · <span style="color:#666">{time_str}</span><br>
                        {alert['message']}</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div style="background:#FFF8E1; padding:12px; border-radius:8px;
                        border-left:4px solid #FFA000; margin-bottom:8px;">
                        <strong>🟡 {alert['city']}</strong> · <span style="color:#666">{time_str}</span><br>
                        {alert['message']}</div>""", unsafe_allow_html=True)

    # Charts
    st.markdown("### Alert Distribution")
    alert_df = pd.DataFrame(alerts)
    ac1, ac2 = st.columns(2)

    cat_counts = alert_df["category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    cat_counts["Category"] = cat_counts["Category"].map({
        "temperature_anomaly": "🌡️ Temperature", "rapid_change": "⚡ Rapid Change",
        "pressure_drop": "🌪️ Storm", "air_quality": "😷 AQI", "uv_index": "☀️ UV"
    }).fillna(cat_counts["Category"])
    fig1 = px.pie(cat_counts, names="Category", values="Count", title="By Category",
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig1.update_layout(height=350)
    ac1.plotly_chart(fig1, use_container_width=True)

    city_counts = alert_df["city"].value_counts().head(10).reset_index()
    city_counts.columns = ["City", "Alerts"]
    fig2 = px.bar(city_counts, x="City", y="Alerts", title="Top Cities",
                  color="Alerts", color_continuous_scale="OrRd")
    fig2.update_layout(height=350)
    ac2.plotly_chart(fig2, use_container_width=True)


# ============================================
# Page 5: Hourly Forecast (Fixed daily high/low)
# ============================================

def render_hourly_forecast():
    st.title("🕐 Hourly Forecast")

    cities = [c["name"] for c in get_all_cities()]
    selected = st.selectbox("Select a City", cities,
                            index=cities.index("New York") if "New York" in cities else 0)

    df = load_hourly_forecast(selected)
    if df.empty:
        st.warning(f"No data for {selected}")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["feels_like"] = df.apply(
        lambda r: calc_feels_like(r["temperature_2m"], r["relative_humidity"], r["wind_speed"]), axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["temperature_2m"], mode="lines+markers",
                             name="Temperature", line=dict(color="#FF6B6B", width=3),
                             marker=dict(size=5), fill="tozeroy", fillcolor="rgba(255,107,107,0.1)"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["feels_like"],
                             name="Feels Like", line=dict(color="#FFA07A", width=2, dash="dot")))
    fig.update_layout(title=f"48-Hour Forecast — {selected}", height=400,
                      hovermode="x unified", xaxis_title="Time", yaxis_title="°C")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df["timestamp"], y=df["precipitation"], marker_color="#74B9FF"))
    fig2.update_layout(title="Precipitation (mm)", height=300)
    c1.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df["timestamp"], y=df["wind_speed"], fill="tozeroy",
                              line=dict(color="#FDCB6E"), fillcolor="rgba(253,203,110,0.2)"))
    fig3.update_layout(title="Wind Speed (km/h)", height=300)
    c2.plotly_chart(fig3, use_container_width=True)

    c3, c4 = st.columns(2)
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df["timestamp"], y=df["relative_humidity"], fill="tozeroy",
                              line=dict(color="#00B894"), fillcolor="rgba(0,184,148,0.2)"))
    fig4.update_layout(title="Humidity (%)", height=300)
    c3.plotly_chart(fig4, use_container_width=True)

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=df["timestamp"], y=df["cloud_cover"], fill="tozeroy",
                              line=dict(color="#636E72"), fillcolor="rgba(99,110,114,0.2)"))
    fig5.update_layout(title="Cloud Cover (%)", height=300)
    c4.plotly_chart(fig5, use_container_width=True)

    # Daily overview — FIXED: today only, chart shows last 7 days
    daily_df = load_daily_forecast(selected)
    if not daily_df.empty:
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        daily_df = daily_df.sort_values("date")

        today_rows = daily_df[daily_df["date"].dt.date == date.today()]
        if today_rows.empty:
            today_rows = daily_df.tail(1)
        td = today_rows.iloc[0]

        st.markdown(f"### Today — {selected}")
        d1, d2, d3 = st.columns(3)
        d1.metric("🌡️ High / Low", f"{td['temp_max']:.0f}°C / {td['temp_min']:.0f}°C")
        if td["uv_index_max"] is not None and not pd.isna(td["uv_index_max"]):
            d2.metric("☀️ UV Index", f"{td['uv_index_max']:.1f} ({get_uv_label(td['uv_index_max'])})")
        if td["sunrise"] is not None:
            sr = str(td["sunrise"]).split("T")[-1][:5] if "T" in str(td["sunrise"]) else ""
            ss = str(td["sunset"]).split("T")[-1][:5] if "T" in str(td["sunset"]) else ""
            if sr and ss:
                d3.metric("🌅 Sunrise / Sunset", f"{sr} / {ss}")

        # Chart: last 7 days only
        recent = daily_df.tail(9)
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=recent["date"], y=recent["temp_max"],
                                   name="High", line=dict(color="#FF6B6B", width=2), mode="lines+markers"))
        fig_d.add_trace(go.Scatter(x=recent["date"], y=recent["temp_min"],
                                   name="Low", line=dict(color="#74B9FF", width=2), mode="lines+markers",
                                   fill="tonexty", fillcolor="rgba(116,185,255,0.15)"))
        fig_d.update_layout(title="Daily High / Low (Recent)", height=300, yaxis_title="°C")
        st.plotly_chart(fig_d, use_container_width=True)


# ============================================
# Page 6: City Comparison
# ============================================

def render_city_comparison():
    st.title("🔀 City Comparison")
    st.markdown("Compare weather patterns across cities side by side.")

    cities = [c["name"] for c in get_all_cities()]
    selected = st.multiselect("Select 2-4 Cities", cities,
                              default=["New York", "Miami", "Chicago"], max_selections=4)

    if len(selected) < 2:
        st.info("Select at least 2 cities to compare.")
        return

    hours = st.slider("Hours", 24, 168, 72, 24)

    all_data = []
    for city in selected:
        df = load_city_timeseries(city, hours)
        if not df.empty:
            df["city_name"] = city
            all_data.append(df)

    if not all_data:
        st.warning("No data for selected cities.")
        return

    combined = pd.concat(all_data, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.sort_values("timestamp")

    fig1 = px.line(combined, x="timestamp", y="temperature_2m", color="city_name",
                   title="Temperature Comparison (°C)")
    fig1.update_layout(height=400, hovermode="x unified")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(combined, x="timestamp", y="temp_zscore", color="city_name",
                   title="Z-Score Comparison")
    fig2.add_hline(y=3, line_dash="dash", line_color="red")
    fig2.add_hline(y=-3, line_dash="dash", line_color="red")
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)

    cc1, cc2 = st.columns(2)
    fig3 = px.line(combined, x="timestamp", y="wind_speed", color="city_name", title="Wind (km/h)")
    fig3.update_layout(height=300)
    cc1.plotly_chart(fig3, use_container_width=True)

    fig4 = px.line(combined, x="timestamp", y="relative_humidity", color="city_name", title="Humidity (%)")
    fig4.update_layout(height=300)
    cc2.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Summary")
    rows = []
    for city in selected:
        cdf = combined[combined["city_name"] == city]
        if not cdf.empty:
            rows.append({
                "City": city, "Avg Temp": f"{cdf['temperature_2m'].mean():.1f}°C",
                "Max": f"{cdf['temperature_2m'].max():.1f}°C",
                "Min": f"{cdf['temperature_2m'].min():.1f}°C",
                "Avg Wind": f"{cdf['wind_speed'].mean():.1f} km/h",
                "Anomalies": int((cdf["temp_zscore"].abs() > 3).sum()),
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


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
    c4.metric("🕐 Latest", str(stats["latest"])[:16] if stats["latest"] else "N/A")

    if stats["aqi_records"] > 0 or stats["forecast_records"] > 0:
        c5, c6 = st.columns(2)
        c5.metric("😷 AQI Records", f"{stats['aqi_records']:,}")
        c6.metric("📅 Forecasts", f"{stats['forecast_records']:,}")

    st.markdown("---")
    st.markdown("### Medallion Architecture")
    st.dataframe(pd.DataFrame({
        "Layer": ["🥉 Bronze (Raw)", "🥈 Silver (Clean)", "🥇 Gold (Features)"],
        "Records": [stats["bronze"], stats["silver"], stats["gold"]],
        "Description": ["Raw API data", "Validated & quality-scored", "13 ML features"],
    }), use_container_width=True, hide_index=True)

    st.markdown("### Pipeline Flow")
    st.code("Open-Meteo API → Bronze → Silver → Gold → ML Models → Dashboard\n"
            "     ↑                                              ↓\n"
            "Every 6 hours                        Anomaly Detection + Forecasting")

    st.markdown("### Tech Stack")
    t1, t2, t3 = st.columns(3)
    t1.markdown("**Data**\n- Python, Pandas\n- PostgreSQL\n- Open-Meteo API\n- Medallion Architecture")
    t2.markdown("**ML**\n- Isolation Forest\n- XGBoost\n- MLflow (10 experiments)")
    t3.markdown("**Deploy**\n- FastAPI, Streamlit\n- Docker Compose\n- AWS EC2")


# Router
if page == "🗺️ Live Weather Map": render_live_map()
elif page == "🚨 Anomaly Feed": render_anomaly_feed()
elif page == "📊 City Deep Dive": render_city_deep_dive()
elif page == "⚠️ Weather Alerts": render_weather_alerts()
elif page == "🕐 Hourly Forecast": render_hourly_forecast()
elif page == "🔀 City Comparison": render_city_comparison()
elif page == "⚙️ Platform Stats": render_platform_stats()