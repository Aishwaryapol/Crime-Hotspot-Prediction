"""Professional Streamlit UI for Crime Hotspot Prediction."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import predict_hotspot


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Crime Hotspot Prediction",
    page_icon="🚨",
    layout="wide",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CSV_COLUMNS = [
    "timestamp",
    "YEAR",
    "MONTH",
    "DAY_OF_WEEK",
    "HOUR",
    "AREA",
    "AREA_NAME",
    "LAT_GRID",
    "LON_GRID",
    "DOMINANT_CRIME_TYPE",
    "DOMINANT_PREMISE",
    "PREV_WEEK_CRIME_COUNT",
    "PREV_2_WEEK_CRIME_COUNT",
    "PREV_3_WEEK_CRIME_COUNT",
    "ROLLING_4_WEEK_CRIME_COUNT",
    "GRID_TOTAL_PREVIOUS_CRIMES",
    "prediction",
    "prediction_label",
    "probability",
]


# ============================================================
# STYLES
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f8fbff 0%, #f3f7ff 100%);
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5ebf5;
        border-radius: 14px;
        padding: 0.75rem 0.9rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .result-card {
        background: white;
        border-radius: 16px;
        padding: 1.2rem;
        border: 1px solid #e5ebf5;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOADERS
# ============================================================

@st.cache_data(show_spinner=False)
def load_training_categories():
    """Load category values from the processed training dataset."""
    x_train_path = PROCESSED_DIR / "X_train.csv"

    if not x_train_path.exists():
        return [], [], []

    try:
        df = pd.read_csv(
            x_train_path,
            usecols=["AREA NAME", "DOMINANT_CRIME_TYPE", "DOMINANT_PREMISE"],
        )

        area_names = sorted(df["AREA NAME"].dropna().astype(str).unique())
        crime_types = sorted(df["DOMINANT_CRIME_TYPE"].dropna().astype(str).unique())
        premises = sorted(df["DOMINANT_PREMISE"].dropna().astype(str).unique())
        return area_names, crime_types, premises
    except Exception:
        return [], [], []


# ============================================================
# HELPERS
# ============================================================


def initialize_session_state():
    """Initialize session-based prediction history for the current browser session."""
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = pd.DataFrame(columns=CSV_COLUMNS)
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def get_session_history():
    """Return the current user's prediction history from session state."""
    initialize_session_state()
    history_df = st.session_state.prediction_history
    if history_df is None:
        history_df = pd.DataFrame(columns=CSV_COLUMNS)
    return history_df.copy()


def append_prediction_to_session(input_data, result):
    """Append a prediction row to the current browser session history."""
    history_df = get_session_history()
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "YEAR": input_data["YEAR"],
        "MONTH": input_data["MONTH"],
        "DAY_OF_WEEK": input_data["DAY_OF_WEEK"],
        "HOUR": input_data["HOUR"],
        "AREA": input_data["AREA"],
        "AREA_NAME": input_data["AREA NAME"],
        "LAT_GRID": input_data["LAT_GRID"],
        "LON_GRID": input_data["LON_GRID"],
        "DOMINANT_CRIME_TYPE": input_data["DOMINANT_CRIME_TYPE"],
        "DOMINANT_PREMISE": input_data["DOMINANT_PREMISE"],
        "PREV_WEEK_CRIME_COUNT": input_data["PREV_WEEK_CRIME_COUNT"],
        "PREV_2_WEEK_CRIME_COUNT": input_data["PREV_2_WEEK_CRIME_COUNT"],
        "PREV_3_WEEK_CRIME_COUNT": input_data["PREV_3_WEEK_CRIME_COUNT"],
        "ROLLING_4_WEEK_CRIME_COUNT": input_data["ROLLING_4_WEEK_CRIME_COUNT"],
        "GRID_TOTAL_PREVIOUS_CRIMES": input_data["GRID_TOTAL_PREVIOUS_CRIMES"],
        "prediction": int(result["prediction"]),
        "prediction_label": result["prediction_label"],
        "probability": float(result["probability"]),
    }
    row_df = pd.DataFrame([row], columns=CSV_COLUMNS)
    if history_df.empty:
        history_df = row_df.copy()
    else:
        history_df = pd.concat([history_df, row_df], ignore_index=True)
    st.session_state.prediction_history = history_df
    return history_df


def build_history_csv_bytes(history_df):
    """Create CSV bytes for the current session history."""
    if history_df.empty:
        empty_df = pd.DataFrame(columns=CSV_COLUMNS)
        return empty_df.to_csv(index=False).encode("utf-8")
    return history_df.to_csv(index=False).encode("utf-8")

MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

DAY_OPTIONS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

MONTH_OPTIONS = list(MONTH_MAP.keys())


def build_input_data(year, month_name, day_of_week, hour, area, area_name, lat_grid, lon_grid,
                     dominant_crime_type, dominant_premise, prev_week, prev_2_week,
                     prev_3_week, rolling_4, grid_total_prev):
    """Create the exact input dictionary expected by the prediction pipeline."""
    month_value = MONTH_MAP[month_name]
    return {
        "YEAR": int(year),
        "MONTH": int(month_value),
        "DAY_OF_WEEK": day_of_week,
        "HOUR": int(hour),
        "AREA": int(area),
        "AREA NAME": area_name,
        "LAT_GRID": float(lat_grid),
        "LON_GRID": float(lon_grid),
        "DOMINANT_CRIME_TYPE": dominant_crime_type,
        "DOMINANT_PREMISE": dominant_premise,
        "PREV_WEEK_CRIME_COUNT": int(prev_week),
        "PREV_2_WEEK_CRIME_COUNT": int(prev_2_week),
        "PREV_3_WEEK_CRIME_COUNT": int(prev_3_week),
        "ROLLING_4_WEEK_CRIME_COUNT": int(rolling_4),
        "GRID_TOTAL_PREVIOUS_CRIMES": int(grid_total_prev),
    }


def show_result_card(result):
    """Display a polished prediction result card."""
    probability_percent = float(result["probability"]) * 100
    is_hotspot = result["prediction"] == 1

    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.subheader("🎯 Latest Prediction")

    if is_hotspot:
        st.error("🔴 HOTSPOT")
    else:
        st.success("🟢 NOT A HOTSPOT")

    st.metric("Hotspot Probability", f"{probability_percent:.2f}%")
    st.progress(probability_percent / 100)
    st.caption(
        "Model prediction. This classification is based on the supplied historical crime, location, and time features."
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    initialize_session_state()

    st.title("🚨 Crime Hotspot Prediction System")
    st.markdown("### Machine Learning Based Crime Hotspot Prediction")
    st.caption(
        "This dashboard uses the trained Random Forest model to assess whether a geographic grid cell is likely to be a crime hotspot based on historical activity, location, time, and crime characteristics."
    )

    st.sidebar.header("🧭 Navigation")
    navigation = st.sidebar.radio(
        "Select a view",
        ["🏠 Prediction", "📊 Dashboard", "📋 Prediction History", "ℹ️ Model Information"],
        index=0,
    )

    area_names, crime_types, premises = load_training_categories()
    if not area_names:
        st.sidebar.warning("Training category values could not be loaded from X_train.csv.")

    if navigation == "📊 Dashboard":
        render_dashboard()
        return

    if navigation == "📋 Prediction History":
        render_history_page()
        return

    if navigation == "ℹ️ Model Information":
        render_model_information()
        return

    render_prediction_page(area_names, crime_types, premises)


def render_prediction_page(area_names, crime_types, premises):
    st.info("Enter the crime, location, time, and historical activity information below and click the prediction button.")

    with st.container():
        with st.form("prediction_form"):
            st.subheader("📝 Prediction Input")
            st.caption("The form uses the same feature names required by the prediction pipeline and avoids leakage fields such as crime count and hotspot labels.")

            st.subheader("📅 Date & Time")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                year = st.number_input("YEAR", min_value=2020, max_value=2100, value=2024, step=1)
            with col2:
                month_name = st.selectbox("MONTH", MONTH_OPTIONS, index=11)
            with col3:
                day_of_week = st.selectbox("DAY_OF_WEEK", DAY_OPTIONS, index=5)
            with col4:
                hour = st.number_input("HOUR", min_value=0, max_value=23, value=22, step=1)

            st.subheader("📍 Location")
            col1, col2 = st.columns(2)
            with col1:
                area = st.number_input("AREA", min_value=1, max_value=25, value=1, step=1)
                if area_names:
                    area_name = st.selectbox("AREA NAME", area_names)
                else:
                    area_name = st.text_input("AREA NAME", value="Central")
            with col2:
                lat_grid = st.number_input("LAT_GRID", min_value=33.0, max_value=35.0, value=34.05, format="%.5f")
                lon_grid = st.number_input("LON_GRID", min_value=-119.0, max_value=-117.0, value=-118.25, format="%.5f")

            st.subheader("🔎 Crime Information")
            col1, col2 = st.columns(2)
            with col1:
                if crime_types:
                    dominant_crime_type = st.selectbox("DOMINANT_CRIME_TYPE", crime_types)
                else:
                    dominant_crime_type = st.text_input("DOMINANT_CRIME_TYPE", value="VEHICLE - STOLEN")
            with col2:
                if premises:
                    dominant_premise = st.selectbox("DOMINANT_PREMISE", premises)
                else:
                    dominant_premise = st.text_input("DOMINANT_PREMISE", value="STREET")

            st.subheader("📊 Historical Crime Activity")
            st.caption("These historical features are important because they capture recent crime pressure in the grid cell and help the model estimate hotspot likelihood.")
            col1, col2, col3 = st.columns(3)
            with col1:
                prev_week = st.number_input("PREV_WEEK_CRIME_COUNT", min_value=0, value=8, step=1)
            with col2:
                prev_2_week = st.number_input("PREV_2_WEEK_CRIME_COUNT", min_value=0, value=6, step=1)
            with col3:
                prev_3_week = st.number_input("PREV_3_WEEK_CRIME_COUNT", min_value=0, value=7, step=1)
            col1, col2 = st.columns(2)
            with col1:
                rolling_4 = st.number_input("ROLLING_4_WEEK_CRIME_COUNT", min_value=0, value=25, step=1)
            with col2:
                grid_total_prev = st.number_input("GRID_TOTAL_PREVIOUS_CRIMES", min_value=0, value=150, step=1)

            st.markdown("---")
            submitted = st.form_submit_button("🔮 Predict Crime Hotspot", use_container_width=True, type="primary")

    if submitted:
        try:
            input_data = build_input_data(
                year, month_name, day_of_week, hour, area, area_name, lat_grid, lon_grid,
                dominant_crime_type, dominant_premise, prev_week, prev_2_week, prev_3_week,
                rolling_4, grid_total_prev,
            )

            with st.spinner("Running Random Forest prediction..."):
                result = predict_hotspot(input_data)

            append_prediction_to_session(input_data, result)
            st.session_state.last_result = result
            st.success("✅ Prediction added to your current session history.")
        except Exception as exc:
            st.error(f"Prediction could not be completed. {exc}")

    if st.session_state.last_result is not None:
        st.markdown("---")
        show_result_card(st.session_state.last_result)


def render_dashboard():
    st.subheader("📊 Dashboard Overview")
    history_df = get_session_history()

    if history_df.empty:
        st.info("No prediction history is available yet. Make a prediction to populate this session dashboard.")
        return

    history_df = history_df.copy()
    history_df["probability"] = pd.to_numeric(history_df["probability"], errors="coerce")
    history_df["prediction_label"] = history_df["prediction_label"].fillna("UNKNOWN")

    total_predictions = len(history_df)
    hotspot_count = int((history_df["prediction"] == 1).sum())
    non_hotspot_count = int((history_df["prediction"] == 0).sum())
    average_probability = float(history_df["probability"].mean()) if not history_df["probability"].empty else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Predictions", total_predictions)
    col2.metric("HOTSPOT Predictions", hotspot_count)
    col3.metric("NOT A HOTSPOT Predictions", non_hotspot_count)
    col4.metric("Average Hotspot Probability", f"{average_probability * 100:.2f}%")

    overview_tab, map_tab = st.tabs(["Overview", "Prediction Map"])

    with overview_tab:
        chart_data = pd.DataFrame(
            {
                "HOTSPOT": [hotspot_count],
                "NOT A HOTSPOT": [non_hotspot_count],
            }
        )
        st.bar_chart(chart_data, use_container_width=True)

        bins = pd.cut(history_df["probability"], bins=10, include_lowest=True)
        density_df = pd.DataFrame({"Probability Bucket": bins.value_counts().sort_index().index.astype(str), "Count": bins.value_counts().sort_index().values})
        st.subheader("Probability Distribution")
        st.bar_chart(density_df.set_index("Probability Bucket"), use_container_width=True)

        try:
            history_df["timestamp"] = pd.to_datetime(history_df["timestamp"], errors="coerce")
            trend = history_df.dropna(subset=["timestamp"]).groupby(history_df["timestamp"].dt.date).size()
            st.subheader("Predictions Over Time")
            st.line_chart(trend, use_container_width=True)
        except Exception:
            st.info("Prediction timestamps are not available for trend visualization.")

    with map_tab:
        map_df = history_df.dropna(subset=["LAT_GRID", "LON_GRID"]).copy()
        map_df = map_df.rename(columns={"LAT_GRID": "lat", "LON_GRID": "lon"})
        map_df = map_df.loc[:, ["lat", "lon", "prediction_label"]]
        map_df = map_df[pd.to_numeric(map_df["lat"], errors="coerce").notna() & pd.to_numeric(map_df["lon"], errors="coerce").notna()]

        if not map_df.empty:
            st.caption("Prediction locations from your current session history are shown below.")
            st.map(map_df[["lat", "lon"]])
        else:
            st.info("Not enough valid latitude and longitude values are available for the map view.")


def render_history_page():
    st.subheader("📋 Prediction History")
    history_df = get_session_history()

    if history_df.empty:
        st.info("No predictions have been made in this browser session yet.")
        return

    history_df = history_df.copy()
    history_df["probability"] = pd.to_numeric(history_df["probability"], errors="coerce")
    history_df = history_df.sort_values(by="timestamp", ascending=False, na_position="last")

    col1, col2, col3 = st.columns(3)
    with col1:
        prediction_filter = st.selectbox("Prediction type", ["All", "HOTSPOT", "NOT A HOTSPOT"])
    with col2:
        area_filter = st.selectbox("Area", ["All"] + sorted(history_df["AREA_NAME"].fillna("").astype(str).unique().tolist()))
    with col3:
        probability_range = st.slider("Probability range", 0.0, 1.0, (0.0, 1.0), step=0.01)

    try:
        if "timestamp" in history_df.columns:
            date_values = pd.to_datetime(history_df["timestamp"], errors="coerce").dropna()
            if not date_values.empty:
                min_date = date_values.min().date()
                max_date = date_values.max().date()
                date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            else:
                date_range = None
        else:
            date_range = None
    except Exception:
        date_range = None

    filtered_df = history_df.copy()
    if prediction_filter != "All":
        filtered_df = filtered_df[filtered_df["prediction_label"] == prediction_filter]
    if area_filter != "All":
        filtered_df = filtered_df[filtered_df["AREA_NAME"].fillna("").astype(str) == area_filter]
    filtered_df = filtered_df[(filtered_df["probability"] >= probability_range[0]) & (filtered_df["probability"] <= probability_range[1])]

    if date_range is not None and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[pd.to_datetime(filtered_df["timestamp"], errors="coerce").dt.date.between(start_date, end_date)]

    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    csv_bytes = build_history_csv_bytes(history_df)
    st.download_button("⬇️ Download Prediction History CSV", data=csv_bytes, file_name="prediction_history.csv", mime="text/csv", use_container_width=True)


def render_model_information():
    st.subheader("ℹ️ Model Information")
    st.write(
        """
        The Random Forest Classifier used here was trained on historical crime activity aggregated by geographic grid cells and weeks. It combines location, time, crime characteristics, and recent crime history to estimate hotspot likelihood.
        """
    )

    with st.expander("Model Details", expanded=True):
        st.write(
            """
            - Algorithm: Random Forest Classifier
            - Number of estimators: 150
            - Max depth: 15
            - Minimum samples leaf: 5
            - Class weight: balanced
            - Random state: 42
            - Training period: 2020–2023
            - Testing period: 2024
            - Test accuracy: approximately 82%
            - Hotspot recall: approximately 86%
            """
        )

    st.markdown("---")
    st.subheader("📈 Project Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training Records", "171,992")
    col2.metric("Testing Records", "35,041")
    col3.metric("Geographic Grid Cells", "1,279")
    col4.metric("Training Period", "2020–2023")


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()