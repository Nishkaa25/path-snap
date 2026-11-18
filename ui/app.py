import os
import requests
import folium
import streamlit as st
from streamlit_folium import st_folium

# Configuration
API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="PathSnap | Map Matching Evaluator",
    page_icon="🗺️",
    layout="wide"
)

# Styling
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.main-title {
    font-size: 2.5rem;
    font-weight: 750;
    margin-bottom: 0;
}
.subtitle {
    color: #9CA3AF;
    font-size: 1rem;
    margin-top: 4px;
}
.metric-card {
    background: rgba(128,128,128,0.08);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 12px;
    padding: 16px;
}
.metric-label {
    font-size: 0.85rem;
    color: #9CA3AF;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-title">🗺️ PathSnap</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">GPS Map-Matching & Route Reconstruction Evaluator</p>',
    unsafe_allow_html=True
)

st.warning(
    "DEMO MODE: Currently using temporary stub API responses. "
    "Results are illustrative, not actual model predictions."
)

st.divider()

# Sidebar
with st.sidebar:
    st.title("Evaluation Panel")
    st.caption("Select a student-test trip")

    st.write("**API Connection**")
    st.code(API)

    if st.button("Check API Connection", use_container_width=True):
        try:
            response = requests.get(f"{API}/trips", timeout=10)
            response.raise_for_status()
            st.success("API connected successfully!")
        except requests.RequestException as e:
            st.error(f"Connection failed: {e}")

# Fetch trips
try:
    response = requests.get(f"{API}/trips", timeout=10)
    response.raise_for_status()
    trips = response.json()["trips"]

    if not trips:
        st.error("No evaluation trips returned by the API.")
        st.stop()

except (requests.RequestException, ValueError, KeyError) as e:
    st.error(f"Cannot retrieve evaluation trips: {e}")
    st.stop()

# Trip selection
with st.sidebar:
    selected_trip = st.selectbox(
        "Select Evaluation Trip",
        trips,
        format_func=lambda x: str(x)
    )

    run_prediction = st.button(
        "▶ Run Prediction",
        type="primary",
        use_container_width=True
    )

    st.divider()
    st.caption(f"Available evaluation trips: {len(trips)}")
    st.caption("PathSnap | Member A UI")

# Prediction
if run_prediction:
    try:
        with st.spinner("Processing GPS observations and reconstructing route..."):
            r = requests.post(
                f"{API}/predict",
                json={"trip_id": selected_trip},
                timeout=600
            )
            r.raise_for_status()
            result = r.json()

            required = ["metrics", "timing", "center", "gps", "pred_polyline"]
            missing = [key for key in required if key not in result]

            if missing:
                raise ValueError(f"API response missing fields: {missing}")

            st.session_state["result"] = result
            st.session_state["result_trip"] = selected_trip
            st.session_state.pop("prediction_error", None)

    except (requests.RequestException, ValueError, KeyError) as e:
        st.session_state["prediction_error"] = str(e)
        st.session_state.pop("result", None)

if st.session_state.get("prediction_error"):
    st.error(f"Prediction failed: {st.session_state['prediction_error']}")

# Display results
result = st.session_state.get("result")

if result:
    metrics = result["metrics"]
    timing = result["timing"]

    st.subheader("Prediction Results")
    st.caption(f"Evaluation Trip: {st.session_state['result_trip']}")

    columns = st.columns(4)

    metric_data = [
        ("Classifier Top-1 Accuracy", metrics["top1_classifier"]),
        ("Classifier + HMM Top-1", metrics["top1_hmm"]),
        ("Route Continuity", metrics["continuity_hmm"]),
        ("API Processing Time", timing["total_s"])
    ]

    for col, (label, value) in zip(columns, metric_data):
        if label == "API Processing Time":
            display_value = f"{float(value):.2f} s"
        else:
            display_value = f"{float(value):.3f}"

        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{display_value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()
    st.subheader("Geographical Route Visualization")
    st.caption("Blue: GPS observations | Red: Predicted route")

    fmap = folium.Map(
        location=result["center"],
        zoom_start=15,
        control_scale=True
    )

    folium.PolyLine(
        result["gps"],
        color="#2685D8",
        weight=3,
        opacity=0.8,
        tooltip="GPS Observations"
    ).add_to(fmap)

    folium.PolyLine(
        result["pred_polyline"],
        color="#E53935",
        weight=5,
        opacity=0.9,
        tooltip="Predicted Route"
    ).add_to(fmap)

    if result["gps"] or result["pred_polyline"]:
        all_points = result["gps"] + result["pred_polyline"]
        if all_points:
            fmap.fit_bounds(all_points)

    folium.LayerControl().add_to(fmap)

    st_folium(
        fmap,
        width=None,
        height=600,
        use_container_width=True
    )

else:
    st.info(
        "Select an evaluation trip from the sidebar and click "
        "'Run Prediction' to view the results."
    )

st.divider()
st.caption("PathSnap | GPS Map-Matching Evaluation System")
