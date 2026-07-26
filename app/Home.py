import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

st.set_page_config(
    page_title="Apogee — Release Risk Intelligence",
    page_icon="🔍",
    layout="wide",
)

st.title("Apogee — Release Risk Intelligence")
st.markdown(
    "Predictive bug severity intelligence to help engineering teams ship with confidence."
)

st.divider()

# Fixed-height card text + a scoped 4-across -> 2x2 breakpoint below, so the
# four "Open X" buttons stay aligned regardless of description length or
# window width (st.columns alone doesn't equalize content height, and at
# in-between widths a plain 4-up flex row squeezes every card's text into
# many more wrapped lines instead of reflowing).
st.markdown(
    """
    <style>
    @media (max-width: 1100px) and (min-width: 641px) {
        .st-key-home-cards [data-testid="stColumn"] {
            flex: 1 1 calc(50% - 16px) !important;
            width: calc(50% - 16px) !important;
            min-width: 0 !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CARD_HEIGHT = 270

with st.container(key="home-cards"):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(height=CARD_HEIGHT):
            st.subheader("Industry Benchmark")
            st.markdown(
                "Compare High/Critical bug rates across industries, with a client "
                "roster for each — consultative, cross-customer context in one view."
            )
        if st.button("Open Industry Benchmark →", use_container_width=True):
            st.switch_page("pages/1_Industry_Benchmark.py")

    with col2:
        with st.container(height=CARD_HEIGHT):
            st.subheader("Risk Dashboard")
            st.markdown(
                "Ranked view of historically risky components, platforms, and "
                "environments. No inputs required — always-on signal from historical data."
            )
        if st.button("Open Risk Dashboard →", use_container_width=True):
            st.switch_page("pages/2_Risk_Dashboard.py")

    with col3:
        with st.container(height=CARD_HEIGHT):
            st.subheader("Release Predictor")
            st.markdown(
                "Tell us about your upcoming release and get a predicted High/Critical "
                "bug probability, plus a breakdown of where bugs are most likely to appear."
            )
        if st.button("Open Release Predictor →", use_container_width=True):
            st.switch_page("pages/3_Release_Predictor.py")

    with col4:
        with st.container(height=CARD_HEIGHT):
            st.subheader("Monthly Digest")
            st.markdown(
                "Historical pattern analysis — trend lines, top recurring risk factors, "
                "and period-over-period changes. Refreshed monthly."
            )
        if st.button("Open Monthly Digest →", use_container_width=True):
            st.switch_page("pages/4_Monthly_Digest.py")

st.divider()

st.subheader("Data Upload")
st.markdown(
    "Self-serve CSV upload is coming soon. Contact the Apogee team to load new or "
    "refreshed client data in the meantime."
)
if st.button("Open Data Upload →", use_container_width=True):
    st.switch_page("pages/6_Data_Upload.py")

st.divider()

from model.predict import artifacts_exist
if not artifacts_exist():
    st.warning(
        "Model artifacts not found. Upload data from the **Data Upload** page "
        "to train the model before using the predictor.",
        icon="⚠️",
    )
else:
    st.success("Model is trained and ready.", icon="✅")
