import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

st.set_page_config(page_title="Data Upload — Apogee", page_icon="📤", layout="wide")
st.title("Data Upload")

st.info(
    "**A CSV uploader is coming soon.** The goal: load any BTS (bug tracking system) "
    "export as a CSV straight into the training set, and see your data measured "
    "against industry benchmarks — no fixed spreadsheet format required.",
    icon="🚧",
)

st.markdown(
    "In the meantime, reach out to the Apogee team to load new or refreshed client "
    "data and retrain the model."
)
