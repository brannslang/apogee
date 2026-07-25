import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

from config import ENGAGEMENT_TYPES


def require_customer() -> tuple:
    """
    Render required Customer and Engagement Type selectors in the sidebar.
    Persists selections in session_state across page navigations.
    Calls st.stop() if no customer is selected, so pages can call this
    at the top and rely on the returned values being valid.

    Returns (customer, engagement_type) — engagement_type is one of
    ENGAGEMENT_TYPES ("Functional", "Accessibility", "Security"), scoping
    every downstream chart to that team's bugs so one team's findings don't
    get blended into another's risk profile.
    """
    from model.predict import artifacts_exist, get_customers

    if not artifacts_exist():
        st.error("Model artifacts not found. Upload data from the **Data Upload** page to train the model.")
        st.stop()

    customers = get_customers()
    if not customers:
        st.error("No customers found in training data.")
        st.stop()

    with st.sidebar:
        st.markdown("### Customer")
        current = st.session_state.get("selected_customer")
        idx = (customers.index(current) + 1) if current in customers else 0

        selection = st.selectbox(
            "Customer",
            options=["— Select a customer —"] + customers,
            index=idx,
            label_visibility="collapsed",
            key="customer_selector",
        )

        if selection == "— Select a customer —":
            st.session_state["selected_customer"] = None
        else:
            st.session_state["selected_customer"] = selection

        st.markdown("### Engagement Type")
        current_type = st.session_state.get("selected_engagement_type")
        type_idx = ENGAGEMENT_TYPES.index(current_type) if current_type in ENGAGEMENT_TYPES else 0

        engagement_type = st.radio(
            "Engagement Type",
            options=ENGAGEMENT_TYPES,
            index=type_idx,
            label_visibility="collapsed",
            key="engagement_type_selector",
            help="Accessibility, Security, and Functional bugs are each tracked separately — different teams, different risk profiles.",
        )
        st.session_state["selected_engagement_type"] = engagement_type

        st.divider()

    customer = st.session_state.get("selected_customer")
    if not customer:
        st.info("Select a customer from the sidebar to view data.", icon="👈")
        st.stop()

    return customer, engagement_type
