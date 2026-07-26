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
    from model.predict import artifacts_exist, get_customers, get_customer_engagement_types

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

        # Many customers are only ever engaged for one Engagement Type (e.g.
        # Accessibility-only or Security-only) and have zero Functional bugs.
        # If we let the radio stay on a type with no data for this customer,
        # every dropdown silently falls back to the global, unscoped category
        # lists (hundreds of App Components) with no indication why. So when
        # the customer selection changes, snap the type to one that actually
        # has data for them.
        available_types = get_customer_engagement_types(selection) if selection in customers else []
        prev_customer = st.session_state.get("_engagement_type_customer")
        if selection != prev_customer:
            current_type = st.session_state.get("selected_engagement_type")
            if available_types and current_type not in available_types:
                st.session_state["selected_engagement_type"] = available_types[0]
            st.session_state["_engagement_type_customer"] = selection

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

        if selection in customers and engagement_type not in available_types:
            if available_types:
                st.caption(
                    f"⚠️ {selection} has no {engagement_type} bug history "
                    f"(data available for: {', '.join(available_types)}). Dropdowns "
                    "below will show unscoped, all-customer options."
                )
            else:
                st.caption(
                    f"⚠️ Not enough {selection} bug history to narrow dropdowns to "
                    "this customer. Showing unscoped, all-customer options."
                )

        st.divider()

    customer = st.session_state.get("selected_customer")
    if not customer:
        st.info("Select a customer from the sidebar to view data.", icon="👈")
        st.stop()

    return customer, engagement_type
