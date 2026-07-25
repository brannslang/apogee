import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from config import ENGAGEMENT_TYPES, MIN_BUGS_FOR_TABLE
from model.predict import (
    artifacts_exist,
    get_customer_roster_by_industry,
    get_industries,
    get_industry_monthly_series,
    get_industry_rank_trend,
    get_industry_risk_tables,
)

st.set_page_config(page_title="Industry Benchmark — Apogee", page_icon="🏭", layout="wide")
st.title("Industry Benchmark")
st.caption(
    "Cross-customer comparison of High/Critical bug rates by industry, computed over the full "
    "multi-customer dataset. Scoped to a single Engagement Type (Functional / Accessibility / "
    "Security) — the same team split used everywhere else in Apogee."
)

if not artifacts_exist():
    st.error("Model artifacts not found. Upload data from the **Data Upload** page to train the model.")
    st.stop()

industries = get_industries()
if not industries:
    st.error("No industries found in training data.")
    st.stop()

with st.sidebar:
    st.markdown("### Industry")
    current_industry = st.session_state.get("selected_industry")
    industry_idx = industries.index(current_industry) if current_industry in industries else 0

    industry = st.selectbox(
        "Industry",
        options=industries,
        index=industry_idx,
        label_visibility="collapsed",
        key="industry_selector",
    )
    st.session_state["selected_industry"] = industry

    st.markdown("### Engagement Type")
    current_type = st.session_state.get("selected_industry_engagement_type")
    type_idx = ENGAGEMENT_TYPES.index(current_type) if current_type in ENGAGEMENT_TYPES else 0

    engagement_type = st.radio(
        "Engagement Type",
        options=ENGAGEMENT_TYPES,
        index=type_idx,
        label_visibility="collapsed",
        key="industry_engagement_type_selector",
        help="Accessibility, Security, and Functional bugs are each tracked separately — different teams, different risk profiles.",
    )
    st.session_state["selected_industry_engagement_type"] = engagement_type

    st.divider()

tables = get_industry_risk_tables(industry, engagement_type)

if not tables:
    st.warning(f"No {engagement_type.lower()} training data found for the **{industry}** industry.")
    st.stop()

st.subheader(f"Industry: {industry} — {engagement_type}")
baseline = tables["baseline"]
total    = tables["total_bugs"]

m1, m2 = st.columns(2)
m1.metric("H/C Baseline Rate", f"{baseline:.1%}")
m2.metric("Total Bugs", f"{total:,}")

st.divider()

# --- Quality & Process Metrics ---
st.subheader("Quality & Process Metrics")
st.caption(
    "Signals beyond the H/C rate — requirements clarity, QA verification loop closure, "
    "and structured test coverage."
)


def _fmt_pct(v):
    return f"{v:.1%}" if v is not None else "n/a"


q1, q2 = st.columns(2)
q1.metric(
    "Fix verification rate",
    _fmt_pct(tables.get("fix_verification_requested_rate")),
    help="Share of bugs where fix verification was ever requested (i.e. didn't sit at 'Not Requested'). "
         "Low = fixes shipping without a closed QA loop.",
)
q2.metric(
    "Rejection rate",
    _fmt_pct(tables.get("rejection_rate")),
    help="Share of all submitted bugs that were Rejected or Discarded.",
)

q3, q4 = st.columns(2)
q3.metric(
    "Works-as-designed share",
    _fmt_pct(tables.get("wad_share")),
    help="Of rejected bugs, the share rejected as 'Works as designed'. High = requirements/acceptance-"
         "criteria misalignment rather than a QA quality problem.",
)
q4.metric(
    "Coverage gap score",
    f"{tables['coverage_gap_score']:.1f}" if tables.get("coverage_gap_score") is not None else "n/a",
    help="Exploratory bugs ÷ (Structured bugs + 1). Higher = less structured regression coverage.",
)
st.caption(
    "Coverage Gap Score is derived from Bug Source Type (Structured vs. Exploratory), which is not "
    "consistently logged across all customers and testers — treat this score as directional, not precise."
)

st.divider()

# --- Client Roster (info-only) ---
st.subheader("Clients in This Industry")
st.caption(
    "Informational only — not a filter. Ranked by bug volume, a proxy for engagement scale "
    "(Apogee doesn't track company size or revenue directly); distinct test cycles shown "
    "as a secondary signal."
)
roster = get_customer_roster_by_industry(industry)
if roster:
    roster_df = pd.DataFrame(roster)
    roster_df.insert(0, "Rank", range(1, len(roster_df) + 1))
    roster_df = roster_df.rename(columns={
        "bug_volume":  "Bug Volume",
        "test_cycles": "Test Cycles",
        "Customer":    "Customer",
    })
    st.dataframe(roster_df, use_container_width=True, hide_index=True)
else:
    st.info("No client roster available for this industry.")

st.divider()


# --- Breakdown charts ---
def risk_bar(df: pd.DataFrame, dim_col: str, title: str, top_n: int = 20):
    if df.empty:
        st.info(f"No data available for {title}.")
        return
    plot_df = df.head(top_n).copy()
    plot_df["color"] = plot_df["hc_rate"].apply(
        lambda r: "#d62728" if r > 0.50 else "#ff7f0e" if r > 0.35 else "#1f77b4"
    )
    plot_df["label"] = (
        plot_df["hc_rate"].map("{:.0%}".format)
        + " (n="
        + plot_df["n_bugs"].astype(str)
        + ")"
    )
    fig = go.Figure(
        go.Bar(
            x=plot_df["hc_rate"],
            y=plot_df[dim_col].astype(str),
            orientation="h",
            marker_color=plot_df["color"],
            text=plot_df["label"],
            textposition="outside",
            textfont=dict(color="#444444"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "H/C Rate: %{x:.1%}<br>"
                "<extra></extra>"
            ),
        )
    )
    fig.add_vline(
        x=baseline,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Baseline {baseline:.1%}",
        annotation_position="top right",
    )
    fig.update_layout(
        title=title,
        xaxis_title="High/Critical Rate",
        yaxis=dict(autorange="reversed"),
        height=max(350, len(plot_df) * 28 + 80),
        margin=dict(l=10, r=120, t=50, b=40),
        plot_bgcolor="white",
        xaxis=dict(tickformat=".0%", range=[0, min(1.0, plot_df["hc_rate"].max() * 1.35)]),
    )
    st.plotly_chart(fig, use_container_width=True)


st.subheader("Breakdowns")
st.caption(
    "App Component and Parent App Component are customer-specific vocabulary (often not even "
    "captured) and don't roll up meaningfully across an industry, so this tab uses dimensions "
    "that are genuinely comparable across customers instead."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "By Bug Type",
    "By Sub-Industry",
    "By Platform",
    "By Bug Source Type",
])

with tab1:
    risk_bar(
        tables.get("Bug Type", pd.DataFrame()),
        "Bug Type",
        "High/Critical Rate by Bug Type",
    )

with tab2:
    st.caption(
        f"CAPDB Sub-Industry is a finer taxonomy beneath {industry}. Many industries map to a "
        "single sub-industry — a one-row chart there just means the two are equivalent for this industry."
    )
    risk_bar(
        tables.get("CAPDB Sub-Industry", pd.DataFrame()),
        "CAPDB Sub-Industry",
        f"High/Critical Rate by Sub-Industry within {industry}",
    )

with tab3:
    risk_bar(
        tables.get("Platform Group", pd.DataFrame()),
        "Platform Group",
        "High/Critical Rate by Platform (Android / iOS / Web / OTT / Other)",
    )

with tab4:
    st.caption(
        "⚠️ Bug Source Type (Structured vs. Exploratory) is not consistently logged across all "
        "customers and testers — treat this breakdown as directional, not precise."
    )
    risk_bar(
        tables.get("Bug Source Type", pd.DataFrame()),
        "Bug Source Type",
        "High/Critical Rate by Bug Source Type",
    )

st.divider()

# --- Monthly trend for this industry, vs. peer industries ---
monthly = tables.get("monthly_trend")
all_monthly = get_industry_monthly_series(engagement_type)

if monthly is not None and not monthly.empty:
    st.subheader("H/C Rate Over Time")

    peer_band = pd.DataFrame()
    if not all_monthly.empty:
        peers = all_monthly[all_monthly["Industry"] != industry]
        if not peers.empty:
            peer_band = (
                peers.groupby("month")["hc_rate"]
                .quantile([0.25, 0.5, 0.75])
                .unstack()
                .rename(columns={0.25: "p25", 0.5: "p50", 0.75: "p75"})
                .reset_index()
                .sort_values("month")
            )

    fig_trend = go.Figure()

    if not peer_band.empty:
        merged = monthly[["month"]].merge(peer_band, on="month", how="left")
        fig_trend.add_trace(go.Scatter(
            x=merged["month"], y=merged["p75"], mode="lines",
            line=dict(width=0), hoverinfo="skip", showlegend=False,
        ))
        fig_trend.add_trace(go.Scatter(
            x=merged["month"], y=merged["p25"], mode="lines",
            line=dict(width=0), fill="tonexty", fillcolor="rgba(128,128,128,0.25)",
            hoverinfo="skip", showlegend=False, name="Peer P25–P75 range",
        ))
        fig_trend.add_trace(go.Scatter(
            x=merged["month"], y=merged["p50"], mode="lines",
            line=dict(color="gray", width=1.5, dash="dot"),
            name="Peer median", hovertemplate="Peer median: %{y:.1%}<extra></extra>",
        ))

    fig_trend.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["hc_rate"],
            mode="lines+markers",
            name=industry,
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
            hovertemplate="Month: %{x}<br>H/C Rate: %{y:.1%}<extra></extra>",
        )
    )
    fig_trend.add_hline(
        y=baseline,
        line_dash="dash",
        line_color="black",
        annotation_text=f"{industry} baseline {baseline:.1%}",
        annotation_position="top left",
    )
    fig_trend.update_layout(
        xaxis_title="Month",
        yaxis_title="High/Critical Rate",
        yaxis=dict(tickformat=".0%"),
        height=380,
        plot_bgcolor="white",
        margin=dict(t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    if not peer_band.empty:
        st.caption(
            f"Shaded band = the middle 50% (P25–P75) of every other industry's H/C rate each month; "
            f"dotted gray line = peer median. Shows whether {industry} is moving inside or outside the "
            "normal range for its peers, not just up or down in isolation."
        )
    st.divider()

# --- Industry Trend Leaderboard ---
rank_trend = get_industry_rank_trend(engagement_type)
trendable = [r for r in rank_trend if r.get("trend_slope") is not None]

if trendable:
    st.subheader("Industry Trend Leaderboard")
    st.caption(
        "Industries with the steepest H/C rate slope over their available monthly history, "
        "shown as sparklines — tiny trend lines with no axis, showing shape rather than "
        "precise values."
    )

    total_ranked = len(rank_trend)
    max_n = max(1, len(trendable) // 2)
    top_n = st.slider(
        "Industries to show per side (most degrading / most improving)",
        1, min(8, max_n), min(4, max_n),
    )

    degrading = sorted(trendable, key=lambda r: r["trend_slope"], reverse=True)[:top_n]
    improving = sorted(trendable, key=lambda r: r["trend_slope"])[:top_n]

    def _peer_band_for(ind_name: str) -> pd.DataFrame:
        if all_monthly.empty:
            return pd.DataFrame()
        peers = all_monthly[all_monthly["Industry"] != ind_name]
        if peers.empty:
            return pd.DataFrame()
        return (
            peers.groupby("month")["hc_rate"]
            .quantile([0.25, 0.75])
            .unstack()
            .rename(columns={0.25: "p25", 0.75: "p75"})
            .reset_index()
            .sort_values("month")
        )

    def render_trend_tile(entry: dict):
        ind_name = entry["Industry"]
        ind_monthly = all_monthly[all_monthly["Industry"] == ind_name].sort_values("month")
        if ind_monthly.empty:
            return

        x = list(range(len(ind_monthly)))
        y = ind_monthly["hc_rate"].tolist()
        low_conf = (ind_monthly["n_bugs"] < MIN_BUGS_FOR_TABLE).tolist()

        direction = entry.get("trend_direction")
        line_color = "#d62728" if direction == "Degrading" else "#2ca02c" if direction == "Improving" else "#888888"

        fig = go.Figure()

        band = _peer_band_for(ind_name)
        if not band.empty:
            merged = ind_monthly[["month"]].merge(band, on="month", how="left")
            fig.add_trace(go.Scatter(
                x=x, y=merged["p75"], mode="lines", line=dict(width=0),
                hoverinfo="skip", showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=x, y=merged["p25"], mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(128,128,128,0.25)",
                hoverinfo="skip", showlegend=False,
            ))

        # Confidence-weighted line: dashed segments where monthly bug volume
        # is below the reliability threshold, solid where it's robust.
        i = 0
        n = len(x)
        while i < n:
            j = i
            while j + 1 < n and low_conf[j + 1] == low_conf[i]:
                j += 1
            start = i - 1 if i > 0 else i
            end = j + 1 if j < n - 1 else j
            fig.add_trace(go.Scatter(
                x=x[start:end + 1], y=y[start:end + 1], mode="lines",
                line=dict(color=line_color, width=2, dash="dot" if low_conf[i] else "solid"),
                opacity=0.55 if low_conf[i] else 1.0,
                hoverinfo="skip", showlegend=False,
            ))
            i = j + 1

        fig.add_annotation(x=x[0], y=y[0], text=f"{y[0]:.0%}", showarrow=False,
                            xanchor="right", xshift=-4, font=dict(size=10, color="#888888"))
        fig.add_annotation(x=x[-1], y=y[-1], text=f"{y[-1]:.0%}", showarrow=False,
                            xanchor="left", xshift=4, font=dict(size=10, color=line_color))

        fig.update_layout(
            height=90,
            margin=dict(l=28, r=28, t=8, b=8),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )

        rank_now = entry["rank_now"]
        rank_6mo = entry.get("rank_6mo_ago")
        if rank_6mo is not None and rank_6mo != rank_now:
            arrow = "↑" if rank_now < rank_6mo else "↓"
            rank_text = f":red[{arrow}]" if rank_now < rank_6mo else f":green[{arrow}]"
            st.markdown(f"**{ind_name}**")
            st.caption(f"{entry['baseline']:.1%} H/C  ·  Rank {rank_now} of {total_ranked} ({rank_text} was {rank_6mo})")
        else:
            st.markdown(f"**{ind_name}**")
            st.caption(f"{entry['baseline']:.1%} H/C  ·  Rank {rank_now} of {total_ranked}")

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    col_deg, col_imp = st.columns(2)
    with col_deg:
        st.markdown("**Most degrading**")
        for entry in degrading:
            render_trend_tile(entry)
    with col_imp:
        st.markdown("**Most improving**")
        for entry in improving:
            render_trend_tile(entry)

    st.caption(
        "How to read these: the gray shaded band is the P25–P75 range of every other industry each "
        "month — a line leaving the band means that industry is now outside the normal peer range, "
        "not just moving. Dotted segments mark months with fewer than "
        f"{MIN_BUGS_FOR_TABLE} bugs — treat those stretches as low-confidence, not a real signal. "
        "The rank badge compares each industry's current H/C-rate rank among all industries to its rank "
        "6 months ago; ↑ red means it climbed toward riskier, ↓ green means it dropped toward safer."
    )
    st.divider()

# --- Cross-industry comparison ---
st.subheader("All Industries at a Glance")
st.caption(f"Baseline H/C rate across all industries for **{engagement_type}** engagements.")

compare_rows = []
for ind in industries:
    ind_tables = get_industry_risk_tables(ind, engagement_type)
    if ind_tables:
        compare_rows.append({
            "Industry": ind,
            "hc_rate":  ind_tables["baseline"],
            "n_bugs":   ind_tables["total_bugs"],
        })

if compare_rows:
    compare_df = pd.DataFrame(compare_rows).sort_values("hc_rate", ascending=True)
    compare_df["Selected"] = compare_df["Industry"] == industry

    fig_compare = px.bar(
        compare_df,
        x="hc_rate",
        y="Industry",
        orientation="h",
        color="hc_rate",
        color_continuous_scale=["#2ca02c", "#ff7f0e", "#d62728"],
        range_color=[0, 1],
        text=compare_df["hc_rate"].map("{:.0%}".format),
        hover_data={"n_bugs": True, "hc_rate": ":.1%", "Selected": False},
        labels={"hc_rate": "H/C Baseline Rate", "Industry": ""},
    )
    fig_compare.update_layout(
        height=max(300, len(compare_df) * 35 + 80),
        xaxis=dict(tickformat=".0%"),
        coloraxis_showscale=False,
        plot_bgcolor="white",
        margin=dict(l=10, r=80, t=20, b=40),
    )
    fig_compare.update_traces(
        textposition="outside",
        textfont_color="#444444",
        marker_line_color=["black" if s else "rgba(0,0,0,0)" for s in compare_df["Selected"]],
        marker_line_width=[3 if s else 0 for s in compare_df["Selected"]],
    )
    st.plotly_chart(fig_compare, use_container_width=True)
    st.caption(f"Black outline marks the selected industry (**{industry}**).")
else:
    st.info("Not enough data to compare industries.")

st.divider()
st.caption(
    "Red = >50% H/C rate  |  Orange = 35–50%  |  Blue = <35%  |  Dashed line = industry baseline. "
    "Minimum 10 bugs per row/industry split."
)
