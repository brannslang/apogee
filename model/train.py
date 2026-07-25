"""
Train the Apogee risk classifier and precompute risk tables.

Run once (and monthly) to refresh model artifacts:
    python model/train.py
"""

import re
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # not installed in the cloud env; skip embedding features
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from config import (
    ARTIFACTS_DIR, CATEGORICAL_FEATURES, DATA_DIR,
    ENGAGEMENT_TYPE_OVERRIDES, ENGAGEMENT_TYPES, EXCLUDED_TESTING_TYPES,
    GRAPH_FEATURES, KEYWORD_GROUPS, MIN_BUGS_FOR_TABLE,
    N_EMB_COMPONENTS, N_NMF_FACTORS, N_SVD_COMPONENTS,
    NMF_ENTITY_COLS, NMF_FEATURES, NUMERIC_FEATURES,
    TARGET, TEXT_EMB_FEATURES, TEXT_FLAG_FEATURES,
    TEXT_SVD_FEATURES,
)
from xlsx_io import read_excel_all_sheets

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# "Platform Product Name" is free text combining app name + platform (e.g.
# "ConsumerApp (Android)", "MediaCo OTT App - tvOS") and is unique per customer, so
# it can't be compared across customers/industries. This regex buckets it
# into a small, cross-customer-comparable set for the Industry Benchmark tab.
# OTT is checked first since some OTT device names (e.g. "Android TV") would
# otherwise false-match the Android/iOS keyword checks below.
_OTT_PATTERN = re.compile(
    r"(?i)\b(ott|tvos|ftv|fire ?tv|apple ?tv|\batv\b|chromecast|roku|"
    r"smart ?tv|android ?tv|samsung ?tv|xbox|playstation|ps4|ps5)\b"
)


def _normalize_platform_group(name) -> str:
    if pd.isna(name):
        return "Other"
    n = str(name).lower()
    if _OTT_PATTERN.search(n):
        return "OTT"
    if "android" in n:
        return "Android"
    if "ios" in n or "iphone" in n or "ipad" in n:
        return "iOS"
    if "web" in n or "website" in n or ".com" in n:
        return "Web"
    return "Other"


def load_data() -> pd.DataFrame:
    bugs = pd.read_excel(os.path.join(DATA_DIR, "bugdetails.xlsx"), engine="openpyxl")
    cycles = pd.read_excel(os.path.join(DATA_DIR, "testcycles.xlsx"), engine="openpyxl")

    cycle_cols = [
        "Test Cycle Id",
        "Test Cycle Testing Type",
        "Test Cycle Duration Activation to Lock/Close/Today",
        "Testing Approach",
    ]
    df = bugs.merge(cycles[cycle_cols], on="Test Cycle Id", how="left")

    # Usability/UX engagements deliver interviews, surveys, and qualitative
    # feedback rather than bugs — drop them from training/analysis entirely.
    # A client with other (Functional/Accessibility) engagements is unaffected;
    # this only removes their UX-specific rows.
    df = df[~df["Test Cycle Testing Type"].isin(EXCLUDED_TESTING_TYPES)].copy()

    # Accessibility and Security each get their own bucket (separate teams,
    # separate risk profiles). Everything else — Functional, Localization,
    # and Payment-Transaction-compensated bugs (tagged Functional/"Payment
    # Testing") — falls under Managed Testing Services and is bucketed
    # as "Functional".
    df["Engagement Type"] = (
        df["Test Cycle Testing Type"].map(ENGAGEMENT_TYPE_OVERRIDES).fillna("Functional")
    )

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET] = df["Bug Severity"].isin(["Critical", "High"]).astype(int)

    if "Platform Product Name" in df.columns:
        df["Platform Group"] = df["Platform Product Name"].map(_normalize_platform_group)

    return df


def build_text_features(df: pd.DataFrame) -> tuple:
    """Stream A: keyword flags, TF-IDF + SVD, sentence embeddings + PCA."""
    subject = df["Bug Subject"].fillna("") if "Bug Subject" in df.columns else pd.Series("", index=df.index)
    result  = df["Bug Result"].fillna("")  if "Bug Result"  in df.columns else pd.Series("", index=df.index)
    text = (subject + " " + result).str.strip()

    # Layer 1: keyword flags
    flag_df = pd.DataFrame(index=df.index)
    for col, pattern in KEYWORD_GROUPS.items():
        flag_df[col] = text.str.contains(pattern, case=False, regex=True).astype(int)

    # Layer 2: TF-IDF + Truncated SVD
    tfidf = TfidfVectorizer(
        min_df=5, max_features=2000, stop_words="english", ngram_range=(1, 2)
    )
    tfidf_matrix = tfidf.fit_transform(text)
    n_svd = min(N_SVD_COMPONENTS, tfidf_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_svd, random_state=42)
    svd_arr = svd.fit_transform(tfidf_matrix)
    svd_df = pd.DataFrame(index=df.index)
    for i in range(N_SVD_COMPONENTS):
        svd_df[f"text_svd_{i}"] = svd_arr[:, i] if i < n_svd else 0.0

    # Layer 3: sentence embeddings + PCA (skipped if sentence-transformers isn't installed,
    # e.g. when retraining runs inside the lightweight Streamlit Cloud environment)
    emb_df = pd.DataFrame(0.0, index=df.index, columns=[f"text_emb_{i}" for i in range(N_EMB_COMPONENTS)])
    pca = None
    n_emb = 0
    sentence_model_name = None
    if SentenceTransformer is not None:
        print("    Encoding bug text with sentence transformer (this may take a minute)...")
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = st_model.encode(text.tolist(), show_progress_bar=True, batch_size=128)
        n_emb = min(N_EMB_COMPONENTS, embeddings.shape[0] - 1, embeddings.shape[1])
        pca = PCA(n_components=n_emb, random_state=42)
        pca_arr = pca.fit_transform(embeddings)
        for i in range(N_EMB_COMPONENTS):
            emb_df[f"text_emb_{i}"] = pca_arr[:, i] if i < n_emb else 0.0
        sentence_model_name = "all-MiniLM-L6-v2"
    else:
        print("    sentence-transformers not installed — skipping sentence embedding features "
              "(keyword flags + TF-IDF/SVD text features still apply).")

    feature_df = pd.concat([flag_df, svd_df, emb_df], axis=1)
    artifacts = {
        "tfidf": tfidf,
        "svd": svd,
        "pca": pca,
        "n_svd": n_svd,
        "n_emb": n_emb,
        "sentence_model_name": sentence_model_name,
        "keyword_groups": KEYWORD_GROUPS,
    }
    return feature_df, artifacts


def build_nmf_features(df: pd.DataFrame) -> tuple:
    """Stream B: NMF latent factors from entity co-occurrence matrix."""
    available_cols = [c for c in NMF_ENTITY_COLS if c in df.columns]

    encoded_parts = []
    for col in available_cols:
        dummies = pd.get_dummies(df[col].fillna("Unknown"), prefix=col)
        encoded_parts.append(dummies)

    entity_matrix = pd.concat(encoded_parts, axis=1).astype(float)
    feature_names = list(entity_matrix.columns)

    nmf = NMF(n_components=N_NMF_FACTORS, random_state=42, max_iter=500)
    nmf_arr = nmf.fit_transform(entity_matrix)
    feature_df = pd.DataFrame(nmf_arr, columns=NMF_FEATURES, index=df.index)

    artifacts = {
        "nmf": nmf,
        "entity_cols": available_cols,
        "feature_names": feature_names,
    }
    return feature_df, artifacts


def build_graph_features(df: pd.DataFrame) -> tuple:
    """Stream C: property graph — entity co-occurrence network weighted by above-baseline H/C rate."""
    baseline = df[TARGET].mean()
    G = nx.Graph()

    entity_pairs = [
        ("App Component", "Platform Product Name"),
        ("App Component", "Customer"),
        ("App Component", "Development Stage"),
        ("App Component", "Testing Approach"),
        ("Platform Product Name", "Customer"),
    ]

    for col_a, col_b in entity_pairs:
        if col_a not in df.columns or col_b not in df.columns:
            continue
        agg = (
            df.groupby([col_a, col_b])[TARGET]
            .agg(["mean", "count"])
            .reset_index()
        )
        agg = agg[agg["count"] >= MIN_BUGS_FOR_TABLE]
        agg["weight"] = (agg["mean"] - baseline).clip(lower=0)
        for _, row in agg.iterrows():
            if row["weight"] > 0:
                node_a = f"{col_a}:{row[col_a]}"
                node_b = f"{col_b}:{row[col_b]}"
                if G.has_edge(node_a, node_b):
                    G[node_a][node_b]["weight"] = max(
                        G[node_a][node_b]["weight"], row["weight"]
                    )
                else:
                    G.add_edge(node_a, node_b, weight=row["weight"])

    node_metrics = {}
    if len(G.nodes()) > 0:
        pr = nx.pagerank(G, weight="weight")
        dc = nx.degree_centrality(G)
        try:
            cl = nx.clustering(G, weight="weight")
        except Exception:
            cl = {n: 0.0 for n in G.nodes()}
        for node in G.nodes():
            node_metrics[node] = {
                "pagerank": pr.get(node, 0.0),
                "degree_centrality": dc.get(node, 0.0),
                "clustering": cl.get(node, 0.0),
            }

    def make_lookup(prefix, metric):
        return {
            k.split(":", 1)[1]: v[metric]
            for k, v in node_metrics.items()
            if k.startswith(f"{prefix}:")
        }

    comp_pr = make_lookup("App Component", "pagerank")
    comp_dc = make_lookup("App Component", "degree_centrality")
    comp_cl = make_lookup("App Component", "clustering")
    plat_pr = make_lookup("Platform Product Name", "pagerank")
    cust_pr = make_lookup("Customer", "pagerank")

    feature_df = pd.DataFrame(index=df.index)
    feature_df["graph_comp_pagerank"]     = df["App Component"].map(comp_pr).fillna(0.0)          if "App Component"         in df.columns else 0.0
    feature_df["graph_comp_degree"]       = df["App Component"].map(comp_dc).fillna(0.0)          if "App Component"         in df.columns else 0.0
    feature_df["graph_comp_clustering"]   = df["App Component"].map(comp_cl).fillna(0.0)          if "App Component"         in df.columns else 0.0
    feature_df["graph_platform_pagerank"] = df["Platform Product Name"].map(plat_pr).fillna(0.0)  if "Platform Product Name" in df.columns else 0.0
    feature_df["graph_customer_pagerank"] = df["Customer"].map(cust_pr).fillna(0.0)               if "Customer"              in df.columns else 0.0

    return feature_df, {"node_metrics": node_metrics}


def compute_text_profiles(df: pd.DataFrame) -> dict:
    """
    Compute per-component and per-platform mean text feature profiles.

    Two sub-profiles per entity:
      'all' — mean across all bugs (used to impute text features at prediction time)
      'hc'  — mean across H/C bugs only (used to surface risk signals in the UI)
    """
    all_text_cols = [c for c in TEXT_FLAG_FEATURES + TEXT_SVD_FEATURES + TEXT_EMB_FEATURES if c in df.columns]
    flag_cols     = [c for c in TEXT_FLAG_FEATURES if c in df.columns]

    def means(subset: pd.DataFrame) -> dict:
        return subset[all_text_cols].mean().to_dict() if len(subset) > 0 else {}

    global_profile = {
        "all": means(df),
        "hc":  means(df[df[TARGET] == 1]),
    }

    by_component = {}
    if "App Component" in df.columns:
        for comp, grp in df.groupby("App Component"):
            hc_grp = grp[grp[TARGET] == 1]
            by_component[comp] = {
                "all":    means(grp),
                "hc":     means(hc_grp) if len(hc_grp) >= MIN_BUGS_FOR_TABLE else {},
                "n_bugs": len(grp),
                "n_hc":   int(hc_grp[TARGET].sum()),
            }

    by_platform = {}
    if "Platform Product Name" in df.columns:
        for plat, grp in df.groupby("Platform Product Name"):
            by_platform[plat] = {"all": means(grp)}

    # Monthly keyword flag trends — global, and split per customer + engagement
    # type so the Monthly Digest's language-trend chart can be scoped to the
    # selected customer instead of blending every customer's bug language together.
    date_col = next(
        (c for c in df.columns if "create" in c.lower() and "date" in c.lower()), None
    )

    def _monthly_flag_trends(subset: pd.DataFrame):
        if not date_col or not flag_cols or subset.empty:
            return None
        _df = subset.copy()
        _df["_month"] = pd.to_datetime(_df[date_col], errors="coerce").dt.to_period("M")
        agg = {f: "mean" for f in flag_cols}
        agg[TARGET] = "count"
        trend = (
            _df.groupby("_month")
            .agg(agg)
            .reset_index()
            .rename(columns={"_month": "month", TARGET: "n_bugs"})
        )
        trend["month"] = trend["month"].astype(str)
        return trend

    monthly_flag_trends = _monthly_flag_trends(df)

    monthly_flag_trends_by_customer = {}
    if "Customer" in df.columns and "Engagement Type" in df.columns:
        for customer, cust_grp in df.groupby("Customer"):
            per_type = {}
            for etype, type_grp in cust_grp.groupby("Engagement Type"):
                trend = _monthly_flag_trends(type_grp)
                if trend is not None and not trend.empty:
                    per_type[etype] = trend
            if per_type:
                monthly_flag_trends_by_customer[customer] = per_type

    return {
        "by_component":       by_component,
        "by_platform":        by_platform,
        "global":             global_profile,
        "flag_cols":          flag_cols,
        "all_text_cols":      all_text_cols,
        "monthly_flag_trends": monthly_flag_trends,
        "monthly_flag_trends_by_customer": monthly_flag_trends_by_customer,
    }


REJECTED_STATUSES = ["Rejected", "Discard"]


def _quality_process_metrics(df: pd.DataFrame) -> dict:
    """
    Metrics beyond H/C rate, for consultative industry-level comparison:

    - rejection_rate / wad_share: what fraction of bugs get rejected, and of
      those, what fraction are "Works as designed" — high WAD share signals
      requirements/acceptance-criteria misalignment rather than a QA quality
      problem, a different story than the H/C rate tells.
    - fix_verification_requested_rate: what fraction of bugs ever leave the
      "Not Requested" state — most bugs across this whole dataset never do,
      so this is a real closed-loop-QA signal, not a rounding error.
    - coverage_gap_score: Exploratory bugs / (Structured bugs + 1). High =
      little regression safety net. Bug Source Type is not consistently
      logged across all customers/testers, so treat this as directional.
    """
    metrics = {}

    if "Bug Status" in df.columns:
        is_rejected = df["Bug Status"].isin(REJECTED_STATUSES)
        metrics["rejection_rate"] = float(is_rejected.mean()) if len(df) else None
        if "Bug Reject Reason" in df.columns and is_rejected.sum() > 0:
            wad = (df.loc[is_rejected, "Bug Reject Reason"] == "Works as designed").mean()
            metrics["wad_share"] = float(wad)
        else:
            metrics["wad_share"] = None
    else:
        metrics["rejection_rate"] = None
        metrics["wad_share"] = None

    if "Bug Fix Verification Status" in df.columns and len(df):
        requested = df["Bug Fix Verification Status"] != "Not Requested"
        metrics["fix_verification_requested_rate"] = float(requested.mean())
    else:
        metrics["fix_verification_requested_rate"] = None

    if "Bug Source Type" in df.columns:
        exp_n = int((df["Bug Source Type"] == "Exploratory").sum())
        str_n = int((df["Bug Source Type"] == "Structured").sum())
        metrics["coverage_gap_score"] = round(exp_n / (str_n + 1), 2)
    else:
        metrics["coverage_gap_score"] = None

    return metrics


def _trend_slope(monthly: pd.DataFrame) -> dict:
    """
    Linear regression slope of monthly H/C rate over successive months.
    Needs >=3 months of data to be meaningful; returns Nones otherwise.
    """
    if monthly is None or len(monthly) < 3:
        return {"trend_slope": None, "trend_direction": None}
    x = np.arange(len(monthly))
    y = monthly["hc_rate"].to_numpy()
    slope, _, _, _, _ = stats.linregress(x, y)
    direction = "Degrading" if slope > 0.005 else "Improving" if slope < -0.005 else "Stable"
    return {"trend_slope": round(float(slope), 5), "trend_direction": direction}


def _risk_tables_for(df: pd.DataFrame) -> dict:
    """Compute the standard risk table dict for any bug DataFrame slice."""
    baseline = df[TARGET].mean()
    tables = {"baseline": baseline, "total_bugs": len(df)}
    tables.update(_quality_process_metrics(df))

    for col in RISK_DIMS:
        if col not in df.columns:
            continue
        tbl = (
            df.groupby(col)[TARGET]
            .agg(["mean", "count", "sum"])
            .rename(columns={"mean": "hc_rate", "count": "n_bugs", "sum": "n_hc"})
            .reset_index()
        )
        tbl["vs_baseline"] = tbl["hc_rate"] - baseline
        tbl = tbl[tbl["n_bugs"] >= MIN_BUGS_FOR_TABLE].sort_values(
            "hc_rate", ascending=False
        )
        tables[col] = tbl

    date_col = next(
        (c for c in df.columns if "create" in c.lower() and "date" in c.lower()), None
    )
    if date_col:
        _df = df.copy()
        _df["_month"] = pd.to_datetime(_df[date_col], errors="coerce").dt.to_period("M")
        monthly = (
            _df.groupby("_month")[TARGET]
            .agg(["mean", "count"])
            .rename(columns={"mean": "hc_rate", "count": "n_bugs"})
            .reset_index()
        )
        monthly["_month"] = monthly["_month"].astype(str)
        monthly = monthly.rename(columns={"_month": "month"})
        tables["monthly_trend"] = monthly
        tables.update(_trend_slope(monthly))
    else:
        tables.update({"trend_slope": None, "trend_direction": None})

    return tables


def _categories_for(df: pd.DataFrame, cat_cols: list) -> dict:
    """Distinct dropdown option values per categorical column, for any bug DataFrame slice."""
    return {
        col: sorted(df[col].dropna().astype(str).unique().tolist())
        for col in cat_cols
        if col in df.columns
    }


def compute_risk_tables(df: pd.DataFrame) -> dict:
    """
    Bug-based roster: a customer appears in the dashboard if they have at
    least one bug (of any Engagement Type surviving EXCLUDED_TESTING_TYPES).
    A client with a UX-only engagement and nothing else has zero rows left
    at this point and correctly won't appear — but a client with UX *and*
    Functional/Accessibility work still shows up via their non-UX bugs.

    Per customer and globally, risk tables are additionally split by
    Engagement Type (Functional / Accessibility / Security — each run by a
    different team) so one team's findings don't get blended into another's
    risk profile. Each split is independently gated by MIN_BUGS_FOR_TABLE,
    same as the old single-table threshold.

    Dropdown option lists (categories_by_customer) are computed on the same
    per-customer/per-Engagement-Type splits so the Release Predictor's filters
    only ever offer values that actually occur for the selected customer —
    e.g. picking a customer narrows App Component, Platform, etc. to just
    that customer's values instead of every customer's.

    Industry Benchmark tables (by_industry / categories_by_industry) mirror
    the by_customer split but grouped by Industry x Engagement Type over the
    full multi-customer dataset, so one industry's risk profile can be
    compared against another's. customer_roster_by_industry is an info-only
    client list per industry (not used for filtering); customer_industry is
    a Customer -> Industry lookup used to silently enrich the classifier's
    Industry feature at prediction time when the caller only supplies Customer.
    """
    tables = _risk_tables_for(df)
    available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    has_customer_col = "Customer" in df.columns
    customers = sorted(df["Customer"].dropna().unique().tolist()) if has_customer_col else []

    by_customer = {}
    categories_by_customer = {}
    if has_customer_col:
        for customer in customers:
            cust_subset = df[df["Customer"] == customer]
            type_tables = {}
            type_categories = {}
            for etype in ENGAGEMENT_TYPES:
                type_subset = cust_subset[cust_subset["Engagement Type"] == etype]
                if len(type_subset) >= MIN_BUGS_FOR_TABLE:
                    type_tables[etype] = _risk_tables_for(type_subset)
                    type_categories[etype] = _categories_for(type_subset, available_cat)
            if type_tables:
                by_customer[customer] = type_tables
            if type_categories:
                categories_by_customer[customer] = type_categories

    by_type = {}
    for etype in ENGAGEMENT_TYPES:
        type_subset = df[df["Engagement Type"] == etype]
        if len(type_subset) >= MIN_BUGS_FOR_TABLE:
            by_type[etype] = _risk_tables_for(type_subset)

    has_industry_col = "Industry" in df.columns
    industries = sorted(df["Industry"].dropna().unique().tolist()) if has_industry_col else []

    by_industry = {}
    categories_by_industry = {}
    customer_roster_by_industry = {}
    customer_industry = {}
    if has_industry_col and has_customer_col:
        has_cycle_col = "Test Cycle Id" in df.columns
        for industry in industries:
            ind_subset = df[df["Industry"] == industry]
            type_tables = {}
            type_categories = {}
            for etype in ENGAGEMENT_TYPES:
                type_subset = ind_subset[ind_subset["Engagement Type"] == etype]
                if len(type_subset) >= MIN_BUGS_FOR_TABLE:
                    type_tables[etype] = _risk_tables_for(type_subset)
                    type_categories[etype] = _categories_for(type_subset, available_cat)
            if type_tables:
                by_industry[industry] = type_tables
            if type_categories:
                categories_by_industry[industry] = type_categories

            roster = ind_subset.groupby("Customer").size().rename("bug_volume").reset_index()
            if has_cycle_col:
                cycles = (
                    ind_subset.groupby("Customer")["Test Cycle Id"]
                    .nunique()
                    .rename("test_cycles")
                    .reset_index()
                )
                roster = roster.merge(cycles, on="Customer", how="left")
            else:
                roster["test_cycles"] = None
            roster = roster.sort_values("bug_volume", ascending=False).reset_index(drop=True)
            customer_roster_by_industry[industry] = roster.to_dict("records")

        customer_industry = (
            df.groupby("Customer")["Industry"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
            .dropna()
            .to_dict()
        )

    # Industry trend leaderboard: current rank + rank as of 6 months ago (for
    # the rank-change badge) and a combined per-industry monthly series (used
    # to derive the peer P25-P75 band, computed at render time so it can
    # exclude whichever industry is currently selected).
    industry_rank_trend = {}
    industry_monthly_series = {}
    if has_industry_col and has_customer_col:
        date_col = next(
            (c for c in df.columns if "create" in c.lower() and "date" in c.lower()), None
        )
        cutoff = None
        if date_col:
            max_date = pd.to_datetime(df[date_col], errors="coerce").max()
            if pd.notna(max_date):
                cutoff = max_date - pd.DateOffset(months=6)

        for etype in ENGAGEMENT_TYPES:
            rows = []
            monthly_frames = []
            for industry, type_tables_for_industry in by_industry.items():
                scoped = type_tables_for_industry.get(etype)
                if not scoped:
                    continue

                row = {
                    "Industry": industry,
                    "baseline": scoped["baseline"],
                    "n_bugs": scoped["total_bugs"],
                    "trend_slope": scoped.get("trend_slope"),
                    "trend_direction": scoped.get("trend_direction"),
                    "baseline_6mo_ago": None,
                    "n_bugs_6mo_ago": None,
                }
                if cutoff is not None:
                    ind_type_subset = df[
                        (df["Industry"] == industry) & (df["Engagement Type"] == etype)
                    ]
                    trailing_dates = pd.to_datetime(ind_type_subset[date_col], errors="coerce")
                    trailing = ind_type_subset[trailing_dates <= cutoff]
                    if len(trailing) >= MIN_BUGS_FOR_TABLE:
                        row["baseline_6mo_ago"] = float(trailing[TARGET].mean())
                        row["n_bugs_6mo_ago"] = len(trailing)
                rows.append(row)

                monthly = scoped.get("monthly_trend")
                if monthly is not None and not monthly.empty:
                    m = monthly.copy()
                    m["Industry"] = industry
                    monthly_frames.append(m)

            if rows:
                rank_df = pd.DataFrame(rows).sort_values("baseline", ascending=False).reset_index(drop=True)
                rank_df["rank_now"] = rank_df.index + 1

                have_6mo = rank_df["baseline_6mo_ago"].notna()
                rank_6mo_df = (
                    rank_df[have_6mo]
                    .sort_values("baseline_6mo_ago", ascending=False)
                    .reset_index(drop=True)
                )
                rank_6mo_df["rank_6mo_ago"] = rank_6mo_df.index + 1
                rank_df = rank_df.merge(
                    rank_6mo_df[["Industry", "rank_6mo_ago"]], on="Industry", how="left"
                )

                records = []
                for _, r in rank_df.iterrows():
                    rec = r.to_dict()
                    for k, v in rec.items():
                        if isinstance(v, float) and np.isnan(v):
                            rec[k] = None
                    records.append(rec)
                industry_rank_trend[etype] = records

            if monthly_frames:
                industry_monthly_series[etype] = pd.concat(monthly_frames, ignore_index=True)

    tables["customers"] = customers
    tables["by_customer"] = by_customer
    tables["by_type"] = by_type
    tables["categories_by_customer"] = categories_by_customer
    tables["industries"] = industries
    tables["by_industry"] = by_industry
    tables["categories_by_industry"] = categories_by_industry
    tables["customer_roster_by_industry"] = customer_roster_by_industry
    tables["customer_industry"] = customer_industry
    tables["industry_rank_trend"] = industry_rank_trend
    tables["industry_monthly_series"] = industry_monthly_series
    return tables


def train_classifier(df: pd.DataFrame):
    all_numeric = (
        NUMERIC_FEATURES
        + TEXT_FLAG_FEATURES
        + TEXT_SVD_FEATURES
        + TEXT_EMB_FEATURES
        + NMF_FEATURES
        + GRAPH_FEATURES
    )
    available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    available_num = [c for c in all_numeric if c in df.columns]
    features = available_cat + available_num

    X = df[features].copy()
    for col in available_num:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    y = df[TARGET]

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )),
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocessor = ColumnTransformer([
        ("cat", cat_pipe, available_cat),
        ("num", num_pipe, available_num),
    ])

    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            # n_jobs=1: parallel fitting forks worker processes via joblib, which
            # crashes with SIGSEGV on macOS if a networked library (e.g. the
            # sentence transformer's Hugging Face Hub check, done earlier in this
            # same run) has already touched Network.framework. Not worth the
            # fork-safety risk for a 300-tree forest on this dataset size.
            n_jobs=1,
        )),
    ])

    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc", n_jobs=1)
    print(f"  CV ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}")

    clf.fit(X, y)

    feature_info = {
        "features":     features,
        "cat_cols":     available_cat,
        "num_cols":     available_num,
        "base_num_cols": [c for c in NUMERIC_FEATURES if c in df.columns],
        "categories": {
            col: sorted(df[col].dropna().astype(str).unique().tolist())
            for col in available_cat
        },
        "has_multimodal": True,
    }
    return clf, feature_info


SEVERITY_MAP = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
FAILURE_RATE_WEIGHT = 50
BUG_WEIGHT = 10
FAILURE_RATE_THRESHOLD = 0.10
SEVERITY_THRESHOLD = 10

CLUSTER_COLORS = {
    "Critical Hotspot":              "#d62728",
    "Nuisance Zone (High Fail, Low Sev)": "#ff7f0e",
    "Stable Yielder":                "#2ca02c",
    "Low ROI":                       "#aec7e8",
}

RISK_DIMS = [
    "App Component",
    "Parent App Component",
    "Platform Product Name",
    "Development Stage",
    "Testing Approach",
    "Bug Source Type",
    "Bug Type",
    "CAPDB Sub-Industry",
    "Platform Group",
]

CLUSTER_NAMES = {0: "Stable", 1: "Nuisance Zone", 2: "Critical Hazard"}
CLUSTER_COLORS = {
    "Critical Hazard": "#d62728",
    "Nuisance Zone":   "#ff7f0e",
    "Stable":          "#2ca02c",
}


def compute_bubble_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    QA Predictive Radar — Device-Level Architecture.

    Scoring:
      Severity_Index        = sum of per-bug severity weights (Critical=4, High=3, Med=2, Low=1)
      Predictive_Risk_Score = (Historical_Failure_Rate × 50) + (Severity_Index × 10)

    Clusters (rule-based fixed thresholds):
      Critical Hotspot              failure_rate > 10% AND severity_index > 10
      Nuisance Zone (High Fail, Low Sev)  failure_rate > 10% AND severity_index ≤ 10
      Low ROI                       failure_rate == 0 AND total_bugs == 0
      Stable Yielder                everything else
    """
    device_bugs = pd.read_excel(os.path.join(DATA_DIR, "devicebugs.xlsx"), engine="openpyxl")
    device_runs = pd.read_excel(os.path.join(DATA_DIR, "devicetestruns.xlsx"), engine="openpyxl")

    # Test Cycle Id -> Engagement Type, for tagging the run side the same way
    # load_data() already tagged df's bugs (Usability excluded; Accessibility
    # and Security bucketed separately, everything else as Functional).
    cycle_types = pd.read_excel(
        os.path.join(DATA_DIR, "testcycles.xlsx"), engine="openpyxl",
        usecols=["Test Cycle Id", "Test Cycle Testing Type"],
    )
    cycle_type_map = cycle_types.set_index("Test Cycle Id")["Test Cycle Testing Type"]

    # --- NODE A: bug severity per (Customer, Device, Engagement Type) ---
    bug_lookup = df[["Bug", "Bug Severity", "Customer", "Engagement Type"]].rename(columns={"Bug": "Bug Id"})
    master_bugs = device_bugs.merge(bug_lookup, on="Bug Id", how="left")
    master_bugs["Severity_Weight"] = master_bugs["Bug Severity"].map(SEVERITY_MAP).fillna(1.5)

    comp_col = "App Component Name"
    master_bugs[comp_col] = master_bugs[comp_col].fillna("Unknown").replace("-", "Unknown")

    def _top_value(s):
        vc = s.value_counts()
        return vc.index[0] if not vc.empty else "Unknown"

    bug_agg = (
        master_bugs.groupby(["Customer", "Device", "Engagement Type"])
        .agg(
            Total_Bugs=("Bug Id", "count"),
            Severity_Index=("Severity_Weight", "sum"),
            Primary_Failing_Component=(comp_col, _top_value),
        )
        .reset_index()
    )

    # --- NODE B: run failure rate per (Customer, Device, Engagement Type) ---
    device_runs["_raw_testing_type"] = device_runs["Test Cycle Id"].map(cycle_type_map)
    device_runs = device_runs[~device_runs["_raw_testing_type"].isin(EXCLUDED_TESTING_TYPES)].copy()
    device_runs["Engagement Type"] = (
        device_runs["_raw_testing_type"].map(ENGAGEMENT_TYPE_OVERRIDES).fillna("Functional")
    )
    device_runs = device_runs.drop(columns=["_raw_testing_type"])

    try:
        tc_results = read_excel_all_sheets(
            os.path.join(DATA_DIR, "testcaseresults.xlsx"),
            usecols=["Test Run Result Id", "Customer"],
        )
        master_runs = device_runs.merge(tc_results, on="Test Run Result Id", how="left")
    except Exception:
        master_runs = device_runs.copy()
        master_runs["Customer"] = np.nan

    run_group = ["Customer", "Device", "Engagement Type"] if "Customer" in master_runs.columns else ["Device", "Engagement Type"]
    run_agg = (
        master_runs.groupby(run_group)
        .agg(
            Total_Runs=("Result Status", "count"),
            Failed_Runs=("Result Status", lambda x: (x == "Failed").sum()),
        )
        .reset_index()
    )
    run_agg["Historical_Failure_Rate"] = (
        run_agg["Failed_Runs"] / run_agg["Total_Runs"].replace(0, 1)
    )

    # --- ASSEMBLE ---
    master_df = bug_agg.merge(run_agg, on=["Customer", "Device", "Engagement Type"], how="outer").fillna(0)
    master_df["Predictive_Risk_Score"] = (
        master_df["Historical_Failure_Rate"] * FAILURE_RATE_WEIGHT
        + master_df["Severity_Index"] * BUG_WEIGHT
    ).round(3)

    # --- CLUSTER (rule-based) ---
    def _cluster(row):
        if row["Historical_Failure_Rate"] > FAILURE_RATE_THRESHOLD and row["Severity_Index"] > SEVERITY_THRESHOLD:
            return "Critical Hotspot"
        elif row["Historical_Failure_Rate"] > FAILURE_RATE_THRESHOLD:
            return "Nuisance Zone (High Fail, Low Sev)"
        elif row["Historical_Failure_Rate"] == 0 and row["Total_Bugs"] == 0:
            return "Low ROI"
        return "Stable Yielder"

    master_df["Optimization_Cluster"] = master_df.apply(_cluster, axis=1)
    master_df["Cluster_Color"] = master_df["Optimization_Cluster"].map(CLUSTER_COLORS).fillna("#aec7e8")

    bubble = (
        master_df[master_df["Predictive_Risk_Score"] > 0]
        .sort_values("Predictive_Risk_Score", ascending=False)
        .reset_index(drop=True)
    )
    return bubble


def main():
    print("Loading data...")
    df = load_data()
    print(f"  {len(df):,} bugs  |  baseline H/C rate: {df[TARGET].mean():.1%}")

    print("Building text features (keyword flags, TF-IDF/SVD, embeddings)...")
    text_features, text_artifacts = build_text_features(df)
    df = pd.concat([df, text_features], axis=1)
    print(f"  {len(text_features.columns)} text feature columns added")

    print("Computing per-entity text profiles...")
    text_profiles = compute_text_profiles(df)
    print(f"  {len(text_profiles['by_component'])} component profiles  |  {len(text_profiles['by_platform'])} platform profiles")

    print("Building NMF association features...")
    nmf_features, nmf_artifacts = build_nmf_features(df)
    df = pd.concat([df, nmf_features], axis=1)
    print(f"  {N_NMF_FACTORS} NMF factors added")

    print("Building property graph features...")
    graph_features, graph_artifacts = build_graph_features(df)
    df = pd.concat([df, graph_features], axis=1)
    print(f"  {len(graph_artifacts['node_metrics'])} graph nodes  |  {len(graph_features.columns)} graph feature columns added")

    print("Computing risk tables...")
    risk_tables = compute_risk_tables(df)
    type_counts = df["Engagement Type"].value_counts()
    print(f"  {len(risk_tables['customers'])} customers  |  " +
          ", ".join(f"{count:,} {etype.lower()}-bucket bugs" for etype, count in type_counts.items()))

    print("Training classifier...")
    clf, feature_info = train_classifier(df)

    print("Computing bubble map data (QA Predictive Radar)...")
    bubble_data = compute_bubble_data(df)
    print(f"  {len(bubble_data):,} device bubbles  |  clusters: {bubble_data['Optimization_Cluster'].value_counts().to_dict()}")

    joblib.dump(clf,            os.path.join(ARTIFACTS_DIR, "classifier.joblib"))
    joblib.dump(risk_tables,    os.path.join(ARTIFACTS_DIR, "risk_tables.joblib"))
    joblib.dump(feature_info,   os.path.join(ARTIFACTS_DIR, "feature_info.joblib"))
    joblib.dump(text_artifacts, os.path.join(ARTIFACTS_DIR, "text_pipeline.joblib"))
    joblib.dump(text_profiles,  os.path.join(ARTIFACTS_DIR, "text_profiles.joblib"))
    joblib.dump(nmf_artifacts,  os.path.join(ARTIFACTS_DIR, "nmf_model.joblib"))
    joblib.dump(graph_artifacts, os.path.join(ARTIFACTS_DIR, "graph_artifacts.joblib"))
    joblib.dump(bubble_data,    os.path.join(ARTIFACTS_DIR, "bubble_data.joblib"))

    print(f"\nArtifacts saved to {ARTIFACTS_DIR}/")
    print("  Done. Run 'streamlit run app/Home.py' to launch the dashboard.")


if __name__ == "__main__":
    main()
