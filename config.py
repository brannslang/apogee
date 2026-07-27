import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# Two datasets live side by side: "full-data" is the complete, untrimmed
# dataset (used for full-scale training/research) and "demo-data" is a
# curated top-30-customer subset (see model/build_demo_data.py) used by the
# shipped app. DEFAULT_DATASET is "demo-data" so that anything launching the
# app directly — e.g. Streamlit Cloud, which never invokes the Makefile —
# shows the curated roster rather than silently serving the full 300+
# organizations. Training explicitly opts into "full-data" via `make train`.
DATASET_ENV_VAR = "APOGEE_DATASET"
VALID_DATASETS = ("full-data", "demo-data")
DEFAULT_DATASET = "demo-data"

ACTIVE_DATASET = os.environ.get(DATASET_ENV_VAR, DEFAULT_DATASET)
if ACTIVE_DATASET not in VALID_DATASETS:
    raise ValueError(f"{DATASET_ENV_VAR}={ACTIVE_DATASET!r} is not one of {VALID_DATASETS}")

FULL_DATA_DIR = os.path.join(ROOT, "data", "full-data")
DEMO_DATA_DIR = os.path.join(ROOT, "data", "demo-data")

DATA_DIR = os.path.join(ROOT, "data", ACTIVE_DATASET)
ARTIFACTS_DIR = os.path.join(ROOT, "model", "artifacts", ACTIVE_DATASET)

CATEGORICAL_FEATURES = [
    "App Component",
    "Parent App Component",
    "Platform Product Name",
    "Development Stage",
    "Bug Request Source",
    "Bug Source Type",
    "Testing Approach",
    "Engagement Type",
    "Industry",
]

NUMERIC_FEATURES = [
    "Bug Rate Amount",
    "Test Cycle Duration Activation to Lock/Close/Today",
]

TARGET = "is_high_crit"
MIN_BUGS_FOR_TABLE = 10

# --- Demo dataset selection (model/build_demo_data.py) ---
# Top-N customers, ranked by a combined bug-volume + bug-diversity score, are
# carried from full-data into demo-data. Weights and basis are plain config
# constants so the selection can be retuned without touching code.
TOP_N_DEMO_CUSTOMERS = 30
DEMO_VOLUME_WEIGHT = 0.5
DEMO_DIVERSITY_WEIGHT = 0.5
DEMO_DIVERSITY_BASIS = CATEGORICAL_FEATURES + ["Bug Severity", "Bug Type"]

# Test Cycle Testing Type values (from testcycles.xlsx) that should be dropped
# entirely from training/analysis. Usability/UX engagements deliver interviews,
# surveys, and qualitative feedback rather than trainable bug data.
EXCLUDED_TESTING_TYPES = ["Usability"]

# Testing Types that get their own separate analysis lens everywhere (Risk
# Dashboard, Release Predictor, Monthly Digest, Risk Map) instead of being
# folded into Functional. Accessibility and Security are each run by their
# own dedicated team, distinct from Managed Testing Services — Security
# in particular is never touched by the functional team, so lumping it into
# "Functional" would misattribute another team's bugs to that risk profile.
#
# Localization and Payment-Transaction-compensated bugs (tagged
# Functional/"Payment Testing") DO fall under Managed Testing
# Services' purview, so they're intentionally left bucketed as "Functional".
ACCESSIBILITY_TESTING_TYPE = "Accessibility"
SECURITY_TESTING_TYPE = "Security"
ENGAGEMENT_TYPE_OVERRIDES = {
    ACCESSIBILITY_TESTING_TYPE: ACCESSIBILITY_TESTING_TYPE,
    SECURITY_TESTING_TYPE: SECURITY_TESTING_TYPE,
}
ENGAGEMENT_TYPES = ["Functional", "Accessibility", "Security"]

# Standardized filenames the training pipeline and the data uploader both
# recognize. Uploading a subset is fine — only the matching files get
# replaced and everything else in DATA_DIR is left as-is.
EXPECTED_DATA_FILES = [
    "bugdetails.xlsx",
    "testcycles.xlsx",
    "devicebugs.xlsx",
    "devicetestruns.xlsx",
    "testcaseresults.xlsx",
    "testcasedetails.xlsx",
    "testcaseentitlements.xlsx",
    "entitlementdetails.xlsx",
]

# --- Multi-modal enrichment constants ---

N_SVD_COMPONENTS = 25
N_EMB_COMPONENTS = 20
N_NMF_FACTORS = 15

KEYWORD_GROUPS = {
    "text_flag_crash": r"\b(?:crash|freeze|hang|unresponsive|force.?close)\b",
    "text_flag_data_integrity": r"\b(?:data.?loss|incorrect|missing|wrong|corrupt(?:ed)?)\b",
    "text_flag_error": r"\b(?:error|exception|null|undefined|failed.?to.?load)\b",
    "text_flag_security": r"\b(?:security|unauthorized|unauthorised|exposed|bypass)\b",
    "text_flag_visibility": r"\b(?:blank|white.?screen|not.?loading|broken)\b",
    "text_flag_performance": r"\b(?:slow|timeout|time.?out|performance|lag|latency)\b",
    "text_flag_access": r"\b(?:login|auth(?:entication)?|permission|access.?denied|session)\b",
}

TEXT_FLAG_FEATURES = list(KEYWORD_GROUPS.keys())
TEXT_SVD_FEATURES = [f"text_svd_{i}" for i in range(N_SVD_COMPONENTS)]
TEXT_EMB_FEATURES = [f"text_emb_{i}" for i in range(N_EMB_COMPONENTS)]
NMF_FEATURES = [f"nmf_factor_{i}" for i in range(N_NMF_FACTORS)]

GRAPH_FEATURES = [
    "graph_comp_pagerank",
    "graph_comp_degree",
    "graph_comp_clustering",
    "graph_platform_pagerank",
    "graph_customer_pagerank",
]

NMF_ENTITY_COLS = [
    "App Component",
    "Platform Product Name",
    "Development Stage",
    "Testing Approach",
    "Bug Source Type",
    "Customer",
]
