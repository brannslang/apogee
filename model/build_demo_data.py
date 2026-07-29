"""
Build the trimmed "demo-data" dataset from the full "full-data" dataset.

The shipped app is meant to showcase the tool with a manageable roster, not
the full 300+-organization dataset. This script selects the top
TOP_N_DEMO_CUSTOMERS customers by a combined bug-volume + bug-diversity
score, then filters all 8 raw tables down to just those customers' rows
(respecting the joins between tables), and writes the result to
data/demo-data/.

full-data itself is never modified. Run this, then `make train-demo`, to
refresh the artifacts the live app actually serves.

    python model/build_demo_data.py
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from config import (
    DEMO_DATA_DIR,
    DEMO_DIVERSITY_BASIS,
    DEMO_DIVERSITY_WEIGHT,
    DEMO_VOLUME_WEIGHT,
    EXPECTED_DATA_FILES,
    FULL_DATA_DIR,
    TOP_N_DEMO_CUSTOMERS,
)
from xlsx_io import read_excel_all_sheets, write_excel_chunked

# Files large enough to risk crossing Excel's single-sheet row limit — read
# and write these through the chunked helpers rather than plain read_excel.
CHUNKED_FILES = {"testcaseresults.xlsx"}


def _select_top_customers(bugs: pd.DataFrame) -> list:
    bug_volume = bugs.groupby("Customer").size().rename("bug_volume")

    basis = [c for c in DEMO_DIVERSITY_BASIS if c in bugs.columns]
    diversity = bugs.groupby("Customer")[basis].nunique().sum(axis=1).rename("diversity_score")

    scores = pd.concat([bug_volume, diversity], axis=1).fillna(0)

    def _normalize(s: pd.Series) -> pd.Series:
        span = s.max() - s.min()
        return (s - s.min()) / span if span else pd.Series(0.0, index=s.index)

    scores["combined_score"] = (
        DEMO_VOLUME_WEIGHT * _normalize(scores["bug_volume"])
        + DEMO_DIVERSITY_WEIGHT * _normalize(scores["diversity_score"])
    )
    scores.index.name = "Customer"

    # Tie-break by raw bug_volume desc, then Customer name asc, for determinism.
    scores = scores.reset_index().sort_values(
        ["combined_score", "bug_volume", "Customer"],
        ascending=[False, False, True],
    ).set_index("Customer")

    top = scores.head(TOP_N_DEMO_CUSTOMERS)
    print("Selected demo-data customers (combined_score desc):")
    print(top.to_string())
    return top.index.tolist()


def _filter_by_column(path: str, column: str, keep_values) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    before = len(df)
    if column in df.columns:
        df = df[df[column].isin(keep_values)].copy()
    print(f"{os.path.basename(path)}: {before} -> {len(df)} rows (filtered on {column!r})")
    return df


def _copy_or_filter_by_customer(filename: str, retained_customers) -> None:
    src = os.path.join(FULL_DATA_DIR, filename)
    dst = os.path.join(DEMO_DATA_DIR, filename)
    df = pd.read_excel(src, engine="openpyxl")
    if "Customer" in df.columns:
        before = len(df)
        df = df[df["Customer"].isin(retained_customers)].copy()
        print(f"{filename}: {before} -> {len(df)} rows (filtered on 'Customer')")
        df.to_excel(dst, engine="openpyxl", index=False)
    else:
        print(f"{filename}: no 'Customer' column found, copied unchanged")
        shutil.copyfile(src, dst)


def main() -> None:
    os.makedirs(DEMO_DATA_DIR, exist_ok=True)

    bugs = pd.read_excel(os.path.join(FULL_DATA_DIR, "bugdetails.xlsx"), engine="openpyxl")
    retained_customers = _select_top_customers(bugs)

    bugs_demo = bugs[bugs["Customer"].isin(retained_customers)].copy()
    print(f"bugdetails.xlsx: {len(bugs)} -> {len(bugs_demo)} rows (filtered on 'Customer')")
    bugs_demo.to_excel(os.path.join(DEMO_DATA_DIR, "bugdetails.xlsx"), engine="openpyxl", index=False)

    retained_cycle_ids = bugs_demo["Test Cycle Id"].dropna().unique()
    retained_bug_ids = bugs_demo["Bug"].dropna().unique()

    for filename, column, keep_values in [
        ("testcycles.xlsx", "Test Cycle Id", retained_cycle_ids),
        ("devicetestruns.xlsx", "Test Cycle Id", retained_cycle_ids),
        ("devicebugs.xlsx", "Bug Id", retained_bug_ids),
    ]:
        df = _filter_by_column(os.path.join(FULL_DATA_DIR, filename), column, keep_values)
        df.to_excel(os.path.join(DEMO_DATA_DIR, filename), engine="openpyxl", index=False)

    # testcaseresults.xlsx has its own Customer column and can exceed Excel's
    # single-sheet row limit in full-data, so it needs the chunked helpers.
    results = read_excel_all_sheets(os.path.join(FULL_DATA_DIR, "testcaseresults.xlsx"))
    before = len(results)
    results = results[results["Customer"].isin(retained_customers)].copy()
    print(f"testcaseresults.xlsx: {before} -> {len(results)} rows (filtered on 'Customer')")
    write_excel_chunked(results, os.path.join(DEMO_DATA_DIR, "testcaseresults.xlsx"))

    # Remaining files aren't read by any code path today; best-effort filter
    # by Customer where that column exists, otherwise copy unchanged.
    handled = {
        "bugdetails.xlsx", "testcycles.xlsx", "devicetestruns.xlsx",
        "devicebugs.xlsx", "testcaseresults.xlsx",
    }
    for filename in EXPECTED_DATA_FILES:
        if filename in handled:
            continue
        _copy_or_filter_by_customer(filename, retained_customers)

    print(f"\ndemo-data built: {len(retained_customers)} customers -> {DEMO_DATA_DIR}")


if __name__ == "__main__":
    main()
