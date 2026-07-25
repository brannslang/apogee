"""
Multi-sheet helpers for data/*.xlsx tables that may outgrow Excel's
per-sheet row limit (1,048,576, including the header row).

Currently only testcaseresults.xlsx crosses that limit, but any table can
in the future as more data gets appended -- read and write through these
so growth past one sheet doesn't silently truncate data.
"""
import pandas as pd

EXCEL_MAX_ROWS = 1_048_576 - 1  # minus the header row


def read_excel_all_sheets(path: str, **kwargs) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl", **kwargs)
    return pd.concat(sheets.values(), ignore_index=True)


def write_excel_chunked(df: pd.DataFrame, path: str) -> None:
    n_chunks = max(1, -(-len(df) // EXCEL_MAX_ROWS))  # ceil division
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for i in range(n_chunks):
            chunk = df.iloc[i * EXCEL_MAX_ROWS : (i + 1) * EXCEL_MAX_ROWS]
            chunk.to_excel(writer, sheet_name=f"Sheet{i + 1}", index=False)
