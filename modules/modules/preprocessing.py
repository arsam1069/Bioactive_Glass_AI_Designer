from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_OXIDES = [
    "SiO2", "TiO2", "Al2O3", "FeO(T)", "MnO", "MgO", "CaO", "Na2O", "K2O", "P2O5"
]


def clean_composition_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean composition table with a Sample column and numeric oxide columns."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "Sample" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "Sample"})

    df["Sample"] = df["Sample"].astype(str).str.strip()
    df = df[df["Sample"].notna() & (df["Sample"].str.lower() != "nan")]

    for col in df.columns:
        if col != "Sample":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep known oxide columns when present. Unknown numeric columns are retained separately only if user selects them.
    if "Total" not in df.columns:
        numeric_cols = [c for c in df.columns if c != "Sample" and pd.api.types.is_numeric_dtype(df[c])]
        df["Total"] = df[numeric_cols].sum(axis=1)

    return df.reset_index(drop=True)


def get_numeric_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "Sample" and c != "Total" and pd.api.types.is_numeric_dtype(df[c])]


def normalize_rows_to_100(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Normalize selected composition features in each row to sum to 100."""
    out = df.copy()
    if not feature_cols:
        return out
    values = out[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    row_sum = values.sum(axis=1).replace(0, np.nan)
    normalized = values.div(row_sum, axis=0) * 100.0
    out[feature_cols] = normalized.fillna(0.0)
    out["Total"] = out[feature_cols].sum(axis=1)
    return out


def recompute_total(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    if feature_cols:
        out["Total"] = out[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    return out
