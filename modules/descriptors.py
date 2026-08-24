from __future__ import annotations

import numpy as np
import pandas as pd


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def calculate_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate chemistry-inspired descriptors for bioactive glass design.

    These are heuristic descriptors, not measured property predictions.
    """
    sample = df["Sample"] if "Sample" in df.columns else pd.Series([f"S{i+1}" for i in range(len(df))])
    si = _col(df, "SiO2")
    p = _col(df, "P2O5")
    ca = _col(df, "CaO")
    na = _col(df, "Na2O")
    k = _col(df, "K2O")
    mg = _col(df, "MgO")
    al = _col(df, "Al2O3")
    ti = _col(df, "TiO2")
    fe = _col(df, "FeO(T)")
    mn = _col(df, "MnO")

    formers = si + p
    modifiers = ca + na + k + mg
    intermediates = al + ti + fe + mn

    safe_p = p.replace(0, np.nan)
    safe_formers = formers.replace(0, np.nan)

    out = pd.DataFrame({
        "Sample": sample,
        "Network_formers_SiO2_P2O5": formers,
        "Network_modifiers_CaO_Na2O_K2O_MgO": modifiers,
        "Intermediates_Al2O3_TiO2_FeO_MnO": intermediates,
        "CaO_to_P2O5_ratio": (ca / safe_p).replace([np.inf, -np.inf], np.nan),
        "Modifier_to_Former_ratio": (modifiers / safe_formers).replace([np.inf, -np.inf], np.nan),
        "Silica_fraction": si,
        "Calcium_fraction": ca,
        "Phosphate_fraction": p,
        "Alkali_fraction_Na2O_K2O": na + k,
    })
    return out
