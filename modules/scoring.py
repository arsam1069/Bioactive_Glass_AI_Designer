from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_CONSTRAINTS = pd.DataFrame([
    {"Feature": "SiO2", "Min": 35.0, "Max": 80.0},
    {"Feature": "TiO2", "Min": 0.0, "Max": 10.0},
    {"Feature": "Al2O3", "Min": 0.0, "Max": 20.0},
    {"Feature": "FeO(T)", "Min": 0.0, "Max": 10.0},
    {"Feature": "MnO", "Min": 0.0, "Max": 5.0},
    {"Feature": "MgO", "Min": 0.0, "Max": 20.0},
    {"Feature": "CaO", "Min": 0.0, "Max": 40.0},
    {"Feature": "Na2O", "Min": 0.0, "Max": 30.0},
    {"Feature": "K2O", "Min": 0.0, "Max": 15.0},
    {"Feature": "P2O5", "Min": 0.0, "Max": 15.0},
])


def range_score(x: pd.Series, low: float, high: float, ideal_low: float | None = None, ideal_high: float | None = None) -> pd.Series:
    """Score 0-100. Full score inside ideal range; linearly decays outside allowed range."""
    x = pd.to_numeric(x, errors="coerce").fillna(0.0)
    ideal_low = low if ideal_low is None else ideal_low
    ideal_high = high if ideal_high is None else ideal_high
    score = pd.Series(100.0, index=x.index)
    below = x < ideal_low
    above = x > ideal_high
    if ideal_low > low:
        score[below] = 100 * (x[below] - low) / (ideal_low - low)
    else:
        score[below] = 0
    if high > ideal_high:
        score[above] = 100 * (high - x[above]) / (high - ideal_high)
    else:
        score[above] = 0
    return score.clip(0, 100)


def check_constraints(df: pd.DataFrame, constraints: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-sample constraint score and detailed issue table."""
    issues = []
    scores = []
    sample_col = df["Sample"] if "Sample" in df.columns else pd.Series([f"S{i+1}" for i in range(len(df))])

    for idx, sample in sample_col.items():
        row_issues = 0
        checked = 0
        for _, c in constraints.iterrows():
            feat = str(c["Feature"])
            if feat not in df.columns:
                continue
            checked += 1
            val = pd.to_numeric(pd.Series([df.loc[idx, feat]]), errors="coerce").iloc[0]
            min_v = float(c["Min"])
            max_v = float(c["Max"])
            if pd.isna(val):
                row_issues += 1
                issues.append({"Sample": sample, "Feature": feat, "Value": np.nan, "Issue": "Missing value", "Allowed": f"{min_v:g} to {max_v:g}"})
            elif val < min_v:
                row_issues += 1
                issues.append({"Sample": sample, "Feature": feat, "Value": val, "Issue": "Below minimum", "Allowed": f"{min_v:g} to {max_v:g}"})
            elif val > max_v:
                row_issues += 1
                issues.append({"Sample": sample, "Feature": feat, "Value": val, "Issue": "Above maximum", "Allowed": f"{min_v:g} to {max_v:g}"})
        score = 100.0 if checked == 0 else max(0.0, 100.0 - (row_issues / checked) * 100.0)
        scores.append({"Sample": sample, "Constraint_score": score, "Constraint_issues": row_issues, "Constraint_status": "Pass" if row_issues == 0 else "Review"})

    return pd.DataFrame(scores), pd.DataFrame(issues)


def score_designs(df: pd.DataFrame, descriptors: pd.DataFrame, constraints: pd.DataFrame, pca_scores: pd.DataFrame | None = None) -> pd.DataFrame:
    sample = df["Sample"] if "Sample" in df.columns else pd.Series([f"S{i+1}" for i in range(len(df))])
    tmp = descriptors.copy()
    tmp["Sample"] = sample.values

    si = pd.to_numeric(df.get("SiO2", 0), errors="coerce").fillna(0.0) if "SiO2" in df.columns else pd.Series(0.0, index=df.index)
    ca = pd.to_numeric(df.get("CaO", 0), errors="coerce").fillna(0.0) if "CaO" in df.columns else pd.Series(0.0, index=df.index)
    p = pd.to_numeric(df.get("P2O5", 0), errors="coerce").fillna(0.0) if "P2O5" in df.columns else pd.Series(0.0, index=df.index)
    al = pd.to_numeric(df.get("Al2O3", 0), errors="coerce").fillna(0.0) if "Al2O3" in df.columns else pd.Series(0.0, index=df.index)

    modifier_former = pd.to_numeric(tmp.get("Modifier_to_Former_ratio", 0), errors="coerce").fillna(0.0)
    ca_p = pd.to_numeric(tmp.get("CaO_to_P2O5_ratio", 0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    formers = pd.to_numeric(tmp.get("Network_formers_SiO2_P2O5", 0), errors="coerce").fillna(0.0)
    modifiers = pd.to_numeric(tmp.get("Network_modifiers_CaO_Na2O_K2O_MgO", 0), errors="coerce").fillna(0.0)

    # Heuristic, transparent scoring. This is NOT a trained property predictor.
    bioactivity = (
        0.30 * range_score(ca, 0, 45, 10, 35)
        + 0.25 * range_score(p, 0, 15, 2, 10)
        + 0.20 * range_score(modifier_former, 0.0, 1.6, 0.35, 1.2)
        + 0.15 * range_score(ca_p, 0, 16, 3, 12)
        + 0.10 * range_score(si, 25, 80, 35, 65)
    )

    stability = (
        0.40 * range_score(formers, 30, 85, 45, 75)
        + 0.25 * range_score(si, 25, 80, 40, 70)
        + 0.20 * range_score(modifiers, 0, 70, 5, 50)
        + 0.15 * range_score(al, 0, 25, 0, 12)
    )

    constraint_scores, issue_table = check_constraints(df, constraints)

    if pca_scores is not None and {"PC1", "PC2"}.issubset(pca_scores.columns):
        dist = np.sqrt((pca_scores["PC1"] - pca_scores["PC1"].mean()) ** 2 + (pca_scores["PC2"] - pca_scores["PC2"].mean()) ** 2)
        ref = np.nanpercentile(dist, 75) if len(dist) > 2 else np.nanmax(dist)
        ref = ref if ref and ref > 0 else 1.0
        pca_space_score = pd.Series(100 - (dist / (ref * 1.8)) * 100).clip(0, 100).reset_index(drop=True)
    else:
        pca_space_score = pd.Series(75.0, index=df.index)

    ranked = pd.DataFrame({
        "Sample": sample.values,
        "Bioactivity_potential_score": bioactivity.round(2),
        "Network_stability_score": stability.round(2),
        "Constraint_score": constraint_scores["Constraint_score"].values.round(2),
        "PCA_space_score": pca_space_score.round(2),
    })
    ranked["Final_design_score"] = (
        0.35 * ranked["Bioactivity_potential_score"]
        + 0.25 * ranked["Network_stability_score"]
        + 0.25 * ranked["Constraint_score"]
        + 0.15 * ranked["PCA_space_score"]
    ).round(2)
    ranked = ranked.merge(constraint_scores[["Sample", "Constraint_status", "Constraint_issues"]], on="Sample", how="left")
    ranked = ranked.sort_values("Final_design_score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "Rank", np.arange(1, len(ranked) + 1))
    return ranked


def generate_candidates(constraints: pd.DataFrame, n: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    features = constraints["Feature"].astype(str).tolist()
    mins = constraints["Min"].astype(float).values
    maxs = constraints["Max"].astype(float).values
    for i in range(n):
        vals = rng.uniform(mins, maxs)
        total = vals.sum()
        if total <= 0:
            vals = np.zeros_like(vals)
        else:
            vals = vals / total * 100.0
        row = {"Sample": f"Candidate_{i+1:02d}"}
        row.update({feat: float(val) for feat, val in zip(features, vals)})
        row["Total"] = float(sum(vals))
        rows.append(row)
    return pd.DataFrame(rows)
