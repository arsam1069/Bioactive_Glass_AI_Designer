from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca(df: pd.DataFrame, feature_cols: list[str], n_components: int = 2) -> dict:
    """Run standardized PCA and return scores, loadings, variance and fitted objects."""
    if len(feature_cols) < 2:
        raise ValueError("Select at least two numeric feature columns for PCA.")
    if len(df) < 2:
        raise ValueError("PCA needs at least two samples.")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    zero_var = [c for c in feature_cols if X[c].nunique(dropna=False) <= 1]
    used_features = [c for c in feature_cols if c not in zero_var]
    if len(used_features) < 2:
        raise ValueError("After removing zero-variance features, less than two features remain.")

    X = X[used_features]
    n_components = max(2, min(n_components, len(used_features), len(df)))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X_scaled)

    score_cols = [f"PC{i+1}" for i in range(n_components)]
    scores_df = pd.DataFrame(scores, columns=score_cols)
    if "Sample" in df.columns:
        scores_df.insert(0, "Sample", df["Sample"].values)

    loadings_df = pd.DataFrame(pca.components_.T, index=used_features, columns=score_cols).reset_index()
    loadings_df = loadings_df.rename(columns={"index": "Feature"})

    variance_df = pd.DataFrame({
        "PC": score_cols,
        "Explained_variance_ratio": pca.explained_variance_ratio_,
        "Explained_variance_percent": pca.explained_variance_ratio_ * 100,
        "Cumulative_percent": np.cumsum(pca.explained_variance_ratio_) * 100,
    })

    # Contribution per feature across PC1/PC2, normalized to 100.
    pc_cols = [c for c in ["PC1", "PC2"] if c in loadings_df.columns]
    contribution = loadings_df.set_index("Feature")[pc_cols].pow(2).sum(axis=1)
    contribution_df = (contribution / contribution.sum() * 100).reset_index()
    contribution_df.columns = ["Feature", "Contribution_percent"]
    contribution_df = contribution_df.sort_values("Contribution_percent", ascending=False)

    return {
        "scores": scores_df,
        "loadings": loadings_df,
        "variance": variance_df,
        "contribution": contribution_df,
        "model": pca,
        "scaler": scaler,
        "used_features": used_features,
        "zero_variance_features": zero_var,
        "scaled_matrix": X_scaled,
    }
