from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_pca_scatter(scores_df: pd.DataFrame, variance_df: pd.DataFrame, ranking_df: pd.DataFrame | None = None):
    df = scores_df.copy()
    if ranking_df is not None and "Sample" in df.columns:
        df = df.merge(ranking_df[["Sample", "Final_design_score", "Constraint_status"]], on="Sample", how="left")
    pc1 = variance_df.loc[variance_df["PC"] == "PC1", "Explained_variance_percent"].iloc[0]
    pc2 = variance_df.loc[variance_df["PC"] == "PC2", "Explained_variance_percent"].iloc[0]
    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        text="Sample" if "Sample" in df.columns else None,
        color="Final_design_score" if "Final_design_score" in df.columns else None,
        hover_data=df.columns,
        title="PCA score plot",
    )
    fig.update_traces(textposition="top center", marker=dict(size=11, line=dict(width=0.7, color="DarkSlateGrey")))
    fig.update_layout(xaxis_title=f"PC1 ({pc1:.2f}% variance)", yaxis_title=f"PC2 ({pc2:.2f}% variance)", height=650)
    fig.add_hline(y=0, line_width=1, line_dash="dash")
    fig.add_vline(x=0, line_width=1, line_dash="dash")
    return fig


def plot_biplot(scores_df: pd.DataFrame, loadings_df: pd.DataFrame, variance_df: pd.DataFrame):
    pc1 = variance_df.loc[variance_df["PC"] == "PC1", "Explained_variance_percent"].iloc[0]
    pc2 = variance_df.loc[variance_df["PC"] == "PC2", "Explained_variance_percent"].iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=scores_df["PC1"], y=scores_df["PC2"], mode="markers+text",
        text=scores_df["Sample"] if "Sample" in scores_df.columns else None,
        textposition="top center", name="Samples",
        marker=dict(size=10, line=dict(width=0.7, color="DarkSlateGrey")),
        hovertemplate="%{text}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>"
    ))
    x_span = scores_df["PC1"].max() - scores_df["PC1"].min()
    y_span = scores_df["PC2"].max() - scores_df["PC2"].min()
    scale = max(min(x_span, y_span) * 0.45, 1)
    for _, row in loadings_df.iterrows():
        x = row["PC1"] * scale
        y = row["PC2"] * scale
        fig.add_annotation(x=x, y=y, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1.4)
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="text", text=[row["Feature"]], name=row["Feature"], showlegend=False))
    fig.add_hline(y=0, line_width=1, line_dash="dash")
    fig.add_vline(x=0, line_width=1, line_dash="dash")
    fig.update_layout(title="PCA biplot", xaxis_title=f"PC1 ({pc1:.2f}% variance)", yaxis_title=f"PC2 ({pc2:.2f}% variance)", height=700)
    return fig


def plot_explained_variance(variance_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=variance_df["PC"],
        y=variance_df["Explained_variance_percent"],
        text=variance_df["Explained_variance_percent"],
        texttemplate="%{text:.1f}%",
        textposition="outside",
        name="Explained variance %",
    ))
    fig.add_trace(go.Scatter(
        x=variance_df["PC"],
        y=variance_df["Cumulative_percent"],
        mode="lines+markers",
        name="Cumulative %",
        hovertemplate="%{x}<br>Cumulative=%{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Explained variance",
        yaxis_title="Variance (%)",
        height=450,
        yaxis=dict(range=[0, max(105, float(variance_df["Cumulative_percent"].max()) + 5)]),
    )
    return fig


def plot_feature_contribution(contribution_df: pd.DataFrame):
    df = contribution_df.sort_values("Contribution_percent", ascending=True)
    fig = px.bar(df, x="Contribution_percent", y="Feature", orientation="h", text="Contribution_percent", title="Feature contribution to PC1/PC2")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(xaxis_title="Contribution (%)", yaxis_title="Feature", height=520)
    return fig


def plot_score_ranking(ranking_df: pd.DataFrame):
    top = ranking_df.sort_values("Final_design_score", ascending=True)
    fig = px.bar(top, x="Final_design_score", y="Sample", orientation="h", color="Constraint_status", title="Final design score ranking")
    fig.update_layout(xaxis_title="Final design score", yaxis_title="Sample", height=max(500, len(top) * 28))
    return fig


def plot_composition_fingerprint(df: pd.DataFrame, sample_name: str, feature_cols: list[str]):
    if sample_name not in set(df["Sample"]):
        sample_name = df["Sample"].iloc[0]
    row = df[df["Sample"] == sample_name].iloc[0]
    plot_df = pd.DataFrame({"Feature": feature_cols, "Value": [row.get(c, 0) for c in feature_cols]})
    fig = px.bar(plot_df, x="Feature", y="Value", title=f"Composition fingerprint: {sample_name}")
    fig.update_layout(yaxis_title="Composition (%)", height=450)
    return fig
