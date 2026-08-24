from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from modules.preprocessing import (
    DEFAULT_OXIDES,
    clean_composition_table,
    get_numeric_features,
    normalize_rows_to_100,
    recompute_total,
)
from modules.descriptors import calculate_descriptors
from modules.pca_analysis import run_pca
from modules.scoring import DEFAULT_CONSTRAINTS, check_constraints, score_designs, generate_candidates
from modules.plotting import (
    plot_pca_scatter,
    plot_biplot,
    plot_explained_variance,
    plot_feature_contribution,
    plot_score_ranking,
    plot_composition_fingerprint,
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

st.set_page_config(
    page_title="Bioactive Glass AI Designer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_default_data(dataset_kind: str) -> pd.DataFrame:
    file_name = "default_wt_percent.csv" if dataset_kind == "wt%" else "default_mol_percent.csv"
    return clean_composition_table(pd.read_csv(DATA_DIR / file_name))


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return clean_composition_table(pd.read_csv(uploaded_file))
    if name.endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(uploaded_file, sheet_name=None)
        # Use the first sheet that looks like a composition table.
        for _, sdf in sheets.items():
            cols = [str(c) for c in sdf.columns]
            if any("SiO2" in c for c in cols) or len(sdf.columns) >= 5:
                return clean_composition_table(sdf)
        first = next(iter(sheets.values()))
        return clean_composition_table(first)
    raise ValueError("Upload a CSV or Excel file.")


def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def make_text_report(clean_df, variance_df, ranking_df, issues_df, used_features) -> str:
    pc1 = variance_df.loc[variance_df["PC"] == "PC1", "Explained_variance_percent"].iloc[0]
    pc2 = variance_df.loc[variance_df["PC"] == "PC2", "Explained_variance_percent"].iloc[0]
    top_rows = ranking_df.head(5)[["Rank", "Sample", "Final_design_score", "Bioactivity_potential_score", "Network_stability_score", "Constraint_status"]]
    report = []
    report.append("Bioactive Glass AI Designer - PCA and Design Screening Report")
    report.append("=" * 68)
    report.append("")
    report.append(f"Samples analyzed: {len(clean_df)}")
    report.append(f"Features used for PCA: {', '.join(used_features)}")
    report.append(f"Explained variance: PC1={pc1:.2f}%, PC2={pc2:.2f}%, PC1+PC2={pc1+pc2:.2f}%")
    report.append(f"Constraint issues detected: {len(issues_df)}")
    report.append("")
    report.append("Top candidate ranking:")
    report.append(top_rows.to_string(index=False))
    report.append("")
    report.append("Important note:")
    report.append("The current scoring system is AI-assisted and rule-based. It is useful for exploratory design and ranking, but it is not a trained performance prediction model until measured glass-characterization targets are added.")
    return "\n".join(report)


def make_excel_bytes(tables: dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet_name, table in tables.items():
            safe_name = sheet_name[:31]
            table.to_excel(writer, sheet_name=safe_name, index=False)
            worksheet = writer.sheets[safe_name]
            for idx, col in enumerate(table.columns):
                max_len = max([len(str(col))] + [len(str(x)) for x in table[col].head(50).values])
                worksheet.set_column(idx, idx, min(max(max_len + 2, 10), 28))
    return output.getvalue()


# ----------------------- UI -----------------------
st.title("🧪 Bioactive Glass AI Designer")
st.caption("Python Streamlit research dashboard for PCA, chemical descriptors, constraints, and AI-assisted candidate ranking.")

with st.sidebar:
    st.header("1) Dataset")
    dataset_kind = st.radio("Default dataset", ["wt%", "mol%"], horizontal=True, help="Use the wt% or mol% composition extracted from your MvsG Excel file.")
    uploaded_file = st.file_uploader("Optional: upload new CSV/Excel", type=["csv", "xlsx", "xls"])

    st.header("2) Processing")
    auto_normalize = st.checkbox("Normalize rows to 100 before PCA", value=True)
    show_reference = st.checkbox("Keep pure SiO2 reference sample", value=True)

    st.header("3) Candidate generator")
    n_candidates = st.slider("Number of generated candidates", min_value=5, max_value=100, value=20, step=5)
    seed = st.number_input("Random seed", min_value=1, max_value=99999, value=42, step=1)

if uploaded_file is not None:
    try:
        base_df = load_uploaded_file(uploaded_file)
        st.sidebar.success(f"Loaded uploaded file: {uploaded_file.name}")
    except Exception as exc:
        st.sidebar.error(f"Could not read uploaded file: {exc}")
        base_df = load_default_data(dataset_kind)
else:
    base_df = load_default_data(dataset_kind)

if not show_reference and "Sample" in base_df.columns:
    base_df = base_df[base_df["Sample"].astype(str).str.upper() != "SIO2"].reset_index(drop=True)

base_features = get_numeric_features(base_df)
default_features = [c for c in DEFAULT_OXIDES if c in base_features]
if not default_features:
    default_features = base_features

# Editable data lives in session state, reset when dataset source changes.
source_key = f"{dataset_kind}_{uploaded_file.name if uploaded_file is not None else 'default'}_{show_reference}"
if st.session_state.get("source_key") != source_key:
    st.session_state["source_key"] = source_key
    st.session_state["composition_df"] = base_df.copy()
    st.session_state["constraints_df"] = DEFAULT_CONSTRAINTS[DEFAULT_CONSTRAINTS["Feature"].isin(default_features)].copy()

st.subheader("Quick status")
metric_cols = st.columns(5)
metric_cols[0].metric("Samples", len(st.session_state["composition_df"]))
metric_cols[1].metric("Numeric features", len(base_features))
metric_cols[2].metric("Dataset", dataset_kind)
metric_cols[3].metric("Mode", "Normalized" if auto_normalize else "Raw totals")
metric_cols[4].metric("App type", "Python Streamlit")

# Main tabs
tab_data, tab_pca, tab_scores, tab_constraints, tab_candidates, tab_export, tab_about = st.tabs([
    "1. Data editor",
    "2. PCA graphs",
    "3. Scores & descriptors",
    "4. Constraints",
    "5. Candidate generator",
    "6. Export",
    "7. Method note",
])

with tab_data:
    st.markdown("### Editable composition table")
    st.info("Change any value in the table. Streamlit reruns the app automatically, so PCA graphs and scores update after the edit.")

    edited_df = st.data_editor(
        st.session_state["composition_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="main_data_editor",
    )
    edited_df = clean_composition_table(edited_df)
    all_features = get_numeric_features(edited_df)

    selected_features = st.multiselect(
        "Select oxide/features for PCA and normalization",
        options=all_features,
        default=[c for c in default_features if c in all_features] or all_features,
    )

    col_a, col_b, col_c = st.columns(3)
    if col_a.button("Normalize displayed data to 100 now"):
        st.session_state["composition_df"] = normalize_rows_to_100(edited_df, selected_features)
        st.rerun()
    if col_b.button("Recalculate Total only"):
        st.session_state["composition_df"] = recompute_total(edited_df, selected_features)
        st.rerun()
    if col_c.button("Reset to default dataset"):
        st.session_state["composition_df"] = base_df.copy()
        st.rerun()

    work_df = edited_df.copy()
    if auto_normalize:
        work_df = normalize_rows_to_100(work_df, selected_features)
    else:
        work_df = recompute_total(work_df, selected_features)

    st.markdown("### Data used for calculations")
    st.dataframe(work_df, use_container_width=True)

# Build computation tables once after data editor.
edited_df_global = clean_composition_table(st.session_state.get("composition_df", base_df))
feature_options_global = get_numeric_features(edited_df_global)
# If selected_features was created in tab scope, use it. Otherwise use defaults.
try:
    selected_features_global = selected_features
except NameError:
    selected_features_global = [c for c in default_features if c in feature_options_global] or feature_options_global

if auto_normalize:
    calc_df = normalize_rows_to_100(edited_df_global, selected_features_global)
else:
    calc_df = recompute_total(edited_df_global, selected_features_global)

descriptors_df = calculate_descriptors(calc_df)

with tab_constraints:
    st.markdown("### Design constraints")
    st.write("Set min/max allowed values for each oxide. These limits control warnings and candidate ranking.")
    constraints_in = st.data_editor(
        st.session_state["constraints_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="constraints_editor",
    )
    constraints_in = constraints_in.dropna(subset=["Feature"]).copy()
    constraints_in["Min"] = pd.to_numeric(constraints_in["Min"], errors="coerce").fillna(0.0)
    constraints_in["Max"] = pd.to_numeric(constraints_in["Max"], errors="coerce").fillna(100.0)
    constraints_in.loc[constraints_in["Max"] < constraints_in["Min"], "Max"] = constraints_in["Min"]
    st.session_state["constraints_df"] = constraints_in

    constraint_scores_df, issues_df = check_constraints(calc_df, constraints_in)
    st.metric("Constraint issues", len(issues_df))
    if len(issues_df):
        st.warning("Some samples are outside your chosen design limits. This is a warning, not a software error.")
        st.dataframe(issues_df, use_container_width=True)
    else:
        st.success("All selected samples pass the current constraints.")

# Run PCA and scoring after constraints are known.
try:
    pca_result = run_pca(calc_df, selected_features_global, n_components=4)
    pca_scores_df = pca_result["scores"]
    loadings_df = pca_result["loadings"]
    variance_df = pca_result["variance"]
    contribution_df = pca_result["contribution"]
    used_features = pca_result["used_features"]
    zero_var_features = pca_result["zero_variance_features"]
    pca_ok = True
    pca_error = ""
except Exception as exc:
    pca_scores_df = pd.DataFrame()
    loadings_df = pd.DataFrame()
    variance_df = pd.DataFrame()
    contribution_df = pd.DataFrame()
    used_features = []
    zero_var_features = []
    pca_ok = False
    pca_error = str(exc)

if pca_ok:
    ranking_df = score_designs(calc_df, descriptors_df, st.session_state["constraints_df"], pca_scores_df)
else:
    ranking_df = pd.DataFrame()

with tab_pca:
    st.markdown("### PCA graphs")
    if not pca_ok:
        st.error(pca_error)
    else:
        if zero_var_features:
            st.warning("Removed zero-variance features from PCA: " + ", ".join(zero_var_features))
        pc1 = variance_df.loc[variance_df["PC"] == "PC1", "Explained_variance_percent"].iloc[0]
        pc2 = variance_df.loc[variance_df["PC"] == "PC2", "Explained_variance_percent"].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("PC1 variance", f"{pc1:.2f}%")
        c2.metric("PC2 variance", f"{pc2:.2f}%")
        c3.metric("PC1 + PC2", f"{pc1 + pc2:.2f}%")

        st.plotly_chart(plot_pca_scatter(pca_scores_df, variance_df, ranking_df), use_container_width=True)
        st.plotly_chart(plot_biplot(pca_scores_df, loadings_df, variance_df), use_container_width=True)
        chart_cols = st.columns(2)
        chart_cols[0].plotly_chart(plot_explained_variance(variance_df), use_container_width=True)
        chart_cols[1].plotly_chart(plot_feature_contribution(contribution_df), use_container_width=True)

        st.markdown("### PCA tables")
        t1, t2, t3 = st.tabs(["Scores", "Loadings", "Explained variance"])
        with t1:
            st.dataframe(pca_scores_df, use_container_width=True)
        with t2:
            st.dataframe(loadings_df, use_container_width=True)
        with t3:
            st.dataframe(variance_df, use_container_width=True)

with tab_scores:
    st.markdown("### Chemical descriptors and design ranking")
    st.caption("These scores are heuristic and transparent. They support candidate screening; they are not trained ML predictions yet.")
    if pca_ok:
        st.plotly_chart(plot_score_ranking(ranking_df), use_container_width=True)
        st.dataframe(ranking_df, use_container_width=True)
    else:
        st.error("PCA failed, so PCA-space scoring is unavailable.")
    st.markdown("### Descriptors")
    st.dataframe(descriptors_df, use_container_width=True)

    if "Sample" in calc_df.columns and len(calc_df):
        sample_for_chart = st.selectbox("Select sample for composition fingerprint", calc_df["Sample"].astype(str).tolist())
        st.plotly_chart(plot_composition_fingerprint(calc_df, sample_for_chart, selected_features_global), use_container_width=True)

with tab_candidates:
    st.markdown("### Generate new candidate compositions")
    st.write("The generator samples inside your current min/max constraints, normalizes each candidate to 100, then scores and ranks them.")
    if st.button("Generate candidate compositions"):
        candidates_df = generate_candidates(st.session_state["constraints_df"], n=int(n_candidates), seed=int(seed))
        candidate_features = [c for c in candidates_df.columns if c not in ["Sample", "Total"]]
        candidate_desc = calculate_descriptors(candidates_df)
        try:
            candidate_pca = run_pca(pd.concat([calc_df[used_features + ["Sample", "Total"]] if set(["Sample", "Total"]).issubset(calc_df.columns) else calc_df, candidates_df], ignore_index=True, sort=False).fillna(0), used_features, n_components=4)
            combined_scores = candidate_pca["scores"]
            candidate_scores_only = combined_scores[combined_scores["Sample"].astype(str).str.startswith("Candidate_")].reset_index(drop=True)
        except Exception:
            candidate_scores_only = None
        candidate_rank = score_designs(candidates_df, candidate_desc, st.session_state["constraints_df"], candidate_scores_only)
        st.session_state["candidate_df"] = candidates_df
        st.session_state["candidate_rank"] = candidate_rank

    if "candidate_df" in st.session_state:
        st.markdown("#### Generated candidates")
        st.dataframe(st.session_state["candidate_df"], use_container_width=True)
        st.markdown("#### Candidate ranking")
        st.dataframe(st.session_state["candidate_rank"], use_container_width=True)
        st.download_button("Download generated candidates CSV", to_csv_download(st.session_state["candidate_df"]), "generated_candidates.csv", "text/csv")
        st.download_button("Download candidate ranking CSV", to_csv_download(st.session_state["candidate_rank"]), "candidate_ranking.csv", "text/csv")

with tab_export:
    st.markdown("### Export results")
    if pca_ok:
        constraint_scores_df, issues_df = check_constraints(calc_df, st.session_state["constraints_df"])
        tables = {
            "composition_used": calc_df,
            "descriptors": descriptors_df,
            "pca_scores": pca_scores_df,
            "pca_loadings": loadings_df,
            "explained_variance": variance_df,
            "feature_contribution": contribution_df,
            "design_ranking": ranking_df,
            "constraint_issues": issues_df,
            "constraints": st.session_state["constraints_df"],
        }
        report_text = make_text_report(calc_df, variance_df, ranking_df, issues_df, used_features)
        st.download_button("Download edited composition CSV", to_csv_download(calc_df), "edited_composition_used.csv", "text/csv")
        st.download_button("Download PCA scores CSV", to_csv_download(pca_scores_df), "pca_scores.csv", "text/csv")
        st.download_button("Download PCA loadings CSV", to_csv_download(loadings_df), "pca_loadings.csv", "text/csv")
        st.download_button("Download design ranking CSV", to_csv_download(ranking_df), "design_ranking.csv", "text/csv")
        st.download_button("Download full Excel results", make_excel_bytes(tables), "bioactive_glass_ai_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download text report", report_text.encode("utf-8"), "bioactive_glass_ai_report.txt", "text/plain")
        st.text_area("Report preview", report_text, height=280)
    else:
        st.error("Fix PCA input first, then exports will be available.")

with tab_about:
    st.markdown(
        """
        ### What this system does

        This is a **Python Streamlit research dashboard** for bioactive glass composition design. It combines:

        - editable chemical composition data
        - row normalization to 100%
        - PCA score plot, biplot, explained variance and loadings
        - chemical descriptors such as network formers, modifiers and CaO/P2O5 ratio
        - constraint checking
        - rule-based AI-assisted design ranking
        - candidate composition generation

        ### Important scientific limitation

        The current design score is **not a trained ML prediction model**. It is a transparent, chemistry-inspired screening score.
        For real ML prediction, add measured target columns such as bioactivity, apatite formation, dissolution rate, pH change,
        hardness, density, antibacterial activity or compressive strength.
        """
    )
