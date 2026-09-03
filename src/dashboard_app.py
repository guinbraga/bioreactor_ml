import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import streamlit as st

# Set Streamlit page config
st.set_page_config(
    page_title="Bioreactor Regression Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Base directories
BASE_DIR = Path("results/40_batch_regression")
COMPILED_CSV_PATH = Path("results/compiled_regression_metrics.csv")

# Custom CSS for premium aesthetics
st.markdown(
    """
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Title styling */
    .title-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .title-container h1 {
        margin: 0;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .title-container p {
        margin: 5px 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.25rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1rem;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.06);
        border-color: #2a5298;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #666666;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.8rem;
        color: #1e3c72;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .metric-subtitle {
        font-size: 0.8rem;
        color: #888888;
    }
    
    /* Section headers */
    .section-header {
        font-family: 'Inter', sans-serif;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        color: #1e3c72;
        font-weight: 600;
    }
    
    /* Tab formatting */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-bottom: 2px solid transparent;
        color: #555555;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #1e3c72 !important;
        border-bottom-color: #1e3c72 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Function to compile metrics dynamically
def compile_all_results():
    if not BASE_DIR.exists():
        return pd.DataFrame()

    all_runs = []
    csv_files = list(BASE_DIR.glob("**/*_predictions_summary.csv"))

    for csv_path in csv_files:
        rel_path = csv_path.relative_to(BASE_DIR)
        parts = rel_path.parts

        # Structure parsing:
        # Case 1 (all features): all_features/{target}/{regressor}/{filename}
        # Case 2 (feature selection): {experiment}/{classification_model}/{target}/{regressor}/{filename}
        if parts[0] == "all_features":
            experiment = "all_features"
            feature_selector = "all_features"
            target = parts[1]
            regressor = parts[2]
        else:
            experiment = parts[0]
            feature_selector = parts[1]
            target = parts[2]
            regressor = parts[3]

        try:
            df = pd.read_csv(csv_path)
            if df.empty or len(df) < 2:
                continue

            y_true = df["True Y"].values
            y_pred = df["Predicted Y"].values

            # Compute metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            rrmse = rmse / np.mean(y_true)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            # Pearson & Spearman
            if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
                pearson_val, _ = pearsonr(y_true, y_pred)
                spearman_val, _ = spearmanr(y_true, y_pred)
            else:
                pearson_val = np.nan
                spearman_val = np.nan

            all_runs.append(
                {
                    "experiment": experiment,
                    "feature_selector": feature_selector,
                    "target": target,
                    "regressor": regressor,
                    "path": str(csv_path),
                    "n_samples": len(df),
                    "rmse": rmse,
                    "rrmse": rrmse,
                    "mae": mae,
                    "r2": r2,
                    "pearson_r": pearson_val,
                    "spearman_rho": spearman_val,
                }
            )
        except Exception:
            # Skip invalid files
            continue

    df_results = pd.DataFrame(all_runs)
    if not df_results.empty:
        # Save to cache
        df_results.to_csv(COMPILED_CSV_PATH, index=False)
    return df_results


# Load compiled metrics
@st.cache_data
def load_metrics(force_refresh=False):
    if force_refresh or not COMPILED_CSV_PATH.exists():
        df = compile_all_results()
    else:
        df = pd.read_csv(COMPILED_CSV_PATH)
    return df


# Main title block
st.markdown(
    """
<div class="title-container">
    <h1>🧬 Bioreactor Process Regression Dashboard</h1>
    <p>Compare process parameter prediction results based on microbial and functional feature selection subsets.</p>
</div>
""",
    unsafe_allow_html=True,
)

# Load data
try:
    df_metrics = load_metrics()
except Exception as e:
    st.error(f"Error loading metrics: {e}. Attempting to recompile...")
    df_metrics = compile_all_results()

if df_metrics.empty:
    st.warning(
        "No regression summary files found in `results/batch_regression/`! Make sure you run your pipeline first."
    )
    st.stop()

# Sidebar configuration
st.sidebar.markdown("### ⚙️ Navigation & Actions")
page = st.sidebar.radio(
    "Select View",
    [
        "📊 Target Performance Comparison",
        "🔬 Detailed Model Explorer",
        "📈 Cross-Target Matrix",
    ],
)

# Refresh button
if st.sidebar.button("🔄 Refresh Data Cache"):
    with st.spinner("Re-compiling all metrics..."):
        df_metrics = load_metrics(force_refresh=True)
        st.cache_data.clear()
        st.success("Data recompiled and cache cleared!")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total runs loaded:** {len(df_metrics)}")
st.sidebar.markdown(f"**Targets detected:** {df_metrics['target'].nunique()}")
st.sidebar.markdown(
    f"**Feature Conditions:** {df_metrics['feature_selector'].nunique()} selectors × {df_metrics['experiment'].nunique()} experiments"
)


# Helper to construct paths
def get_run_dir(row):
    if row["experiment"] == "all_features":
        return BASE_DIR / "all_features" / row["target"] / row["regressor"]
    else:
        return (
            BASE_DIR
            / row["experiment"]
            / row["feature_selector"]
            / row["target"]
            / row["regressor"]
        )


# ---------------- PAGE 1: Target Performance Comparison ----------------
if page == "📊 Target Performance Comparison":
    st.markdown(
        "<h2 class='section-header'>📊 Compare Runs for a Single Parameter</h2>",
        unsafe_allow_html=True,
    )

    # Target selection
    available_targets = sorted(df_metrics["target"].unique())
    selected_target = st.selectbox(
        "Select Target Parameter to Analyze:", available_targets
    )

    # Filter for target
    df_target = df_metrics[df_metrics["target"] == selected_target].copy()

    # Unique feature condition label for plotting
    df_target["feature_condition"] = df_target.apply(
        lambda r: (
            "All Features"
            if r["experiment"] == "all_features"
            else f"{r['experiment']} / {r['feature_selector']}"
        ),
        axis=1,
    )

    # Calculate best models
    best_r2_row = (
        df_target.loc[df_target["r2"].idxmax()]
        if not df_target["r2"].isna().all()
        else None
    )
    best_rmse_row = (
        df_target.loc[df_target["rmse"].idxmin()]
        if not df_target["rmse"].isna().all()
        else None
    )

    # Metrics Row
    col1, col2, col3 = st.columns(3)

    with col1:
        if best_r2_row is not None:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-title">🏆 Best R² Score</div>
                <div class="metric-value">{best_r2_row["r2"]:.4f}</div>
                <div class="metric-subtitle"><b>Model:</b> {best_r2_row["regressor"]}</div>
                <div class="metric-subtitle"><b>Features:</b> {best_r2_row["feature_condition"]}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div class="metric-card">
                <div class="metric-title">🏆 Best R² Score</div>
                <div class="metric-value">N/A</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        if best_rmse_row is not None:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-title">📉 Lowest RMSE (RRMSE)</div>
                <div class="metric-value">{best_rmse_row["rmse"]:.4f} ({best_rmse_row["rrmse"] * 100:.2f}%)</div>
                <div class="metric-subtitle"><b>Model:</b> {best_rmse_row["regressor"]}</div>
                <div class="metric-subtitle"><b>Features:</b> {best_rmse_row["feature_condition"]}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div class="metric-card">
                <div class="metric-title">📉 Best (Lowest) RMSE</div>
                <div class="metric-value">N/A</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-title">📋 Run Summary</div>
            <div class="metric-value">{len(df_target)} / 32</div>
            <div class="metric-subtitle">Runs evaluated for this target parameter.</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Heatmap visualizer
    st.markdown(
        "<h4 style='color: #1e3c72; font-weight:600; margin-top:1.5rem;'>Model Performance Heatmap</h4>",
        unsafe_allow_html=True,
    )
    metric_choice = st.radio(
        "Choose performance metric to plot:",
        ["r2", "rmse", "mae", "pearson_r", "spearman_rho"],
        horizontal=True,
    )

    # Pivot target df for heatmap
    pivot_df = df_target.pivot(
        index="feature_condition", columns="regressor", values=metric_choice
    )

    # Plotly heatmap
    fig_heatmap = px.imshow(
        pivot_df,
        labels=dict(
            x="Regressor Model",
            y="Feature Selection Source",
            color=metric_choice.upper(),
        ),
        color_continuous_scale="Viridis"
        if metric_choice in ["rmse", "mae"]
        else "RdBu_r",
        aspect="auto",
    )
    fig_heatmap.update_layout(
        title=f"{metric_choice.upper()} across Feature Sources and Regressors",
        height=500,
        margin=dict(l=150, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Sorted Bar Chart
    st.markdown(
        "<h4 style='color: #1e3c72; font-weight:600; margin-top:1.5rem;'>Comparison of All Runs</h4>",
        unsafe_allow_html=True,
    )
    df_sorted = df_target.sort_values(
        by=metric_choice, ascending=(metric_choice in ["rmse", "mae"])
    )

    fig_bar = px.bar(
        df_sorted,
        x=metric_choice,
        y="feature_condition",
        color="regressor",
        barmode="group",
        orientation="h",
        labels={
            metric_choice: metric_choice.upper(),
            "feature_condition": "Feature Selection Source",
        },
        color_discrete_sequence=px.colors.qualitative.D3,
        hover_data=["experiment", "feature_selector", "r2", "rmse"],
    )
    fig_bar.update_layout(
        height=600,
        margin=dict(l=150, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Show data table
    st.markdown(
        "<h4 style='color: #1e3c72; font-weight:600; margin-top:1.5rem;'>Detailed Performance Table</h4>",
        unsafe_allow_html=True,
    )
    df_display = df_target[
        [
            "experiment",
            "feature_selector",
            "regressor",
            "r2",
            "rmse",
            "rrmse",
            "mae",
            "pearson_r",
            "spearman_rho",
            "n_samples",
        ]
    ].copy()
    df_display = df_display.sort_values(by="r2", ascending=False)

    st.dataframe(
        df_display.style.background_gradient(
            subset=["r2"], cmap="Greens"
        ).background_gradient(subset=["rmse"], cmap="Reds_r"),
        use_container_width=True,
    )


# ---------------- PAGE 2: Detailed Model Explorer ----------------
elif page == "🔬 Detailed Model Explorer":
    st.markdown(
        "<h2 class='section-header'>🔬 Detailed Run Analyzer & Explanations</h2>",
        unsafe_allow_html=True,
    )

    # 4-way filter
    available_targets = sorted(df_metrics["target"].unique())
    target_sel = st.selectbox("1. Target Column:", available_targets)

    df_filt1 = df_metrics[df_metrics["target"] == target_sel]

    available_exps = sorted(df_filt1["experiment"].unique())
    exp_sel = st.selectbox("2. Feature Selection Experiment:", available_exps)

    df_filt2 = df_filt1[df_filt1["experiment"] == exp_sel]

    available_selectors = sorted(df_filt2["feature_selector"].unique())
    sel_sel = st.selectbox("3. Feature Selection Model:", available_selectors)

    df_filt3 = df_filt2[df_filt2["feature_selector"] == sel_sel]

    available_regressors = sorted(df_filt3["regressor"].unique())
    reg_sel = st.selectbox("4. Regressor Model:", available_regressors)

    # Get the specific run row
    run_row_search = df_filt3[df_filt3["regressor"] == reg_sel]

    if run_row_search.empty:
        st.error("No run found for this specific combination!")
    else:
        run_row = run_row_search.iloc[0]
        run_dir = get_run_dir(run_row)

        # Display metrics
        st.markdown("### 📊 Metrics Summary")
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        mcol1.metric(
            "R² Score", f"{run_row['r2']:.4f}" if not pd.isna(run_row["r2"]) else "N/A"
        )
        mcol2.metric(
            "RMSE", f"{run_row['rmse']:.4f}" if not pd.isna(run_row["rmse"]) else "N/A"
        )
        mcol3.metric(
            "MAE", f"{run_row['mae']:.4f}" if not pd.isna(run_row["mae"]) else "N/A"
        )
        mcol4.metric(
            "Pearson r",
            f"{run_row['pearson_r']:.4f}"
            if not pd.isna(run_row["pearson_r"])
            else "N/A",
        )
        mcol5.metric(
            "Spearman rho",
            f"{run_row['spearman_rho']:.4f}"
            if not pd.isna(run_row["spearman_rho"])
            else "N/A",
        )

        # Load prediction CSV
        pred_csv = Path(run_row["path"])
        if pred_csv.exists():
            df_preds = pd.read_csv(pred_csv)

            tab1, tab2, tab3 = st.tabs(
                [
                    "🎯 Predictions & Errors",
                    "🌍 Global Interpretations (SHAP)",
                    "👤 Individual Sample SHAP",
                ]
            )

            # --- TAB 1: Predictions vs True ---
            with tab1:
                col_left, col_right = st.columns([3, 2])

                with col_left:
                    # True vs Predicted Scatter Plot
                    fig_scatter = px.scatter(
                        df_preds,
                        x="True Y",
                        y="Predicted Y",
                        hover_data=["Test Sample", "Root Mean Squared Error"],
                        title=f"True vs. Predicted {run_row['target']}",
                        labels={
                            "True Y": "True Value",
                            "Predicted Y": "Predicted Value",
                        },
                    )

                    # Add y = x diagonal
                    min_val = min(
                        df_preds["True Y"].min(), df_preds["Predicted Y"].min()
                    )
                    max_val = max(
                        df_preds["True Y"].max(), df_preds["Predicted Y"].max()
                    )
                    fig_scatter.add_shape(
                        type="line",
                        x0=min_val,
                        y0=min_val,
                        x1=max_val,
                        y1=max_val,
                        line=dict(color="Red", dash="dash"),
                        name="Perfect Prediction",
                    )
                    fig_scatter.update_layout(height=500)
                    st.plotly_chart(fig_scatter, use_container_width=True)

                with col_right:
                    # Display RMSE Boxplot image if exists
                    boxplot_path = (
                        run_dir / "plots" / f"{run_row['regressor']}_rmse_boxplot.png"
                    )
                    if boxplot_path.exists():
                        st.markdown(
                            "<p style='text-align: center; font-weight:600;'>Distribution of Errors (CV Splits)</p>",
                            unsafe_allow_html=True,
                        )
                        st.image(str(boxplot_path), use_container_width=True)
                    else:
                        st.info("No pre-saved error boxplot found.")

                st.markdown("#### Predictions Table")
                st.dataframe(
                    df_preds[
                        [
                            "Split",
                            "Test Sample",
                            "True Y",
                            "Predicted Y",
                            "Root Mean Squared Error",
                            "Best Pipeline Params",
                        ]
                    ],
                    use_container_width=True,
                )

            # --- TAB 2: Global Interpretations ---
            with tab2:
                col_b, col_ci = st.columns(2)

                with col_b:
                    beeswarm_path = (
                        run_dir
                        / "plots"
                        / f"{run_row['regressor']}_beeswarm_regression.png"
                    )
                    if beeswarm_path.exists():
                        st.markdown(
                            "<p style='text-align: center; font-weight: 600;'>SHAP Beeswarm Plot (Global Feature Effects)</p>",
                            unsafe_allow_html=True,
                        )
                        st.image(str(beeswarm_path), use_container_width=True)
                    else:
                        st.info("No pre-saved beeswarm plot found for this run.")

                with col_ci:
                    ci_path = (
                        run_dir
                        / "plots"
                        / f"{run_row['regressor']}_cluster_importance_Owen_regression.png"
                    )
                    if ci_path.exists():
                        st.markdown(
                            "<p style='text-align: center; font-weight: 600;'>Feature/Cluster Importance (Owen SHAP values)</p>",
                            unsafe_allow_html=True,
                        )
                        st.image(str(ci_path), use_container_width=True)
                    else:
                        st.info(
                            "No pre-saved cluster importance plot found for this run."
                        )

            # --- TAB 3: Individual Sample SHAP ---
            with tab3:
                st.markdown(
                    "#### Select split sample to view localized SHAP Waterfall explanation:"
                )
                available_samples = sorted(df_preds["Test Sample"].unique())
                sample_id = st.selectbox("Select Sample ID:", available_samples)

                sample_row = df_preds[df_preds["Test Sample"] == sample_id].iloc[0]

                scol1, scol2, scol3 = st.columns(3)
                scol1.metric("True Y Value", f"{sample_row['True Y']:.4f}")
                scol2.metric("Predicted Y Value", f"{sample_row['Predicted Y']:.4f}")
                scol3.metric(
                    "Absolute Error (RMSE)",
                    f"{sample_row['Root Mean Squared Error']:.4f}",
                )

                # Check waterfall plot
                waterfall_path = (
                    run_dir
                    / "plots"
                    / f"{run_row['regressor']}_waterfall_{sample_id}_regression.png"
                )
                if waterfall_path.exists():
                    st.image(str(waterfall_path), use_container_width=True)
                else:
                    st.info(f"Waterfall plot not found at: {waterfall_path.name}")
        else:
            st.error(
                f"Could not load predictions summary file at {pred_csv}. File may not exist."
            )


# ---------------- PAGE 3: Cross-Target Matrix ----------------
elif page == "📈 Cross-Target Matrix":
    st.markdown(
        "<h2 class='section-header'>📈 Cross-Target Performance Comparison</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Analyze how the different feature selection methods and regressor models generalize across all predicted parameter targets."
    )

    reg_choices = sorted(df_metrics["regressor"].unique())
    selected_regressor = st.selectbox("Select Regressor Model to Compare:", reg_choices)

    df_reg = df_metrics[df_metrics["regressor"] == selected_regressor].copy()

    df_reg["feature_condition"] = df_reg.apply(
        lambda r: (
            "All Features"
            if r["experiment"] == "all_features"
            else f"{r['experiment']} / {r['feature_selector']}"
        ),
        axis=1,
    )

    ct_metric = st.radio(
        "Select Evaluation Metric:",
        ["r2", "rmse", "mae", "pearson_r", "spearman_rho"],
        key="ct_metric",
    )

    pivot_ct = df_reg.pivot(
        index="feature_condition", columns="target", values=ct_metric
    )

    fig_ct_heatmap = px.imshow(
        pivot_ct,
        labels=dict(
            x="Regression Target Parameter",
            y="Feature Selection Source",
            color=ct_metric.upper(),
        ),
        color_continuous_scale="Viridis" if ct_metric in ["rmse", "mae"] else "RdBu_r",
        aspect="auto",
    )
    fig_ct_heatmap.update_layout(
        title=f"{ct_metric.upper()} Performance Matrix ({selected_regressor})",
        height=600,
        margin=dict(l=150, r=20, t=40, b=100),
    )
    st.plotly_chart(fig_ct_heatmap, use_container_width=True)

    st.markdown("#### Performance Grid Table")
    st.dataframe(
        pivot_ct.style.background_gradient(cmap="coolwarm"), use_container_width=True
    )
