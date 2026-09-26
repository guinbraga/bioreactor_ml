#!/usr/bin/env python3
"""Boxplot comparing per-sample RMSE of the Elastic Net Regressor on the
'Specific biogas production (mLNorm gVS-1)_corrected' target across:

  - all_features baseline
  - 15 feature-selection variants:
      EXPERIMENT / Gut Compartment / TRANSFER
        x { Elastic Net, L1 Logistic Regression, Random Forest,
            SVM-linear, SVM-radial }

Only the Elastic Net Regressor predictions are considered (each selection
folder contains an 'Elastic Net Regressor' subfolder with a
'..._predictions_summary.csv' that holds per-sample RMSE). The Random Forest
Regressor predictions in those folders are deliberately ignored.

Output: same directory as this script,
'boxplot_rmse_biogas_elasticnet.png' (wide figure to fit 16 boxes).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Scale all matplotlib fonts ~1.3x relative to defaults so the wide 16-box
# figure stays readable. Defaults: labels/ticks=10, title=12, legend=8.
plt.rcParams.update(
    {
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.titlesize": 16,
        "legend.fontsize": 10,
    }
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
CLUSTER_DIR = PROJECT_ROOT / "results" / "38_cluster_importances"
ALL_FEATURES_POOL = (
    PROJECT_ROOT / "results" / "21_out_clustering" / "experiment_clusters.csv"
)
TARGET = "Specific biogas production (mLNorm gVS-1)_corrected"
PRED_FILE = "Elastic Net Regressor/Elastic Net Regressor_predictions_summary.csv"
RMSE_COL = "Root Mean Squared Error"

# (group, selection) for the 15 feature-selection variants, in display order:
# grouped by selection method so the plot reads method-across-groups.
SELECTIONS = [
    # "Elastic Net",
    "L1 Logistic Regression",
    # "Random Forest",
    # "SVM-linear",
    # "SVM-radial",
]
GROUPS = ["EXPERIMENT", "Gut Compartment", "TRANSFER"]


def _label(group: str | None, selection: str | None) -> str:
    if group is None:
        return "All features"
    return f"{group}\n{selection}"


def _rmse_series(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    if RMSE_COL not in df.columns:
        raise ValueError(f"Missing column '{RMSE_COL}' in {csv_path}")
    return pd.to_numeric(df[RMSE_COL], errors="coerce").dropna()


def _count_features(csv_path: Path) -> int:
    """Number of features = non-header rows in a '*_top_features.csv' with a
    single 'feature' column. Returns -1 if the file is missing/uncountable."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return -1
    df = pd.read_csv(csv_path)
    if df.empty:
        return 0
    # Single-column file named 'feature' (per cluster-importances layout);
    # fall back to row count if columns differ.
    if "feature" in df.columns:
        return int(df["feature"].notna().sum())
    return int(len(df.dropna(how="all")))


def _features_csv_for(grp: str | None, sel: str | None) -> Path:
    """Path to the *_top_features.csv holding the features used for a given
    (group, selection) — sources from results/38_cluster_importances.
    For all-features the count is computed from the canonical cluster pool
    in results/21_out_clustering/experiment_clusters.csv (all_features'
    regressor top_features.csv omits the ~165 features its L1 penalty
    zeroed out, so it undercounts)."""
    if grp is None:
        return ALL_FEATURES_POOL
    return CLUSTER_DIR / grp / sel / f"{sel}_top_features.csv"


def main() -> int:
    # Build ordered list of (label, rmse values, n_features)
    data: list[tuple[str, pd.Series, int]] = []

    # 1) All-features baseline (first box)
    all_csv = BASE_DIR / "all_features" / TARGET / PRED_FILE
    if not all_csv.exists():
        print(f"Missing predictions file: {all_csv}", file=sys.stderr)
        return 1
    all_features_count = _count_features(_features_csv_for(None, None))
    data.append((_label(None, None), _rmse_series(all_csv), all_features_count))

    # 2) 15 selections. Order requested by the user: for each GROUP
    #    (EXPERIMENT, Gut Compartment, TRANSFER), walk all 5 selection
    #    methods. So the layout is:
    #       [all_features] [EXPERIMENT x5] [Gut Compartment x5] [TRANSFER x5]
    for grp in GROUPS:
        for sel in SELECTIONS:
            csv_path = BASE_DIR / grp / sel / TARGET / PRED_FILE
            if not csv_path.exists():
                print(f"Missing predictions file: {csv_path}", file=sys.stderr)
                return 1
            n_features = _count_features(_features_csv_for(grp, sel))
            data.append((_label(grp, sel), _rmse_series(csv_path), n_features))

    labels = [lbl for lbl, _, _ in data]
    values = [s.values for _, s, _ in data]
    n_features_list = [nf for _, _, nf in data]

    # Find the box with the lowest mean RMSE — draw a red dotted line there.
    means = [s.mean() if len(s) else float("inf") for _, s, _ in data]
    best_idx = int(min(range(len(means)), key=lambda i: means[i]))
    best_val = means[best_idx]
    best_label = labels[best_idx].replace("\n", " / ")

    # Wide figure: 16 boxes -> generous width
    fig, ax = plt.subplots(figsize=(10, 9))
    bp = ax.boxplot(
        values,
        labels=labels,
        showmeans=True,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(color="black", linewidth=1.4),
        meanprops=dict(
            marker="D", markerfacecolor="white", markeredgecolor="black", markersize=5
        ),
        flierprops=dict(
            marker="o",
            markerfacecolor="none",
            markeredgecolor="gray",
            markersize=4,
            alpha=0.8,
        ),
    )

    # Color the first (all_features) box distinctly from the 15 selections
    cmap = plt.get_cmap("tab10")
    group_colors = {
        "EXPERIMENT": cmap(0),
        "Gut Compartment": cmap(1),
        "TRANSFER": cmap(2),
    }
    for i, (lbl, _, _nf) in enumerate(data):
        box = bp["boxes"][i]
        if i == 0:
            box.set_facecolor("#cccccc")
            box.set_edgecolor("#444444")
        else:
            grp = lbl.split("\n")[0]
            c = group_colors.get(grp, "#dddddd")
            box.set_facecolor(c)
            box.set_edgecolor("#333333")
            box.set_alpha(0.65)

    ax.axhline(0, color="lightgray", lw=0.8, zorder=0)
    ax.axhline(
        best_val,
        color="red",
        linestyle=":",
        linewidth=1.4,
        zorder=2,
        label=f"lowest mean RMSE = {best_val:.2f} ({best_label})",
    )
    # Annotation inside the plot (anchored high, y~225) with an arrow pointing
    # to the mean (diamond marker) of the best box. Box positions are 1-based.
    best_box_x = best_idx + 1
    ax.annotate(
        f"lowest mean = {best_val:.2f}\n({best_label})",
        xy=(best_box_x, best_val),
        xytext=(best_box_x + 0.3, 205),
        color="red",
        fontsize=12,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.4),
    )
    ax.set_ylabel("Per-sample RMSE (mLNorm gVS$^{-1}$)")
    ax.set_xlabel("Feature selection method")
    ax.set_title(
        "Elastic Net Regressor — per-sample absolute error for\n"
        f"'{TARGET}'\n"
        "All features vs. features selected by L1 Logistic Regression on 3 experiment classes \n"
        "n = number of input features used by the regressor"
    )

    # Annotate feature counts (n_features) under each tick label
    ticklabels = [
        f"{lbl}\n(n={nf if nf >= 0 else '?'})"
        for lbl, nf in zip(labels, n_features_list)
    ]
    ax.set_xticklabels(ticklabels, rotation=45, ha="right", fontsize=12)

    # Add vertical separators between groups:
    # [all_features | EXPERIMENT x5 | Gut Compartment x5 | TRANSFER x5]
    # separators fall after positions 1, 6, 11.

    # Legend mapping group -> color
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#cccccc", edgecolor="#444444", label="All features"),
        Patch(
            facecolor=group_colors["EXPERIMENT"],
            edgecolor="#333333",
            alpha=0.65,
            label="EXPERIMENT",
        ),
        Patch(
            facecolor=group_colors["Gut Compartment"],
            edgecolor="#333333",
            alpha=0.65,
            label="Gut Compartment",
        ),
        Patch(
            facecolor=group_colors["TRANSFER"],
            edgecolor="#333333",
            alpha=0.65,
            label="TRANSFER",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.95)

    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    out_path = BASE_DIR / "boxplot_rmse_biogas_elasticnet_L1LR.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")
    print("Box count:", len(values))
    for lbl, s, nf in data:
        nf_str = str(nf) if nf >= 0 else "?"
        if len(s):
            print(
                f"  {lbl.replace(chr(10), ' / ')}: n_features={nf_str} "
                f"n_samples={len(s)} "
                f"median={s.median():.3f} mean={s.mean():.3f} "
                f"min={s.min():.3f} max={s.max():.3f}"
            )
        else:
            print(f"  {lbl.replace(chr(10), ' / ')}: NO DATA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
