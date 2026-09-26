#!/usr/bin/env python3
"""True-vs-predicted plots for the best models listed in ``.good_molecules.csv``.

For every row of ``.good_molecules.csv`` (``metric, best_model, features``) this
script locates the matching batch-regression result folder::

    results/40_batch_regression/<features>/<metric>/<best_model>/
        <best_model>_predictions_summary.csv

and plots per-sample ``True Y`` against ``Predicted Y`` as a scatter, together
with the ordinary-least-squares fit line (red) and its 95% confidence band
(grey), plus the goodness-of-fit metrics (RMSE, MAE, R^2, n) in the axis title.

Outputs (written next to this script, in ``true_vs_pred/``):

  - ``true_vs_pred_<metric_slug>.png`` - one figure per metric.
  - ``true_vs_pred_all.png``           - a grid combining all metrics.

Both ``features`` and ``metric`` are resolved case-insensitively (e.g. the CSV
metric ``OLR (gvS.l.d)`` matches the folder ``OLR (gVS.L.d)``), because folder
casing in the results tree is not consistent with the CSV.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as scipy_stats  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
GOOD_MOLECULES_CSV = PROJECT_ROOT / ".good_molecules.csv"
RESULTS_DIR = BASE_DIR
OUT_DIR = BASE_DIR / "true_vs_pred"

TRUE_COL = "True Y"
PRED_COL = "Predicted Y"


def _resolve_child_case_insensitive(parent: Path, name: str) -> Path | None:
    """Return ``parent/<name>`` crying-case-insensitively, or None if absent."""
    if not parent.is_dir():
        return None
    for child in parent.iterdir():
        if child.name.lower() == name.lower():
            return child
    return None


def _slug(text: str) -> str:
    """Filesystem-safe slug for a metric name."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return slug or "metric"


def _feature_parts(features: str) -> list[str]:
    """Split the 'A / B' features field into path components."""
    return [part.strip() for part in features.split("/") if part.strip()]


def _predictions_path(metric: str, model: str, features: str) -> Path | None:
    """Resolve the predictions-summary CSV for one CSV row, or None."""
    group_dir = RESULTS_DIR
    for part in _feature_parts(features):
        group_dir = _resolve_child_case_insensitive(group_dir, part)
        if group_dir is None:
            return None

    metric_dir = _resolve_child_case_insensitive(group_dir, metric)
    if metric_dir is None:
        return None

    model_dir = _resolve_child_case_insensitive(metric_dir, model)
    if model_dir is None:
        return None

    exact = model_dir / f"{model}_predictions_summary.csv"
    if exact.exists():
        return exact
    matches = sorted(model_dir.glob("*_predictions_summary.csv"))
    return matches[0] if matches else None


def _load_predictions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in (TRUE_COL, PRED_COL) if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing column(s) {missing}")
    out = df[[TRUE_COL, PRED_COL]].apply(pd.to_numeric, errors="coerce").dropna()
    if out.empty:
        raise ValueError(f"{csv_path}: no numeric True/Predicted rows")
    return out


def _metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = true - pred
    rmse = float(np.sqrt(np.mean(residual**2)))
    true_mean = float(true.mean())
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((true - true_mean) ** 2))
    return {
        "rmse": rmse,
        "rrmse": rmse / true_mean * 100.0 if true_mean != 0 else float("nan"),
        "mae": float(np.mean(np.abs(residual))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n": int(true.size),
    }


def _scatter(ax, df: pd.DataFrame, title: str) -> dict[str, float]:
    true = df[TRUE_COL].to_numpy(dtype=float)
    pred = df[PRED_COL].to_numpy(dtype=float)
    stats = _metrics(true, pred)

    ax.scatter(true, pred, s=42, edgecolor="black", linewidth=0.5, alpha=0.8, zorder=4)

    lo = float(min(true.min(), pred.min()))
    hi = float(max(true.max(), pred.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    axis_lo, axis_hi = lo - pad, hi + pad
    ax.set_xlim(axis_lo, axis_hi)
    ax.set_ylim(axis_lo, axis_hi)

    slope, intercept = np.polyfit(true, pred, 1)
    fit_x = np.linspace(axis_lo, axis_hi, 200)
    fit_y = slope * fit_x + intercept

    # 95% CI of the OLS mean response:
    #   y_hat(x0) +/- t(0.975, n-2) * s * sqrt(1/n + (x0 - xbar)^2 / Sxx)
    n = true.size
    residuals = pred - (slope * true + intercept)
    s_err = float(np.sqrt(np.sum(residuals**2) / (n - 2))) if n > 2 else 0.0
    sxx = float(np.sum((true - true.mean()) ** 2))
    if s_err > 0 and sxx > 0:
        t_crit = float(scipy_stats.t.ppf(0.975, df=n - 2))
        ci = t_crit * s_err * np.sqrt(1.0 / n + (fit_x - true.mean()) ** 2 / sxx)
    else:
        ci = np.zeros_like(fit_x)

    ax.fill_between(
        fit_x,
        fit_y - ci,
        fit_y + ci,
        color="grey",
        alpha=0.3,
        zorder=1,
        label="95% CI of the OLS fit",
    )
    ax.plot(
        fit_x,
        fit_y,
        color="red",
        linewidth=1.5,
        zorder=3,
        label=f"OLS fit: y = {slope:.3g}x + {intercept:.3g}",
    )

    ax.set_xlabel("True value")
    ax.set_ylabel("Predicted value")
    ax.set_title(
        f"{title}\n"
        f"RRMSE = {stats['rrmse']:.3g}%   MAE = {stats['mae']:.3g}   "
        f"R$^2$ = {stats['r2']:.3f}   n = {stats['n']}",
        fontsize=11,
    )
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(linestyle=":", alpha=0.5)
    return stats


def main() -> int:
    if not GOOD_MOLECULES_CSV.exists():
        print(f"Missing {GOOD_MOLECULES_CSV}", file=sys.stderr)
        return 1

    good = pd.read_csv(GOOD_MOLECULES_CSV)
    required = {"metric", "best_model", "features"}
    if not required.issubset(good.columns):
        print(f"{GOOD_MOLECULES_CSV} must have columns {sorted(required)}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panels: list[tuple[str, pd.DataFrame, dict[str, float]]] = []
    for row in good.itertuples(index=False):
        csv_path = _predictions_path(row.metric, row.best_model, row.features)
        if csv_path is None:
            print(
                f"WARNING: no predictions found for "
                f"'{row.metric}' / '{row.best_model}' / '{row.features}'",
                file=sys.stderr,
            )
            continue

        df = _load_predictions(csv_path)
        title = f"{row.metric} - {row.best_model}\n({row.features})"

        # Individual figure per metric.
        fig, ax = plt.subplots(figsize=(6.5, 6))
        stats = _scatter(ax, df, title)
        fig.tight_layout()
        out_path = OUT_DIR / f"true_vs_pred_{_slug(row.metric)}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(
            f"Wrote {out_path.relative_to(PROJECT_ROOT)}  "
            f"(n={stats['n']}, RRMSE={stats['rrmse']:.4g}%, R2={stats['r2']:.3f})"
        )

        panels.append((title, df, stats))

    if not panels:
        print("No plots produced.", file=sys.stderr)
        return 1

    # Combined grid figure.
    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, df, _stats) in zip(axes, panels):
        _scatter(ax, df, title)
    for ax in axes[len(panels) :]:
        ax.axis("off")
    fig.tight_layout()
    grid_path = OUT_DIR / "true_vs_pred_all.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {grid_path.relative_to(PROJECT_ROOT)}  ({len(panels)} panels)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
