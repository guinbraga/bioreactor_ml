#!/usr/bin/env python3
"""Regenerate SHAP plots and cluster-importance JSON with GTDB taxonomy labels.

For every ``shap_explanations_table.parquet`` found under
``results/40_batch_regression`` this script rebuilds the per-sample SHAP
``Explanation`` objects from the saved table and writes new PNGs to a sibling
``plots_taxonomy/`` folder so that the y-axis feature labels show the OTU's
taxonomic identification (finest available GTDB rank) instead of the raw OTU
identifier. The original ``plots/`` folder is never modified.

It additionally regenerates the ``*_cluster_importance_Owen*.png`` plots from
the same saved artifacts (see ``_cluster_importances`` for the exact recipe,
which mirrors ``TopFeaturesPicker.pick_top_features`` +
``ResultsPlotManager.generate_cluster_importance_plot`` in ``src/``) and writes
a taxonomy version of every ``experiment_cluster.json`` named
``experiment_cluster_taxonomy.json`` (new file; the OTU-keyed original is never
modified).

Because two different OTUs can share the same finest-rank taxonomy, labels are
disambiguated folder-wide by appending the short bin token (``_short_token``) to
*every* OTU whose base label collides. This keeps plot labels unambiguous and,
crucially, keeps the JSON keys unique (without it, colliding medoids would
silently overwrite each other).

Mapping source: ``data/all_metrics/gtdbtk_all.tsv`` (column ``user_genome`` ->
column ``classification``).

Label rule (chosen by the user): use the finest available, unprefixed rank,
i.e. species if present, otherwise genus, otherwise family, otherwise
order/class/phylum/domain. When two different OTUs in the same result folder
collapse to the same label, the short bin token is appended in parentheses
(e.g. ``Sporobacter (B06-bin.24)``) so no two rows are ambiguous.

The SHAP values / raw feature values come from the parquet table; the exact
per-sample base value comes from the sibling ``*_shap_explanations_obj.joblib``
when present (falling back to ``Predicted Y - sum(SHAP)`` from
``*_predictions_summary.csv``). Sample order is taken from the same predictions
CSV, whose rows are aligned with the parquet rows.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from joblib import load as joblib_load  # noqa: E402
from shap import Explanation  # noqa: E402

ROOT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
GTDBTK_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "all_metrics",
    "gtdbtk_all.tsv",
)

MAX_DISPLAY = 15
N_CLUSTERS = 20
DPI = 300
FIG_SIZE = (10, 8)
OUT_SUBDIR = "plots_taxonomy"
CLUSTER_JSON_NAME = "experiment_cluster.json"
CLUSTER_JSON_OUT_NAME = "experiment_cluster_taxonomy.json"

# Finest-to-coarsest rank order used for label fallback.
RANK_ORDER = ("s", "g", "f", "o", "c", "p", "d")

_TAXONOMY: dict[str, str] = {}


def load_taxonomy(path: str) -> dict[str, str]:
    """Return ``{user_genome: finest unprefixed rank name}`` from the GTDB-TSV."""
    table = pd.read_csv(path, sep="\t")
    return {
        str(genome): _finest_rank_label(str(classification))
        for genome, classification in zip(
            table["user_genome"], table["classification"]
        )
    }


def _finest_rank_label(classification: str) -> str:
    ranks: dict[str, str] = {}
    for part in classification.split(";"):
        if "__" in part:
            key, value = part.split("__", 1)
            ranks[key] = value.strip()
    for key in RANK_ORDER:
        if ranks.get(key):
            return ranks[key]
    return ""


def _short_token(otu: str) -> str:
    """Compact unique-ish token for an OTU, e.g. ``B06-bin.24``."""
    if "_" in otu:
        tail = otu.rsplit("_", 1)[1]
        if "-bin." in tail:
            return tail
    return otu


def _build_labels(otus: list[str], taxonomy: dict[str, str]) -> list[str]:
    base = [taxonomy.get(otu, "").strip() or otu for otu in otus]
    counts = Counter(base)
    return [
        f"{label} ({_short_token(otu)})" if counts[label] > 1 else label
        for otu, label in zip(otus, base)
    ]


def _folder_label_map(
    target_dir: str, taxonomy: dict[str, str]
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return ``(medoids_dict, label_map)`` for one result folder.

    ``medoids_dict`` is the raw ``experiment_cluster.json`` content
    (``{medoid: [member_OTUs...]}``). ``label_map`` maps every OTU in the folder
    to its folder-wide, collision-disambiguated taxonomy label.
    """
    json_path = os.path.join(target_dir, CLUSTER_JSON_NAME)
    with open(json_path, encoding="utf-8") as handle:
        medoids_dict = json.load(handle)

    features = [member for members in medoids_dict.values() for member in members]
    labels = _build_labels(features, taxonomy)
    return medoids_dict, dict(zip(features, labels))


def _load_clusters(medoids_dict: dict[str, list[str]]) -> Series:
    """Invert ``{medoid: [members]}`` into a ``member -> medoid`` Series."""
    mapping = {
        member: medoid
        for medoid, members in medoids_dict.items()
        for member in members
    }
    return pd.Series(mapping)


def _cluster_importances(
    df: pd.DataFrame, clusters_series: Series, percentage: float = 0.95
) -> Series:
    """Rebuild the cluster importances exactly as the source pipeline does.

    Mirrors ``ResultsDataManager._mean_abs_shap`` (per-feature mean of |SHAP|),
    ``TopFeaturesPicker.pick_top_features`` (sum |SHAP| per cluster representative,
    keep the clusters covering ``percentage`` of total importance), and the
    ``n_clusters`` head selection from
    ``ResultsPlotManager.generate_cluster_importance_plot``.
    """
    shap_cols = [column for column in df.columns if column.endswith("_SHAP")]
    feature_importances = pd.Series(
        df[shap_cols].abs().mean().values,
        index=[column[:-5] for column in shap_cols],
    )

    # Features absent from the cluster mapping are mapped to themselves, matching
    # TopFeaturesPicker, so nothing is dropped or misaligned.
    missing_features = feature_importances.index.difference(clusters_series.index)
    if not missing_features.empty:
        extra_mapping = pd.Series(missing_features, index=missing_features)
        clusters_series = pd.concat([clusters_series, extra_mapping])

    cluster_importances = feature_importances.groupby(clusters_series).sum()
    sorted_cluster_importances = cluster_importances.sort_values(ascending=False)
    importance_threshold = sorted_cluster_importances.sum() * percentage
    cumulative_sum = sorted_cluster_importances.cumsum()
    cutoff_index = (cumulative_sum >= importance_threshold).to_numpy().argmax()

    return sorted_cluster_importances.iloc[: cutoff_index + 1].head(N_CLUSTERS)


def _init_worker(taxonomy_path: str) -> None:
    global _TAXONOMY
    _TAXONOMY = load_taxonomy(taxonomy_path)


def _load_sample_ids(model_dir: str, expected: int) -> list[str]:
    pred_files = glob.glob(os.path.join(model_dir, "*_predictions_summary.csv"))
    if not pred_files:
        raise FileNotFoundError(f"no predictions CSV in {model_dir}")
    preds = pd.read_csv(pred_files[0])
    if len(preds) != expected:
        raise ValueError(
            f"{model_dir}: predictions rows {len(preds)} != parquet rows {expected}"
        )
    return preds["Test Sample"].astype(str).tolist(), preds


def _load_base_values(
    model_dir: str, values: np.ndarray, preds: pd.DataFrame
) -> np.ndarray:
    obj_files = glob.glob(os.path.join(model_dir, "*_shap_explanations_obj.joblib"))
    if obj_files:
        try:
            objs = joblib_load(obj_files[0])
            if len(objs) == values.shape[0]:
                base = np.array([float(o.base_values) for o in objs], dtype=float)
                # Guard against a stale/mismatched joblib: require additivity.
                derived = preds["Predicted Y"].to_numpy(dtype=float) - values.sum(axis=1)
                if np.allclose(base, derived, rtol=1e-4, atol=1e-3):
                    return base
        except Exception:  # noqa: BLE001 - fall back to derived base values
            pass
    return preds["Predicted Y"].to_numpy(dtype=float) - values.sum(axis=1)


def _save(fig, path: str, name_suffix: str) -> str:
    if name_suffix:
        root, ext = os.path.splitext(path)
        path = f"{root}{name_suffix}{ext}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _render_cluster_importance(model_dir: str, out_subdir: str) -> str:
    """Write the taxonomy-labeled cluster-importance plot for one model folder."""
    parquet_path = os.path.join(model_dir, "shap_explanations_table.parquet")
    df = pd.read_parquet(parquet_path)

    target_dir = os.path.dirname(model_dir)
    medoids_dict, label_map = _folder_label_map(target_dir, _TAXONOMY)
    clusters_series = _load_clusters(medoids_dict)

    selected = _cluster_importances(df, clusters_series).sort_values(ascending=True)
    counts = clusters_series.value_counts()
    selected.index = [
        f"{label_map.get(medoid, medoid)} ({counts.get(medoid, 1)})"
        for medoid in selected.index
    ]

    model_name = os.path.basename(model_dir)
    target_col = os.path.basename(target_dir)
    class_label = str(df["Class"].iloc[0]) if "Class" in df.columns else ""
    title = (
        f"{model_name} Importances for Top {N_CLUSTERS} Selected Clusters "
        f"for predicting {target_col}"
    )
    if class_label:
        title += f" as {class_label}"

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    selected.plot.barh(ax=ax, title=title)

    existing = glob.glob(
        os.path.join(model_dir, "plots", "*_cluster_importance_Owen*.png")
    )
    if len(existing) == 1:
        filename = os.path.basename(existing[0])
    else:
        name_class = f"_{class_label}" if class_label else ""
        filename = f"{model_name}_cluster_importance_Owen{name_class}.png"
    return _save(fig, os.path.join(model_dir, out_subdir, filename), "")


def _render_shap_plots(
    model_dir: str, out_subdir: str = OUT_SUBDIR, name_suffix: str = ""
) -> list[str]:
    """Regenerate the beeswarm and all waterfall plots for one model folder."""
    parquet_path = os.path.join(model_dir, "shap_explanations_table.parquet")
    df = pd.read_parquet(parquet_path)

    has_class = "Class" in df.columns
    class_label = str(df["Class"].iloc[0]) if has_class else ""
    title_class = f" as {class_label}" if class_label else ""
    name_class = f"_{class_label}" if class_label else ""

    shap_cols = [c for c in df.columns if c.endswith("_SHAP")]
    raw_cols = [c[:-5] + "_RAW" for c in shap_cols]
    otus = [c[:-5] for c in shap_cols]
    labels = _build_labels(otus, _TAXONOMY)

    values = df[shap_cols].to_numpy(dtype=float)
    data = df[raw_cols].to_numpy(dtype=float)

    samples, preds = _load_sample_ids(model_dir, len(df))
    base_values = _load_base_values(model_dir, values, preds)

    model_name = os.path.basename(model_dir)
    target_col = os.path.basename(os.path.dirname(model_dir))
    plot_dir = os.path.join(model_dir, out_subdir)
    os.makedirs(plot_dir, exist_ok=True)

    written: list[str] = []

    # -- Global beeswarm -----------------------------------------------------
    beeswarm_explanation = Explanation(
        base_values=float(np.mean(base_values)),
        feature_names=labels,
        values=values,
        data=data,
    )
    plt.figure(figsize=FIG_SIZE)
    shap.plots.beeswarm(beeswarm_explanation, show=False, max_display=MAX_DISPLAY)
    plt.title(
        f"{model_name} Global Beeswarm plot for predicting {target_col}{title_class}"
    )
    written.append(
        _save(
            plt.gcf(),
            os.path.join(plot_dir, f"{model_name}_beeswarm{name_class}.png"),
            name_suffix,
        )
    )

    # -- Per-sample waterfalls ----------------------------------------------
    for i, sample_id in enumerate(samples):
        explanation = Explanation(
            base_values=float(base_values[i]),
            feature_names=labels,
            values=values[i],
            data=data[i],
        )
        plt.figure(figsize=FIG_SIZE)
        shap.plots.waterfall(explanation, show=False, max_display=MAX_DISPLAY)
        plt.title(
            f"{model_name} - Feature Importances for {sample_id} "
            f"Prediction of {target_col}{title_class}"
        )
        written.append(
            _save(
                plt.gcf(),
                os.path.join(
                    plot_dir, f"{model_name}_waterfall_{sample_id}{name_class}.png"
                ),
                name_suffix,
            )
        )

    return written


def process_dir(
    model_dir: str,
    out_subdir: str = OUT_SUBDIR,
    name_suffix: str = "",
    do_shap: bool = True,
    do_cluster: bool = True,
) -> dict:
    """Regenerate the enabled plot families for one model folder."""
    written: list[str] = []
    if do_shap:
        written.extend(_render_shap_plots(model_dir, out_subdir, name_suffix))
    if do_cluster:
        written.append(_render_cluster_importance(model_dir, out_subdir))
    return {"dir": model_dir, "written": len(written)}


def process_cluster_json(target_dir: str) -> str:
    """Write ``experiment_cluster_taxonomy.json`` for one target folder."""
    medoids_dict, label_map = _folder_label_map(target_dir, _TAXONOMY)
    taxonomy_dict = {
        label_map[medoid]: [label_map[member] for member in members]
        for medoid, members in medoids_dict.items()
    }
    out_path = os.path.join(target_dir, CLUSTER_JSON_OUT_NAME)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(taxonomy_dict, handle, indent=4)
    return out_path


def discover_dirs(root: str) -> list[str]:
    parquet_files = glob.glob(
        os.path.join(root, "**", "shap_explanations_table.parquet"), recursive=True
    )
    return sorted(os.path.dirname(p) for p in parquet_files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=ROOT_DEFAULT, help="batch results root")
    parser.add_argument("--taxonomy", default=GTDBTK_DEFAULT, help="gtdbtk TSV")
    parser.add_argument(
        "--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="workers"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="only process the first N folders"
    )
    parser.add_argument(
        "--out-subdir",
        default=OUT_SUBDIR,
        help="sub-folder inside each result dir for the new plots",
    )
    parser.add_argument(
        "--name-suffix", default="", help="suffix inserted before .png"
    )
    parser.add_argument(
        "--only", default="", help="only process folders whose path contains this"
    )
    parser.add_argument(
        "--skip-shap", action="store_true", help="skip beeswarm/waterfall plots"
    )
    parser.add_argument(
        "--skip-cluster", action="store_true", help="skip cluster-importance plots"
    )
    parser.add_argument(
        "--skip-json",
        action="store_true",
        help="skip experiment_cluster_taxonomy.json generation",
    )
    args = parser.parse_args()

    dirs = discover_dirs(args.root)
    if args.only:
        dirs = [d for d in dirs if args.only in d]
    if args.limit:
        dirs = dirs[: args.limit]
    if not dirs:
        print("No shap_explanations_table.parquet found.", file=sys.stderr)
        return 1

    do_shap = not args.skip_shap
    do_cluster = not args.skip_cluster
    do_json = not args.skip_json
    target_dirs = sorted({os.path.dirname(d) for d in dirs})

    print(
        f"Processing {len(dirs)} folders (shap={do_shap}, cluster={do_cluster}) "
        f"and {len(target_dirs)} target folders (json={do_json}) "
        f"with {args.jobs} workers..."
    )

    errors: list[tuple[str, str]] = []
    total_plots = 0
    total_json = 0
    with ProcessPoolExecutor(
        max_workers=args.jobs, initializer=_init_worker, initargs=(args.taxonomy,)
    ) as pool:
        futures: dict = {}
        if do_shap or do_cluster:
            for model_dir in dirs:
                futures[
                    pool.submit(
                        process_dir,
                        model_dir,
                        args.out_subdir,
                        args.name_suffix,
                        do_shap,
                        do_cluster,
                    )
                ] = ("plot", model_dir)
        if do_json:
            for target_dir in target_dirs:
                futures[pool.submit(process_cluster_json, target_dir)] = (
                    "json",
                    target_dir,
                )

        for idx, future in enumerate(as_completed(futures), start=1):
            kind, path = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - collect and keep going
                errors.append((path, f"{type(exc).__name__}: {exc}"))
                print(f"[{idx}/{len(futures)}] FAILED {path}: {exc}", flush=True)
                continue
            if kind == "plot":
                total_plots += result["written"]
                print(
                    f"[{idx}/{len(futures)}] {result['written']} plots  "
                    f"{os.path.relpath(path, args.root)}",
                    flush=True,
                )
            else:
                total_json += 1
                print(
                    f"[{idx}/{len(futures)}] json  {os.path.relpath(path, args.root)}",
                    flush=True,
                )

    print(
        f"\nDone. {total_plots} plots and {total_json} taxonomy JSONs written; "
        f"{len(errors)} folders failed."
    )
    for path, err in errors:
        print(f"  FAILED {path}: {err}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
