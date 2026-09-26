#!/usr/bin/env python3
"""Migrate existing _top_features.csv files to _top_clusters.csv and
generate expanded _top_features.csv from experiment_cluster.json."""

import json
import os
from pathlib import Path

RESULTS_DIR = "/home/guilherme/git/sci_init/04_bioreactor_ml_project/results/38_cluster_importances"


def migrate_target_dir(target_dir: Path) -> None:
    cluster_file = target_dir / "experiment_cluster.json"
    if not cluster_file.exists():
        print(f"Skipping {target_dir}: no experiment_cluster.json")
        return

    with open(cluster_file) as f:
        clusters = json.load(f)

    for model_dir in sorted(target_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        top_feat_csv = model_dir / f"{model_dir.name}_top_features.csv"
        if not top_feat_csv.exists():
            continue

        cluster_reprs = []
        with open(top_feat_csv) as f:
            header = f.readline()
            for line in f:
                feature = line.strip().rstrip(",")
                if feature:
                    cluster_reprs.append(feature)

        top_clusters_csv = model_dir / f"{model_dir.name}_top_clusters.csv"
        top_feat_csv.rename(top_clusters_csv)
        print(f"Renamed: {top_feat_csv} -> {top_clusters_csv}")

        expanded = []
        for rep in cluster_reprs:
            if rep in clusters:
                expanded.extend(clusters[rep])
            else:
                expanded.append(rep)

        new_top_feat_csv = model_dir / f"{model_dir.name}_top_features.csv"
        with open(new_top_feat_csv, "w") as f:
            f.write("feature,\n")
            for feat in expanded:
                f.write(f"{feat},\n")
        print(f"Created: {new_top_feat_csv} ({len(expanded)} features)")


def main():
    results_path = Path(RESULTS_DIR)
    if not results_path.exists():
        print(f"Directory not found: {results_path}")
        return

    for target_dir in sorted(results_path.iterdir()):
        if target_dir.is_dir() and (target_dir / "experiment_cluster.json").exists():
            print(f"\n=== Processing {target_dir.name} ===")
            migrate_target_dir(target_dir)


if __name__ == "__main__":
    main()
