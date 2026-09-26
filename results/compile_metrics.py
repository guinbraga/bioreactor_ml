import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr


def compile_all_results():
    base_dir = Path("results/40_batch_regression")
    all_runs = []

    # We find all csv files named "*_predictions_summary.csv"
    csv_files = list(base_dir.glob("**/*_predictions_summary.csv"))
    print(f"Found {len(csv_files)} predictions summary files.")

    for csv_path in csv_files:
        # Resolve path relative to base_dir
        rel_path = csv_path.relative_to(base_dir)
        parts = rel_path.parts

        # Structure could be:
        # Case 1 (Feature selection): {experiment}/{classification_model}/{target}/{regressor_model}/{filename}
        # e.g. EXPERIMENT/L1 Logistic Regression/Acetate (mg L-1)/Elastic Net Regressor/Elastic Net Regressor_predictions_summary.csv
        # Case 2 (All features): all_features/{target}/{regressor_model}/{filename}
        # e.g. all_features/Acetate (mg L-1)/Elastic Net Regressor/Elastic Net Regressor_predictions_summary.csv

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

        # Load CSV
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

            # Pearson and Spearman
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
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")

    df_results = pd.DataFrame(all_runs)
    print(f"Compiled {len(df_results)} runs.")
    print("Unique targets found:")
    print(df_results["target"].value_counts())
    print("\nUnique experiments found:")
    print(df_results["experiment"].value_counts())
    print("\nUnique feature selectors found:")
    print(df_results["feature_selector"].value_counts())
    print("\nUnique regressors found:")
    print(df_results["regressor"].value_counts())

    # Save compiled results
    df_results.to_csv("results/compiled_regression_metrics.csv", index=False)
    print("\nSaved compiled results to results/compiled_regression_metrics.csv")


if __name__ == "__main__":
    compile_all_results()
