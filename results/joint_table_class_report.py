import pandas as pd
from pathlib import Path
import argparse


def generate_joint_table(experiment_path: str):
    """
    Reads all model_name_classification_report.csv files in the given
    experiment directory and combines them into a single joint DataFrame.
    """
    base_dir = Path(experiment_path)
    report_dict = {}

    if not base_dir.exists():
        print(f"Error: The directory '{base_dir}' does not exist.")
        return

    # Iterate through all subdirectories (which correspond to model_names)
    for model_dir in base_dir.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            csv_filename = f"{model_name}_classification_report.csv"
            csv_path = model_dir / csv_filename

            if csv_path.exists():
                # Read CSV, assuming the first column contains 'precision', 'recall', etc.
                try:
                    df = pd.read_csv(csv_path, index_col=0)
                    report_dict[model_name] = df
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")
            else:
                print(f"Skipping {model_name}: {csv_filename} not found.")

    if not report_dict:
        print("No CSV reports were found. Check your directory structure.")
        return None

    # Concatenate all individual DataFrames into a joint table
    # This creates a MultiIndex with Model Name as the first level and the Metric as the second
    joint_df = pd.concat(report_dict, names=["Model", "Metric"])

    return joint_df


if __name__ == "__main__":
    # Example usage:
    # Set your experiment directory path here
    EXPERIMENT_DIR = "29_gut_compartment_full"

    print(f"Aggregating results from {EXPERIMENT_DIR}...\n")
    joint_table = generate_joint_table(EXPERIMENT_DIR)

    if joint_table is not None:
        print("Joint Classification Report:")
        print("-" * 50)
        print(joint_table)
        print("-" * 50)

        # Optional: Save the joint table to a new CSV or LaTeX file
        output_csv = Path(EXPERIMENT_DIR) / "joint_classification_report.csv"
        joint_table.to_csv(output_csv)
        print(f"\nSaved joint table to: {output_csv}")

        # Optional: Output to LaTeX
        output_tex = Path(EXPERIMENT_DIR) / "joint_classification_report.tex"
        joint_table.to_latex(output_tex)
        print(f"Saved joint LaTeX table to: {output_tex}")
