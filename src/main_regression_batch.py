"""Batch regression pipeline runner.

Runs the regression pipeline for every combination of:
- User-selected target columns
- Feature subsets from each classification experiment × model
  (from results/38_cluster_importances/)
- All-features baseline (no feature selection)

Results are saved to results/batch_regression/ with a directory hierarchy
that encodes: {target}/{experiment}/{model}/
"""

from pathlib import Path

import questionary
from rich.console import Console

from controller import (
    evaluate_experiment_regression,
    fetch_metadata_columns,
    fetch_pipeline_components,
    setup_data,
)

console = Console()

# ====== Prompt the user to setup the experiment data files ====== #

genomic_file_path = questionary.path("Specify path to genomic data:").ask()
metadata_file_path = questionary.path("Specify path to experiment metadata:").ask()
available_columns = fetch_metadata_columns(metadata_file_path)
target_columns = questionary.checkbox(
    "Select target columns to be predicted from metadata file:",
    choices=available_columns,
).ask()
for column in target_columns:
    available_columns.remove(column)
metadata_file_index = questionary.select(
    "Select index column from metadata file:", choices=available_columns
).ask()
available_columns.remove(metadata_file_index)

console.print("[yellow]Validating and merging datasets...[/yellow]")

merge_results, data_manager = setup_data(
    genomic_file_path=genomic_file_path,
    metadata_file_path=metadata_file_path,
    target_columns=target_columns,
    metadata_file_index=metadata_file_index,
)

console.print("\n[bold]Presenting Data Summary on Merge:[/bold]")
console.print(f"# Genomic samples found: {merge_results['genomic_n_samples']}")
console.print(f"# Metadata samples found: {merge_results['metadata_n_samples']}")
console.print(f"# Merged samples: {merge_results['merged_n_samples']}")
console.print(f"Excluded Samples: {merge_results['excluded_samples']}")

# ====== Prompt the user to setup the experiment to be run ====== #

available_components = fetch_pipeline_components()

groups = None
splitter = questionary.select(
    "What splitting method to use for cross-validation?",
    choices=available_components["cross_validators"],
).ask()
if splitter == "Leave One Group Out":
    groups = questionary.select(
        "What column represents the groups?", choices=available_columns
    ).ask()

regression_models = [m for m in available_components["models"] if "Regressor" in m]
selected_models = questionary.checkbox(
    "Select regression models to evaluate:",
    choices=regression_models,
    validate=lambda x: len(x) > 0,
).ask()

models_to_evaluate = {}
for model in selected_models:
    adding_sequences = True
    selected_scaler_sequences = []
    erase_step_msg = "[Erase] -- erase last step --"
    done_msg = "[Done] -- finish this sequence --"
    scaling_choices = available_components["scalers"] + [erase_step_msg, done_msg]
    while adding_sequences:
        scaler_sequence = []
        step = 1
        while True:
            current_sequence_repr = " -> ".join(scaler_sequence)
            console.print("[yellow]Current scaling sequence:[/yellow]")
            console.print(current_sequence_repr)
            scaling_step = questionary.select(
                f"\n[{model}] Select step {step} of the scaling sequence: ",
                choices=scaling_choices,
            ).ask()
            if scaling_step == done_msg or None:
                break
            if scaling_step == erase_step_msg:
                scaler_sequence.pop()
                step -= 1
            else:
                scaler_sequence.append(scaling_step)
                step += 1
        selected_scaler_sequences.append(tuple(scaler_sequence))
        adding_sequences = questionary.confirm("Add another sequence?").ask()

    selected_selectors = questionary.checkbox(
        f"[{model}] Select feature selection techniques to evaluate:",
        choices=available_components["selectors"],
        validate=lambda x: len(x) > 0,
    ).ask()
    models_to_evaluate[model] = {
        "selected_scaler_sequences": selected_scaler_sequences,
        "selected_selectors": selected_selectors,
        "cv": splitter,
        "groups": groups,
    }

# ====== Discover feature subsets ====== #

BASE_FEATURE_DIR = Path("results/38_cluster_importances")
BATCH_RESULTS_DIR = "results/40_batch_regression"

feature_subsets: list[dict[str, str]] = []
for experiment_dir in sorted(BASE_FEATURE_DIR.iterdir()):
    if not experiment_dir.is_dir():
        continue
    experiment_name = experiment_dir.name
    for model_dir in sorted(experiment_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        top_features_csv = model_dir / f"{model_name}_top_features.csv"
        if top_features_csv.exists():
            feature_subsets.append(
                {
                    "experiment": experiment_name,
                    "model": model_name,
                    "csv_path": str(top_features_csv),
                }
            )

console.print(
    f"\n[bold]Discovered {len(feature_subsets)} feature subsets "
    f"across {len(set(s['experiment'] for s in feature_subsets))} experiments.[/bold]"
)
for subset in feature_subsets:
    console.print(f"  {subset['experiment']} / {subset['model']}")

# ====== Run batch evaluation ====== #

status_spinner = console.status("[bold green]Initializing experiment...[/bold green]")


def on_model_begin(model_name: str) -> None:
    console.print(f"[green]Starting Evaluation for {model_name}...[/green]")


def on_split_begin(i: int, n_splits: int) -> None:
    console.print(f"Starting pipeline for split {i + 1} out of {n_splits}")


def on_persist(model_name: str) -> None:
    status_spinner.start()
    status_spinner.update(
        f"[bold cyan]Saving results and rendering plots for {model_name}...[/bold cyan]"
    )


def on_complete(model_name: str) -> None:
    status_spinner.stop()
    console.print(f"[green]Finished Evaluation for {model_name}![/green]")


total_runs = len(target_columns) * (1 + len(feature_subsets))
run_idx = 0

for target_column in target_columns:
    # ---- All-features baseline ----
    # controller.py:178 appends "/{target_column}/{model_name}", so
    # results_dir encodes experiment/source without duplicating the target.
    run_idx += 1
    console.print(
        f"\n[bold cyan]({run_idx}/{total_runs}) Target: {target_column} "
        f"— all features[/bold cyan]"
    )
    evaluate_experiment_regression(
        data_manager=data_manager,
        target_column=target_column,
        selected_models=models_to_evaluate,
        results_dir=f"{BATCH_RESULTS_DIR}/all_features",
        selected_features_csv=None,
        on_model_begin=on_model_begin,
        on_split_begin=on_split_begin,
        on_complete=on_complete,
        on_persist=on_persist,
    )

    # ---- Feature subsets ----
    for subset in feature_subsets:
        run_idx += 1
        label = f"{subset['experiment']} / {subset['model']}"
        console.print(
            f"\n[bold cyan]({run_idx}/{total_runs}) Target: {target_column} "
            f"— features from {label}[/bold cyan]"
        )
        evaluate_experiment_regression(
            data_manager=data_manager,
            target_column=target_column,
            selected_models=models_to_evaluate,
            results_dir=(
                f"{BATCH_RESULTS_DIR}/{subset['experiment']}/{subset['model']}"
            ),
            selected_features_csv=subset["csv_path"],
            on_model_begin=on_model_begin,
            on_split_begin=on_split_begin,
            on_complete=on_complete,
            on_persist=on_persist,
        )

console.print("\n[bold green]Batch regression complete![/bold green]")
