import questionary
from rich.console import Console
from controller import (
    evaluate_experiment,
    fetch_metadata_columns,
    fetch_pipeline_components,
    setup_data,
)

console = Console()

# ======= Prompt the user to setup the experiment data files ======== #
genomic_file_path = questionary.path("Specify path to genomic data:").ask()
metadata_file_path = questionary.path("Specify path to experiment metadata:").ask()
available_columns = fetch_metadata_columns(metadata_file_path)
target_column = questionary.select(
    "Select target column to be predicted from metadata file:",
    choices=available_columns,
).ask()
available_columns.remove(target_column)
metadata_file_index = questionary.select(
    "Select index column from metadata file:", choices=available_columns
).ask()

console.print("[yellow]Validating and merging datasets...[/yellow]")

merge_results, data_manager = setup_data(
    genomic_file_path=genomic_file_path,
    metadata_file_path=metadata_file_path,
    target_column=target_column,
    metadata_file_index=metadata_file_index,
)

console.print("\n[bold]Presenting Data Summary on Merge:[/bold]")
console.print(f"# Genomic samples found: {merge_results['genomic_n_samples']}")
console.print(f"# Metadata samples found: {merge_results['metadata_n_samples']}")
console.print(f"# Merged samples: {merge_results['merged_n_samples']}")
console.print(f"Excluded Samples: {merge_results['excluded_samples']}")

# ====== Prompt the user to setup the experiment to be run ========= #

available_components = fetch_pipeline_components()

selected_models = questionary.checkbox(
    "Select models to evaluate:", choices=available_components["models"]
).ask()

models_to_evaluate = {}
for model in selected_models:
    selected_scalers = questionary.checkbox(
        f"[{model}] Select feature scaling techniques to evaluate:",
        choices=available_components["scalers"],
        validate=lambda x: len(x) > 0
    ).ask()

    selected_selectors = questionary.checkbox(
        f"[{model}] Select feature selection techniques to evaluate:",
        choices=available_components["selectors"],
        validate=lambda x: len(x) > 0,
    ).ask()
    models_to_evaluate[model] = {
        "selected_scalers": selected_scalers,
        "selected_selectors": selected_selectors,
    }


results_directory = questionary.path("Which directory to save results?").ask()

def on_begin(model_name: str):
    console.print(f"[green]Starting Evaluation for {model_name}...[/green]")
def on_complete(model_name: str):
    console.print(f"[green]Finished Evaluation for {model_name}![/green]")

evaluate_experiment(
    data_manager=data_manager,
    selected_models=models_to_evaluate,
    results_directory=results_directory,
    on_begin=on_begin,
    on_complete=on_complete
)
