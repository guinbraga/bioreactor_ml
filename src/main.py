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

# ====== Prompt the user to setup the experiment to be run ========= #

available_components = fetch_pipeline_components()

groups = None
splitter = questionary.select(
    "What splitting method to use for train-validation-test?",
    choices=available_components["cross_validators"],
).ask()
if splitter == "Leave One Group Out":
    groups = questionary.select(
        "What column represents the groups?", choices=available_columns
    ).ask()

selected_models = questionary.checkbox(
    "Select models to evaluate:",
    choices=available_components["models"],
    validate=lambda x: len(x) > 0,
).ask()

models_to_evaluate = {}
for model in selected_models:
    # First we choose scaling sequences to evaluate
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
            console.print(f"[yellow]Current scaling sequence:[/yellow]")
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


results_dir = questionary.path("Which directory to save results?").ask()

status_spinner = console.status("[bold green]Initializing experiment...[/bold green]")


def on_model_begin(model_name: str):
    console.print(f"[green]Starting Evaluation for {model_name}...[/green]")


def on_split_begin(i, n_splits):
    console.print(f"Starting pipeline for split {i + 1} out of {n_splits}")


def on_persist(model_name: str):
    status_spinner.start()
    status_spinner.update(
        f"[bold cyan]Saving results and rendering plots for {model_name}...[/bold cyan]"
    )


def on_complete(model_name: str):
    status_spinner.stop()
    console.print(f"[green]Finished Evaluation for {model_name}![/green]")

for target_column in target_columns:
    evaluate_experiment(
        data_manager=data_manager,
        target_column=target_column,
        selected_models=models_to_evaluate,
        results_dir=results_dir,
        on_model_begin=on_model_begin,
        on_split_begin=on_split_begin,
        on_complete=on_complete,
        on_persist=on_persist,
    )
