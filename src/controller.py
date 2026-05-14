from pathlib import Path
from typing import Callable
from data_manager import DataManager
from experiment_evaluator import ExperimentEvaluator
from pipeline_factory import PipelineFactory


def fetch_metadata_columns(metadata_file_path: Path | str) -> list:
    return DataManager.get_metadata_columns(metadata_file_path)


def fetch_pipeline_components() -> dict[str, list[str]]:
    pipeline_factory = PipelineFactory()
    available_models = pipeline_factory.get_available_models()
    available_scalers = pipeline_factory.get_available_scalers()
    available_selectors = pipeline_factory.get_available_selectors()
    payload = {
        "models": available_models,
        "scalers": available_scalers,
        "selectors": available_selectors,
    }
    return payload


def setup_data(
    genomic_file_path: Path | str,
    metadata_file_path: Path | str,
    target_column: str,
    metadata_file_index: str,
) -> tuple[dict, DataManager]:
    data_manager = DataManager(
        genomic_file_path=genomic_file_path,
        metadata_file_path=metadata_file_path,
        target_column=target_column,
        metadata_file_index=metadata_file_index,
    )
    merge_results = data_manager.merge_datasets()

    # we return the data manager instance so we don't have to reinstantiate it later.
    # when we move to a web app, we'll have to refactor this, probably with temp files
    return merge_results, data_manager


def evaluate_experiment(
    data_manager: DataManager,
    selected_models: dict[str, dict["str", list[str]]],
    results_directory: str,
    on_complete: Callable | None = None,
    on_begin: Callable | None = None,
):
    X, y = data_manager.get_X_y()

    for model_name, model_config in selected_models.items():
        if on_begin:
            on_begin(model_name)
        evaluator = ExperimentEvaluator(
            model_name=model_name,
            selected_scalers=model_config["selected_scalers"],
            selected_selectors=model_config["selected_selectors"],
            results_directory=results_directory,
        )
        evaluator.evaluate(X, y)
        if on_complete:
            on_complete(model_name)
