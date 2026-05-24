from pathlib import Path
from typing import Any, Callable, TypedDict
from data_manager import DataManager
from experiment_evaluator import ExperimentEvaluator
from pipeline_factory import PipelineFactory


class ModelConfig(TypedDict):
    selected_scaler_sequences: list[tuple[str, ...]]
    selected_selectors: list[str]
    cv: str
    groups: Any


def fetch_metadata_columns(metadata_file_path: Path | str) -> list:
    return DataManager.get_metadata_columns(metadata_file_path)


def fetch_pipeline_components() -> dict[str, list[str]]:
    pipeline_factory = PipelineFactory()
    available_models = pipeline_factory.get_available_models()
    available_scalers = pipeline_factory.get_available_scalers()
    available_selectors = pipeline_factory.get_available_selectors()
    available_cross_validators = pipeline_factory.get_available_cv()
    payload = {
        "models": available_models,
        "scalers": available_scalers,
        "selectors": available_selectors,
        "cross_validators": available_cross_validators,
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
    selected_models: dict[str, ModelConfig],
    results_dir: str,
    on_complete: Callable | None = None,
    on_begin: Callable | None = None,
):
    X, y = data_manager.get_X_y()
    pipeline_factory = PipelineFactory()

    for model_name, model_config in selected_models.items():
        if on_begin:
            on_begin(model_name)
        cv_registry = pipeline_factory.cv_registry
        cv_obj = cv_registry[model_config["cv"]]
        groups = data_manager.get_groups(model_config["groups"])
        evaluator = ExperimentEvaluator(
            model_name=model_name,
            selected_scaler_sequences=model_config["selected_scaler_sequences"],
            selected_selectors=model_config["selected_selectors"],
            cv=cv_obj,
            groups=groups,
            results_dir=results_dir,
        )
        evaluator.evaluate(X, y)
        if on_complete:
            on_complete(model_name)
