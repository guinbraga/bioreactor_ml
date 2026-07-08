import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Callable, TypedDict
from data_manager import DataManager
from classification_evaluator import ClassificationEvaluator
from persistence_manager import DataPersistenceManager, PlotPersistenceManager
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
    on_model_begin: Callable | None = None,
    on_split_begin: Callable | None = None,
    on_persist: Callable | None = None,
    persist_to_disk: bool = True,
):
    X, y = data_manager.get_X_y()
    pipeline_factory = PipelineFactory()

    for model_name, model_config in selected_models.items():
        if on_model_begin:
            on_model_begin(model_name)
        cv_registry = pipeline_factory.cv_registry
        cv_obj = cv_registry[model_config["cv"]]
        groups = data_manager.get_groups(model_config["groups"])
        evaluator = ClassificationEvaluator(
            model_name=model_name,
            selected_scaler_sequences=model_config["selected_scaler_sequences"],
            selected_selectors=model_config["selected_selectors"],
            cv=cv_obj,
            groups=groups,
            on_split_begin=on_split_begin,
        )
        results_payload = evaluator.evaluate(X, y)

        if persist_to_disk:
            if on_persist:
                on_persist(model_name)
            model_results_dir = f"{results_dir}/{model_name}"

            data_persister = DataPersistenceManager(
                results_data_manager=results_payload["data_manager"],
                results_dir=model_results_dir,
                model_name=model_name,
            )
            data_persister.save_final_csv()
            data_persister.save_clusters(results_payload["cluster_selector"])
            data_persister.save_shap_dataframes()
            data_persister.save_shap_objects()
            data_persister.save_classification_report()
            data_persister.save_top_feat_importances(results_payload["top_features"])
            data_persister.save_experiment_setup(
                selected_scaler_sequences=model_config["selected_scaler_sequences"],
                selected_selectors=model_config["selected_selectors"],
                cv=cv_obj,
                groups=groups,
                target_col=str(y.name),
            )

            plots_persister = PlotPersistenceManager(model_results_dir, model_name)
            plots = results_payload["plots"]
            for sample_id, fig in plots["waterfall_plots"].items():
                if fig:
                    plots_persister.persist_shap_waterfall(fig, sample_id)
                    plt.close(fig)

            if plots["beeswarm_plot"]:
                plots_persister.persist_beeswarm_plot(plots["beeswarm_plot"])
                plt.close(plots["beeswarm_plot"])

            # if plots["coefficients_plot"]:
            #     plots_persister.persist_coef_plot(plots["coefficients_plot"])
            #     plt.close(plots["coefficients_plot"])

            if plots["confusion_matrix"]:
                plots_persister.persist_confusion_matrix(plots["confusion_matrix"])
                plt.close(plots["confusion_matrix"])

            if plots["cluster_importances_plot"]:
                plots_persister.persist_cluster_importance_plot(
                    plots["cluster_importances_plot"]
                )
                plt.close(plots["cluster_importances_plot"])

        if on_complete:
            on_complete(model_name)
