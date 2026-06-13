from typing import Callable
import numpy as np
from pandas import DataFrame, Series
from sklearn.metrics import log_loss
from sklearn.model_selection import BaseCrossValidator, LeaveOneGroupOut, LeaveOneOut

from correlation_cluster_selector import CorrelationClusterSelector
from pipeline_optimizer import PipelineOptimizer
from results_manager import ResultsDataManager, ResultsPlotManager


class ExperimentEvaluator:
    def __init__(
        self,
        model_name: str,
        selected_scaler_sequences: list[tuple[str, ...]],
        selected_selectors: list[str],
        cv: BaseCrossValidator = LeaveOneOut(),
        groups: Series | None = None,
        on_split_begin: Callable | None = None,
    ) -> None:
        self.model_name = model_name
        self.cv = cv
        self.selected_scaler_sequences = selected_scaler_sequences
        self.selected_selectors = selected_selectors
        self.groups = groups
        self.on_split_begin = on_split_begin
        self.cv_registry: dict[str, BaseCrossValidator] = {
            "Leave One Out": LeaveOneOut(),
            "Leave One Group Out": LeaveOneGroupOut(),
        }

    def evaluate(self, X: DataFrame, y: Series) -> dict:
        target_col = str(y.name)

        data_manager = ResultsDataManager(self.model_name)
        plot_manager = ResultsPlotManager(self.model_name, target_col=target_col)

        cluster_selector = CorrelationClusterSelector(threshold=0.95, linkage="ward")
        cluster_selector.set_output(transform="pandas")

        X_filtered = cluster_selector.fit_transform(X)

        cv = self.cv
        splits = cv.split(X_filtered, y, groups=self.groups)
        waterfall_plots = {}

        for i, (train_index, test_index) in enumerate(splits):

            if self.on_split_begin:
                n_splits = cv.get_n_splits(X_filtered, y, self.groups)
                self.on_split_begin(i, n_splits)

            X_train = X_filtered.iloc[train_index]
            y_train = y.iloc[train_index]
            X_test = X_filtered.iloc[test_index]
            y_test = y.iloc[test_index]
            groups_train = None

            if self.groups is not None and not isinstance(self.groups, Series):
                raise TypeError(
                    "Groups is not Series or None. Perhaps you passed a DataFrame?"
                )

            if type(self.groups) is Series:
                groups_train = self.groups.iloc[train_index]

            pipeline_optimizer = PipelineOptimizer()
            pipeline, study = pipeline_optimizer.optimize_pipeline(
                X_train=X_train,
                y_train=y_train,
                model_name=self.model_name,
                cv=self.cv,
                groups=groups_train,
                selected_scaler_sequences=self.selected_scaler_sequences,
                selected_selectors=self.selected_selectors,
            )

            split = i + 1
            test_sample = X_test.index[0]
            y_true = y_test.iloc[0]
            predictions = pipeline.predict(X_test)
            predictions_proba = pipeline.predict_proba(X_test)
            classes = np.unique(y)
            log_loss_score = log_loss(y_test, predictions_proba, labels=classes)
            best_params_dict = study.best_params

            data_manager.record_split_metrics(
                {
                    "Split": split,
                    "Test Sample": test_sample,
                    "True Class": y_true,
                    "Predicted Class": predictions[0],
                    "Predicted Probabilities": predictions_proba[0].tolist(),
                    "Log Loss Score": log_loss_score,
                    "Best Pipeline Params": str(best_params_dict),
                }
            )
            data_manager.record_split_coefs(pipeline, test_sample)

            explanation = data_manager.compute_and_record_shap(
                pipeline=pipeline, X_train=X_train, X_test=X_test
            )

            waterfall_plot = plot_manager.generate_shap_waterfall(
                explanation, test_sample
            )
            waterfall_plots[test_sample] = waterfall_plot

        beeswarm_plot = plot_manager.generate_bee_swarm_plot(
            data_manager.all_shap_explanations
        )
        coefficients_plot = plot_manager.generate_coef_plot(
            data_manager.coeff_results, n_samples=15
        )

        results_payload = {
            "plots": {
                "waterfall_plots": waterfall_plots,
                "beeswarm_plot": beeswarm_plot,
                "coefficients_plot": coefficients_plot,
            },
            "data_manager": data_manager,
            "cluster_selector": cluster_selector,
        }

        return results_payload

    def get_available_cv(self) -> list[str]:
        available_cv = [cv for cv in self.cv_registry.keys()]
        return available_cv
