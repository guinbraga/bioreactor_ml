from pandas import DataFrame, Series
import numpy as np
from sklearn.model_selection import BaseCrossValidator, LeaveOneOut, LeaveOneGroupOut
from sklearn.metrics import log_loss
from pipeline_optimizer import PipelineOptimizer
from results_manager import ResultsManager


class ExperimentEvaluator:
    def __init__(
        self,
        model_name: str,
        results_dir: str,
        selected_scaler_sequences: list[tuple[str, ...]],
        selected_selectors: list[str],
        cv: BaseCrossValidator = LeaveOneOut(),
        groups: Series | None = None,
    ) -> None:
        self.model_name = model_name
        self.cv = cv
        self.results_dir = results_dir
        self.selected_scaler_sequences = selected_scaler_sequences
        self.selected_selectors = selected_selectors
        self.groups = groups
        self.cv_registry: dict[str, BaseCrossValidator] = {
            "Leave One Out": LeaveOneOut(),
            "Leave One Group Out": LeaveOneGroupOut(),
        }

    def evaluate(self, X: DataFrame, y: Series):
        results_manager = ResultsManager(
            model_name=self.model_name,
            results_dir=f"{self.results_dir}/{self.model_name}",
            target_col=str(y.name),
        )

        cv = self.cv
        splits = cv.split(X, y, groups=self.groups)
        for i, (train_index, test_index) in enumerate(splits):
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_test = X.iloc[test_index]
            y_test = y.iloc[test_index]
            groups_train = None

            if self.groups is not None and not isinstance(self.groups, Series):
                raise TypeError(
                    "Groups is not Series or None. Perhaps you passed a DataFrame?"
                )

            if type(self.groups) is Series:
                groups_train = self.groups.iloc[train_index]

            print(
                f"Starting pipeline for split {i + 1} out of {cv.get_n_splits(X, y, self.groups)}"
            )

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

            results_manager.record_split_metrics(
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

            results_manager.generate_shap_waterfall(
                pipeline=pipeline, X_train=X_train, X_test=X_test
            )
            results_manager.record_split_coefs(pipeline, test_sample)

        results_manager.generate_bee_swarm_plot()
        results_manager.save_shap_objects()
        results_manager.save_shap_dataframes()
        results_manager.generate_coef_plot(15)
        results_manager.save_final_csv()
        results_manager.record_experiment_setup(
            self.selected_scaler_sequences,
            self.selected_selectors,
            self.cv,
            self.groups,
        )

    def get_available_cv(self) -> list[str]:
        available_cv = [cv for cv in self.cv_registry.keys()]
        return available_cv
