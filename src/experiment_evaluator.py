from pandas import DataFrame, Series
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import log_loss
from pipeline_optimizer import PipelineOptimizer
from results_manager import ResultsManager


class ExperimentEvaluator:
    def __init__(
        self,
        model_name: str,
        results_dir: str,
        selected_scalers: list[str],
        selected_selectors: list[str],
        cv: int | object = LeaveOneOut(),
    ) -> None:
        self.model_name = model_name
        self.cv = cv
        self.results_dir = results_dir
        self.selected_scalers = selected_scalers
        self.selected_selectors = selected_selectors

    def evaluate(self, X: DataFrame, y: Series):
        n_samples = len(X)
        classes = np.unique(y)

        loo = LeaveOneOut()
        results_manager = ResultsManager(
            model_name=self.model_name,
            results_dir=f"{self.results_dir}/{self.model_name}",
        )
        splits = loo.split(X, y)
        for i, (train_index, test_index) in enumerate(splits):
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_test = X.iloc[test_index]
            y_test = y.iloc[test_index]

            print(f"Starting pipeline for split {i + 1} out of {n_samples}")

            pipeline_optimizer = PipelineOptimizer()
            pipeline, study = pipeline_optimizer.optimize_pipeline(
                X_train=X_train,
                y_train=y_train,
                model_name=self.model_name,
                cv=self.cv,
                selected_scalers=self.selected_scalers,
                selected_selectors=self.selected_selectors,
            )

            split = i + 1
            test_sample = X_test.index[0]
            y_true = y_test.iloc[0]
            predictions = pipeline.predict(X_test)
            predictions_proba = pipeline.predict_proba(X_test)
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

            results_manager.process_split_data(
                pipeline=pipeline, X_train=X_train, X_test=X_test
            )

        results_manager.generate_bee_swarm_plot()
        results_manager.save_shap_objects()
        results_manager.save_shap_dataframes()
        results_manager.save_final_csv()
