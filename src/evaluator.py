from pandas import DataFrame, Series
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import log_loss
from src.pipeline_optimizer import PipelineOptimizer


class ExperimentEvaluator:
    def __init__(
        self, model_name: str, cv: int | object, results_directory: str
    ) -> None:
        self.model_name = model_name
        self.cv = cv
        self.results_directory = results_directory

    def evaluate(self, X: DataFrame, y: Series):
        loo = LeaveOneOut()
        splits = loo.split(X, y)
        n_samples = len(X)
        classes = np.unique(y)
        log_loss_scores = np.zeros(n_samples)
        for i, (train_index, test_index) in enumerate(splits):
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_test = X.iloc[test_index]
            y_test = y.iloc[test_index]

            pipeline_optimizer = PipelineOptimizer()
            pipeline = pipeline_optimizer.optimize_pipeline(
                X_train=X_train,
                y_train=y_train,
                model_name=self.model_name,
                cv=self.cv,
                scoring="neg_log_loss",
            )

            predictions = pipeline.predict_proba(X_test)
            log_loss_scores[i] = log_loss(y_test, predictions, labels=classes)

        np.savetxt(
            self.results_directory + "/" + self.model_name + "all_tests.csv",
            log_loss_scores,
            delimiter=",",
        )
        np.savetxt(
            self.results_directory + "/" + self.model_name + "_mean_score.txt",
            np.mean(log_loss_scores),
        )
