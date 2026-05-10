import optuna
import numpy as np
from optuna.trial import Trial
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.scoring import log_loss
from pandas import DataFrame, Series
from src.pipeline_factory import PipelineFactory


class objective:
    def __init__(
        self,
        X_train: DataFrame,
        y_train: DataFrame | Series,
        model_name,
        cv,
        scoring="neg_log_loss",
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.model_name = model_name
        self.cv = cv
        self.scoring = scoring
        self.pipeline_factory = PipelineFactory()

    def __call__(self, trial: Trial) -> float:
        pipeline = self.pipeline_factory.build_pipeline(trial, self.model_name)

        scores = cross_val_score(
            pipeline,
            self.X_train,
            self.y_train,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
        )

        return np.mean(scores)


class HyperparameterOptimizer:
    def __init__(self, n_trials: int = 50, direction: str = "maximize") -> None:
        self.n_trials = n_trials
        self.direction = direction

    def optimize_pipeline(
        self, X_train: DataFrame, y_train: DataFrame | Series, model_name: str
    ) -> Pipeline:

        study = optuna.create_study(direction="minimize")
        objective = Objective(trial, X_train, y_train, )
        study.optimize(objective, n_trials=100)
