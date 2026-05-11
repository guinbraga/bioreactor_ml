import optuna
import numpy as np
from optuna.trial import Trial
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import make_scorer, log_loss
from pandas import DataFrame, Series
from src.pipeline_factory import PipelineFactory


class Objective:
    def __init__(
        self,
        X_train: DataFrame,
        y_train: DataFrame | Series,
        model_name: str,
        cv: int | object = LeaveOneOut(),
        scoring: int | object = "neg_log_loss",
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.model_name = model_name
        self.cv = cv
        self.scoring = scoring
        self.pipeline_factory = PipelineFactory()

    # __call__ function is executed when the object is instanciated
    def __call__(self, trial: Trial) -> float:
        pipeline = self.pipeline_factory.build_pipeline(trial, self.model_name)

        classes = np.unique(
            self.y_train
        )  # needed for log_loss with only one value, as in LOO

        custom_scorer = make_scorer(
            log_loss,
            greater_is_better=False,
            response_method="predict_proba",
            labels=classes,
        )

        scores = cross_val_score(
            pipeline,
            self.X_train,
            self.y_train,
            cv=self.cv,
            scoring=custom_scorer,
            n_jobs=-1,
        )

        return np.mean(scores)


class PipelineOptimizer:
    def __init__(self, n_trials: int = 50, direction: str = "maximize") -> None:
        self.n_trials = n_trials
        self.direction = direction

    def optimize_pipeline(
        self,
        X_train: DataFrame,
        y_train: DataFrame | Series,
        model_name: str,
        cv: int | object,
        scoring: str | object,
    ) -> Pipeline:
        study = optuna.create_study(
            direction="maximize"
        )  # might need to change direction depending on cv evaluation metric
        objective = Objective(X_train, y_train, model_name, cv, scoring)

        print("Running Optuna for finding best hyperparameters and pipeline...")
        study.optimize(objective, n_trials=self.n_trials)

        print(f"Best cross-validation score: {study.best_value}")

        best_params_dict = study.best_params
        # FixedTrial allows us to create a specific pipeline from the best params
        fixed_trial = optuna.trial.FixedTrial(best_params_dict)

        best_pipeline = PipelineFactory().build_pipeline(fixed_trial, model_name)
        best_pipeline.fit(X_train, y_train)

        return best_pipeline
