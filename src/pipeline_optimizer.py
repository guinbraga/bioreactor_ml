import optuna
import warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np
from optuna.study import Study
from optuna.trial import Trial
from sklearn.pipeline import Pipeline
from sklearn.model_selection import BaseCrossValidator, cross_val_score, LeaveOneOut
from sklearn.metrics import make_scorer, log_loss
from pandas import DataFrame, Series
from pipeline_factory import PipelineFactory


class Objective:
    """This class presents the objective function taken as argument by optuna
    to evaluate the best hyperparameters. More specifically, the function is
    defined in its __call__ dunder method.
    """

    def __init__(
        self,
        X_train: DataFrame,
        y_train: DataFrame | Series,
        model_name: str,
        selected_scaler_sequences: list[tuple[str, ...]],
        selected_selectors: list[str],
        scoring: int | str | object = None,
        cv: BaseCrossValidator = LeaveOneOut(),
        groups: Series | None = None,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.model_name = model_name
        self.selected_selectors = selected_selectors
        self.selected_scaler_sequences = selected_scaler_sequences
        self.cv = cv
        self.groups = groups
        self.scoring = scoring
        self.pipeline_factory = PipelineFactory()

    # __call__ function is executed when the object is instantiated
    def __call__(self, trial: Trial) -> float:
        pipeline = self.pipeline_factory.build_pipeline(
            trial,
            self.model_name,
            self.selected_scaler_sequences,
            self.selected_selectors,
        )

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
            groups=self.groups,
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
        cv: BaseCrossValidator,
        selected_scaler_sequences: list[tuple[str, ...]],
        selected_selectors: list[str],
        groups: Series | None = None,
        scoring: str | object = None,
        direction: str = "maximize",
    ) -> tuple[Pipeline, Study]:  # returns both pipeline and study for record purposes
        sampler = optuna.samplers.TPESampler(seed=47)
        study = optuna.create_study(
            direction=direction,
            sampler=sampler
        )  # might need to change direction depending on cv evaluation metric
        objective = Objective(
            X_train=X_train,
            y_train=y_train,
            model_name=model_name,
            selected_scaler_sequences=selected_scaler_sequences,
            selected_selectors=selected_selectors,
            cv=cv,
            groups=groups,
            scoring=scoring,
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)  # to not print each trial
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        best_params_dict = study.best_params
        # FixedTrial allows us to create a specific pipeline from the best params
        fixed_trial = optuna.trial.FixedTrial(best_params_dict)

        best_pipeline = PipelineFactory().build_pipeline(
            fixed_trial, model_name, selected_scaler_sequences, selected_selectors
        )
        best_pipeline.fit(X_train, y_train)

        return best_pipeline, study
