from typing import Protocol
from optuna import Trial
from optuna.trial import FixedTrial
from sklearn.base import TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.pipeline import FunctionTransformer


# ==== Base Selector Strategy ====
class BaseSelectorStrategy(Protocol):
    def create_selector(
        self, trial: Trial | FixedTrial, is_regression: bool = False
    ) -> object:
        pass


# ==== Elastic Net Selector Implementation ====
class ElasticNetSelector:
    def create_selector(
        self, trial: Trial | FixedTrial, is_regression: bool = False
    ) -> object:
        selector_C = trial.suggest_float("selector_C", 1e-3, 10.0, log=True)
        selector_l1_ratio = trial.suggest_float("selector_l1_ratio", 0.0, 1.0)
        selector_threshold = trial.suggest_categorical(
            "selector_threshold", ["mean", "median"]
        )

        if is_regression:
            # alpha ≈ 1/C: maps classification-strength C to regression alpha
            selector_model = ElasticNet(
                alpha=1.0 / selector_C,
                l1_ratio=selector_l1_ratio,
                random_state=47,
                max_iter=2000,
            )
        else:
            selector_model = LogisticRegression(
                solver="saga",
                C=selector_C,
                l1_ratio=selector_l1_ratio,
                tol=1e-3,
                random_state=47,
                max_iter=2000,
            )

        feature_selector = SelectFromModel(
            estimator=selector_model, threshold=selector_threshold
        )

        return feature_selector


class PassthroughSelector:
    def create_selector(
        self, trial: Trial | FixedTrial, is_regression: bool = False
    ) -> TransformerMixin:
        return FunctionTransformer(func=None, feature_names_out="one-to-one")
