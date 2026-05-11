from sklearn.pipeline import Pipeline
from optuna.trial import FixedTrial, Trial
from src.pipeline_components.selectors import BaseSelectorStrategy, ElasticNetSelector
from src.pipeline_components.models import (
    BaseModelStrategy,
    RandomForestStrategy,
)
from src.pipeline_components.scalers import BaseScalerStrategy, CLRTransformer
from sklearn.pipeline import make_pipeline


class PipelineFactory:
    def __init__(self):
        self.model_registry: dict[str, BaseModelStrategy] = {
            "RandomForest": RandomForestStrategy()
        }

        self.scaler_registry: dict[str, BaseScalerStrategy] = {
            "CLRTransformer": CLRTransformer()
        }

        self.selector_registry: dict[str, BaseSelectorStrategy] = {
            "ElasticNetSelector": ElasticNetSelector()
        }

    def build_pipeline(
        self, trial: Trial | FixedTrial, target_model_name: str
    ) -> Pipeline:
        # ==== Optuna selects the best scalers and selector from registry ==== #
        scalers_list = list(self.scaler_registry.keys())
        selectors_list = list(self.selector_registry.keys())

        chosen_scaler = trial.suggest_categorical("scaler", scalers_list)
        chosen_selector = trial.suggest_categorical("feature_selector", selectors_list)

        # ==== We instantiate their strategies and build them ==== #
        scaler_strategy = self.scaler_registry[chosen_scaler]
        selector_strategy = self.selector_registry[chosen_selector]
        estimator_strategy = self.model_registry[target_model_name]

        scaler = scaler_strategy.create_scaler(trial)
        selector = selector_strategy.create_selector(trial)
        estimator = estimator_strategy.create_model(trial)

        return make_pipeline(scaler, selector, estimator)
