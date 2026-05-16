from sklearn.pipeline import Pipeline
from optuna.trial import FixedTrial, Trial
from pipeline_components.models.elastic_net import ElasticNetStrategy
from pipeline_components.models.l1_logreg import L1LogisticRegressionStrategy
from pipeline_components.selectors import (
    BaseSelectorStrategy,
    ElasticNetSelector,
    PassthroughSelector,
)
from pipeline_components.models import (
    BaseModelStrategy,
    RandomForestStrategy,
)
from pipeline_components.scalers import (
    BaseScalerStrategy,
    CLRTransformer,
    PassthroughScaler,
)
from sklearn.pipeline import make_pipeline


class PipelineFactory:
    def __init__(self):
        self.model_registry: dict[str, BaseModelStrategy] = {
            "Random Forest": RandomForestStrategy(),
            "L2 Logistic Regression": L1LogisticRegressionStrategy(),
            "Elastic Net": ElasticNetStrategy(),
        }

        self.scaler_registry: dict[str, BaseScalerStrategy] = {
            "No Feature Scaling": PassthroughScaler(),
            "CLR Transformer": CLRTransformer(),
        }

        self.selector_registry: dict[str, BaseSelectorStrategy] = {
            "No Feature Selection": PassthroughSelector(),
            "Elastic Net Selector": ElasticNetSelector(),
        }

    def build_pipeline(
        self,
        trial: Trial | FixedTrial,
        model_name: str,
        selected_scalers: list[str],
        selected_selectors: list[str],
    ) -> Pipeline:
        # ==== Optuna selects the best scalers and selector from registry ==== #
        chosen_scaler = trial.suggest_categorical("scaler", selected_scalers)
        chosen_selector = trial.suggest_categorical(
            "feature_selector", selected_selectors
        )

        # ==== We instantiate their strategies and build them ==== #
        scaler_strategy = self.scaler_registry[chosen_scaler]
        selector_strategy = self.selector_registry[chosen_selector]
        estimator_strategy = self.model_registry[model_name]

        scaler = scaler_strategy.create_scaler(trial)
        selector = selector_strategy.create_selector(trial)
        estimator = estimator_strategy.create_model(trial)

        return make_pipeline(scaler, selector, estimator)

    def get_available_models(self) -> list[str]:
        available_models = [model for model in self.model_registry.keys()]
        return available_models

    def get_available_scalers(self) -> list[str]:
        available_scalers = [scaler for scaler in self.scaler_registry.keys()]
        return available_scalers

    def get_available_selectors(self) -> list[str]:
        available_selectors = [selector for selector in self.selector_registry.keys()]
        return available_selectors
