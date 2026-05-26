from sklearn.model_selection import BaseCrossValidator, LeaveOneGroupOut, LeaveOneOut
from sklearn.pipeline import Pipeline
from optuna.trial import FixedTrial, Trial
from pipeline_components.models.elastic_net import ElasticNetStrategy
from pipeline_components.models.l1_logreg import L1LogisticRegressionStrategy
from pipeline_components.models.linear_svm import LinearSVMStrategy
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
    BinarizerScaler,
    CLRTransformer,
    PassthroughScaler,
    RelativeAbundanceScaler,
    Standardizer,
)
from sklearn.pipeline import make_pipeline


class PipelineFactory:
    def __init__(self):
        self.model_registry: dict[str, BaseModelStrategy] = {
            "Random Forest": RandomForestStrategy(),
            "L1 Logistic Regression": L1LogisticRegressionStrategy(),
            "Elastic Net": ElasticNetStrategy(),
            "SVM-linear": LinearSVMStrategy(),
        }

        self.scaler_registry: dict[str, BaseScalerStrategy] = {
            "No Feature Scaling": PassthroughScaler(),
            "CLR Transformer": CLRTransformer(),
            "Presence/Abscence Transformer": BinarizerScaler(),
            "Relative Abundance": RelativeAbundanceScaler(),
            "Standard Scaler": Standardizer(),
        }

        self.selector_registry: dict[str, BaseSelectorStrategy] = {
            "No Feature Selection": PassthroughSelector(),
            "Elastic Net Selector": ElasticNetSelector(),
        }

        self.cv_registry: dict[str, BaseCrossValidator] = {
            "Leave One Out": LeaveOneOut(),
            "Leave One Group Out": LeaveOneGroupOut(),
        }

    def build_pipeline(
        self,
        trial: Trial | FixedTrial,
        model_name: str,
        selected_scaler_sequences: list[tuple[str, ...]],
        selected_selectors: list[str],
    ) -> Pipeline:
        # ==== Optuna selects the best scalers and selector from registry ==== #
        # For it to choose a scaler sequence, we must pass a string to suggest_categorical.
        # Thus, we'll create a mapping
        sequence_mapping = {
            " -> ".join(sequence) if sequence else "No Feature Scaling": sequence
            for sequence in selected_scaler_sequences
        }
        sequences: list[str] = list(sequence_mapping.keys())
        chosen_scaler_sequence = trial.suggest_categorical("scaler_sequence", sequences)
        chosen_selector = trial.suggest_categorical(
            "feature_selector", selected_selectors
        )

        # ==== We instantiate their strategies and build them ==== #
        # Since scalers are a sequence, we instantiate each one at a time
        instantiated_scalers = []
        for scaler in sequence_mapping[chosen_scaler_sequence]:
            scaler_strategy = self.scaler_registry[scaler]
            instantiated_scalers.append(scaler_strategy.create_scaler(trial))

        selector_strategy = self.selector_registry[chosen_selector]
        selector = selector_strategy.create_selector(trial)

        estimator_strategy = self.model_registry[model_name]
        estimator = estimator_strategy.create_model(trial)

        return make_pipeline(*instantiated_scalers, selector, estimator)

    def get_available_models(self) -> list[str]:
        available_models = [model for model in self.model_registry.keys()]
        return available_models

    def get_available_scalers(self) -> list[str]:
        available_scalers = [scaler for scaler in self.scaler_registry.keys()]
        return available_scalers

    def get_available_selectors(self) -> list[str]:
        available_selectors = [selector for selector in self.selector_registry.keys()]
        return available_selectors

    def get_available_cv(self) -> list[str]:
        available_cv = [cv for cv in self.cv_registry.keys()]
        return available_cv
