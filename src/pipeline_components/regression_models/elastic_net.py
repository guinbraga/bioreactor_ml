from optuna import Trial
from optuna.trial import FixedTrial
from sklearn.linear_model import ElasticNet


class ElasticNetStrategy:
    def create_model(self, trial: Trial | FixedTrial) -> ElasticNet:
        alpha = trial.suggest_float("alpha", 1e-3, 10.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

        model = ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, random_state=47, max_iter=2000
        )

        return model
