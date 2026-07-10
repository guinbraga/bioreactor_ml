from optuna import Trial
from optuna.trial import FixedTrial
from sklearn.linear_model import LogisticRegression


class L1LogisticRegressionStrategy:
    def create_model(self, trial: Trial | FixedTrial) -> LogisticRegression:
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)

        model = LogisticRegression(
            solver="saga", C=C, l1_ratio=1, random_state=47, max_iter=2000
        )

        return model
