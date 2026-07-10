from numpy import inf
from sklearn.ensemble import GradientBoostingClassifier
from optuna.trial import Trial

class XGBoostStrategy:
    def create_model(self, trial: Trial):
        learning_rate = trial.suggest_float("learning_rate", 0, inf, log=True)

