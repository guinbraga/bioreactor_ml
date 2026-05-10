from sklearn.ensemble import RandomForestClassifier
from optuna.trial import Trial


class RandomForestStrategy:
    def create_model(self, trial: Trial) -> RandomForestClassifier:
        max_depth = trial.suggest_int("rf_max_depth", 1, 3)
        min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 10)
        min_samples_split = trial.suggest_int("rf_min_samples_split", 1, 10)

        forest_classifier = RandomForestClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
        )

        return forest_classifier
