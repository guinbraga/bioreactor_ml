from optuna import Trial
from optuna.trial import FixedTrial
from sklearn.svm import SVC

class LinearSVMStrategy:
    def create_model(self, trial: Trial | FixedTrial) -> SVC:
        C = trial.suggest_float("C", 1e-3, 1000, log=True)
        model = SVC(kernel='linear', C=C, probability=True)
        return model
        
