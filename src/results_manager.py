import os
import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from pandas import DataFrame


class ResultsManager:
    def __init__(self, model_name: str, results_dir: str) -> None:
        self.rows_result: list = []
        self.model_name: str = model_name
        self.results_dir: str = f"{results_dir}/{model_name}"
        self.plot_dir: str = f"{self.results_dir}/plots"
        self.shap_values = None
        os.makedirs(self.plot_dir, exist_ok=True)

    def record_split_metrics(self, split_data: dict):
        self.rows_result.append(split_data)

    def generate_waterfall_plot(
        self, pipeline: Pipeline, X_train: DataFrame, X_test: DataFrame, test_index: int
    ):
        explainer = shap.Explainer(pipeline.predict_proba, X_train)
        shap_values = explainer(X_test, max_evals=1500)

        if isinstance(shap_values, list):
            # 1 is for our positive class, which is procedure 2
            explanation = shap_values[1][test_index]

        else:
            if len(shap_values.shape) == 3:
                explanation = shap_values[0, :, 1]

            else:
                explanation = shap_values[test_index]

        sample_id = X_test.index[0]
        shap.plots.waterfall(explanation, show=False, max_display=15)
        plt.title(f"{self.model_name} - Feature Importances for {sample_id}")

        plt.savefig(
            f"{self.plot_dir}/waterfall/waterfall_{sample_id}",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


    def save_final_csv(self):
        df = pd.DataFrame(self.rows_result)
        df.to_csv(f"{self.results_dir}/{self.model_name}.csv")
