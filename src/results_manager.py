import os
import shap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from pandas import DataFrame
from shap import Explanation
from sklearn.utils.parallel import joblib


class ResultsManager:
    def __init__(self, model_name: str, results_dir: str, target_col: str) -> None:
        self.rows_result: list[dict] = []
        self.rows_shap: dict = {}
        self.model_name: str = model_name
        self.results_dir: str = results_dir
        self.target_col: str = target_col
        self.plot_dir: str = f"{self.results_dir}/plots"
        self.all_shap_explanations: list[Explanation] = []
        os.makedirs(self.plot_dir, exist_ok=True)

    def record_split_metrics(self, split_data: dict) -> None:
        self.rows_result.append(split_data)

    def process_split_data(
        self, pipeline: Pipeline, X_train: DataFrame, X_test: DataFrame
    ) -> None:
        """
        Records the SHAP explanation for each LOOCV split and also generates
        the waterfall plot for that test sample
        """

        explainer = shap.Explainer(pipeline.predict_proba, X_train)
        shap_values = explainer(X_test, max_evals=1500)

        # Shap values have different shapes depending on what function we pass them.
        # Here, we expect to fall into the 3-dimensional case (sample, shap_values, class)
        if isinstance(shap_values, list):
            explanation = shap_values[1][0]

        else:
            if len(shap_values.shape) == 3:
                explanation = shap_values[0, :, 1]

            else:
                explanation = shap_values[0]
        self.all_shap_explanations.append(explanation)

        shap.plots.waterfall(explanation, show=False, max_display=15)
        sample_id = X_test.index[0]
        plt.title(f"{self.model_name} - Feature Importances for {sample_id} Prediction")

        plt.savefig(
            f"{self.results_dir}/plots/{self.model_name}_waterfall_{sample_id}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    def generate_bee_swarm_plot(self) -> None:
        """
        Takes each Explanation object stored by processing each train-test split
        and creates a beeswarm plot. Base values are the average of all base values.
        """

        avg_base_values = float(
            np.mean(
                [explanation.base_values for explanation in self.all_shap_explanations]
            )
        )
        feature_names = self.all_shap_explanations[0].feature_names
        merged_values = [
            explanation.values for explanation in self.all_shap_explanations
        ]
        merged_data = [explanation.data for explanation in self.all_shap_explanations]

        stacked_values = np.vstack(merged_values)
        stacked_data = np.vstack(merged_data)

        global_explanation = Explanation(
            base_values=avg_base_values,
            feature_names=feature_names,
            values=stacked_values,
            data=stacked_data,
        )

        shap.plots.beeswarm(global_explanation, show=False, max_display=15)
        plt.title(f"{self.model_name} Global Beeswarm plot (LOOCV) for predicting {self.target_col}")
        plt.savefig(
            f"{self.results_dir}/plots/{self.model_name}_beeswarm.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    def save_shap_dataframes(self) -> None:
        merged_values = np.vstack(
            [explanation.values for explanation in self.all_shap_explanations]
        )

        merged_data = np.vstack(
            [explanation.data for explanation in self.all_shap_explanations]
        )

        feature_names = self.all_shap_explanations[0].feature_names

        shap_cols = [f"{feature}_SHAP" for feature in feature_names]
        raw_cols = [f"{feature}_RAW" for feature in feature_names]

        df_shap = pd.DataFrame(merged_values, columns=shap_cols)
        df_raw = pd.DataFrame(merged_data, columns=raw_cols)

        all_df = pd.concat([df_shap, df_raw], axis=1)
        file_path = f"{self.results_dir}/shap_record_table.parquet"
        all_df.to_parquet(file_path, index=False)

    def save_shap_objects(self) -> None:
        obj_path = f"{self.results_dir}/{self.model_name}_shap_explanations.joblib"
        joblib.dump(self.all_shap_explanations, obj_path)

    def save_final_csv(self) -> None:
        df_summary = pd.DataFrame(self.rows_result)
        df_summary.to_csv(
            f"{self.results_dir}/{self.model_name}_predictions_summary.csv"
        )
