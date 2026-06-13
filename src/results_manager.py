import json
import os

import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import BaseCrossValidator

from correlation_cluster_selector import CorrelationClusterSelector

# so we don't have problems generating plots while running processes on all cores
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandas import DataFrame
from shap import Explanation
from sklearn.pipeline import Pipeline
from sklearn.utils.parallel import joblib


class ResultsDataManager:
    """Handles tracking metrics and saving text-based/data results (CSV, Parquet, JSON, Joblib)."""

    def __init__(self, model_name: str, results_dir: str, target_col: str) -> None:
        self.model_name = model_name
        self.results_dir = results_dir
        self.target_col = target_col
        self.rows_result: list[dict] = []
        self.coeff_results: list[dict] = []
        self.all_shap_explanations: list[Explanation] = []

        os.makedirs(self.results_dir, exist_ok=True)

    def record_experiment_setup(
        self,
        selected_scaler_sequences: list[tuple[str, ...]],
        selected_selectors: list[str],
        cv: BaseCrossValidator,
        groups,
    ) -> None:
        if groups is not None:
            groups = groups.name

        experiment_setup = {
            "Model": self.model_name,
            "Target Column": self.target_col,
            "Scaler Sequences Evaluated": selected_scaler_sequences,
            "Feature Selection Techniques Evaluated": selected_selectors,
            "Cross Validation Method": cv.__str__(),
            "Groups": str(groups),
        }

        with open(f"{self.results_dir}/experiment_config.json", "w") as config_file:
            json.dump(experiment_setup, config_file, indent=4)

    def record_split_metrics(self, split_data: dict) -> None:
        self.rows_result.append(split_data)

    def record_split_coefs(self, pipeline: Pipeline, sample_id: str) -> None:
        estimator = pipeline[-1]

        if not hasattr(estimator, "coef_"):
            return

        feature_names = pipeline[:-1].get_feature_names_out()
        coefficients = estimator.coef_[0]

        coef_dict = {"Test Sample": sample_id}
        coef_dict.update(dict(zip(feature_names, coefficients)))

        self.coeff_results.append(coef_dict)

    def compute_and_record_shap(
        self, pipeline: Pipeline, X_train: DataFrame, X_test: DataFrame
    ) -> Explanation:
        """Computes the SHAP explanation, stores it in memory, and returns it."""
        preprocessing = pipeline[:-1]
        feature_names = preprocessing.get_feature_names_out()
        X_transformed = preprocessing.transform(X_train)
        X_test_transformed = preprocessing.transform(X_test)

        X_train_df = pd.DataFrame(
            X_transformed, columns=feature_names, index=X_train.index
        )
        X_test_df = pd.DataFrame(
            X_test_transformed, columns=feature_names, index=X_test.index
        )

        explainer = shap.Explainer(model=pipeline[-1], masker=X_train_df)
        shap_values = explainer(X_test_df)

        # different Explainer objects return different-shaped objects
        if isinstance(shap_values, list):
            explanation = shap_values[1][0]
        else:
            if len(shap_values.shape) == 3:
                explanation = shap_values[0, :, 1]
            else:
                explanation = shap_values[0]

        self.all_shap_explanations.append(explanation)
        return explanation

    def save_shap_dataframes(self) -> None:
        if not self.all_shap_explanations:
            return

        exp_values = [
            dict(zip(exp.feature_names, exp.values))
            for exp in self.all_shap_explanations
        ]
        exp_data = [
            dict(zip(exp.feature_names, exp.data)) for exp in self.all_shap_explanations
        ]
        df_values = pd.DataFrame(exp_values)
        df_values.fillna(0, inplace=True)
        df_data = pd.DataFrame(exp_data)

        # make sure they're aligned column-wise
        df_data = df_data[df_values.columns]
        feature_names = df_values.columns.to_list()

        shap_cols = [f"{feature}_SHAP" for feature in feature_names]
        raw_cols = [f"{feature}_RAW" for feature in feature_names]

        df_shap = pd.DataFrame(df_values.values, columns=shap_cols)
        df_raw = pd.DataFrame(df_data.values, columns=raw_cols)

        all_df = pd.concat([df_shap, df_raw], axis=1)
        file_path = f"{self.results_dir}/shap_explanations_table.parquet"
        all_df.to_parquet(file_path, index=False)

    def save_shap_objects(self) -> None:
        if not self.all_shap_explanations:
            return
        obj_path = f"{self.results_dir}/{self.model_name}_shap_explanations_obj.joblib"
        joblib.dump(self.all_shap_explanations, obj_path)

    def save_clusters(self, cluster_selector: CorrelationClusterSelector) -> None:
        """
        Saves clusters composition info as json for later analysis
        """
        clusters = cluster_selector.clusters
        medoids = clusters.unique()
        medoids_dict = {
            medoid: list(clusters[clusters == medoid].index) for medoid in medoids
        }
        with open(
            f"{self.results_dir}/../experiment_cluster.json", "w", encoding="utf-8"
        ) as path:
            json.dump(medoids_dict, path, indent=4)

    def save_final_csv(self) -> None:
        df_summary = pd.DataFrame(self.rows_result)
        df_summary.to_csv(
            f"{self.results_dir}/{self.model_name}_predictions_summary.csv", index=False
        )

        if self.coeff_results:
            df_coeff = pd.DataFrame(self.coeff_results)
            df_coeff.to_csv(
                f"{self.results_dir}/{self.model_name}_coefficients.csv", index=False
            )


class ResultsPlotManager:
    """Handles creating and saving all figures and plots."""

    def __init__(self, model_name: str, results_dir: str, target_col: str) -> None:
        self.model_name = model_name
        self.target_col = target_col
        self.plot_dir = f"{results_dir}/plots"

        os.makedirs(self.plot_dir, exist_ok=True)

    def generate_shap_waterfall(self, explanation: Explanation, sample_id: str) -> None:
        shap.plots.waterfall(explanation, show=False, max_display=15)
        plt.title(
            f"{self.model_name} - Feature Importances for {sample_id} Prediction of {self.target_col}"
        )

        plt.savefig(
            f"{self.plot_dir}/{self.model_name}_waterfall_{sample_id}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def generate_bee_swarm_plot(self, all_shap_explanations: list[Explanation]) -> None:
        """
        Genearates a Beeswarm plot at the end of the experiment. As we are dealing
        with Leave One Out or Leave One Group Out, SHAP values are stacked from
        all splits, and base values are the average of all base values.
        """
        if not all_shap_explanations:
            return

        avg_base_values = float(
            np.mean([explanation.base_values for explanation in all_shap_explanations])
        )

        explanation_values = [
            dict(zip(exp.feature_names, exp.values)) for exp in all_shap_explanations
        ]
        explanation_data = [
            dict(zip(exp.feature_names, exp.data)) for exp in all_shap_explanations
        ]
        df_values = pd.DataFrame(explanation_values)
        df_values.fillna(0, inplace=True)
        df_data = pd.DataFrame(explanation_data)
        df_data = df_data[df_values.columns]
        feature_names = df_values.columns.to_list()

        global_explanation = Explanation(
            base_values=avg_base_values,
            feature_names=feature_names,
            values=df_values.values,
            data=df_data.values,
        )

        shap.plots.beeswarm(global_explanation, show=False, max_display=15)
        plt.title(
            f"{self.model_name} Global Beeswarm plot for predicting {self.target_col}"
        )
        plt.savefig(
            f"{self.plot_dir}/{self.model_name}_beeswarm.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def generate_coef_plot(
        self, coeff_results: list[dict], n_samples: int = 15
    ) -> None:
        if not coeff_results:
            return

        coef_df = pd.DataFrame(coeff_results)
        coef_df.set_index("Test Sample", inplace=True)
        coef_sorted = coef_df.abs().mean().sort_values(ascending=False).head(n_samples)
        coef_asc = coef_sorted.sort_values(ascending=True)

        fig = coef_asc.plot.barh(
            title=f"{self.model_name} Top {n_samples} Coefficient Plot for predicting {self.target_col}"
        ).get_figure()

        fig.savefig(
            f"{self.plot_dir}/{self.model_name}_coefficients_plot.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
