import json
import os

import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import BaseCrossValidator

from correlation_cluster_selector import CorrelationClusterSelector

matplotlib.use(
    "Agg"
)  # so that we don't have problems generationg plots while running processes on all cores
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from shap import Explanation
from sklearn.pipeline import Pipeline
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
        self.coeff_results = []

        os.makedirs(self.plot_dir, exist_ok=True)

    def record_experiment_setup(
        self,
        selected_scaler_sequences: list[tuple[str, ...]],
        selected_selectors: list[str],
        cv: BaseCrossValidator,
        groups,
    ) -> None:
        if groups:
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
            json.dump(experiment_setup, config_file)

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

    def generate_shap_waterfall(
        self, pipeline: Pipeline, X_train: DataFrame, X_test: DataFrame
    ) -> None:
        """
        Records the SHAP explanation for each LOOCV split and also generates
        the waterfall plot for that test sample
        """

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

        # Shap values have different shapes depending on what function we pass them.
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
        plt.title(
            f"{self.model_name} - Feature Importances for {sample_id} Prediction of {self.target_col}"
        )

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

        explanation_values = [
            dict(zip(exp.feature_names, exp.values))
            for exp in self.all_shap_explanations
        ]
        explanation_data = [
            dict(zip(exp.feature_names, exp.data)) for exp in self.all_shap_explanations
        ]
        df_values = pd.DataFrame(explanation_values)
        df_values.fillna(0, inplace=True)
        df_data = pd.DataFrame(explanation_data)
        # make sure they're aligned column-wise
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
            f"{self.results_dir}/plots/{self.model_name}_beeswarm.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    def generate_coef_plot(self, n_samples) -> None:
        if not self.coeff_results:
            return
        coef_df = pd.DataFrame(self.coeff_results)  # must set index col
        coef_df.set_index("Test Sample", inplace=True)
        coef_sorted = coef_df.abs().mean().sort_values(ascending=False).head(n_samples)
        coef_asc = coef_sorted.sort_values(ascending=True)
        fig = coef_asc.plot.barh(
            title=f"{self.model_name} Top 15 Coefficient Plot for predicting {self.target_col}"
        ).get_figure()
        fig.savefig(
            f"{self.results_dir}/plots/{self.model_name}_coefficients_plot.png",
            dpi=300,
            bbox_inches="tight",
        )

    def save_shap_dataframes(self) -> None:
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
        obj_path = f"{self.results_dir}/{self.model_name}_shap_explanations_obj.joblib"
        joblib.dump(self.all_shap_explanations, obj_path)

    def save_clusters(self, cluster_selector: CorrelationClusterSelector) -> None:
        """
        Takes the cluster selector object, accesses it's clusters attribute
        (a Series of ints representing a cluster) and saves a csv with the
        cluster id and cluster medoid that each OTU belongs to.
        """
        clusters = cluster_selector.clusters
        medoids = clusters.unique()
        medoids_dict = {
            medoid: list(clusters[clusters == medoid].index) for medoid in medoids
        }
        with open(
            f"{self.results_dir}/../experiment_cluster.json", "w", encoding="utf-8"
        ) as path:
            json.dump(medoids_dict, path)

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
