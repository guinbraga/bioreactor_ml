import json
import os

import matplotlib
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import shap
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix


# so we don't have problems generating plots while running processes on all cores
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandas import DataFrame
from shap import Explanation
from sklearn.pipeline import Pipeline


class ResultsDataManager:
    """Handles tracking metrics and saving text-based/data results (CSV, Parquet, JSON, Joblib)."""

    def __init__(self, target_col: str) -> None:
        self.target_col = target_col
        self.rows_result: list[dict] = []
        self.coeff_results: list[dict] = []
        self.all_shap_explanations: list[Explanation] = []

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

    def record_classification_report(self) -> dict:
        results_df = pd.DataFrame(self.rows_result)
        y_true = results_df["True Class"].values
        y_pred = results_df["Predicted Class"].values
        labels = np.unique(y_true) #type: ignore
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
        return report #type: ignore


class ResultsPlotManager:
    """Handles creating and saving all figures and plots."""

    def __init__(self, model_name: str, target_col: str) -> None:
        self.model_name = model_name
        self.target_col = target_col

    def generate_shap_waterfall(
        self, explanation: Explanation, sample_id: str
    ) -> Figure:
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(explanation, show=False, max_display=15)
        plt.title(
            f"{self.model_name} - Feature Importances for {sample_id} Prediction of {self.target_col}"
        )
        fig = plt.gcf()
        return fig

    def generate_bee_swarm_plot(
        self, all_shap_explanations: list[Explanation]
    ) -> Figure | None:
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

        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(global_explanation, show=False, max_display=15)
        plt.title(
            f"{self.model_name} Global Beeswarm plot for predicting {self.target_col}"
        )
        fig = plt.gcf()
        return fig

    def generate_coef_plot(
        self, coeff_results: list[dict], n_samples: int = 15
    ) -> None | Figure:
        if not coeff_results:
            return

        coef_df = pd.DataFrame(coeff_results)
        coef_df.set_index("Test Sample", inplace=True)
        coef_sorted = coef_df.abs().mean().sort_values(ascending=False).head(n_samples)
        coef_asc = coef_sorted.sort_values(ascending=True)

        plt.figure(figsize=(10, 8))
        fig = coef_asc.plot.barh(
            title=f"{self.model_name} Top {n_samples} Coefficient Plot for predicting {self.target_col}"
        ).get_figure()

        return fig

    def generate_confusion_matrix(self, rows_result: list[dict]) -> Figure:
        df_results = pd.DataFrame(rows_result)
        y_true = df_results["True Class"].values
        y_pred = df_results["Predicted Class"].values
        labels = np.unique(y_true)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(10, 8))

        disp.plot(ax=ax, cmap="Blues")
        plt.title(f"{self.model_name} Confusion Matrix for {self.target_col}")

        return fig
