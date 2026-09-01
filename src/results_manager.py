from collections import defaultdict
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import shap
from matplotlib.figure import Figure
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from correlation_cluster_selector import CorrelationClusterSelector

# Sentinel label used when the model has no `classes_` attribute (regression /
# single-output pipelines). Explanations for those pipelines are produced under
# this key so that all the per-class code paths work uniformly.
REGRESSION_CLASS_LABEL = "regression"

# so we don't have problems generating plots while running processes on all cores
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from shap import Explanation
from sklearn.pipeline import Pipeline


class ResultsDataManager:
    """Handles tracking metrics and saving text-based/data results (CSV, Parquet, JSON, Joblib)."""

    def __init__(self, target_col: str) -> None:
        self.target_col = target_col
        self.rows_result: list[dict] = []
        self.coeff_results: list[dict] = []
        self.all_shap_explanations: list[Explanation] = []
        self.shap_classes: list[Any] = []
        self.shap_explanations_by_class: dict[Any, list[Explanation]] = defaultdict(list)

    def record_split_metrics(self, split_data: dict) -> None:
        self.rows_result.append(split_data)

    def record_split_coefs(self, pipeline: Pipeline, sample_id: str) -> None:
        estimator = pipeline[-1]

        if not hasattr(estimator, "coef_"):
            return

        feature_names = pipeline[:-1].get_feature_names_out()
        coefficients = estimator.coef_[0]  # type: ignore

        coef_dict = {"Test Sample": sample_id}
        coef_dict.update(dict(zip(feature_names, coefficients)))  # type: ignore

        self.coeff_results.append(coef_dict)

    def compute_and_record_shap(
        self,
        pipeline: Pipeline,
        X_train: DataFrame,
        X_test: DataFrame,
        cluster_selector: CorrelationClusterSelector,
    ) -> dict[Any, Explanation]:
        """Computes the Owen explanation for every output class of the fitted
        pipeline, stores them in memory (per-class and flat), and returns a
        mapping ``{class_label -> Explanation}`` for this split's test sample.

        ``class_label`` is read from the estimator's ``classes_`` attribute
        (per-fold), not from the global ``np.unique(y)``, so the label is
        always the one the SHAP column actually refers to -- this eliminates
        the drift that occurs when a fold's training set omits a class.
        """

        model = pipeline[-1]
        predict_function = (
            pipeline.predict_proba
            if hasattr(model, "predict_proba")
            else pipeline.predict
        )

        partition_tree = cluster_selector.partition_tree_

        partition_mask = shap.maskers.Partition(X_train, clustering=partition_tree)
        explainer = shap.PartitionExplainer(
            predict_function, partition_mask, partition_tree=partition_tree
        )

        shap_values = explainer(X_test)

        if hasattr(model, "classes_"):
            class_labels = list(model.classes_)
        else:
            class_labels = [REGRESSION_CLASS_LABEL]

        per_class_explanations: dict[Any, Explanation] = {}

        if isinstance(shap_values, list):
            n_outputs = min(len(shap_values), len(class_labels))
            for k in range(n_outputs):
                per_class_explanations[class_labels[k]] = shap_values[k][0]
        elif len(shap_values.shape) == 3:
            n_outputs = min(shap_values.shape[-1], len(class_labels))
            for k in range(n_outputs):
                per_class_explanations[class_labels[k]] = shap_values[0, :, k]
        else:
            per_class_explanations[class_labels[0]] = shap_values[0]

        for label, explanation in per_class_explanations.items():
            self.all_shap_explanations.append(explanation)
            self.shap_classes.append(label)
            self.shap_explanations_by_class[label].append(explanation)

        return per_class_explanations

    def compute_coalition_shap(self, clusters: Series):
        pass

    def create_classification_report(self) -> dict:
        results_df = pd.DataFrame(self.rows_result)
        y_true = results_df["True Class"].values
        y_pred = results_df["Predicted Class"].values
        labels = np.unique(y_true)  # type: ignore
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
        return report  # type: ignore

    def get_feature_importances(self) -> dict[Any, Series]:
        return {
            class_label: self._mean_abs_shap(explanations)
            for class_label, explanations in self.shap_explanations_by_class.items()
        }

    @staticmethod
    def _mean_abs_shap(explanations: list[Explanation]) -> Series:
        if not explanations:
            return Series(dtype=float)
        explanation_values = [
            dict(zip(exp.feature_names, exp.values)) for exp in explanations
        ]
        shap_df = pd.DataFrame(explanation_values)
        shap_df.fillna(0, inplace=True)
        return shap_df.abs().mean()


class ResultsPlotManager:
    """Handles creating and saving all figures and plots."""

    def __init__(self, model_name: str, target_col: str) -> None:
        self.model_name = model_name
        self.target_col = target_col

    def generate_shap_waterfall(
        self, explanation: Explanation, sample_id: str, class_name: str
    ) -> Figure:
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(explanation, show=False, max_display=15)
        plt.title(
            f"{self.model_name} - Feature Importances for {sample_id} Prediction of {self.target_col} as {class_name}"
        )
        fig = plt.gcf()
        return fig

    def generate_bee_swarm_plot(
        self,
        all_shap_explanations: list[Explanation],
        class_name: str,
    ) -> Figure | None:
        """
        Genearates a Beeswarm plot at the end of the experiment. As we are dealing
        with Leave One Out or Leave One Group Out, SHAP values are stacked from
        all splits, and base values are the average of all base values.

        Explanations passed in must belong to a single class; the caller is
        responsible for grouping `ResultsDataManager.shap_explanations_by_class`.
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
        df_data = df_data[df_values.columns] # make sure both dfs are aligned
        feature_names = df_values.columns.to_list()

        global_explanation = Explanation(
            base_values=avg_base_values,
            feature_names=feature_names,
            values=df_values.values,
            data=df_data.values,  # type: ignore
        )

        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(global_explanation, show=False, max_display=15)
        plt.title(
            f"{self.model_name} Global Beeswarm plot for predicting {self.target_col} as {class_name}"
        )
        fig = plt.gcf()
        return fig

    def generate_cluster_importance_plot(
        self,
        feature_importances: Series | DataFrame,
        class_name: str,
        clusters: dict | Series | None = None,
        n_clusters: int = 20,
    ) -> Figure | None:
        # Select the top n_clusters importances by sorting descending
        top_importances = feature_importances.sort_values(ascending=False).head(
            n_clusters
        )
        # Sort ascending so that pandas barh plots the largest values at the top of the chart
        top_importances_asc = top_importances.sort_values(ascending=True)

        if clusters is not None:
            if isinstance(clusters, dict):
                clusters_series = Series(clusters)
            else:
                clusters_series = clusters

            # Count occurrences of each cluster representative
            counts = clusters_series.value_counts()

            # Map index labels: append '(n)' representing the cluster size
            new_index = [
                f"{idx} ({counts.get(idx, 1)})" for idx in top_importances_asc.index
            ]
            top_importances_asc.index = new_index

        fig, ax = plt.subplots(figsize=(10, 8))
        top_importances_asc.plot.barh(
            ax=ax,
            title=f"{self.model_name} Importances for Top {n_clusters} Selected Clusters for predicting {self.target_col} as {class_name}",
        )
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
        labels = np.unique(y_true)  # type: ignore
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(10, 8))

        disp.plot(ax=ax, cmap="Blues")
        plt.title(f"{self.model_name} Confusion Matrix for {self.target_col}")

        return fig

    def generate_rmse_boxplot(self, rmse_values: list[float]) -> Figure:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(rmse_values, vert=True, patch_artist=True)
        ax.set_xticklabels([self.model_name])
        ax.set_ylabel("RMSE")
        ax.set_title(
            f"RMSE Distribution for {self.model_name} predicting {self.target_col}"
        )

        mean_val = np.mean(rmse_values)
        ax.axhline(
            y=mean_val,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {mean_val:.4f}",
        )
        ax.legend()

        return fig
