import json
from matplotlib.figure import Figure
import pandas as pd
from pandas import Series, DataFrame
import os
from sklearn.utils.parallel import joblib
from correlation_cluster_selector import CorrelationClusterSelector
from sklearn.model_selection import BaseCrossValidator

from results_manager import ResultsDataManager


class DataPersistenceManager:
    def __init__(
        self,
        results_data_manager: ResultsDataManager,
        results_dir: str,
        model_name: str,
        target_column: str,
    ) -> None:
        self.results_data_manager = results_data_manager
        self.results_dir = f"{results_dir}/"
        self.model_name = model_name

        os.makedirs(self.results_dir, exist_ok=True)

    def save_final_csv(self) -> None:
        df_summary = pd.DataFrame(self.results_data_manager.rows_result)
        df_summary.to_csv(
            f"{self.results_dir}/{self.model_name}_predictions_summary.csv", index=False
        )

        if self.results_data_manager.coeff_results:
            df_coeff = pd.DataFrame(self.results_data_manager.coeff_results)
            df_coeff.to_csv(
                f"{self.results_dir}/{self.model_name}_coefficients.csv", index=False
            )

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

    def save_shap_objects(self) -> None:
        if not self.results_data_manager.all_shap_explanations:
            return
        obj_path = f"{self.results_dir}/{self.model_name}_shap_explanations_obj.joblib"
        joblib.dump(self.results_data_manager.all_shap_explanations, obj_path)

    def save_shap_dataframes(self) -> None:
        if not self.results_data_manager.all_shap_explanations:
            return

        rdm = self.results_data_manager
        exp_values = [
            dict(zip(exp.feature_names, exp.values))
            for exp in rdm.all_shap_explanations
        ]
        exp_data = [
            dict(zip(exp.feature_names, exp.data)) for exp in rdm.all_shap_explanations
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
        all_df.insert(0, "Class", list(rdm.shap_classes))
        file_path = f"{self.results_dir}/shap_explanations_table.parquet"
        all_df.to_parquet(file_path, index=False)

    def save_experiment_setup(
        self,
        selected_scaler_sequences: list[tuple[str, ...]],
        selected_selectors: list[str],
        cv: BaseCrossValidator,
        target_col: str,
        groups,
    ) -> None:
        if groups is not None:
            groups = groups.name

        experiment_setup = {
            "Model": self.model_name,
            "Target Column": target_col,
            "Scaler Sequences Evaluated": selected_scaler_sequences,
            "Feature Selection Techniques Evaluated": selected_selectors,
            "Cross Validation Method": cv.__str__(),
            "Groups": str(groups),
        }

        with open(f"{self.results_dir}/experiment_config.json", "w") as config_file:
            json.dump(experiment_setup, config_file, indent=4)

    def save_classification_report(self) -> None:
        classification_report = self.results_data_manager.create_classification_report()
        df_report = pd.DataFrame(classification_report)
        df_report.to_csv(
            f"{self.results_dir}/{self.model_name}_classification_report.csv"
        )
        df_report.to_latex(
            f"{self.results_dir}/{self.model_name}_classification_report.tex"
        )

    def save_top_feat_importances(
        self,
        top_features: Series | DataFrame,
        class_label,
        cluster_selector: CorrelationClusterSelector | None = None,
    ) -> None:
        cluster_reprs = top_features.index.to_list()

        with open(
            f"{self.results_dir}/{self.model_name}_top_clusters_{class_label}.csv",
            "w",
            encoding="utf-8",
        ) as features_file:
            features_file.write("feature,\n")
            for feature in cluster_reprs:
                features_file.write(feature + ",\n")

        if cluster_selector is not None:
            clusters = cluster_selector.clusters
            expanded = []
            for rep in cluster_reprs:
                members = clusters[clusters == rep].index
                expanded.extend(members)
            with open(
                f"{self.results_dir}/{self.model_name}_top_features_{class_label}.csv",
                "w",
                encoding="utf-8",
            ) as features_file:
                features_file.write("feature,\n")
                for feature in expanded:
                    features_file.write(feature + ",\n")
        else:
            with open(
                f"{self.results_dir}/{self.model_name}_top_features_{class_label}.csv",
                "w",
                encoding="utf-8",
            ) as features_file:
                features_file.write("feature,\n")
                for feature in cluster_reprs:
                    features_file.write(feature + ",\n")


class PlotPersistenceManager:
    def __init__(self, results_dir: str, model_name: str) -> None:
        self.plot_dir = f"{results_dir}/plots"
        self.model_name = model_name

        os.makedirs(self.plot_dir, exist_ok=True)
        pass

    def persist_shap_waterfall(self, fig: Figure, sample_id: str, class_label) -> None:
        fig.savefig(
            f"{self.plot_dir}/{self.model_name}_waterfall_{sample_id}_{class_label}.png",
            dpi=300,
            bbox_inches="tight",
        )

    def persist_beeswarm_plot(self, fig: Figure, class_label) -> None:
        fig.savefig(
            f"{self.plot_dir}/{self.model_name}_beeswarm_{class_label}.png",
            dpi=300,
            bbox_inches="tight",
        )

    def persist_coef_plot(self, fig: Figure) -> None:
        fig.savefig(
            f"{self.plot_dir}/{self.model_name}_coefficients_plot.png",
            dpi=300,
            bbox_inches="tight",
        )

    def persist_cluster_importance_plot(self, fig: Figure, class_label) -> None:
        fig.savefig(
            f"{self.plot_dir}/{self.model_name}_cluster_importance_Owen_{class_label}.png",
            dpi=300,
            bbox_inches="tight",
        )

    def persist_confusion_matrix(self, fig: Figure) -> None:
        fig.savefig(
            f"{self.plot_dir}/{self.model_name}_confusion_matrix.png",
            dpi=300,
            bbox_inches="tight",
        )

    def persist_rmse_boxplot(self, fig: Figure) -> None:
        fig.savefig(
            f"{self.plot_dir}/{self.model_name}_rmse_boxplot.png",
            dpi=300,
            bbox_inches="tight",
        )
