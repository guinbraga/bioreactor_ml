import pandas as pd
from pandas import DataFrame, Series


class TopFeaturesPicker:
    def __init__(self, clusters: dict | Series) -> None:
        self.clusters = clusters

    def pick_top_features(
        self, feature_importances: Series, percentage: float = 0.95
    ) -> DataFrame | Series:
        # Convert clusters to Series if it is a dictionary
        if isinstance(self.clusters, dict):
            clusters_series = Series(self.clusters)
        else:
            clusters_series = self.clusters

        # If any feature in feature_importances is not in the clusters mapping,
        # map it to itself to avoid dropping or misaligning it.
        missing_features = feature_importances.index.difference(clusters_series.index)
        if not missing_features.empty:
            extra_mapping = Series(missing_features, index=missing_features)
            clusters_series = pd.concat([clusters_series, extra_mapping])

        # Group feature importances by their cluster representative (medoid) adding up SHAP Values
        cluster_importances = feature_importances.groupby(clusters_series).sum()

        # Sort the cluster importances in descending order
        sorted_cluster_importances = cluster_importances.sort_values(ascending=False)  # type: ignore
        total_importance = sorted_cluster_importances.sum()

        importance_threshold = total_importance * percentage
        cumulative_sum = sorted_cluster_importances.cumsum()

        # Find the first index that meets or exceeds the threshold
        cutoff_index = (cumulative_sum >= importance_threshold).to_numpy().argmax()

        selected_clusters = sorted_cluster_importances.iloc[: cutoff_index + 1]
        return selected_clusters
