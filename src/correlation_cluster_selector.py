import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from pandas.io.formats.style import non_reducing_slice
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from pipeline_components.scalers import CLRTransformer
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy


def proportionality_rho(X):
    cov_matrix = np.cov(X, rowvar=False, ddof=1)
    variances = np.diag(cov_matrix)

    sum_variances = variances[:, None] + variances[None, :]
    var_log_ratio = sum_variances - 2 * cov_matrix

    rho_matrix = 1 - (var_log_ratio / sum_variances)

    return rho_matrix


class CorrelationClusterSelector(BaseEstimator, SelectorMixin):
    def __init__(self, threshold: float = 0.95, linkage: str = "complete") -> None:
        self.threshold = threshold
        self.linkage = linkage

    def fit(self, X: DataFrame, y=None):
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns.to_list())

        # 1. Define Clusters
        df_dist = self._compute_distance_matrix(X)

        # 2. Compute the hierarchy tree for SHAP
        X_dist_flat = squareform(df_dist.to_numpy())
        self.partition_tree_ = hierarchy.linkage(X_dist_flat, method=self.linkage)

        # 3. Create the boolean array of features to keep
        self.support_mask_ = self._compute_medoids_and_mask(df_dist)

        return self

    def _compute_distance_matrix(self, X: DataFrame) -> DataFrame:
        """
        Transforms data using CLR and computes a rho-proportionality-based distance matrix.
        """

        clr_transformer = CLRTransformer().create_scaler(None)
        X_clr = clr_transformer.fit_transform(X)

        X_rho = proportionality_rho(X_clr)
        X_dist = 1 - X_rho

        np.fill_diagonal(X_dist, 0.0)
        X_dist = np.clip(X_dist, 0.0, None)

        return pd.DataFrame(X_dist, columns=X.columns, index=X.columns)

    def _compute_medoids_and_mask(self, df_dist: DataFrame) -> np.ndarray:
        """Flattens the hierarchical tree to flat clusters and extracts the medoids."""
        inverse_threshold = 1 - self.threshold
        cluster_ids = hierarchy.fcluster(
            self.partition_tree_, t=inverse_threshold, criterion="distance"
        )

        # Track mapping of features to cluster IDs
        self.clusters = pd.Series(cluster_ids, index=df_dist.columns)

        # Identify medoid feature for each cluster group
        cluster_medoids = self.clusters.copy().astype(str)
        for cluster in cluster_medoids.unique():
            cluster_feats = cluster_medoids[cluster_medoids == cluster].index
            cluster_dist = df_dist.loc[cluster_feats, cluster_feats]

            arg_medoid = cluster_dist.sum().argmin()
            medoid = cluster_dist.iloc[arg_medoid].name
            cluster_medoids[cluster_medoids == cluster] = medoid

        # Finalize the exposed public series properties
        self.clusters = cluster_medoids
        self.clusters.index.name = "OTU"
        self.clusters.name = "Cluster Medoid"

        # Generate boolean support mask array
        medoids = cluster_medoids.unique()
        return np.array([column in medoids for column in df_dist.columns])

    def _get_support_mask(self) -> np.ndarray:  # type: ignore[Override]
        return self.support_mask_
