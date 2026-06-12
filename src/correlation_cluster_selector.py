import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from pipeline_components.scalers import CLRTransformer
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from scipy.stats import spearmanr


class CorrelationClusterSelector(BaseEstimator, SelectorMixin):
    def __init__(self, threshold: float = 0.95, linkage: str = "ward") -> None:
        self.threshold = threshold
        self.linkage = linkage

    def fit(self, X: DataFrame, y=None):
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns.to_list())

        # 1. Define Clusters
        clr_transformer = CLRTransformer().create_scaler(None)
        X_clr = clr_transformer.fit_transform(X)
        X_corr = np.abs(spearmanr(X_clr).statistic)
        X_dist = 1 - X_corr
        np.fill_diagonal(X_dist, 0.0)
        X_dist = np.clip(
            X_dist, 0.0, None
        )  # To avoid floating number errors giving negative distances
        X_dist = squareform(X_dist)
        clustering = hierarchy.linkage(X_dist, method=self.linkage)
        inverse_threshold = 1 - self.threshold
        cluster_ids = hierarchy.fcluster(
            clustering, t=inverse_threshold, criterion="distance"
        )
        clusters = dict(zip(X.columns, cluster_ids))
        self.clusters: Series = pd.Series(clusters)
        df_dist = pd.DataFrame(squareform(X_dist), columns=X.columns, index=X.columns)

        # 2. Find cluster medoids
        cluster_medoids = self.clusters.copy().astype("str")
        for cluster in cluster_medoids.unique():
            selected_cluster = cluster_medoids[cluster_medoids == cluster]
            cluster_feats = selected_cluster.index
            cluster_dist = df_dist.loc[cluster_feats, cluster_feats]
            arg_medoid = cluster_dist.sum().argmin()
            medoid = cluster_dist.iloc[arg_medoid].name
            cluster_medoids[cluster_medoids == cluster] = medoid
        medoids = cluster_medoids.unique()
        self.clusters = (
            cluster_medoids  # we keep the series containing the medoid of each cluster
        )
        self.clusters.index.name = "OTU"
        self.clusters.name = "Cluster Medoid"

        # 3. Create the boolean array of features to keep
        mask = np.array([column in medoids for column in X.columns])

        self.support_mask_ = mask

        return self

    def _get_support_mask(self) -> np.ndarray:  # type: ignore[Override]
        return self.support_mask_
