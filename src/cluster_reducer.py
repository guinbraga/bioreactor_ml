import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from src.pipeline_components.scalers import CLRTransformer
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from src.data_manager import DataManager
from src.pipeline_components.scalers import CLRTransformer
from scipy.stats import spearmanr


class ClusterReducer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.95) -> None:
        self.threshold = threshold
        self.support_mask_ = None
        self.feature_names_in_ = None
        self.clusters: Series | None = None

    def fit(self, X: DataFrame, y=None):
        # 1. Define Clusters
        clr_transformer = CLRTransformer().create_scaler(None)
        X_clr = clr_transformer.fit_transform(X)
        X_corr = np.abs(spearmanr(X_clr).statistic)
        X_dist = squareform(1 - X_corr)
        clustering = hierarchy.linkage(X_dist, method="single")
        inverse_threshold = 1 - self.threshold
        cluster_ids = hierarchy.fcluster(
            clustering, t=inverse_threshold, criterion="distance"
        )
        clusters = dict(zip(X.columns, cluster_ids))
        self.clusters = pd.Series(clusters)
        df_dist = pd.DataFrame(squareform(X_dist), columns=X.columns, index=X.columns)

        # 2. Find cluster medoids
        medoids = []
        for cluster in self.clusters.unique():
            cluster_feats = self.clusters[self.clusters == cluster].index
            cluster_dist = df_dist.loc[cluster_feats, cluster_feats]
            arg_medoid = cluster_dist.sum().argmin()
            medoid = cluster_dist.iloc[arg_medoid].name
            medoids.append(medoid)

        # 3. Attribute support_mask_
        self.support_mask_ = np.array(medoids)

        return self
