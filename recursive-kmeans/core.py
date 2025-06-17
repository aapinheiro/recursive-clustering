import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from typing import List, Tuple

class RecursiveClustering:
    """
    Implements recursive KMeans clustering with minimum and maximum cluster size constraints.

    Parameters
    ----------
    geoloc_columns : List[str]
        List of column names for geographic coordinates.
    vars_encode : List[str]
        List of column names to be encoded with cluster labels.
    encode : bool, optional
        Whether to encode cluster labels into a new string column, by default False.
    min_cluster_size : int, optional
        Minimum number of samples per cluster, by default 5.
    max_cluster_size : int, optional
        Maximum number of samples per cluster, by default 15.
    max_subsplit_attempts : int, optional
        Maximum number of recursive attempts to split small clusters, by default 15.
    """

    def __init__(self, 
                 geoloc_columns: List[str], 
                 vars_encode: List[str], 
                 encode: bool = False,
                 min_cluster_size: int = 5, 
                 max_cluster_size: int = 15, 
                 max_subsplit_attempts: int = 15):
        """
        Initialize the RecursiveClustering instance.
        """
        self.min_cluster_size: int = min_cluster_size
        self.max_cluster_size: int = max_cluster_size
        self.geoloc_columns: List[str] = geoloc_columns
        self.clusters = []
        self.cluster_labels = {}
        self.original_df = None
        self.vars_encode = vars_encode
        self.encode = encode
        self.max_subsplit_attempts = max_subsplit_attempts
        self.history = []

    def _recursive_split(self, data, cluster_id, depth=0, subsplit_attempts_remaining=None):
        """
        Recursively split the data using KMeans until all clusters fall within size constraints.

        Parameters
        ----------
        data : np.ndarray
            Input data array.
        cluster_id : int
            Identifier for the current cluster.
        depth : int, optional
            Depth of recursion, by default 0.
        subsplit_attempts_remaining : int, optional
            Remaining attempts to split small clusters.
        """
        if subsplit_attempts_remaining is None:
            subsplit_attempts_remaining = self.max_subsplit_attempts

        if len(data) < self.min_cluster_size:
            if subsplit_attempts_remaining > 0 and len(data) > 1:
                for attempt in range(3):
                    seed = 100 + attempt
                    kmeans = KMeans(n_clusters=2, random_state=seed, n_init=10)
                    labels = kmeans.fit_predict(data[:, :2])
                    cluster1 = data[labels == 0]
                    cluster2 = data[labels == 1]

                    if len(cluster1) == len(data) or len(cluster2) == len(data):
                        continue

                    self._recursive_split(cluster1, cluster_id * 2 + 1, depth + 1, subsplit_attempts_remaining - 1)
                    self._recursive_split(cluster2, cluster_id * 2 + 2, depth + 1, subsplit_attempts_remaining - 1)
                    return
            self.clusters.append((cluster_id, data))
            return

        if self.min_cluster_size <= len(data) <= self.max_cluster_size:
            self.clusters.append((cluster_id, data))
            return

        for attempt in range(15):
            seed = 42 + attempt
            kmeans = KMeans(n_clusters=2, random_state=seed, n_init=10)
            labels = kmeans.fit_predict(data[:, :2])

            cluster1 = data[labels == 0]
            cluster2 = data[labels == 1]

            split_successful = False
            if len(cluster1) >= self.min_cluster_size:
                if len(cluster1) > self.max_cluster_size:
                    self._recursive_split(cluster1, cluster_id * 2 + 1, depth + 1)
                else:
                    self.clusters.append((cluster_id * 2 + 1, cluster1))
                split_successful = True

            if len(cluster2) >= self.min_cluster_size:
                if len(cluster2) > self.max_cluster_size:
                    self._recursive_split(cluster2, cluster_id * 2 + 2, depth + 1)
                else:
                    self.clusters.append((cluster_id * 2 + 2, cluster2))
                split_successful = True

            if split_successful:
                return

        self.clusters.append((cluster_id, data))

    def _encode_cluster_labels(self, df):
        """
        Encodes the cluster labels as strings for downstream applications.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with cluster assignments.

        Returns
        -------
        pd.DataFrame
            Modified DataFrame with encoded cluster labels.
        """
        df = df.copy()
        list_variables = self.vars_encode + ['cluster']
        df[list_variables] = df[list_variables].astype(str)
        df = df.reset_index(drop=True)
        df['cluster_encoded'] = 'cluster_encoded+_' + df['cluster']
        return df

    def fit(self, X, flavor='pandas'):
        """
        Fit the clustering model.

        Parameters
        ----------
        X : Union[pd.DataFrame, pyspark.sql.DataFrame]
            Input data.
        flavor : str, optional
            Type of DataFrame ('pandas' or 'pyspark'), by default 'pandas'.

        Raises
        ------
        Exception
            If flavor is not supported.
        """
        if flavor == 'pandas':
            self._recursive_split(X[self.geoloc_columns].values, cluster_id=0)
            self.original_df = X

        elif flavor == 'pyspark':
            X.cache().count()
            X = X.toPandas()
            self._recursive_split(X[self.geoloc_columns].values, cluster_id=0)
            self.original_df = X

        else:
            raise Exception("Flavor not supported.")

    def get_cluster_dataframe(self):
        """
        Generate a DataFrame with cluster assignments and centroids.

        Returns
        -------
        Tuple[pd.DataFrame, np.ndarray]
            DataFrame with cluster labels, and array with centroids.
        """
        data_list = []
        centroids = []

        for cluster_id, cluster in self.clusters:
            cluster = np.array(cluster)
            centroid = cluster.mean(axis=0)
            centroids.append(centroid)

            for point in cluster:
                data_list.append([point[0], point[1], cluster_id])

        df = pd.DataFrame(data_list, columns=self.geoloc_columns + ['cluster'])
        df = pd.concat([df, self.original_df.drop(columns=self.geoloc_columns)], axis=1)

        if self.encode:
            df = self._encode_cluster_labels(df)

        return df, np.array(centroids)