import logging

import pandas as pd


class ClusterMap:
    logger = logging.getLogger('ClusterMap')

    def read_cluster_file(self, path, format="csv", sep=',', header=0, columns=None):
        if columns is None:
            columns = ['node_hash', 'cluster3', 'pagerank']
        clusters = pd.read_csv(path, sep=sep, header=header)
        return clusters

    def read_map_file(self, path, format="csv", sep=',', header=0):
        map = pd.read_csv(path, sep=sep, header=header)
        return map

    def from_hash_to_id(self, clusters, map):
        clusters_map = clusters.merge(map, left_on='node_hash', right_on='node_hash', how='left')
        clusters_map.sort_values(by=['cluster3'], inplace=True)
        return clusters_map

    def group_and_count_by_cluster(self, clusters):
        clusters_grouped = clusters.groupby("cluster3").size()
        clusters_grouped = clusters_grouped.reset_index()
        return clusters_grouped
