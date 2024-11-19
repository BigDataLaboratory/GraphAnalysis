# -*- coding: utf-8 -*-
import json
import logging
import os
import time
import uuid

import igraph as ig
import leidenalg as la
import modin.pandas as pd
import numpy as np

from Utils.Writer import Writer


class Leiden:
    logger = logging.getLogger('Leiden')

    def __init__(self):
        self.id = uuid.uuid1().hex

    def csv_to_igraph(self, **kwargs):
        self.logger.info("Caricamento grafo da csv in corso...")

        if kwargs.get("input_csv_graph_file_path", None) is not None:
            w = Writer()
            graph = w.read_csv_files_in_folder_parallel(kwargs.get("input_csv_graph_file_path", None))
        elif kwargs.get("graph", None) is not None:
            graph = kwargs.get("graph", None)

        graph[:] = [(src, dst, int(weight), type) for type, src, dst, weight in graph]
        self.logger.info("Import del grafo in formato iGraph in corso...")
        start = time.time()
        # da csv a gml, weight e type attributi degli edge
        data_graph = ig.Graph.TupleList(graph, directed=True, edge_attrs=['weight', 'type'])

        # Iterate over edges and update vertex attributes
        for edge in data_graph.es:
            type = edge["type"]
            if type in ['0', '4', '5']:
                data_graph.vs[edge.source]["type"] = 'u'
                data_graph.vs[edge.target]["type"] = 'u'
            elif type == '2':
                data_graph.vs[edge.source]["type"] = 'u'
                data_graph.vs[edge.target]["type"] = 'h'
            elif type == '3':
                data_graph.vs[edge.source]["type"] = 'h'
                data_graph.vs[edge.target]["type"] = 'h'

        end = time.time()
        self.logger.info('csv importato in formato iGraph!')
        self.logger.info("Elapsed time: " + str(end - start))
        return data_graph

    def compute_pagerank(self, data_graph):
        self.logger.info("Calcolo del PageRank in corso")
        start = time.time()
        data_graph.vs['pagerank'] = data_graph.pagerank(directed=True, weights='weight', implementation="prpack")
        end = time.time()
        self.logger.info('PageRank completato!')
        self.logger.info("Elapsed time: " + str(end - start))
        return data_graph

    def compute_leiden(self, data_graph, resolution_parameter_range=(0.1, 1.0)):
        self.logger.info("Calcolo di Leiden con CPM Quality Function e resolution parameter")
        for rp in np.linspace(resolution_parameter_range[0], resolution_parameter_range[1], num=10):
            start = time.time()

            rp_round = round(rp, 1)
            self.logger.info("Calcolo di Leiden con CPM Quality Function e resolution parameter = {}".format(rp_round))

            partition = la.find_partition(data_graph, la.CPMVertexPartition, resolution_parameter=rp_round,
                                          weights='weight',
                                          seed=0)
            data_graph.vs["{}".format(rp_round)] = partition.membership
            end = time.time()
            self.logger.info(
                "Calcolo di Leiden con CPM Quality Function e resolution parameter = {} completato".format(rp_round))
            self.logger.info("Elapsed time: " + str(end - start))

        return data_graph

    def export_graph(self, g, file_path):
        w = Writer()
        w.create_dir(file_path, self.id)
        file_path = os.sep.join([file_path, self.id, "nodes_with_communities.csv"])
        w.export_nodes_with_attributes(g, file_path)

    def get_cluster_nodes(self, g, cluster_num, cluster_type):
        """
        Get all vertices belonging to an input cluster
        :param g: graph
        :param cluster_num: input cluster
        :param cluster_type: cluster name
        :return: all node belonging to the input cluster_num
        """
        cluster_nodes = []
        for v in g.vs:
            if v[cluster_type] == cluster_num:
                cluster_nodes.append(v["name"])
        return cluster_nodes

    def get_avg_pagerank(self, g):
        """
        Get all vertices belonging to an input graph
        :param g: graph
        :return: all node belonging to the input graph
        """
        for v in g.vs:
            pagerank = pagerank + v["pagerank"]

        result = pagerank / g.vcount()

    def get_clusters_nodes(self, g):
        """
        Get all vertices belonging to an input graph
        :param g: graph
        :return: all node belonging to the input graph
        """
        # cluster1_count = {}
        # cluster2_count = {}
        cluster3_count = {}
        # sum_pr_0 = {}
        # sum_pr_2 = {}
        sum_pr_3 = {}
        for v in g.vs:
            # cluster1_count[v["cluster"]] = cluster1_count.get(v["cluster"], 0) + 1
            # cluster2_count[v["cluster2"]] = cluster2_count.get(v["cluster2"], 0) + 1
            cluster3_count[v["cluster3"]] = cluster3_count.get(v["cluster3"], 0) + 1
            # sum_pr_0[v["cluster"]] = sum_pr_0.get(v["cluster"], 0) + v["pagerank"]
            # sum_pr_2[v["cluster2"]] = sum_pr_2.get(v["cluster2"], 0) + v["pagerank"]
            sum_pr_3[v["cluster3"]] = sum_pr_3.get(v["cluster3"], 0) + v["pagerank"]

        avg_cluster = {"cluster0": {}, "cluster2": {}, "cluster3": {}}
        # for k, v in sum_pr_0.items():
        #    avg_cluster["cluster0"][k] = v/cluster1_count[k]
        # for k, v in sum_pr_2.items():
        #    avg_cluster["cluster2"][k] = v/cluster2_count[k]
        for k, v in sum_pr_3.items():
            avg_cluster["cluster3"][k] = v / cluster3_count[k]

        pr_cluster = {}
        for vs in g.vs:
            pr_cluster[vs["cluster3"]] = [vs["name"], vs["pagerank"]]

        # with open('/home/pasquini/GraphAnalysis/resources/Cluster1_nodes.json', 'w', encoding='utf-8') as f:
        #    json.dump(cluster1_count, f)

        # with open('/home/pasquini/GraphAnalysis/resources/Cluster2_nodes.json', 'w', encoding='utf-8') as f:
        #    json.dump(cluster2_count, f)

        with open('/home/pasquini/GraphAnalysis/resources/Cluster3_nodes.json', 'w', encoding='utf-8') as f:
            json.dump(cluster3_count, f)

        with open('/home/pasquini/GraphAnalysis/resources/Clusters_Pagerank.json', 'w', encoding='utf-8') as f:
            json.dump(pr_cluster, f)

    def max_clusters_nodes(self):
        with open('/home/pasquini/GraphAnalysis/resources/Clusters_Pagerank.json', 'r') as f:
            data = json.load(f)

        dati_ordinati = sorted(data.items(), key=lambda x: x[1][1], reverse=True)

        with open('/home/pasquini/GraphAnalysis/resources/Clusters_avg_Pagerank_ordered.json', 'w',
                  encoding='utf-8') as f:
            json.dump(dati_ordinati, f)

    def get_clusters_as_list(self, g):
        df_clusters = []
        for v in g.vs:
            # n = {'node_hash': v['name'], 'cluster1': v['cluster'], 'cluster2': v['cluster2'],
            # 'cluster3': v['cluster3'], 'pagerank': v['pagerank']}
            n = {'node_hash': v['name'], 'cluster3': v['cluster3'], 'pagerank': v['pagerank']}
            df_clusters.append(n)
        return df_clusters

    def get_clusters_as_dataframe(self, g):
        """
        Get all vertices belonging to an input cluster
        :param g: graph
        :param cluster_num: input cluster
        :param cluster_type: cluster name
        :return: all node belonging to the input cluster_num
        """
        clusters_list = self.get_clusters_as_list(g)
        clusters_df = pd.DataFrame(clusters_list)

        return clusters_df

    def hash_to_name(self, cluster_df, user_map_path_file, hashtag_map_path_file, screen_name_user_id_map_path_file):

        user_map = pd.read_csv(user_map_path_file)
        user_map['type'] = 'user'
        user_map['node_hash'] = user_map['node_hash'].astype(str)

        screen_name_user_id_map = pd.read_csv(screen_name_user_id_map_path_file)
        screen_name_user_id_map['type'] = 'user'

        hashtag_map = pd.read_csv(hashtag_map_path_file)
        hashtag_map['type'] = 'hashtag'
        hashtag_map['node_hash'] = hashtag_map['node_hash'].astype(str)

        user_df = user_map.merge(right=cluster_df, on=['node_hash', 'type'],
                                 how='inner').drop(columns=['node_hash', 'weight'])
        user_df = screen_name_user_id_map.merge(right=user_df, on=['user_id', 'type'],
                                                how='inner').drop(columns=['user_id', 'type', 'weight'])

        hashtag_df = hashtag_map.merge(right=cluster_df, on=['node_hash', 'type'],
                                       how='inner').drop(columns=['node_hash', 'weight', 'type'])

        return [user_df, hashtag_df]


'''
    def read_edge_list_csv(self, filename):
        """
        Legge il grafo dalla edge list fornita come file CSV.

        Args:
        filename (str): Il percorso del file CSV di edge list.

        Returns:
        nk.Graph: Il grafo letto dalla edge list.
        """
        G = nk.Graph(weighted=True, directed=True)
        node_mapping = {}
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Legge l'header, ma non lo usiamo in questo caso
            for row in reader:
                source = int(row[0])
                target = int(row[1])
                weight = float(row[2])
                edge_type = row[3]

                # Aggiungi i nodi se non esistono già, tenendo traccia degli identificatori originali
                if source not in node_mapping:
                    node_mapping[source] = G.addNode()
                if target not in node_mapping:
                    node_mapping[target] = G.addNode()

                # Aggiungi l'arco al grafo
                G.addEdge(node_mapping[source], node_mapping[target], w=weight)
        return G, node_mapping


    def calculate_approx_betweenness(self, graph):
        """
        Calcola la betwennes approssimata per ciascun nodo nel grafo.

        Args:
        graph (nk.Graph): Il grafo su cui calcolare la betwennes.

        Returns:
        dict: Un dizionario che mappa ciascun nodo al suo valore di betwennes approssimata.
        """
        n = graph.numberOfNodes()
        nsamples = int(n*0.2)
        approx_betweenness = nk.centrality.EstimateBetweenness(graph, nSamples=nsamples, parallel=True).run().scores()
        return approx_betweenness
'''
