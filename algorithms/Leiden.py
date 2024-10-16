# -*- coding: utf-8 -*-
import igraph as ig
import leidenalg as la
import modin.pandas as pd
import sys
import time
import gc
import logging
import json
from igraph import *


class Leiden():
    logger = logging.getLogger('Leiden')

    def __init__(self):
        pass

    def csv_to_igraph(self, **kwargs):

        # indica se è necessario aggiungere il tipo di nodo all'interno del file di output
        # se non viene inserito, c'è un risparmio di memoria
        data_type_needed = False

        self.logger.info("Caricamento grafo da csv in corso...")

        if kwargs.get("input_csv_graph_file_path", None) is not None:
            start = time.time()
            graph_list = []
            for s in kwargs.get("input_csv_graph_file_path", None):
                df = pd.read_csv(s)
                graph_list.append(df)
            dataframe_graph = pd.concat(graph_list, ignore_index=True)
            end = time.time()

            self.logger.info("Caricamento grafo completato!")
            self.logger.info("Elapsed time: " + str(end - start))
            self.logger.info(len(dataframe_graph.axes[1]))
        elif kwargs.get("dataframe_graph", None) is not None:
            dataframe_graph = kwargs.get("dataframe_graph", None)

        # Rinominare le variabili leiden necessita di questi nomi delle colonne
        dataframe_graph.columns = ['source', 'target', 'weight', 'type']
        type_dict = {}

        if data_type_needed:

            self.logger.info("Creazione dizionario nodo-tipo in corso...")
            start = time.time()
            count = 0
            number_of_source_nodes = len(dataframe_graph.index)
            # creazione dict nodo--->tipo_di_nodo
            for row in dataframe_graph.itertuples():
                # il nodo source è sempre uno user
                id_source = row.source

                if id_source in type_dict and type_dict[id_source] != "user":
                    self.logger.info("Abbiamo un problema... ID:" + str(
                        id_source) + " è già presente ed era un hashtag, mentre ora è uno user")
                    sys.exit(-1)
                type_dict[id_source] = "user"

                # il nodo target può essere uno user o un hashtag in base al valore della colonna type
                id_target = row.target
                type_data = row.type

                if type_data == "hashtag":
                    if id_target in type_dict and type_dict[id_target] != "hashtag":
                        self.logger.info("Abbiamo un problema... ID:" + str(
                            id_target) + " è già presente ed era uno user, mentre ora è un hashtag")
                    type_dict[id_target] = "hashtag"
                elif type_data == "retweet":
                    if id_target in type_dict and type_dict[id_target] != "user":
                        self.logger.info("Abbiamo un problema... ID:" + str(
                            id_target) + " è già presente ed era un hashtag, mentre ora è uno user")
                    type_dict[id_target] = "user"
                elif type_data == "mention":
                    if id_target in type_dict and type_dict[id_target] != "user":
                        self.logger.info("Abbiamo un problema... ID:" + str(
                            id_target) + " è già presente e non era uno user, mentre ora è uno user")
                    type_dict[id_target] = "user"
                elif type_data == "reply":
                    if id_target in type_dict and type_dict[id_target] != "user":
                        self.logger.info("Abbiamo un problema... ID:" + str(
                            id_target) + " è già presente e non era uno user, mentre ora è uno user")
                    type_dict[id_target] = "user"
                elif type_data == "cooccurrences":
                    if id_target in type_dict and type_dict[id_target] != "hashtag":
                        self.logger.info("Abbiamo un problema... ID:" + str(
                            id_target) + " è già presente ed era uno user, mentre ora è un hashtag")
                    type_dict[id_target] = "hashtag"
                else:
                    self.logger.info(
                        "ERRORE: Il tipo di arco sembra non essere né hashtag né retweet né mentions nè reply nè cooccurrences")
                    sys.exit(
                        "ERRORE: Il tipo di arco sembra non essere né hashtag né retweet né mentions nè reply nè cooccurrences")
                count += 1
                if count % 100000 == 0:
                    self.logger.info(
                        "Eseguiti " + str(count) + " nodi sorgente su " + str(number_of_source_nodes) + " nodi totali")

            end = time.time()
            self.logger.info("Creazione dizionario nodo-tipo completato")
            self.logger.info("Numero di nodi aggiunti: " + str(len(type_dict)))
            self.logger.info("Elapsed time: " + str(end - start))

        self.logger.info("Conversione degli hash in string in corso...")
        start = time.time()
        # trasformare in stringhe gli hash
        dataframe_graph['source'] = dataframe_graph['source'].apply(str)
        dataframe_graph['target'] = dataframe_graph['target'].apply(str)
        end = time.time()
        self.logger.info("Conversione degli hash in string completata")
        self.logger.info("Elapsed time: " + str(end - start))

        self.logger.info(dataframe_graph.head())

        self.logger.info("Conversione del dataframe in tuple in corso...")
        start = time.time()
        tuples = [tuple(x) for x in dataframe_graph.values]
        end = time.time()
        self.logger.info("Conversione del dataframe in tuple completata")
        self.logger.info("Elapsed time: " + str(end - start))

        # libero la memoria
        self.logger.info("Pulizia dell'oggetto dataframe_graph e garbage collector in corso...")
        del dataframe_graph
        gc.collect()
        self.logger.info("Pulizia dell'oggetto dataframe_graph e garbage collector completata!")

        self.logger.info("Import del grafo in formato iGraph in corso...")
        start = time.time()
        # da csv a gml, weight e type attributi degli edge
        data_graph = ig.Graph.TupleList(tuples, directed=True, edge_attrs=['weight', 'type'])
        end = time.time()
        self.logger.info('csv importato in formato iGraph!')
        self.logger.info("Elapsed time: " + str(end - start))

        # libero la memoria
        del tuples
        gc.collect()

        if data_type_needed:
            self.logger.info("Aggiunta del tipo di dato ai nodi in corso...")
            start = time.time()
            count = 0
            num_nodes_graph = data_graph.vcount()
            for id_node, type_node in type_dict.items():
                node = data_graph.vs.find(name=str(id_node))
                id_graph_node = node.index
                data_graph.vs[id_graph_node]["type"] = type_node
                count += 1
                if count % 10000 == 0:
                    self.logger.info("Aggiunto il tipo dei nodi a " + str(count) + " nodi su " + str(
                        num_nodes_graph) + " nodi totali")
            end = time.time()
            self.logger.info('Tipo dei nodi aggiunto!')
            self.logger.info("Elapsed time: " + str(end - start))
            # libero la memoria
            del type_dict
            gc.collect()

        return data_graph

    def add_leiden_to_igraph(self, data_graph):

        self.logger.info("Calcolo del PageRank in corso...")
        start = time.time()
        data_graph.vs['pagerank'] = data_graph.pagerank(directed=True, weights='weight', implementation="prpack")
        end = time.time()
        self.logger.info('PageRank completato!')
        self.logger.info("Elapsed time: " + str(end - start))

        self.logger.info("Calcolo di Leiden con CPM Quality Function in corso...")
        start = time.time()
        # Se non si specifica i weights allora Leiden considera il grafo non pesato
        """
        partition = la.find_partition(data_graph, la.CPMVertexPartition, weights='weight', seed=0)
        # Aggiunge il cluster alle proprietà del nodo
        data_graph.vs['cluster'] = partition.membership
        summary(partition)
        end = time.time()
        self.logger.info('Calcolo di Leiden con CPM Quality Function completato!')
        self.logger.info("Elapsed time: " + str(end - start))

        self.logger.info("Calcolo di Leiden con Modularità in corso...")
        start = time.time()
        partition2 = la.find_partition(data_graph, la.ModularityVertexPartition, weights='weight', seed=0)
        data_graph.vs['cluster2'] = partition2.membership
        summary(partition2)
        end = time.time()
        self.logger.info('Calcolo di Leiden con Modularità completato!')
        self.logger.info("Elapsed time: " + str(end - start))
        """
        self.logger.info("Calcolo di Leiden con CPM Quality Function e resolution parameter (0.4) in corso...")
        start = time.time()
        partition3 = la.find_partition(data_graph, la.CPMVertexPartition, resolution_parameter=0.4, weights='weight',
                                       seed=0)
        data_graph.vs['cluster3'] = partition3.membership
        summary(partition3)
        end = time.time()
        self.logger.info('Calcolo di Leiden con CPM Quality Function e resolution parameter (0.4) completato!')
        self.logger.info("Elapsed time: " + str(end - start))

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
