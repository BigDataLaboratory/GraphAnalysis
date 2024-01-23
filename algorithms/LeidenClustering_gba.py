# -*- coding: utf-8 -*-
import igraph as ig
from igraph import *
import leidenalg as la
import pandas as pd
import sys
import time
import gc

class Leiden:

    def csv_to_igraph(self, input_csv_graph_file_path):

        #indica se è necessario aggiungere il tipo di nodo all'interno del file di output
        #se non viene inserito, c'è un risparmio di memoria
        data_type_needed = True

        print("Caricamento grafo da csv in corso...")

        start = time.time()
        dataframe_graph = pd.read_csv(input_csv_graph_file_path)
        end = time.time()

        print("Caricamento grafo completato!")
        print("Elapsed time: " + str(end - start))
        print(len(dataframe_graph.axes[1]))

        #Rinominare le variabili leiden necessita di questi nomi delle colonne
        dataframe_graph.columns = ['source', 'target', 'weight', 'type']
        type_dict = {}

        if data_type_needed:

            print("Creazione dizionario nodo-tipo in corso...")
            start = time.time()
            count = 0
            number_of_source_nodes = len(dataframe_graph.index)
            #creazione dict nodo--->tipo_di_nodo
            for row in dataframe_graph.itertuples():
                #il nodo source è sempre uno user
                id_source = row.source

                if id_source in type_dict and type_dict[id_source] != "user":
                    print("Abbiamo un problema... ID:" + str(id_source) + " è già presente ed era un hashtag, mentre ora è uno user")
                    sys.exit(-1)
                type_dict[id_source] = "user"

                #il nodo target può essere uno user o un hashtag in base al valore della colonna type
                id_target = row.target
                type_data = row.type

                if type_data == "hashtag":
                    if id_target in type_dict and type_dict[id_target] != "hashtag":
                        print("Abbiamo un problema... ID:" + str(id_target) + " è già presente ed era uno user, mentre ora è un hashtag")
                    type_dict[id_target] = "hashtag"
                elif type_data == "retweet":
                    if id_target in type_dict and type_dict[id_target] != "user":
                        print("Abbiamo un problema... ID:" + str(id_target) + " è già presente ed era un hashtag, mentre ora è uno user")
                    type_dict[id_target] = "user"
                elif type_data == "mention":
                    if id_target in type_dict and type_dict[id_target] != "user":
                        print("Abbiamo un problema... ID:" + str(id_target) + " è già presente e non era uno user, mentre ora è uno user")
                    type_dict[id_target] = "user"
                elif type_data == "reply":
                    if id_target in type_dict and type_dict[id_target] != "user":
                        print("Abbiamo un problema... ID:" + str(id_target) + " è già presente e non era uno user, mentre ora è uno user")
                    type_dict[id_target] = "user"
                elif type_data == "cooccurrences":
                    if id_target in type_dict and type_dict[id_target] != "hashtag":
                        print("Abbiamo un problema... ID:" + str(id_target) + " è già presente ed era uno user, mentre ora è un hashtag")
                    type_dict[id_target] = "hashtag"
                else:
                    print("ERRORE: Il tipo di arco sembra non essere né hashtag né retweet né mentions nè reply nè cooccurrences")
                    sys.exit("ERRORE: Il tipo di arco sembra non essere né hashtag né retweet né mentions nè reply nè cooccurrences")
                count += 1
                if count % 100000 == 0:
                    print("Eseguiti " + str(count) + " nodi sorgente su " + str(number_of_source_nodes) + " nodi totali")

            end = time.time()
            print("Creazione dizionario nodo-tipo completato")
            print("Numero di nodi aggiunti: " + str(len(type_dict)))
            print("Elapsed time: " + str(end - start))

        print("Conversione degli hash in string in corso...")
        start = time.time()
        #trasformare in stringhe gli hash
        dataframe_graph['source'] = dataframe_graph['source'].apply(str)
        dataframe_graph['target'] = dataframe_graph['target'].apply(str)
        end = time.time()
        print("Conversione degli hash in string completata")
        print("Elapsed time: " + str(end - start))

        print(dataframe_graph.head())

        print("Conversione del dataframe in tuple in corso...")
        start = time.time()
        tuples = [tuple(x) for x in dataframe_graph.values]
        end = time.time()
        print("Conversione del dataframe in tuple completata")
        print("Elapsed time: " + str(end - start))

        #libero la memoria
        print("Pulizia dell'oggetto dataframe_graph e garbage collector in corso...")
        del dataframe_graph
        gc.collect()
        print("Pulizia dell'oggetto dataframe_graph e garbage collector completata!")

        print("Import del grafo in formato iGraph in corso...")
        start = time.time()
        #da csv a gml, weight e type attributi degli edge
        data_graph = ig.Graph.TupleList(tuples, directed=True, edge_attrs=['weight', 'type'])
        end = time.time()
        print('csv importato in formato iGraph!')
        print("Elapsed time: " + str(end - start))

        #libero la memoria
        del tuples
        gc.collect()

        if data_type_needed:
            print("Aggiunta del tipo di dato ai nodi in corso...")
            start = time.time()
            count = 0
            num_nodes_graph = data_graph.vcount()
            for id_node, type_node in type_dict.items():
                node = data_graph.vs.find(name=str(id_node))
                id_graph_node = node.index
                data_graph.vs[id_graph_node]["type"] = type_node
                count += 1
                if count % 10000 == 0:
                    print("Aggiunto il tipo dei nodi a " + str(count) + " nodi su " + str(num_nodes_graph) + " nodi totali")
            end = time.time()
            print('Tipo dei nodi aggiunto!')
            print("Elapsed time: " + str(end - start))
            #libero la memoria
            del type_dict
            gc.collect()
        
        return data_graph

    def add_leiden_to_igraph(self, data_graph):

        print("Calcolo di Leiden con CPM Quality Function in corso...")
        start = time.time()
        # Se non si specifica i weights allora Leiden considera il grafo non pesato
        partition = la.find_partition(data_graph, la.CPMVertexPartition, weights='weight', seed=0)
        # Aggiunge il cluster alle proprietà del nodo
        data_graph.vs['cluster'] = partition.membership
        summary(partition)
        end = time.time()
        print('Calcolo di Leiden con CPM Quality Function completato!')
        print("Elapsed time: " + str(end - start))

        print("Calcolo di Leiden con Modularità in corso...")
        start = time.time()
        partition2 = la.find_partition(data_graph, la.ModularityVertexPartition, weights='weight', seed=0)
        data_graph.vs['cluster2'] = partition2.membership
        summary(partition2)
        end = time.time()
        print('Calcolo di Leiden con Modularità completato!')
        print("Elapsed time: " + str(end - start))

        print("Calcolo di Leiden con CPM Quality Function e resolution parameter (0.4) in corso...")
        start = time.time()
        partition3 = la.find_partition(data_graph, la.CPMVertexPartition, resolution_parameter=0.4, weights='weight', seed=0)
        data_graph.vs['cluster3'] = partition3.membership
        summary(partition3)
        end = time.time()
        print('Calcolo di Leiden con CPM Quality Function e resolution parameter (0.4) completato!')
        print("Elapsed time: " + str(end - start))
    
    def igraph_to_file(self, data_graph, output_file_path='../resources/output_graph', ext='gml'):

        print('Salvataggio del grafo in formato gml in corso...')
        start = time.time()
        data_graph.save(output_file_path + '.' + ext)
        end = time.time()
        print('Salvataggio del grafo in formato gml completato!')
        print("Elapsed time: " + str(end - start))


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

    def get_clusters_as_list(self, g):
        df_clusters = []
        for v in g.vs:
            #n = {'node': v['name'], 'cluster1': v['cluster'], 'cluster2': v['cluster2'], 'cluster3': v['cluster3'], 'pagerank': v['pagerank']}
            n = {'node_hash': v['name'], 'cluster1': v['cluster'], 'cluster2': v['cluster2'], 'cluster3': v['cluster3'], 'type': v['type']}
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
    
    def write_clusters(self, clusters_df, path_file):
        clusters_df.to_csv(path_file, sep="\t", header=True, index=False)

    def hash_to_name(self, cluster_df):

        user_map = pd.read_csv('./user_map.csv')
        user_map['type'] = 'user'
        user_map['node_hash'] = user_map['node_hash'].astype(str)

        hashtag_map = pd.read_csv('./hashtag_map.csv')
        hashtag_map['type'] = 'hashtag'
        hashtag_map['node_hash'] = hashtag_map['node_hash'].astype(str)

        user_df = user_map.merge(right=cluster_df, on=['node_hash', 'type'], how='inner').drop(columns=['node_hash', 'weight'])
        user_df.to_csv('./user_clusters.csv', index=False)

        hashtag_df = hashtag_map.merge(right=cluster_df, on=['node_hash', 'type'], how='inner').drop(columns=['node_hash', 'weight'])
        user_df.to_csv('./hashtag_clusters.csv', index=False)
