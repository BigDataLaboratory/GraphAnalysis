# -*- coding: utf-8 -*-
import igraph
import igraph as ig
import leidenalg as la
import pandas as pd
import sys
from igraph import *
import time
import gc

print("Ciao")

# Nel caso si avesse il formato gml
# data_graph = Graph.Read_GraphML("/Users/gba/Downloads/dataset_multigraph_2022_06_13.graphml")

#Costruito il sottografo del cluster 0
#dataframe_graph = pd.read_csv(
#    '/Users/gba/Google Drive/ACCOUNT_RUSSI/data/Analisi_2022_11_15/clustering/dataset_multigraph_gruppo0_2022_06_13.csv',
#    sep='\t')

'''
Script da essere lanciato con il comando
python LeidenClustering_gba.py /path/to/csv/graph/file.csv /path/to/output/gml/file/file.gml
'''
num_arguments = len(sys.argv) - 1
if num_arguments != 2:
    print("""\
Lo script calcola varie comunità su un dato grafo pesato di input attraverso l'algoritmo di Leiden.
E' necessario indicare il percorso al file csv di input e il percorso al file gml di output di questo script.

Usage:  python LeidenClustering_gba.py /path/to/csv/graph/file.csv /path/to/output/gml/file/file.gml
""")
    print("Il numero di argomenti aggiuntivi passati allo script è sbagliato. Il numero di argomenti passato è "
          + str(num_arguments) + " e non 2")
    sys.exit(0)


input_csv_graph_file_path = sys.argv[1]
output_gml_file_path = sys.argv[2]

#indica se è necessario aggiungere il tipo di nodo all'interno del file di output
#se non viene inserito, c'è un risparmio di memoria
data_type_needed = False
print("Caricamento grafo da csv in corso...")
start = time.time()
dataframe_graph = pd.read_csv(
    # './analysis/QCPS_2/2022-12-13/2022-12-13_multigrafo_hashtag_retweet.csv',
    input_csv_graph_file_path)
#dataframe_graph = dataframe_graph[["hash_s", "hash_t", "frequency"]]
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
        # print(id_source)
        if id_source in type_dict and type_dict[id_source] != "user":
            print("Abbiamo un problema... ID:" + str(id_source) + " è già presente ed era un hashtag, mentre ora è uno user")
            sys.exit(-1)
        type_dict[id_source] = "user"
        #il nodo target può essere uno user o un hashtag in base al valore della colonna type
        id_target = row.target
        type_data = row.type
        # print(type_data)
        if type_data == "hashtag":
            if id_target in type_dict and type_dict[id_target] != "hashtag":
                print(
                    "Abbiamo un problema... ID:" + str(id_target) + " è già presente ed era uno user, mentre ora è un hashtag")
                sys.exit(-2)
            type_dict[id_target] = "hashtag"
        elif type_data == "retweet":
            if id_target in type_dict and type_dict[id_target] != "user":
                print(
                    "Abbiamo un problema... ID:" + str(id_target) + " è già presente ed era un hashtag, mentre ora è uno user")
                sys.exit(-3)
            type_dict[id_target] = "user"
        elif type_data == "mentions":
            if id_target in type_dict and type_dict[id_target] != "user":
                print(
                    "Abbiamo un problema... ID:" + str(id_target) + " è già presente e non era uno user, mentre ora è uno user")
                sys.exit(-4)
            type_dict[id_target] = "user"
        else:
            print("ERRORE: Il tipo di arco sembra non essere né hashtag né retweet né mentions")
            sys.exit("ERRORE: Il tipo di arco sembra non essere né hashtag né retweet né mentions")
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
#TODO NECESSARIO?
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
#dataframe_graph = pd.DataFrame()
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

# print(data_graph.get_edgelist()[0:10])
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

# cluster0 = read.csv('/Users/gba/Google Drive/ACCOUNT_RUSSI/data/Analisi_2022-06-13/dataset_multigraph_gruppo0.csv', header=TRUE)
# print(ig.__version__)

# betw = data_graph.community_edge_betweenness()
# summary(betw)

if False:
    print("Calcolo del PageRank in corso...")
    start = time.time()
    data_graph.vs['pagerank'] = data_graph.pagerank(directed=True, weights='weight', niter=1000, eps=0.0001)
    end = time.time()
    print('PageRank completato!')
    print("Elapsed time: " + str(end - start))

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

#data_graph.save('/Users/gba/Google Drive/ACCOUNT_RUSSI/data/Analisi_2022-06-13/dataset_multigraph_gruppo0.csv')
print('Salvataggio del grafo in formato gml in corso...')
start = time.time()
# data_graph.save('./analysis/QCPS_2/2022-12-13/dataset_multigraph_with_node_type_2022_12_13.gml')
data_graph.save(output_gml_file_path)
#data_graph.write_edgelist(output_gml_file_path)
#print(data_graph.get_edge_dataframe())
#data_graph.write(output_gml_file_path, format="graphml")

end = time.time()
print('Salvataggio del grafo in formato gml completato!')
print("Elapsed time: " + str(end - start))
# print(partition2)

# ig.plot(partition2)

def get_cluster_nodes(g, cluster_num, cluster_type="cluster3"):
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

def get_clusters_as_dataframe(g):
    """
    Get all vertices belonging to an input cluster
    :param g: graph
    :param cluster_num: input cluster
    :param cluster_type: cluster name
    :return: all node belonging to the input cluster_num
    """
    clusters_list = get_clusters_as_list(g)
    clusters_df = pd.DataFrame(clusters_list)
    print(clusters_df)
    return clusters_df

def get_clusters_as_list(g):
    df_clusters = []
    for v in g.vs:
        #n = {'node': v['name'], 'cluster1': v['cluster'], 'cluster2': v['cluster2'], 'cluster3': v['cluster3'], 'pagerank': v['pagerank']}
        n = {'node': v['name'], 'cluster1': v['cluster'], 'cluster2': v['cluster2'], 'cluster3': v['cluster3']}
        df_clusters.append(n)
    return df_clusters

def write_clusters(clusters_df, path_file):
    clusters_df.to_csv(path_file, sep="\t", header=True, index=False)