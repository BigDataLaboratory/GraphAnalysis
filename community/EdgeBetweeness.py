import logging
import time
import uuid
import os
import networkx as nx
from networkx.algorithms.community import girvan_newman, modularity
from Utils.Writer import Writer

class EdgeBetweenness:
    logger = logging.getLogger('EdgeBetweenness')

    def __init__(self, data_graph):
        """
        Inizializza l'analizzatore con l'ID univoco e il grafo igraph 
        generato dalla pipeline principale.
        """
        self.id = uuid.uuid1().hex
        self.data_graph = data_graph

    def get_graph(self):
        """
        Restituisce il grafo associato.
        Mantiene la stessa firma del metodo presente in Leiden.py.
        """
        return self.data_graph

    def compute_edge_betweenness(self):
        """
        Esegue l'algoritmo di Girvan-Newman convertendo temporaneamente
        il grafo in NetworkX, per poi mappare i risultati su igraph.
        """
        self.logger.info("Avvio calcolo Edge Betweenness (Girvan-Newman)")
        start = time.time()

        # 1. Conversione da igraph a NetworkX
        self.logger.info("Conversione del grafo in formato NetworkX...")
        nx_graph = self.data_graph.to_networkx()

        # 2. Esecuzione dell'algoritmo
        self.logger.info(f"Esecuzione Girvan-Newman su {nx_graph.number_of_nodes()} nodi...")
        communities_generator = girvan_newman(nx_graph)
        
        best_modularity = -1.0
        optimal_partition = None
        step = 0

        # Iteriamo sulle divisioni per trovare quella che massimizza la modularità
        for communities in communities_generator:
            step += 1
            current_partition = list(communities)
            
            # Calcolo della modularità per questa specifica partizione
            current_modularity = modularity(nx_graph, current_partition)

            self.logger.debug(f"Step {step}: Trovate {len(current_partition)} comunità. Modularità: {current_modularity:.4f}")

            # Salviamo la partizione se la modularità migliora
            if current_modularity > best_modularity:
                best_modularity = current_modularity
                optimal_partition = current_partition
            else:
                self.logger.info(f"Picco di modularità raggiunto allo step {step-1}. Interruzione per risparmiare risorse.")
                break

        # 3. Mappatura dei risultati sul grafo originale
        self.logger.info("Mappatura delle comunità sul grafo igraph originale...")
        
        # Inizializziamo un array con -1 (indica nessuna comunità)
        labels = [-1] * self.data_graph.vcount()
        
        # Assegnamo l'ID della comunità ad ogni nodo
        for community_id, node_set in enumerate(optimal_partition):
            for node in node_set:
                labels[int(node)] = community_id
                
        # Salviamo il risultato come attributo dei vertici
        self.data_graph.vs["edge_betweenness"] = labels
        self.data_graph["eb_modularity"] = best_modularity

        end = time.time()
        self.logger.info("Edge Betweenness completata. Tempo trascorso: " + str(end - start))
        
        return labels

    def export_partition(self, g, file_path, attr=None):
        """
        Esporta la partizione in CSV.
        Firma allineata con Leiden.py per la massima compatibilità.
        """
        Writer.create_dir(file_path, self.id)
        full_path = os.sep.join([file_path, self.id, "communities_edge_betweenness.csv"])
        Writer.export_nodes_with_attributes(g, full_path, attr)

    def export_graph(self, g, file_path):
        """
        Esporta il grafo completo.
        Firma allineata con Leiden.py.
        """
        Writer.create_dir(file_path, self.id)
        full_path = os.sep.join([file_path, self.id, "nodes_with_communities_eb.csv"])
        Writer.export_nodes_with_attributes(g, full_path)