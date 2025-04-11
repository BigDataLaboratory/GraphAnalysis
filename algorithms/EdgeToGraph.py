import logging

import networkx as nx
import igraph as ig

class EdgeToGraph:
    logger = logging.getLogger('EdgeToGraph')

    def __init__(self, type = 'nx'):
        self.type = type
        self.g = nx.MultiDiGraph() if type == 'nx' else ig.Graph(directed=True)

    def get_graph(self):
        return self.g

    def to_graph(self, edge_batch = None):
        if edge_batch == None:
            raise AttributeError('edge_list cannot be None')

        if self.type == 'nx':
            self.g.add_edges_from(
                [(e[1], e[2], e[0], {"weight": int(e[3])}) for e in edge_batch]
            )

            for u, v, edge_key, data in self.g.edges(keys=True, data=True):
                if edge_key in ['0', '4', '5']:
                    self.g.nodes[u]["type"] = 'u'
                    self.g.nodes[v]["type"] = 'u'
                elif edge_key == '2':
                    self.g.nodes[u]["type"] = 'u'
                    self.g.nodes[v]["type"] = 'h'
                elif edge_key == '3':
                    self.g.nodes[u]["type"] = 'h'
                    self.g.nodes[v]["type"] = 'h'
            self.logger.info(f" Finished processing {len(edge_batch)} edges.")
