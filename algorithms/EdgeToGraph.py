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
            raise AttributeError('edge_batch cannot be None')

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
        elif self.type == 'igraph':
            node_names = set()
            for e in edge_batch:
                node_names.update([e[1], e[2]])

            # Add vertices with unique names
            self.g.add_vertices(len(node_names))
            self.g.vs["name"] = list(node_names)
            self.g.vs["type"] = [None] * self.g.vcount()  # initialize node type

            # Add edges using names (igraph resolves name->index internally)
            edge_list = [(e[1], e[2]) for e in edge_batch]
            self.g.add_edges(edge_list)
            self.g.es["type"] = [e[0] for e in edge_batch]
            self.g.es["weight"] = [int(e[3]) for e in edge_batch]

            # Set node types based on edge types
            for e in self.g.es:
                src = e.source
                dst = e.target
                edge_type = e["type"]

                if edge_type in ['0', '4', '5']:
                    self.g.vs[src]["type"] = 'u'
                    self.g.vs[dst]["type"] = 'u'
                elif edge_type == '2':
                    self.g.vs[src]["type"] = 'u'
                    self.g.vs[dst]["type"] = 'h'
                elif edge_type == '3':
                    self.g.vs[src]["type"] = 'h'
                    self.g.vs[dst]["type"] = 'h'

        self.logger.info(f"Finished processing {len(edge_batch)} edges.")