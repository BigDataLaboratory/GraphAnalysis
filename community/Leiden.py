# -*- coding: utf-8 -*-
import concurrent.futures
import logging
import os
import time
import uuid

import igraph as ig
import leidenalg as la
import numpy as np

from Utils.Writer import Writer


class Leiden:
    logger = logging.getLogger('Leiden')

    def __init__(self):
        self.id = uuid.uuid1().hex

    def csv_to_igraph(self, graph):
        graph[:] = [(src, dst, int(weight), type) for type, src, dst, weight in graph]
        self.logger.info("Graph loading from CSV completed")

        self.logger.info("Start converting graph in iGraph format")
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
        self.logger.info('Graph successfully converted in iGraph format')
        self.logger.info("Elapsed time: " + str(end - start))
        return data_graph

    def compute_pagerank(self, data_graph):
        self.logger.info("Start removing nodes of type hashtag")
        # Specify the type of nodes to exclude
        exclude_type = "h"
        # Get the indices of nodes to keep (i.e., nodes not of the excluded type)
        nodes_to_keep = [v.index for v in data_graph.vs if v["type"] != exclude_type]
        # Create the subgraph with the selected nodes
        subgraph = data_graph.induced_subgraph(nodes_to_keep)
        self.logger.info("Start computation of PageRank")
        start = time.time()
        pagerank_values = subgraph.pagerank(directed=True, weights='weight', implementation="prpack")
        end = time.time()
        # Map the PageRank values back to the original graph
        # Initialize all PageRank values in the original graph to None
        data_graph.vs["pagerank"] = [None] * data_graph.vcount()
        # Copy the PageRank values from the subgraph back to the original graph
        for subgraph_node, pagerank in zip(subgraph.vs, pagerank_values):
            original_index = subgraph.vs["name"].index(subgraph_node["name"])  # Get the original index from the subgraph
            data_graph.vs[original_index]["pagerank"] = pagerank
        self.logger.info('Computation of PageRank completed!')
        self.logger.info("Elapsed time: " + str(end - start))

    def worker_process(self, g, resolution_parameter):

        partition = la.find_partition(g, la.CPMVertexPartition, resolution_parameter=resolution_parameter,
                                      weights='weight',
                                      seed=0)
        return resolution_parameter, partition.membership

    def compute_leiden_in_parallel(self, data_graph, resolution_parameter_range=(0.1, 1.0)):
        self.logger.info("Start parallel Leiden computation for each resolution parameter value")
        rps = np.linspace(resolution_parameter_range[0], resolution_parameter_range[1], num=10)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self.worker_process, data_graph, round(rp, 1)): rp for rp in rps}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                resolution_parameter, partitions = result
                data_graph.vs["{}".format(resolution_parameter)] = partitions
                yield resolution_parameter

    def compute_leiden(self, data_graph, resolution_parameter_range=(0.1, 1.0)):
        self.logger.info("Start Leiden computation")
        for rp in np.linspace(resolution_parameter_range[0], resolution_parameter_range[1], num=10):
            start = time.time()

            rp_round = round(rp, 1)
            self.logger.info("Starting Leiden with CPM Quality Function and resolution parameter = {}".format(rp_round))

            partition = la.find_partition(data_graph, la.CPMVertexPartition, resolution_parameter=rp_round,
                                          weights='weight',
                                          seed=0)
            cpm = partition.quality()

            data_graph.vs["{}".format(rp_round)] = partition.membership
            data_graph["cpm_quality"] = cpm
            self.logger.info("Leiden CPM quality value is: {}, resolution parameter is {}".format(cpm, rp_round))

            end = time.time()
            self.logger.info(
                "Finished Leiden with CPM Quality Function and resolution parameter = {}".format(rp_round))
            self.logger.info("Elapsed time: " + str(end - start))
            yield rp_round

    def export_partition(self, g, resolution_parameter, file_path, attr=None):
        Writer.create_dir(file_path, self.id)
        file_path = os.sep.join([file_path, self.id, "communities_rp_{}.csv".format(resolution_parameter)])
        Writer.export_nodes_with_attributes(g, file_path, attr)

    def export_graph(self, g, file_path):
        Writer.create_dir(file_path, self.id)
        file_path = os.sep.join([file_path, self.id, "nodes_with_communities.csv"])
        Writer.export_nodes_with_attributes(g, file_path)
