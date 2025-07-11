# -*- coding: utf-8 -*-
import concurrent.futures
import logging
import os
import time
import uuid

import igraph as ig
import leidenalg as la
import numpy as np
from collections import defaultdict
import datetime as dt

from Utils.Writer import Writer


class Leiden:
    logger = logging.getLogger('Leiden')

    def __init__(self):
        self.id = uuid.uuid1().hex

    def compute_pagerank(self, data_graph):
        """
        Computes the PageRank values for nodes in the graph, excluding nodes of type 'hashtag'.
        This method first creates a subgraph excluding nodes of type 'hashtag', then computes the
        PageRank values for the remaining nodes. The results are mapped back to the original graph.

        :param data_graph: The graph to compute PageRank on, an igraph.Graph instance.
        """
        self.logger.info("Start removing nodes of type hashtag")
        # Specify the type of nodes to exclude
        # Get the indices of nodes to keep (i.e., nodes not of the excluded type)
        # Create the subgraph with the selected nodes
        exclude_type = "h"
        nodes_to_keep = [v.index for v in data_graph.vs if v["type"] != exclude_type]
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


    def build_daily_slices(self, g):
        """
        Builds daily slices of a temporal graph by grouping edges by their timestamp.
        Each slice is a subgraph representing the edges that occurred on a specific day.

        :param g: The temporal graph to slice, an igraph.Graph instance.
        :return: A list of tuples, each containing a date and the corresponding daily subgraph.
        :raises ValueError: If the graph does not have a 'time' attribute for its edges.
        :raises RuntimeError: If the graph is not directed or if the 'time' attribute is not present in the edges.
        :raises Exception: If the graph is not connected or if the partitioning fails.
        """
        # 1) Raggruppa gli archi per data (UTC)
        edges_by_day = defaultdict(list)
        for e in g.es:
            day = dt.datetime.utcfromtimestamp(e['time']).date()
            edges_by_day[day].append((e.tuple, e['weight']))

        # 2) Costruisci un grafo per ogni giorno
        slices = []
        for day in sorted(edges_by_day):                # ordine cronologico
            edge_tuples = [tpl for tpl, _ in edges_by_day[day]]
            weights     = [w for _, w in edges_by_day[day]]

            # nuovo grafo “vuoto” con stesso numero di nodi
            Gd = ig.Graph(n=g.vcount(), edges=edge_tuples, directed=g.is_directed())

            # copia attributi di vertice
            Gd.vs['id']    = g.vs['id']
            Gd.vs['name']  = g.vs['name']
            Gd.vs['type']  = g.vs['type']
            Gd.vs['slice'] = [day.isoformat()] * Gd.vcount()   # etichetta del time‑slice

            # copia attributi di arco
            Gd.es['weight'] = weights

            slices.append((day, Gd))
        return slices

    def compute_leiden_temporal(self, data_graph, resolution_parameter_range=(0.1, 1.0)):
        """
        Computes the Leiden partitioning of a temporal graph using the CPM quality function.
        This method iterates over a range of resolution parameters, applies the Leiden algorithm,
        and updates the graph with the partitioning results for each time slice.

        :param data_graph: The temporal graph to partition, an igraph.Graph instance.
        :param resolution_parameter_range: A tuple specifying the range of resolution parameters to test.

        :return: Yields the resolution parameter used for each partitioning.
        :raises Exception: If the graph is not connected or if the partitioning fails.
        :raises ValueError: If the resolution parameter is not within the specified range.
        :raises RuntimeError: If the Leiden algorithm fails to compute a partition.
        :raises RuntimeError: If the Leiden algorithm fails to compute a partition for the temporal graph
        """
        self.logger.info("Start Leiden computation for temporal data")
        all_memberships = defaultdict(list) # dict: res → [ [labels], … ]

        if 'id' not in data_graph.vs.attribute_names():
            data_graph.vs['id'] = data_graph.vs['name'] 
        for rp in np.linspace(resolution_parameter_range[0], resolution_parameter_range[1], num=10):
            start = time.time()

            rp_round = round(rp, 1)
            self.logger.info("Starting Leiden with CPM Quality Function and resolution parameter = {}".format(rp_round))

            # Build daily slices of the temporal graph
            daily_slices = self.build_daily_slices(data_graph)
            dates = [d for d, _ in daily_slices]

            # Apply the Leiden algorithm to each daily slice
            # and collect the memberships for each resolution parameter
            graphs = [G for _, G in daily_slices]

            memberships, dQ = la.find_partition_temporal(graphs, 
                                            la.CPMVertexPartition, 
                                            interslice_weight=0.2,
                                            resolution_parameter=rp_round,
                                            seed=42)
            all_memberships[rp_round] = memberships
            series_per_node = list(zip(*all_memberships[rp_round]))
            data_graph.vs["{}".format(rp_round)] = [
                    {date.isoformat(): lbl for date, lbl in zip(dates, labels)}
                for labels in series_per_node
            ]

            data_graph[f"cpm_quality_{rp_round}"] = dQ
            self.logger.info("Leiden CPM quality value is: {}, resolution parameter is {}".format(dQ, rp_round))

            end = time.time()

            self.logger.info(
                "Finished Leiden with CPM Quality Function and resolution parameter = {}".format(rp_round))
            self.logger.info("Elapsed time: " + str(end - start))
            yield rp_round

    def compute_leiden(self, data_graph, resolution_parameter_range=(0.1, 1.0)):
        """
        Computes the Leiden partitioning of the graph using the CPM quality function.
        This method iterates over a range of resolution parameters, applies the Leiden algorithm,
        and updates the graph with the partitioning results.
        
        :param data_graph: The graph to partition, an igraph.Graph instance.
        :param resolution_parameter_range: A tuple specifying the range of resolution parameters to test.
        
        :return: Yields the resolution parameter used for each partitioning.
        :raises Exception: If the graph is not connected or if the partitioning fails.
        :raises ValueError: If the resolution parameter is not within the specified range.
        :raises RuntimeError: If the Leiden algorithm fails to compute a partition. 
        """
        self.logger.info("Start Leiden computation")
        for rp in np.linspace(resolution_parameter_range[0], resolution_parameter_range[1], num=10):
            start = time.time()

            rp_round = round(rp, 1)
            self.logger.info("Starting Leiden with CPM Quality Function and resolution parameter = {}".format(rp_round))

            partition = la.find_partition(data_graph, la.RBConfigurationVertexPartition, resolution_parameter=rp_round,
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
        """
        Exports the partition of the graph to a CSV file with nodes and their attributes.
        The file is saved in a directory structure based on the instance ID.
        The directory is created if it does not exist.
        The nodes are exported with their attributes, including 'id', 'name', 'type',
        and 'pagerank' if available.

        :param g: The graph to export, an igraph.Graph instance.
        :param resolution_parameter: The resolution parameter used for the partitioning.
        :param file_path: The base path where the partition will be exported.
        :param attr: Optional list of attributes to include in the export. If None, defaults
        """
        Writer.create_dir(file_path, self.id)
        file_path = os.sep.join([file_path, self.id, "communities_rp_{}.csv".format(resolution_parameter)])
        Writer.export_nodes_with_attributes(g, file_path, attr)

    def export_graph(self, g, file_path):
        """
        Exports the graph to a CSV file with nodes and their attributes.
        The file is saved in a directory structure based on the instance ID.
        The directory is created if it does not exist.
        The nodes are exported with their attributes, including 'id', 'name', 'type',
        and 'pagerank' if available.

        :param g: The graph to export, an igraph.Graph instance.
        :param file_path: The base path where the graph will be exported.
        """
        Writer.create_dir(file_path, self.id)
        file_path = os.sep.join([file_path, self.id, "nodes_with_communities.csv"])
        Writer.export_nodes_with_attributes(g, file_path)
