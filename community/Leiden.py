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

    def __init__(self, data_graph):
        self.id = uuid.uuid1().hex
        self.data_graph = data_graph
    
    def get_graph(self):
        """
        Returns the graph associated with this Leiden instance.
        This method is useful for accessing the graph after it has been processed or modified.
        
        :return: The igraph.Graph instance associated with this Leiden instance.
        """
        return self.data_graph

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


    def build_daily_slices(self, g, first_n_days=10):
        """
        Builds daily slices of a temporal graph by grouping edges by their timestamp.
        Each slice is a subgraph representing the edges that occurred on a specific day.

        :param g: The temporal graph to slice, an igraph.Graph instance.
        :return: A list of tuples, each containing a date and the corresponding daily subgraph.
        :raises ValueError: If the graph does not have a 'time' attribute for its edges.
        :raises RuntimeError: If the graph is not directed or if the 'time' attribute is not present in the edges.
        :raises Exception: If the graph is not connected or if the partitioning fails.
        """
        # Group edges by day
        edges_by_day = defaultdict(list)
        for e in g.es:
            day = dt.datetime.utcfromtimestamp(e['time']).date()
            edges_by_day[day].append((e.tuple, e['weight']))

        # Create daily slices
        if first_n_days > 0:
            sorted_days = sorted(edges_by_day)[:first_n_days] # Get the first 14 days with activity
            self.logger.info(f"Using the first {first_n_days} days with activity: {[day.isoformat() for day in sorted_days]}")
        else:
            sorted_days = sorted(edges_by_day)
            self.logger.info(f"Using all days with activity")

        slices = []
        for day in sorted_days:                # chronological order
            edge_tuples = [tpl for tpl, _ in edges_by_day[day]]
            weights     = [w for _, w in edges_by_day[day]]

            # Create a subgraph for the day
            Gd = ig.Graph(n=g.vcount(), edges=edge_tuples, directed=g.is_directed())

            # copy attributes of vertex
            Gd.vs['id']    = g.vs['id']
            Gd.vs['name']  = g.vs['name']
            Gd.vs['type']  = g.vs['type']
            Gd.vs['slice'] = [day.isoformat()] * Gd.vcount()   # add slice date to vertices

            # copy attributes of edge
            Gd.es['weight'] = weights

            slices.append((day, Gd))
        return slices

    def build_weekly_slices(self, g):
        """
        Builds weekly slices of a temporal directed graph by grouping edges by ISO calendar week.
        For each week, edges with the same source and target are aggregated by summing their weights.

        :param g: A directed igraph.Graph with edge attributes 'time' (Unix timestamp) and 'weight'.
        :return: A list of tuples: ((year, week_number), igraph.Graph for that week).
        """
        # Step 1: Aggregate edge weights per week
        weekly_edge_weights = defaultdict(lambda: defaultdict(int))  # {(year, week): {(src, tgt): total_weight}}

        for e in g.es:
            ts = dt.datetime.utcfromtimestamp(e['time'])
            year, week, _ = ts.isocalendar()
            edge_key = e.tuple  # KEEP ORDER: directed edge (source → target)
            weekly_edge_weights[(year, week)][edge_key] += e['weight']

        # Step 2: Build graph per week
        slices = []
        for (year, week), edge_dict in sorted(weekly_edge_weights.items()):
            edge_list = list(edge_dict.keys())
            weight_list = list(edge_dict.values())

            Gw = ig.Graph(n=g.vcount(), edges=edge_list, directed=True)

            # Copy vertex attributes
            Gw.vs['id']    = g.vs['id']
            Gw.vs['name']  = g.vs['name']
            Gw.vs['type']  = g.vs['type']
            Gw.vs['slice'] = [f"{year}-W{week:02d}"] * Gw.vcount()

            # Copy edge weights
            Gw.es['weight'] = weight_list

            slices.append(((year, week), Gw))

        return slices

    def collapse_nodes(self, labels, min_size=30, dummy=-1):
        """
        Collapses nodes in the graph based on their labels, keeping only those with a size greater than or equal to `min_size`.
        Nodes that do not meet this criterion are replaced with a dummy value.
        This method is useful for simplifying the graph by merging less significant nodes into a single dummy community.

        :param labels: A list or array-like structure containing the labels of the nodes.
        :param min_size: The minimum size for a label to be retained. Nodes with fewer than `min_size` occurrences will be replaced with the dummy value.
        :param dummy: The value to replace nodes community that do not meet the `min_size` criterion.
        """
        self.logger.info("Start collapsing nodes")
        vc = pd.Series(labels).value_counts()
        big = vc[vc >= min_size].index
        self.logger.info("Finished collapsing nodes")
        return np.where(pd.Series(labels).isin(big), labels, dummy)

    def compute_leiden_temporal(self, resolution_parameter_range=(0.4, 0.5)):
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

        if 'id' not in self.data_graph.vs.attribute_names():
            self.data_graph.vs['id'] = self.data_graph.vs['name']
        if True:
            self.logger.info("Start removing edges not equal to 0 or 2")
            for e in self.data_graph.es:
                print(f"Edge {e.index} type: {e['type']}, type is {type(e['type'])}")
            edges_to_keep = [e.index for e in self.data_graph.es if e["type"] == "0" or e["type"] == "2"] 
            self.logger.info("Number of edges to keep: {}".format(len(edges_to_keep)))
            self.data_graph = self.data_graph.subgraph_edges(edges_to_keep, delete_vertices=False)
            self.logger.info("Finished removing edges not equal to 0 or 2, created the subgraph")
        
        if True:
            # Build daily slices of the temporal graph
            self.logger.info(f"Building daily slices of the temporal graph with {self.data_graph.vcount()} nodes and {self.data_graph.ecount()} edges")
            daily_slices = self.build_daily_slices(self.data_graph)
            dates = [d for d, _ in daily_slices]
            self.logger.info("Number of daily slices: {}".format(len(daily_slices)))
            graphs = [G for _, G in daily_slices]
            self.logger.info("Number of graphs in daily slices: {}".format(len(graphs)))
        if False:
            self.logger.info(f"Building weekly slices of the temporal graph with {self.data_graph.vcount()} nodes and {self.data_graph.ecount()} edges")
            weekly_slices = self.build_weekly_slices(self.data_graph)
            # Extract (year, week) identifiers
            weeks = [f"{year}-W{week:02d}" for (year, week), _ in weekly_slices]
            self.logger.info("Number of weekly slices: {}".format(len(weekly_slices)))
            # Extract igraph.Graph objects only
            graphs = [G for _, G in weekly_slices]
            self.logger.info("Number of graphs in weekly slices: {}".format(len(graphs)))

        for rp in np.linspace(resolution_parameter_range[0], resolution_parameter_range[1], num=1):
            start = time.time()

            rp_round = round(rp, 1)
            self.logger.info("Starting Leiden with CPM Quality Function and resolution parameter = {}".format(rp_round))
            # Apply the Leiden algorithm to each daily slice
            # and collect the memberships for each resolution parameter
            memberships, dQ = la.find_partition_temporal(graphs, 
                                            la.CPMVertexPartition, 
                                            interslice_weight=0.2,
                                            resolution_parameter=rp_round,
                                            seed=42)
            all_memberships[rp_round] = memberships
            series_per_node = list(zip(*all_memberships[rp_round]))

            if True:
                self.data_graph.vs["{}".format(rp_round)] = [
                        {date.isoformat(): lbl for date, lbl in zip(dates, labels)}
                    for labels in series_per_node
                ]
            if False:
                # Convert (year, week) → "YYYY-Www" strings
                week_labels = [f"{year}-W{week:02d}" for (year, week), _ in weekly_slices]
                print("Week labels: ", week_labels)

                self.data_graph.vs["{}".format(rp_round)] = [
                    {week: lbl for week, lbl in zip(week_labels, labels)}
                    for labels in series_per_node
                ]

            self.data_graph[f"cpm_quality_{rp_round}"] = dQ
            self.logger.info("Leiden CPM quality value is: {}, resolution parameter is {}".format(dQ, rp_round))

            end = time.time()

            self.logger.info(
                "Finished Leiden with CPM Quality Function and resolution parameter = {}".format(rp_round))
            self.logger.info("Elapsed time: " + str(end - start))
            yield rp_round

    def compute_leiden(self, resolution_parameter_range=(0.1, 1.0), number_of_resolutions=10):
        """
        Computes the Leiden partitioning of the graph using the CPM quality function.
        This method iterates over a range of resolution parameters, applies the Leiden algorithm,
        and updates the graph with the partitioning results.
        
        :param resolution_parameter_range: A tuple specifying the range of resolution parameters to test.
        
        :return: Yields the resolution parameter used for each partitioning.
        :raises Exception: If the graph is not connected or if the partitioning fails.
        :raises ValueError: If the resolution parameter is not within the specified range.
        :raises RuntimeError: If the Leiden algorithm fails to compute a partition. 
        """
        self.logger.info("Start Leiden computation")
        for rp in np.linspace(resolution_parameter_range[0], resolution_parameter_range[1], num=number_of_resolutions):
            start = time.time()

            rp_round = round(rp, 1)
            self.logger.info("Starting Leiden with CPM Quality Function and resolution parameter = {}".format(rp_round))

            partition = la.find_partition(self.data_graph, la.CPMVertexPartition, resolution_parameter=rp_round,
                                          weights='weight',
                                          seed=0)
            cpm = partition.quality()

            self.data_graph.vs["{}".format(rp_round)] = partition.membership
            self.data_graph["cpm_quality"] = cpm
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
