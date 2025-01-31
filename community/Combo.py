import concurrent.futures
import logging
import os
import time
import uuid

import networkx as nx
import numpy as np
import pycombo as pycombo

from Utils.Writer import Writer
from Utils.memory_monitor import memory_tracker, log_memory


class Combo:
    logger = logging.getLogger('Combo')

    def __init__(self):
        self.id = uuid.uuid1().hex

    @memory_tracker
    def csv_to_nx(self, graph, edge_threshold = 0):
        log_memory("Start converting graph in NetworkX format")
        start = time.time()

        data_graph = nx.MultiDiGraph()

        # Add edges to the MultiGraph
        while graph:
            element = graph.pop(0)
            data_graph.add_edge(element[1], element[2], key=element[0], weight=int(element[3]))

        del graph
        # Iterate over the edges to assign node types
        log_memory("Adding node type for each node")
        for u, v, edge_key, data in data_graph.edges(keys=True, data=True):
            if edge_key in ['0', '4', '5']:
                data_graph.nodes[u]["type"] = 'u'
                data_graph.nodes[v]["type"] = 'u'
            elif edge_key == '2':
                data_graph.nodes[u]["type"] = 'u'
                data_graph.nodes[v]["type"] = 'h'
            elif edge_key == '3':
                data_graph.nodes[u]["type"] = 'h'
                data_graph.nodes[v]["type"] = 'h'
        log_memory("Adding node type successfully completed")

        if edge_threshold != 0:
            # Find nodes with a degree (counting all parallel edges) less than edge_threshold
            # Find nodes with the target type that also have degree < edge_threshold
            target_type = "u"
            log_memory("Remove nodes with degree less than edge_threshold")
            nodes_to_remove = [
                node for node in data_graph.nodes
                if data_graph.nodes[node].get("type") == target_type and (data_graph.out_degree(node) + data_graph.in_degree(node)) < edge_threshold
            ]            # Remove nodes with degree less than edge_threshold
            data_graph.remove_nodes_from(nodes_to_remove)
            log_memory("Remove nodes successfully completed")
        end = time.time()
        log_memory("Graph successfully converted in NetworkX format")
        self.logger.info("Elapsed time: " + str(end - start))
        return data_graph

    @memory_tracker
    def worker_process(self, g, resolution_parameter):
        log_memory('Start combo computation with resolution parameter {}'.format(resolution_parameter))
        start = time.time()
        partition, modularity = pycombo.execute(g, weight="weight", modularity_resolution=resolution_parameter,
                                                random_seed=0)
        end = time.time()
        log_memory('Created communities with Combo with resolution parameter {}'.format(resolution_parameter))
        self.logger.info("Elapsed time: " + str(end - start))
        return resolution_parameter, partition, modularity

    def compute_combo_in_parallel(self, data_graph, resolution_parameter_range=(0.1, 1.0)):
        self.logger.info("Start parallel Combo computation for each resolution parameter value")
        rps = np.linspace(resolution_parameter_range[0], resolution_parameter_range[1], num=10)
        with concurrent.futures.ThreadPoolExecutor(max_workers = 10) as executor:
            futures = {executor.submit(self.worker_process, data_graph, round(rp, 1)): rp for rp in rps}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                resolution_parameter, partitions, modularity = result
                data_graph.graph["{}".format(resolution_parameter)] = modularity
                for k, v in partitions.items():
                    data_graph.nodes[k]["{}".format(resolution_parameter)] = v
                yield resolution_parameter

    def export_partition(self, g, resolution_parameter, file_path, attr=None):
        Writer.create_dir(file_path, self.id)
        file_path = os.sep.join([file_path, self.id, "communities_rp_{}.csv".format(resolution_parameter)])
        Writer.export_nx_nodes_with_attributes(g, file_path, attr)

    def export_graph(self, g, file_path):
        Writer.create_dir(file_path, self.id)
        file_path = os.sep.join([file_path, self.id, "nodes_with_communities.csv"])
        Writer.export_nx_nodes_with_attributes(g, file_path)
