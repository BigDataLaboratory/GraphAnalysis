import concurrent.futures
import itertools
import logging
import multiprocessing
import os
import threading
import time
import uuid
from concurrent.futures.thread import ThreadPoolExecutor

import graphscope.nx as nx
import numpy as np
import pycombo as pycombo

from Utils.Writer import Writer
from Utils.memory_monitor import memory_tracker, log_memory


class Combo:
    logger = logging.getLogger('Combo')

    def __init__(self):
        self.id = uuid.uuid1().hex
        self.data_graph = nx.MultiDiGraph()

    # Function to process a batch of edges
    def process_batch(self, index, edge_batch):
        """Processes a batch of edges and adds them to the graph."""
        self.logger.info(f"[Thread-{index}] Processing {len(edge_batch)} edges...")
        # Use bulk insertion instead of adding edges one-by-one
        self.data_graph.add_edges_from(
            [(e[1], e[2], e[0], {"weight": int(e[3])}) for e in edge_batch]
        )
        del edge_batch
        self.logger.info(f"[Thread-{index}] Finished processing {len(edge_batch)} edges.")


    # Function to dynamically split the dataset
    def chunk_list(self, data, num_chunks):
        """Splits data into balanced chunks for efficient parallel processing."""
        avg_chunk_size = max(1, len(data) // num_chunks)  # Prevents zero-sized chunks
        self.logger.info("Average chunk size: {}".format(avg_chunk_size))
        it = iter(data)
        return [list(itertools.islice(it, avg_chunk_size)) for _ in range(num_chunks)]

    @memory_tracker
    def csv_to_nx(self, graph, edge_threshold = 0):
        log_memory("Start converting graph in NetworkX format")
        start = time.time()

        # Auto-tune number of threads
        num_edges = len(graph)
        cpu_cores = multiprocessing.cpu_count()

        # Adaptive threading logic
        if num_edges < 1000:
            num_threads = min(4, cpu_cores-2)  # Use up to 4 threads for small datasets
        elif num_edges < 10000:
            num_threads = min(8, cpu_cores-2)  # Use up to 8 threads for medium datasets
        else:
            num_threads = min(16, cpu_cores-2)  # Use more threads for large datasets

        # Split dataset into optimal chunks
        edge_chunks = self.chunk_list(graph, num_threads)
        del graph
        # Process edges in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            executor.map(self.process_batch, range(len(edge_chunks)), edge_chunks)

        # Add edges to the MultiGraph
        # while graph:
        #     element = graph.pop(0)
        #     data_graph.add_edge(element[1], element[2], key=element[0], weight=int(element[3]))
        #     counter += 1  # Increment counter
        #     if counter % log_interval == 0 or not graph:
        #         log_memory(f"Processed {counter} of {len(graph)} elements so far.")

        # Iterate over the edges to assign node types
        log_memory("Adding node type for each node")
        for u, v, edge_key, data in self.data_graph.edges(keys=True, data=True):
            if edge_key in ['0', '4', '5']:
                self.data_graph.nodes[u]["type"] = 'u'
                self.data_graph.nodes[v]["type"] = 'u'
            elif edge_key == '2':
                self.data_graph.nodes[u]["type"] = 'u'
                self.data_graph.nodes[v]["type"] = 'h'
            elif edge_key == '3':
                self.data_graph.nodes[u]["type"] = 'h'
                self.data_graph.nodes[v]["type"] = 'h'
        log_memory("Adding node type successfully completed")

        end = time.time()
        log_memory("Graph successfully converted in NetworkX format")
        self.logger.info("Elapsed time: " + str(end - start))
        return self.data_graph

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
