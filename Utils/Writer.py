import concurrent.futures
import glob
import multiprocessing
import os
import logging
import csv
import pickle
import time
import uuid
from datetime import datetime

import networkx as nx
import igraph as ig
import pandas as pd
import numpy as np

from Utils.Const import Const as c
from itertools import chain
from collections import defaultdict
from multiprocessing import cpu_count

from algorithms.EdgeToGraph import EdgeToGraph


class Writer:
    logger = logging.getLogger('Writer')

    def __init__(self, graph_type = 'nx', temporal=False):
        self.graph_degree = defaultdict(int)
        self.graph_type = graph_type
        self.temporal_graph = temporal
        self.id = uuid.uuid1().hex

    @staticmethod
    def write_on_csv(file_path, rows):
        """
        Writes a list of rows to a CSV file.
        Each row is expected to be a list of values that will be written as a single row in the CSV file.
        If the file does not exist, it will be created. If it exists, new rows will be appended to the end of the file.

        :param file_path: Path to the CSV file where the rows will be written.
        :param rows: List of rows to write to the CSV file. Each row should be a list of values.
        
        :raises IOError: If there is an error writing to the file.
        :raises FileNotFoundError: If the specified file path does not exist.
        :raises Exception: For any other exceptions that may occur during the file writing process.

        """
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    @staticmethod
    def create_dirs(output_path, uuid):
        """
        Creates directories for checkpoints and output files based on the provided path and UUID.
        If the directories do not exist, they are created.

        :param output_path: Path where the directories should be created.
        :param uuid: Unique identifier for the directories.
        """
        checkpoint_folder = os.sep.join([output_path, c.CHECKPOINT_FOLDER, uuid])
        output_folder = os.sep.join([output_path, uuid])
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    @staticmethod
    def create_dir(output_path, uuid):
        """
        Creates a directory for output files based on the provided path and UUID.
        If the directory does not exist, it is created.

        :param output_path: Path where the directory should be created.
        :param uuid: Unique identifier for the directory.
        """
        output_folder = os.sep.join([output_path, uuid])
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    @staticmethod
    def list_checkpoint_files(dir_path):
        """
        List all checkpoint files in the specified directory.

        :param dir_path: Path to the directory containing checkpoint files.
        :return: List of file paths in the directory.
        """
        return glob.glob(dir_path)

    @staticmethod
    def load_checkpoint_file(file_path):
        """
        Load a checkpoint file and return rows as a list of tuples.
        This method reads a CSV file and returns its content as a list of rows,
        where each row is represented as a list of values.

        :param file_path: Path to the checkpoint file.
        :return: List of rows from the CSV file.
        """
        with open(file_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            return [list(row) for row in reader]

    @staticmethod
    def process_chunk(rows):
        """
        Process a chunk of rows from a CSV file.
        
        :param rows: List of rows to process.
        :return: Processed rows as a list of tuples.
        """
        return [tuple(row) for row in rows if row]

    @staticmethod
    def export_nx_nodes_with_attributes(g, file_path, attr_list=None):
        """
        Exports nodes with attributes from a NetworkX graph to a CSV file.

        :param g: The igraph graph object.
        :param file_path: Path to the output CSV file.
        :param attr_list: List of attributes to export. If None, all attributes are exported
        """
        attributes = set()
        if not attr_list:
            for _, data in g.nodes(data=True):
                attributes.update(data.keys())
        else:
            attributes = attr_list

        # Open the file for writing
        with open(file_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            # Write the header (attribute names)
            writer.writerow(["node"] + [attr for attr in attributes])

            # Write node data
            for node, data in g.nodes(data=True):
                row = [node] + [data.get(attr, "null") for attr in attributes]
                writer.writerow(row)

    @staticmethod
    def export_nodes_with_attributes(g, file_path, attr_list=None):
        """
        Exports nodes with attributes from an igraph graph to a CSV file.

        :param g: The igraph graph object.
        :param file_path: Path to the output CSV file.
        :param attr_list: List of attributes to export. If None, all attributes are exported
        """
        # Get all attributes for vertices
        attributes = g.vs.attributes() if not attr_list else attr_list

        for attr in attributes:
            if attr not in g.vs.attributes():
                print(f"Missing attribute: {attr}")

        # Open the file for writing
        with open(file_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            # Write the header (attribute names)
            writer.writerow(["id"] + attributes)

            # Write vertex data
            for vertex in g.vs:
                row = [vertex.index] + [vertex[attr] for attr in attributes]
                writer.writerow(row)

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
    
    def process_csv_file(self, file_path, chunk_size, header=False):
        """
        Read a single CSV file in chunks and process it.

        Args:
        - file_path: Path to the CSV file.
        - chunk_size: Number of rows per chunk.

        Returns:
        - List of processed chunks for the file.
        """
        processed_data = []
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            if header:
                h = next(reader, None)  # Skip the headers
            rows = []
            for row in reader:
                rows.append(row)
                if row[0] in ['0', '4', '5']:
                    self.graph_degree[row[1]] += 1
                    self.graph_degree[row[2]] += 1
                elif row[0] in ['2']:
                    self.graph_degree[row[1]] += 1
                if len(rows) == chunk_size:
                    processed_data.extend(self.process_chunk(rows))
                    rows = []  # Reset for the next chunk

            # Process remaining rows
            if rows:
                processed_data.extend(self.process_chunk(rows))
        return processed_data

    def process_csv_file_parallel(self, args):
        """
        Wrapper for multiprocessing to handle arguments.

        Args:
        - args: A tuple containing (file_path, chunk_size).

        Returns:
        - Tuple with file name and processed data.
        """
        file_path, chunk_size, header = args
        return self.process_csv_file(file_path, chunk_size, header)

    def process_csv_chunk(self, args, header=False):
        """
        Processes a chunk of a CSV file to extract edges and create a graph.
        This method reads a specified range of rows from the CSV file, processes them,
        and converts them into a graph structure using the EdgeToGraph class.

        :param args: Tuple containing the file path, start row, and end row.
        :param header: Boolean indicating if the CSV file has a header row.
        :return: Graph object created from the processed chunk.
        """
        path, start, end = args
        with open(path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            if header:
                next(reader, None)
            for _ in range(start):
                next(reader, None)

            batch = []
            i = start
            for row in reader:
                if i >= end:
                    break
                batch.append(row)
                i += 1
        e_to_g = EdgeToGraph(self.graph_type, self.temporal_graph)
        e_to_g.to_graph(batch)
        return e_to_g.get_graph()

    def merge_and_serialize(self, global_graph, subgraph, graph_type, step, output_folder, output_file_name, serialize_every=10):

        if graph_type == 'nx':
            global_graph = nx.compose(global_graph, subgraph)
        elif graph_type == 'igraph':
            if global_graph.vcount() == 0 and global_graph.ecount() == 0:
                global_graph = subgraph
            else:
                graphs = [global_graph, subgraph]
                names, types = [], []
                name2idx     = {}

                for g in graphs:
                    for v in g.vs:
                        n = v["name"]
                        if n not in name2idx:               # first time we see this name
                            name2idx[n] = len(names)
                            names.append(n)
                            types.append(v["type"])
                        else:
                            assert v["type"] == types[name2idx[n]], f"vertex {n!r} carries inconsistent type"
                
                master = ig.Graph(directed=True)
                master.add_vertices(len(names))
                master.vs["name"] = names
                master.vs["type"] = types

                sources, targets, weights, types_edge  = [], [], [], []
                times = [] if self.temporal_graph else None

                for g in graphs:
                    # translate local vertex IDs to master IDs *vectorised*
                    src_ids = [name2idx[n] for n in g.vs["name"]]          # list is OK once
                    for e in g.es:
                        sources.append(src_ids[e.source])
                        targets.append(src_ids[e.target])
                    if self.temporal_graph:
                        times.extend(g.es["time"])
                    types_edge.extend(g.es["type"])
                    weights.extend(g.es["weight"])
                master.add_edges(list(zip(sources, targets)))
                if self.temporal_graph:
                    master.es["time"] = times
                master.es["type"]   = types_edge
                master.es["weight"] = weights
                global_graph = master

        if step % serialize_every == 0:
            filename = f"{output_file_name}_snapshot_step_{step}.pkl"
            full_path = os.path.join(output_folder, filename)
            if graph_type == 'nx':
                with open(full_path, 'wb') as f:
                    pickle.dump(global_graph, f)
            elif graph_type == 'igraph':
                global_graph.write_pickle(full_path)
            self.logger.info(f"Serialized at step {step} to {filename}")

            # Optional: reset to free memory (keep just recent state or restart fresh)
            global_graph = nx.MultiDiGraph() if graph_type == 'nx' else ig.Graph(directed=True)

        return global_graph

    def read_csv_in_batch(self, path, output_path, batch_size = 300000, header = False):
        """
        Reads a CSV file in batches and processes it to create a graph.
        The graph is built using NetworkX or igraph based on the specified graph type.

        :param path: Path to the CSV file to read.
        :param output_path: Path to save the serialized graph.
        :param batch_size: Number of rows to read in each batch.
        :param header: Boolean indicating if the CSV file has a header row.

        """
        global_graph = nx.MultiDiGraph() if self.graph_type == 'nx' else ig.Graph(directed=True)
        serialize_every = 10  # Save every 10 steps

        output_folder = output_path
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        else:
            raise FileExistsError(f"{output_folder} already exists!")
        output_file_name = f"{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"

        tasks = []

        total_rows = 0
        try:
            with open(path) as f:
                total_rows = sum(1 for line in f)
                if header:
                    total_rows -= 1
        except FileNotFoundError:
            self.logger.debug(f"Csv file at {path} not found.")

        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            tasks.append((path, start, end))

        # Recycle pool between batches if needed
        step = 1
        available_cpu = cpu_count() - 2
        for i in range(0, len(tasks), available_cpu):
            with multiprocessing.Pool(processes=available_cpu) as pool:
                batch_tasks = tasks[i:i + available_cpu]
                subgraphs = pool.map(self.process_csv_chunk, batch_tasks)

            for subgraph in subgraphs:
                global_graph = self.merge_and_serialize(global_graph, subgraph, self.graph_type, step, output_folder, output_file_name, serialize_every)
                step += 1
        # Final save

        filename = f"{output_file_name}_final.pkl"
        full_path = os.path.join(output_folder, filename)
        with open(full_path, "wb") as f:
            if self.graph_type == 'nx':
                pickle.dump(global_graph, f)
            elif self.graph_type == 'igraph':
                global_graph.write_pickle(full_path)
        self.logger.info("Final graph saved.")

    def read_pickle(self, snapshot_dir):
        """
        Reads a snapshot of the graph from pickle files in the specified directory.

        :param snapshot_dir: Directory containing the snapshot files.
        :return: Full graph constructed from the snapshot files.
        """
        self.logger.info(f"Reading snapshot at {snapshot_dir}")
        files = sorted(glob.glob(os.path.join(snapshot_dir, "*.pkl")))

        full_g = ig.Graph(directed=True)

        for i, fn in enumerate(files):
            if self.graph_type == "igraph":
                self.logger.info(f"Loading subgraph {i} / {len(files)}")
                sg = ig.Graph.Read_Pickle(fn)
                self.logger.info(f"Subgraph {i} loaded with {sg.vcount()} vertices and {sg.ecount()} edges")
                if full_g.vcount() == 0 and full_g.ecount() == 0:
                    full_g = sg
                else:
                    graphs = [full_g, sg]
                    names, types = [], []
                    name2idx     = {}

                    for g in graphs:
                        for v in g.vs:
                            n = v["name"]
                            if n not in name2idx:               # first time we see this name
                                name2idx[n] = len(names)
                                names.append(n)
                                types.append(v["type"])
                            else:
                                assert v["type"] == types[name2idx[n]], f"vertex {n!r} carries inconsistent type"
                    
                    master = ig.Graph(directed=True)
                    master.add_vertices(len(names))
                    master.vs["name"] = names
                    master.vs["type"] = types

                    sources, targets = [], []
                    types_edge,  weights  = [], []

                    times = [] if self.temporal_graph else None

                    for g in graphs:
                        # translate local vertex IDs to master IDs *vectorised*
                        src_ids = [name2idx[n] for n in g.vs["name"]]          # list is OK once
                        for e in g.es:
                            sources.append(src_ids[e.source])
                            targets.append(src_ids[e.target])
                        types_edge.extend(g.es["type"])
                        weights.extend(g.es["weight"])
                        if self.temporal_graph:
                            times.extend(g.es["time"])
                    master.add_edges(list(zip(sources, targets)))
                    master.es["type"]   = types_edge
                    master.es["weight"] = weights
                    if self.temporal_graph:
                        master.es["time"] = times
                    full_g = master
                self.logger.info(f"Graph {i} loaded with {full_g.vcount()} vertices and {full_g.ecount()} edges")
            elif self.graph_type == "nx":
                with open(fn, "rb") as f:
                    sg = pickle.load(f)
                full_g = full_g.union(sg)
        self.logger.info("Full graph loaded with {} vertices and {} edges".format(full_g.vcount(), full_g.ecount()))
        return full_g

    def read_csv_files_in_folder_parallel(self, path, chunk_size=100, header=False):
        """
        Read all CSV files in a folder and process them in parallel using multiprocessing.

        Args:
        - path: Path to the folder containing CSV files.
        - chunk_size: Number of rows per chunk for each CSV file.

        Returns:
        - Dictionary with filenames as keys and processed data as values.
        """
        self.logger.info("Start loading graph from CSV in parallel, number of threads: {}".format(30))

        args = [(file, chunk_size, header) for file in path]
        with concurrent.futures.ThreadPoolExecutor(max_workers = 30) as executor:
            results = executor.map(self.process_csv_file_parallel, args)
        final_result = list(chain.from_iterable(results))

        self.logger.info("Graph loading from CSV completed")

        remove_nodes = {key for key, value in self.graph_degree.items() if value < 11}

        self.logger.info("Created list with nodes to remove")

        self.logger.info("I will analyze {} nodes to remove".format(len(remove_nodes)))
        new_edges = []
        self.logger.info("There are {} edges to analyze".format(len(final_result)))
        start_time = time.time()
        last_log_time = start_time
        for i, e in enumerate(final_result):
            if e[1] not in remove_nodes and e[2] not in remove_nodes:
                new_edges.append(e)
            # Log progress every 1 minute
            if time.time() - last_log_time >= 60:
                self.logger.info(f"Filtered {i:,} edges in {time.time() - start_time:.2f} seconds")
                last_log_time = time.time()
        final_result = new_edges
        del new_edges
        del remove_nodes
        self.logger.info("New number of edges to analyze: {}".format(len(final_result)))
        return final_result
