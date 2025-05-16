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

from Utils.Const import Const as c
from itertools import chain
from collections import defaultdict
from multiprocessing import cpu_count

from algorithms.EdgeToGraph import EdgeToGraph


class Writer:
    logger = logging.getLogger('Writer')

    def __init__(self):
        self.graph_degree = defaultdict(int)
        self.id = uuid.uuid1().hex

    @staticmethod
    def write_on_csv(file_path, rows):
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    @staticmethod
    def create_dirs(output_path, uuid):
        checkpoint_folder = os.sep.join([output_path, c.CHECKPOINT_FOLDER, uuid])
        output_folder = os.sep.join([output_path, uuid])
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    @staticmethod
    def create_dir(output_path, uuid):
        output_folder = os.sep.join([output_path, uuid])
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    @staticmethod
    def list_checkpoint_files(dir_path):
        return glob.glob(dir_path)

    @staticmethod
    def load_checkpoint_file(file_path):
        """Load a checkpoint file and return rows as a list of tuples."""
        with open(file_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            return [list(row) for row in reader]

    @staticmethod
    def process_chunk(rows):
        """
        Process a chunk of rows from a CSV file.

        Args:
        - rows: A list of rows (as lists) from the CSV file.

        Returns:
        - Processed data for the chunk.
        """
        # Example: Transform rows or filter data
        return [tuple(row) for row in rows if row]  # Keep non-empty rows as an example

    @staticmethod
    def export_nx_nodes_with_attributes(g, file_path, attr_list=None):
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
        # Get all attributes for vertices
        attributes = g.vs.attributes() if not attr_list else attr_list

        # Open the file for writing
        with open(file_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            # Write the header (attribute names)
            writer.writerow(["id"] + attributes)

            # Write vertex data
            for vertex in g.vs:
                row = [vertex.index] + [vertex[attr] for attr in attributes]
                writer.writerow(row)


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


    def process_csv_chunk(self, args, graph_type, header=False):
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
        e_to_g = EdgeToGraph(graph_type)
        e_to_g.to_graph(batch)
        return e_to_g.get_graph()

    def merge_and_serialize(self, global_graph, subgraph, graph_type, step, output_folder, output_file_name, serialize_every=10):

        if graph_type == 'nx':
            global_graph = nx.compose(global_graph, subgraph)
        elif graph_type == 'igraph':
            name_to_index = {v["name"]: v.index for v in global_graph.vs}
            for v in subgraph.vs:
                if v["name"] not in name_to_index:
                    global_graph.add_vertex(name=v["name"])
                    name_to_index[v["name"]] = global_graph.vcount() - 1
            for e in subgraph.es:
                src_name = subgraph.vs[e.source]["name"]
                dst_name = subgraph.vs[e.target]["name"]
                global_graph.add_edge(src_name, dst_name, **e.attributes())

        if step % serialize_every == 0:
            filename = f"{output_file_name}_snapshot_step_{step}.pkl"
            full_path = os.path.join(output_folder, filename)
            with open(full_path, 'wb') as f:
                pickle.dump(global_graph, f)
            self.logger.info(f"Serialized at step {step} to {filename}")

            # Optional: reset to free memory (keep just recent state or restart fresh)
            global_graph = nx.MultiDiGraph() if graph_type == 'nx' else ig.Graph(directed=True)

        return global_graph


    def read_csv_in_batch(self, path, output_path, graph_type = 'nx', batch_size = 300000, header = False):
        print(graph_type)
        global_graph = nx.MultiDiGraph() if graph_type == 'nx' else ig.Graph(directed=True)
        serialize_every = 30  # Save every 10 steps

        uuid = self.id
        output_folder = os.sep.join([output_path, uuid])
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
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
        for i in range(0, len(tasks), cpu_count() - 1):
            with multiprocessing.Pool(processes=cpu_count() - 1) as pool:
                batch_tasks = tasks[i:i + cpu_count() - 1]
                print(batch_tasks)
                subgraphs = pool.map(self.process_csv_chunk, batch_tasks, graph_type)

            for subgraph in subgraphs:
                global_graph = self.merge_and_serialize(global_graph, subgraph, graph_type, step, output_folder, output_file_name, serialize_every)
                step += 1
        # Final save
        with open(os.path.join(output_path, output_file_name), "wb") as f:
            pickle.dump(global_graph, f)
        self.logger.info("Final graph saved.")

        return global_graph

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
