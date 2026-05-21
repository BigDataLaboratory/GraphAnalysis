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
import pyarrow as pa
import pyarrow.parquet as pq

from Utils.Const import Const as c
from itertools import chain
from collections import defaultdict
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

from algorithms.EdgeToGraph import EdgeToGraph

# ─────────────────────────────────────────────────────────────────────────────
# Supported formats
# ─────────────────────────────────────────────────────────────────────────────
SUPPORTED_FORMATS = ("csv", "pickle", "parquet", "feather")


def _ext(file_format: str) -> str:
    """Return the file extension for the given format."""
    fmt = (file_format or "csv").lower()
    return {
        "csv": ".csv",
        "pickle": ".pkl",
        "parquet": ".parquet",
        "feather": ".feather"
    }.get(fmt, "." + fmt)


class Writer:
    logger = logging.getLogger('Writer')

    def __init__(self, graph_type='nx', temporal=False):
        self.graph_degree = defaultdict(int)
        self.graph_type = graph_type
        self.temporal_graph = temporal
        self.id = uuid.uuid1().hex

    # ─────────────────────────────────────────────────────────────────────────
    # LOW-LEVEL WRITERS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def write_on_csv(file_path: str, rows: list) -> None:
        """
        Append rows to a CSV file (creates the file if it does not exist).

        :param file_path: Path with or without .csv extension.
        :param rows: List of iterables, one per CSV row.
        """
        path = file_path if file_path.endswith(".csv") else file_path + ".csv"
        with open(path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(rows)

    @staticmethod
    def write_on_pickle(file_path: str, rows: list, columns: list = None) -> None:
        """
        Append a batch of rows to a pickle file.

        :param file_path: Path with or without .pkl extension.
        :param rows: List of rows (lists or tuples).
        :param columns: Ignored — kept for API consistency.
        """
        if not rows:
            return
        path = file_path if file_path.endswith(".pkl") else file_path + ".pkl"
        with open(path, 'ab') as f:
            pickle.dump(rows, f)

    @staticmethod
    def write_on_parquet(file_path: str, rows: list, columns: list = None) -> None:
        """
        Append a batch of rows to a Parquet file (Snappy compression).

        :param file_path: Path with or without .parquet extension.
        :param rows: List of rows (lists or tuples).
        :param columns: Column names; positional fallback if None.
        """
        if not rows:
            return
        path = file_path if file_path.endswith(".parquet") else file_path + ".parquet"
        df = pd.DataFrame(rows, columns=columns)
        # downcast numeric columns to save space
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
                if pd.api.types.is_integer_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], downcast='integer')
            except (ValueError, TypeError):
                pass
        table = pa.Table.from_pandas(df, preserve_index=False)
        mode = 'ab' if os.path.exists(path) else 'wb'
        with pa.OSFile(path, mode) as sink:
            with pq.ParquetWriter(sink, table.schema, compression='snappy') as pw:
                pw.write_table(table)

    @staticmethod
    def write_on_feather(file_path: str, rows: list, columns: list = None) -> None:
        """
        Append a batch of rows to a Feather v2 file (LZ4 compression).
        Feather is optimised for fast sequential read/write — ideal for checkpoints.

        :param file_path: Path with or without .feather extension.
        :param rows: List of rows (lists or tuples).
        :param columns: Column names; positional fallback if None.
        """
        if not rows:
            return
        import pyarrow.feather as feather
        path = file_path if file_path.endswith(".feather") else file_path + ".feather"
        df = pd.DataFrame(rows, columns=columns)
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
                if pd.api.types.is_integer_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], downcast='integer')
            except (ValueError, TypeError):
                pass
        if os.path.exists(path):
            # Feather doesn't support append natively — load, concat, rewrite
            existing = feather.read_feather(path)
            df = pd.concat([existing, df], ignore_index=True)
        feather.write_feather(df, path, compression='lz4')

    # ─────────────────────────────────────────────────────────────────────────
    # UNIFIED WRITE / LOAD
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def write_data(file_path: str, rows: list,
                   columns: list = None, file_format: str = "csv") -> None:
        """
        Write rows to disk in the requested format.
        The correct extension is appended automatically.

        :param file_path: Base path **without** extension.
        :param rows: List of rows to write.
        :param columns: Column names (required for parquet/feather; optional for others).
        :param file_format: One of 'csv', 'pickle', 'parquet', 'feather'.
        :raises ValueError: If file_format is not supported.
        """
        if file_format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file_format '{file_format}'. "
                             f"Choose from {SUPPORTED_FORMATS}.")
        if file_format == "parquet":
            Writer.write_on_parquet(file_path, rows, columns)
        elif file_format == "pickle":
            Writer.write_on_pickle(file_path, rows, columns)
        elif file_format == "feather":
            Writer.write_on_feather(file_path, rows, columns)
        else:
            Writer.write_on_csv(file_path, rows)

    @staticmethod
    def load_checkpoint_file(file_path: str) -> list:
        """
        Load all rows from a checkpoint file regardless of its format.
        The format is inferred from the file extension.

        :param file_path: Full path including extension.
        :return: List of rows (each row is a list).
        """
        if file_path.endswith('.pkl'):
            rows = []
            with open(file_path, 'rb') as f:
                while True:
                    try:
                        rows.extend(pickle.load(f))
                    except EOFError:
                        break
            return rows
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(file_path).values.tolist()
        elif file_path.endswith('.feather'):
            import pyarrow.feather as feather
            return feather.read_feather(file_path).values.tolist()
        else:
            with open(file_path, mode='r', newline='', encoding='utf-8',
                      errors='replace') as f:
                return [list(row) for row in csv.reader(f)]

    # ─────────────────────────────────────────────────────────────────────────
    # DIRECTORY HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def create_dirs(output_path: str, uuid_str: str) -> None:
        """Create checkpoint and output directories."""
        for folder in [
            os.sep.join([output_path, c.CHECKPOINT_FOLDER, uuid_str]),
            os.sep.join([output_path, uuid_str]),
        ]:
            os.makedirs(folder, exist_ok=True)

    @staticmethod
    def create_dir(output_path: str, uuid_str: str) -> None:
        """Create the output directory for a run."""
        os.makedirs(os.sep.join([output_path, uuid_str]), exist_ok=True)

    @staticmethod
    def list_checkpoint_files(pattern: str) -> list:
        """Return all files matching a glob pattern."""
        return glob.glob(pattern)

    # ─────────────────────────────────────────────────────────────────────────
    # GRAPH EXPORT HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def export_nx_nodes_with_attributes(g, file_path: str,
                                        attr_list: list = None) -> None:
        """Export NetworkX node attributes to CSV."""
        attributes = attr_list or sorted(
            {k for _, d in g.nodes(data=True) for k in d})
        with open(file_path, mode="a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["node"] + list(attributes))
            for node, data in g.nodes(data=True):
                w.writerow([node] + [data.get(a, "null") for a in attributes])

    @staticmethod
    def export_nodes_with_attributes(g, file_path: str,
                                     attr_list: list = None) -> None:
        """Export igraph node attributes to CSV."""
        attributes = attr_list or g.vs.attributes()
        with open(file_path, mode="a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["id"] + attributes)
            for v in g.vs:
                w.writerow([v.index] + [v[a] for a in attributes])

    # ─────────────────────────────────────────────────────────────────────────
    # GRAPH MERGE / SERIALIZE
    # ─────────────────────────────────────────────────────────────────────────

    def collapse_nodes(self, labels, min_size: int = 30, dummy: int = -1):
        """Collapse small communities into a dummy label."""
        self.logger.info("Start collapsing nodes")
        vc = pd.Series(labels).value_counts()
        big = vc[vc >= min_size].index
        self.logger.info("Finished collapsing nodes")
        return np.where(pd.Series(labels).isin(big), labels, dummy)

    def process_chunk(self, rows: list) -> list:
        return [tuple(row) for row in rows if row]

    def process_csv_file(self, file_path: str, chunk_size: int,
                         header: bool = False) -> list:
        processed_data = []
        with open(file_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            if header:
                next(reader, None)
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
        e = EdgeToGraph(self.graph_type, self.temporal_graph)
        e.to_graph(batch)
        return e.get_graph()

    def merge_and_serialize(self, global_graph, subgraph, graph_type: str,
                            step: int, output_folder: str,
                            output_file_name: str,
                            serialize_every: int = 10):
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
                sources, targets, weights, types_edge = [], [], [], []
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

    @staticmethod
    def _read_single_graph(path):
        # Worker: read and return igraph.Graph (uses igraph's own unpickle)
        return ig.Graph.Read_Pickle(path)

    def read_pickle_parallel_preserve_time(self, snapshot_dir, max_workers=None):
        """
        Read pickled subgraphs in parallel and merge them, preserving edge attributes
        including 'time' if self.temporal_graph is True.

        Strategy:
        1. Parallel load all pickles into igraph.Graph objects.
        2. Sequentially merge them (consistent name -> index mapping).
        """
        self.logger.info(f"Reading snapshot at {snapshot_dir}")
        files = sorted(glob.glob(os.path.join(snapshot_dir, "*.pkl")))
        if not files:
            return ig.Graph(directed=True)

        # 1) Parallel load
        n_workers = max_workers or min(12, os.cpu_count() or 1)
        self.logger.info(f"Loading {len(files)} files in parallel with {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            graphs = list(ex.map(self._read_single_graph, files))

        self.logger.info("All subgraphs loaded. Starting merge...")

        # 2) Sequential pairwise merge for determinism and simplicity
        full_g = graphs[0]
        for i, sg in enumerate(graphs[1:], start=1):
            self.logger.info(f"Merging graph {i+1}/{len(graphs)}: sg nodes={sg.vcount()}, edges={sg.ecount()}")
            full_g = self._merge_two_igraphs_preserve_time(full_g, sg, temporal=self.temporal_graph)

        self.logger.info(f"Full graph loaded with {full_g.vcount()} vertices and {full_g.ecount()}")
        return full_g

    @staticmethod
    def _merge_two_igraphs_preserve_time(g1: ig.Graph, g2: ig.Graph, temporal: bool = False) -> ig.Graph:
        """
        Merge two igraph Graphs by vertex 'name', preserving edge attributes:
        - 'type' (if present)
        - 'weight' (if present)
        - 'time' (if temporal and present)
        The merged graph will have vertices ordered by first occurrence (g1 then new g2 names).
        """
        # Build name -> idx mapping (deterministic ordering)
        names = []
        types = []
        name2idx = {}

        for g in (g1, g2):
            for v in g.vs:
                n = v["name"]
                if n not in name2idx:
                    name2idx[n] = len(names)
                    names.append(n)
                    # store 'type' if present else None
                    try:
                        types.append(v["type"])
                    except Exception:
                        types.append(None)

        master = ig.Graph(directed=True)
        master.add_vertices(len(names))
        master.vs["name"] = names
        # set vertex 'type' only if at least one non-None
        if any(t is not None for t in types):
            master.vs["type"] = [t if t is not None else "u" for t in types]
        else:
            # no type attribute across vertices; skip setting to avoid attribute errors later
            pass

        # prepare edge lists and attributes
        all_sources = []
        all_targets = []
        all_types = []
        all_weights = []
        all_times = [] if temporal else None

        def extend_from_graph(g):
            # map local vertex ids -> master ids using names
            local_names = list(g.vs["name"])
            local_map = [name2idx[n] for n in local_names]  # list of master indices aligned to local id
            # edges as pairs of master indices
            for (s, t) in g.get_edgelist():
                all_sources.append(local_map[s])
                all_targets.append(local_map[t])
            # attributes: if attribute missing, attempt to provide defaults to keep lists aligned
            if "type" in g.es.attribute_names():
                all_types.extend(list(g.es["type"]))
            else:
                all_types.extend([None] * g.ecount())
            if "weight" in g.es.attribute_names():
                all_weights.extend([float(w) if w is not None else 1.0 for w in g.es["weight"]])
            else:
                all_weights.extend([1.0] * g.ecount())
            if temporal:
                if "time" in g.es.attribute_names():
                    all_times.extend(list(g.es["time"]))
                else:
                    # if temporal requested but missing in subgraph, append None
                    all_times.extend([None] * g.ecount())

        extend_from_graph(g1)
        extend_from_graph(g2)

        # add edges and assign edge attributes
        if all_sources:
            master.add_edges(list(zip(all_sources, all_targets)))
            # set 'type' attribute if any non-None
            if any(t is not None for t in all_types):
                # fallback missing types to a sentinel 'u'
                master.es["type"] = [t if t is not None else "u" for t in all_types]
            else:
                # leave unset or set to default if you prefer
                master.es["type"] = ["u"] * len(all_types)

            master.es["weight"] = all_weights
            if temporal:
                master.es["time"] = all_times

        return master

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