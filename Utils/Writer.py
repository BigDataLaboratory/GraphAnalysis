import glob
import os
import logging
import csv
from Utils.Const import Const as c
from itertools import chain
from multiprocessing import Pool
from multiprocessing import cpu_count


class Writer:
    logger = logging.getLogger('Writer')

    def __init__(self):
        pass

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

    def read_csv_files_in_folder_parallel(self, path, chunk_size=100, header=False):
        """
        Read all CSV files in a folder and process them in parallel using multiprocessing.

        Args:
        - path: Path to the folder containing CSV files.
        - chunk_size: Number of rows per chunk for each CSV file.

        Returns:
        - Dictionary with filenames as keys and processed data as values.
        """
        self.logger.info("Start loading graph from CSV in parallel, cpu cores: {}".format(cpu_count()))

        args = [(file, chunk_size, header) for file in path]
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(self.process_csv_file_parallel, args)
        self.logger.info("Graph loading from CSV completed")
        return list(chain.from_iterable(results))
