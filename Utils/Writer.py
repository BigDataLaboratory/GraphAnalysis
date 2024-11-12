import glob
import os
import logging
import csv

logger = logging.getLogger('Writer')

class Writer:

    def __init__(self, output_path):
        self.output_path = output_path

    def write_on_csv(self, file_path, rows):
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def create_dirs(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            logger.debug("Checkpoints temporary dir doesn't exist. Created folder @ {}", self.output_path)

    def list_checkpoint_files(self, dir_path):
        return glob.glob(dir_path)

    def load_checkpoint_file(self, file_path):
        """Load a checkpoint file and return rows as a list of tuples."""
        with open(file_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            return [list(row) for row in reader]