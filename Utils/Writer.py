import glob
import os
import logging
import csv

logger = logging.getLogger('Writer')

class Writer:

    def __init__(self):
        self.checkpoint_folder = "tmp"

    def write_on_csv(self, file_path, rows):
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def create_dirs(self, output_path, id):
        checkpoint_folder = os.sep.join([output_path, self.checkpoint_folder, id])
        output_folder = os.sep.join([output_path, id])
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
            logger.debug("Checkpoints temporary dir doesn't exist. Created folder @ {}", checkpoint_folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            logger.debug("Output dir doesn't exist. Created folder @ {}", output_folder)

    def list_checkpoint_files(self, dir_path):
        return glob.glob(dir_path)

    def load_checkpoint_file(self, file_path):
        """Load a checkpoint file and return rows as a list of tuples."""
        with open(file_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            return [list(row) for row in reader]