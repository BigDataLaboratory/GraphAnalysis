import logging
import sys


def setup_logging(filepath):
    root = logging.getLogger()
    root.handlers = []  # Clear existing handlers
    root.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(filepath)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    root.addHandler(file_handler)

