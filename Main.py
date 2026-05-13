import argparse
import logging
from pathlib import Path

from config import AppConfig
from Utils.logging_config import setup_logging
from Utils.memory_monitor import setup_memory_logging
from algorithms.GraphAnalysis import GraphAnalysis

import os
os.chdir(Path(__file__).parent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphAnalysis pipeline")
    parser.add_argument("--properties", type=str, default="properties/prop.yaml",
                        help="Path to the YAML properties file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig.from_yaml(args.properties)

    # Setup logging
    setup_logging(config.log.filepath)
    logger = logging.getLogger(__name__)
    setup_memory_logging(config.log.filepath)

    # Guard: skip entirely if nothing is enabled
    cfg = config
    nothing_to_do = not any([
        cfg.graph_generation.to_execute,
        cfg.community_detection.leiden.to_execute,
        cfg.community_detection.combo.to_execute,
        cfg.community_detection.hierarchical.to_execute,
        cfg.get_users_text.to_execute,
        cfg.topic_builder.to_execute,
    ])
    if nothing_to_do:
        logger.warning("Nothing to execute — all stages are disabled in the config.")
        return

    logger.info("Starting GraphAnalysis pipeline")
    GraphAnalysis(config).run()


if __name__ == "__main__":
    main()