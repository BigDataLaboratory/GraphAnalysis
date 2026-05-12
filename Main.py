import argparse
import logging
import os
from collections import namedtuple
from datetime import datetime
from pathlib import Path

from Utils.logging_config import setup_logging
from algorithms.GraphAnalysis import GraphAnalysis
from Utils.memory_monitor import setup_memory_logging


os.chdir(Path(__file__).parent)

def get_properties(file_path="properties/prop.json"):
    """
    Reads properties from a JSON file.
    The properties file should contain configuration settings for the application.

    :param file_path: Path to the properties file. Defaults to 'properties/prop.json'.
    :return: A dictionary containing the properties read from the file.

    :raises FileNotFoundError: If the specified file does not exist.
    :raises json.JSONDecodeError: If the file is not a valid JSON.
    """
    import json
    with open(file_path) as f:
        properties = json.load(f)
    return properties


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Command line args")
    parser.add_argument('--properties', type=str, help='Properties file path')
    args = parser.parse_args()

    if args.properties is not None:
        prop: dict = get_properties(args.properties)
    else:
        prop: dict = get_properties()

    # Setup logging
    setup_logging(prop["log"]["filepath"])
    logger = logging.getLogger(__name__)
    setup_memory_logging(prop["log"]["filepath"])

    # Extract properties for graph analysis
    delete_tmp_after_merge = prop["graph_generation"]["delete_tmp_after_merge"]
    do_graph_generation = prop["graph_generation"]["to_execute"]
    do_community_detection_leiden = prop["community_detection"]["leiden"]["to_execute"]
    do_community_detection_combo = prop["community_detection"]["combo"]["to_execute"]
    do_community_hierarchical = prop.get("community_detection", {}).get("hierarchical", {}).get("to_execute", False)
    do_get_text = prop.get("get_users_text", {}).get("to_execute", False)
    do_topic_builder = prop.get("topic_builder", {}).get("to_execute", False)

    config = prop["graph_generation"]["parameters"]["input"]["conf"]
    checkpoint_every = prop["graph_generation"]["parameters"].get("checkpoint_every", 5000)
    n_workers = prop["graph_generation"]["parameters"].get("n_workers", 4)
    fast_rt_threshold = prop["graph_generation"]["parameters"].get("fast_rt_threshold", 60)
    source_input_type = prop["graph_generation"]["parameters"]["input"]["type"]
        # for docker/podman, to safely inject database credentials
    source_uri            = os.getenv("MONGO_URI", config.get("uri"))
    source_username       = os.getenv("MONGO_USERNAME", config.get("username"))
    source_password       = os.getenv("MONGO_PASSWORD", config.get("password"))
    source_db_name        = os.getenv("MONGO_DATABASE", config.get("db_name"))
    source_collection     = os.getenv("MONGO_COLLECTION", config.get("collection"))
    source_auth_source    = os.getenv("MONGO_AUTH_SOURCE", config.get("authName"))
    source_auth_mechanism = os.getenv("MONGO_AUTH_MECHANISM", config.get("authMechanism"))
    source_chunk_start_date = datetime.strptime(config["chunk_start_date"], '%d/%m/%Y')
    source_chunk_end_date = datetime.strptime(config["chunk_end_date"], '%d/%m/%Y')
    source_method = config["method"]

    do_retweet_graph = prop["graph_generation"]["parameters"]["graph_type"]["retweet"]
    do_tweet_retweet_graph = prop["graph_generation"]["parameters"]["graph_type"]["tweet_retweet"]
    do_hashtag_graph = prop["graph_generation"]["parameters"]["graph_type"]["user_hashtag"]
    do_hashtag_cooccurrences_graph = prop["graph_generation"]["parameters"]["graph_type"]["hashtag_cooccurrences"]
    do_mention_graph = prop["graph_generation"]["parameters"]["graph_type"]["mention"]
    do_response_graph = prop["graph_generation"]["parameters"]["graph_type"]["response"]
    do_user_user_graph = prop["graph_generation"]["parameters"]["graph_type"].get("user_user", False)

    output_graph_path = prop["graph_generation"]["parameters"]["output"]["path"]
    output_retweet_graph_path = prop["graph_generation"]["parameters"]["output"]["graph_file_name"]["retweet"]
    output_response_graph_path = prop["graph_generation"]["parameters"]["output"]["graph_file_name"]["response"]
    output_mention_graph_path = prop["graph_generation"]["parameters"]["output"]["graph_file_name"]["mention"]
    output_hashtag_graph_path = prop["graph_generation"]["parameters"]["output"]["graph_file_name"]["hashtag"]
    output_hashtag_cooccurrences_graph_path = prop["graph_generation"]["parameters"]["output"]["graph_file_name"][
        "hashtag_cooccurrences"]
    output_multi_graph_path = prop["graph_generation"]["parameters"]["output"]["graph_file_name"]["multigraph"]
    output_map_prefix = prop["graph_generation"]["parameters"]["output"]["map_file_name_prefix"]
    # Support both legacy single format and new split intermediate/final formats
    _out_cfg = prop["graph_generation"]["parameters"]["output"]
    output_intermediate_format = _out_cfg.get("intermediate_file_format",
                                               _out_cfg.get("file_format", "parquet"))
    output_final_format        = _out_cfg.get("final_file_format",
                                               _out_cfg.get("file_format", "parquet"))

    # Community detection parameters
    community_config = prop["community_detection"]["parameters"]
    do_read_graph_from_file = community_config["read_from_file"]
    do_read_from_edge_list = community_config["read_from_edge_list"]
    graph_file_path = community_config["graph_file_path"]
    pickle_graph_path = community_config["pickle_graph_file_path"]
    temporal = community_config["temporal"]
    community_combo_prop = prop["community_detection"]["combo"]["parameters"]
    community_leiden_prop = prop["community_detection"]["leiden"]["parameters"]
    community_hierarchical_prop = prop["community_detection"]["hierarchical"]["parameters"]

    # Get text parameters
    users_text_config = prop["get_users_text"]["parameters"]
    community_indexes = users_text_config["communities"]["indexes"]
    community_col_name = users_text_config["communities"]["community_col_name"]
    do_read_communities_from_file = users_text_config["communities"]["read_communities_from_file"]
    community_file_path = users_text_config["communities"]["community_file_path"]

    do_read_maps_from_file = users_text_config["map_file_path"]["read_maps_from_file"]
    user_map = users_text_config["map_file_path"]["user_map"]
    hashtag_map = users_text_config["map_file_path"]["hashtag_map"]
    retweet_user_map = users_text_config["map_file_path"]["retweet_user_map"]

    text_data_config = users_text_config["input"]["conf"]
    td_input_type = users_text_config["input"]["type"]
    td_uri = text_data_config["uri"]
    td_username = text_data_config["username"]
    td_password = text_data_config["password"]
    td_auth_source = text_data_config["authName"]
    td_auth_mechanism = text_data_config["authMechanism"]
    td_db_name = text_data_config["db_name"]
    td_collection = text_data_config["collection"]

    # Topic builder parameters
    topic_config = prop["topic_builder"]["parameters"]
    topics_file_path = topic_config["topics_file_path"]
    docs_file_path = topic_config["docs_file_path"]
    model_path = topic_config["model"]["model_path"]
    model_serialization = topic_config["model"]["serialization"]

    Parameters = namedtuple('Parameters', [
        # General
        "do_graph_generation",
        "do_community_detection_leiden",
        "do_community_detection_combo",
        "do_community_hierarchical",
        "do_get_text",
        "do_topic_builder",
        # Graph generation parameters
        "delete_tmp_after_merge",
        "checkpoint_every",
        "n_workers",
        "fast_rt_threshold",
        "source_input_type",
        "source_uri",
        "source_username",
        "source_password",
        "source_auth_source",
        "source_auth_mechanism",
        "source_db_name",
        "source_collection",
        "source_chunk_start_date",
        "source_chunk_end_date",
        "source_method",
        "do_retweet_graph",
        "do_tweet_retweet_graph",
        "do_hashtag_graph",
        "do_hashtag_cooccurrences_graph",
        "do_mention_graph",
        "do_response_graph",
        "do_user_user_graph",
        "output_graph_path",
        "output_retweet_graph_path",
        "output_response_graph_path",
        "output_mention_graph_path",
        "output_hashtag_graph_path",
        "output_hashtag_cooccurrences_graph_path",
        "output_multi_graph_path",
        "output_map_prefix",
        "output_intermediate_format",
        "output_final_format",
        # Community detection parameters
        "do_read_graph_from_file",
        "do_read_from_edge_list",
        "graph_file_path",
        "pickle_graph_path",
        "temporal",
        "community_combo_prop",
        "community_leiden_prop",
        "community_hierarchical_prop",
        # Get text parameters
        "community_indexes",
        "community_col_name",
        "do_read_communities_from_file",
        "community_file_path",
        "do_read_maps_from_file",
        "user_map",
        "hashtag_map",
        "retweet_user_map",
        "td_input_type",
        "td_uri",
        "td_username",
        "td_password",
        "td_auth_source",
        "td_auth_mechanism",
        "td_db_name",
        "td_collection",
        # Topic builder parameters
        "topics_file_path",
        "docs_file_path",
        "model_path",
        "model_serialization"
    ])

    P = Parameters(
                   # General
                   do_graph_generation,
                   do_community_detection_leiden,
                   do_community_detection_combo,
                   do_community_hierarchical,
                   do_get_text,
                   do_topic_builder,
                   # Graph generation parameters
                   delete_tmp_after_merge,
                   checkpoint_every,
                   n_workers,
                   fast_rt_threshold,
                   source_input_type,
                   source_uri,
                   source_username,
                   source_password,
                   source_auth_source,
                   source_auth_mechanism,
                   source_db_name,
                   source_collection,
                   source_chunk_start_date,
                   source_chunk_end_date,
                   source_method,
                   do_retweet_graph,
                   do_tweet_retweet_graph,
                   do_hashtag_graph,
                   do_hashtag_cooccurrences_graph,
                   do_mention_graph,
                   do_response_graph,
                   do_user_user_graph,
                   output_graph_path,
                   output_retweet_graph_path,
                   output_response_graph_path,
                   output_mention_graph_path,
                   output_hashtag_graph_path,
                   output_hashtag_cooccurrences_graph_path,
                   output_multi_graph_path,
                   output_map_prefix,
                   output_intermediate_format,
                   output_final_format,
                   # Community detection parameters
                   do_read_graph_from_file,
                   do_read_from_edge_list,
                   graph_file_path,
                   pickle_graph_path,
                   temporal,
                   community_combo_prop,
                   community_leiden_prop,
                   community_hierarchical_prop,
                   # Get text parameters
                   community_indexes,
                   community_col_name,
                   do_read_communities_from_file,
                   community_file_path,
                   do_read_maps_from_file,
                   user_map,
                   hashtag_map,
                   retweet_user_map,
                   td_input_type,
                   td_uri,
                   td_username,
                   td_password,
                   td_auth_source,
                   td_auth_mechanism,
                   td_db_name,
                   td_collection,
                   # Topic builder parameters
                   topics_file_path,
                   docs_file_path,
                   model_path,
                   model_serialization
                   )

    if do_graph_generation or do_community_detection_leiden or do_community_detection_combo or do_community_hierarchical or do_get_text or do_topic_builder:
        graph_analysis = GraphAnalysis(P)
        graph_analysis.run()
