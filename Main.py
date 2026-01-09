import argparse
import logging
import os
from collections import namedtuple
from datetime import datetime
from pathlib import Path

from Utils.logging_config import setup_logging
from Utils.memory_monitor import setup_memory_logging
from algorithms.GraphAnalysis import GraphAnalysis

# Ensure relative paths work even when launched from elsewhere
os.chdir(Path(__file__).parent)


def get_properties(file_path: str = "properties/prop.ipazia.json") -> dict:
    """Reads properties from a JSON file."""
    import json
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def _parse_date_or_none(s):
    """Parse 'dd/mm/YYYY' or return None if missing/empty/null."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    return datetime.strptime(s, "%d/%m/%Y")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line args")
    parser.add_argument("--properties", type=str, help="Properties file path")
    args = parser.parse_args()

    prop: dict = get_properties(args.properties) if args.properties else get_properties()

    # ------------------------------------------------------------
    # Setup logging
    # ------------------------------------------------------------
    log_path = prop.get("log", {}).get("filepath", "./logs/log.log")
    setup_logging(log_path)
    logger = logging.getLogger(__name__)
    setup_memory_logging(log_path)

    # ------------------------------------------------------------
    # Flags principali (safe)
    # ------------------------------------------------------------
    do_graph_generation = prop.get("graph_generation", {}).get("to_execute", False)

    cd = prop.get("community_detection", {})
    do_community_detection_leiden = cd.get("leiden", {}).get("to_execute", False)
    do_community_detection_combo = cd.get("combo", {}).get("to_execute", False)
    # IMPORTANT: keep naming consistent with GraphAnalysis.py (use do_community_hierarchical)
    do_community_hierarchical = cd.get("hierarchical", {}).get("to_execute", False)

    do_get_text = prop.get("get_users_text", {}).get("to_execute", False)
    do_topic_builder = prop.get("topic_builder", {}).get("to_execute", False)

    # ------------------------------------------------------------
    # Graph generation (SAFE even if to_execute=False)
    # ------------------------------------------------------------
    gg = prop.get("graph_generation", {})
    gg_params = gg.get("parameters", {})
    gg_input = gg_params.get("input", {})
    source_input_type = gg_input.get("type", "json")
    config = gg_input.get("conf", {}) or {}

    source_uri = config.get("uri", "")
    source_username = config.get("username", None)
    source_password = config.get("password", None)
    source_auth_source = config.get("authName", None)
    source_auth_mechanism = config.get("authMechanism", None)
    source_db_name = config.get("db_name", None)
    source_collection = config.get("collection", None)

    source_chunk_start_date = _parse_date_or_none(config.get("chunk_start_date", None))
    source_chunk_end_date = _parse_date_or_none(config.get("chunk_end_date", None))
    source_method = config.get("method", None)

    graph_type = gg_params.get("graph_type", {}) or {}
    do_retweet_graph = graph_type.get("retweet", False)
    do_tweet_retweet_graph = graph_type.get("tweet_retweet", False)
    do_hashtag_graph = graph_type.get("user_hashtag", False)
    do_hashtag_cooccurrences_graph = graph_type.get("hashtag_cooccurrences", False)
    do_mention_graph = graph_type.get("mention", False)
    do_response_graph = graph_type.get("response", False)

    out_cfg = gg_params.get("output", {}) or {}
    output_graph_path = out_cfg.get("path", "./resources/")
    gnames = out_cfg.get("graph_file_name", {}) or {}

    output_retweet_graph_path = gnames.get("retweet", "retweet")
    output_response_graph_path = gnames.get("response", "response")
    output_mention_graph_path = gnames.get("mention", "mention")
    output_hashtag_graph_path = gnames.get("hashtag", "hashtag")
    output_hashtag_cooccurrences_graph_path = gnames.get("hashtag_cooccurrences", "hashtag_cooccurrences")
    output_multi_graph_path = gnames.get("multigraph", "multigraph")
    output_map_prefix = out_cfg.get("map_file_name_prefix", "map")

    # ------------------------------------------------------------
    # Community detection common parameters (SAFE)
    # ------------------------------------------------------------
    community_config = cd.get("parameters", {}) or {}
    do_read_graph_from_file = community_config.get("read_from_file", True)
    do_read_from_edge_list = community_config.get("read_from_edge_list", False)
    graph_file_path = community_config.get("graph_file_path", [])

    # In your JSON this is "pickle_graph_file_path": keep it but also allow "pickle_graph_path"
    pickle_graph_path = community_config.get(
        "pickle_graph_path",
        community_config.get("pickle_graph_file_path", "./snapshot")
    )

    temporal = community_config.get("temporal", True)

    # Per-algorithm props (SAFE)
    community_combo_prop = cd.get("combo", {}).get("parameters", {}) or {}

    # Leiden: pass ONLY "parameters" here; experiments separately
    leiden_block = cd.get("leiden", {}) or {}
    community_leiden_prop = leiden_block.get("parameters", {}) or {}
    community_leiden_exp_prop = leiden_block.get("experiments", {}) or {}

    community_hierarchical_prop = cd.get("hierarchical", {}).get("parameters", {}) or {}

    # ------------------------------------------------------------
    # Get users text props (SAFE)
    # ------------------------------------------------------------
    if do_get_text:
        users_text_config = prop.get("get_users_text", {}).get("parameters", {}) or {}

        comm_cfg = users_text_config.get("communities", {}) or {}
        community_indexes = comm_cfg.get("indexes", [])
        community_col_name = comm_cfg.get("community_col_name", "cluster")
        do_read_communities_from_file = comm_cfg.get("read_communities_from_file", False)
        community_file_path = comm_cfg.get("community_file_path", "")

        map_cfg = users_text_config.get("map_file_path", {}) or {}
        do_read_maps_from_file = map_cfg.get("read_maps_from_file", False)
        user_map = map_cfg.get("user_map", "")
        hashtag_map = map_cfg.get("hashtag_map", "")
        retweet_user_map = map_cfg.get("retweet_user_map", "")

        input_cfg = users_text_config.get("input", {}) or {}
        td_input_type = input_cfg.get("type", "json")
        text_data_config = input_cfg.get("conf", {}) or {}

        td_uri = text_data_config.get("uri", "")
        td_username = text_data_config.get("username", None)
        td_password = text_data_config.get("password", None)
        td_auth_source = text_data_config.get("authName", None)
        td_auth_mechanism = text_data_config.get("authMechanism", None)
        td_db_name = text_data_config.get("db_name", None)
        td_collection = text_data_config.get("collection", None)
    else:
        # Defaults when module disabled
        community_indexes = []
        community_col_name = "cluster"
        do_read_communities_from_file = False
        community_file_path = ""
        do_read_maps_from_file = False
        user_map = ""
        hashtag_map = ""
        retweet_user_map = ""
        td_input_type = "json"
        td_uri = ""
        td_username = None
        td_password = None
        td_auth_source = None
        td_auth_mechanism = None
        td_db_name = None
        td_collection = None

    # ------------------------------------------------------------
    # Topic builder props (SAFE)
    # ------------------------------------------------------------
    if do_topic_builder:
        topic_config = prop.get("topic_builder", {}).get("parameters", {}) or {}
        topics_file_path = topic_config.get("topics_file_path", "./resources/topics_info")
        docs_file_path = topic_config.get("docs_file_path", "./resources/docs_info")
        model_cfg = topic_config.get("model", {}) or {}
        model_path = model_cfg.get("model_path", "./resources/topic_model")
        model_serialization = model_cfg.get("serialization", "pickle")
    else:
        topics_file_path = "./resources/topics_info"
        docs_file_path = "./resources/docs_info"
        model_path = "./resources/topic_model"
        model_serialization = "pickle"


    # Parameters tuple (COERENTE con GraphAnalysis)
    
    Parameters = namedtuple("Parameters", [
        # flags
        "do_graph_generation",
        "do_community_detection_leiden",
        "do_community_detection_combo",
        "do_community_hierarchical",
        "do_get_text",
        "do_topic_builder",

        # source input
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

        # graph types
        "do_retweet_graph",
        "do_tweet_retweet_graph",
        "do_hashtag_graph",
        "do_hashtag_cooccurrences_graph",
        "do_mention_graph",
        "do_response_graph",

        # output graph
        "output_graph_path",
        "output_retweet_graph_path",
        "output_response_graph_path",
        "output_mention_graph_path",
        "output_hashtag_graph_path",
        "output_hashtag_cooccurrences_graph_path",
        "output_multi_graph_path",
        "output_map_prefix",

        # community common
        "do_read_graph_from_file",
        "do_read_from_edge_list",
        "graph_file_path",
        "pickle_graph_path",
        "temporal",

        # per-algo params
        "community_combo_prop",
        "community_leiden_prop",
        "community_leiden_exp_prop",
        "community_hierarchical_prop",

        # get text
        "community_indexes",
        "community_col_name",
        "do_read_communities_from_file",
        "community_file_path",
        "do_read_maps_from_file",
        "user_map",
        "hashtag_map",
        "retweet_user_map",

        # text data source
        "td_input_type",
        "td_uri",
        "td_username",
        "td_password",
        "td_auth_source",
        "td_auth_mechanism",
        "td_db_name",
        "td_collection",

        # topic builder
        "topics_file_path",
        "docs_file_path",
        "model_path",
        "model_serialization",
    ])

    P = Parameters(
        do_graph_generation,
        do_community_detection_leiden,
        do_community_detection_combo,
        do_community_hierarchical,
        do_get_text,
        do_topic_builder,

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

        output_graph_path,
        output_retweet_graph_path,
        output_response_graph_path,
        output_mention_graph_path,
        output_hashtag_graph_path,
        output_hashtag_cooccurrences_graph_path,
        output_multi_graph_path,
        output_map_prefix,

        do_read_graph_from_file,
        do_read_from_edge_list,
        graph_file_path,
        pickle_graph_path,
        temporal,

        community_combo_prop,
        community_leiden_prop,
        community_leiden_exp_prop,
        community_hierarchical_prop,

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

        topics_file_path,
        docs_file_path,
        model_path,
        model_serialization,
    )

    graph_analysis = GraphAnalysis(P)
    logger.info("Main parameters loaded. Starting GraphAnalysis...")
    graph_analysis.run()
