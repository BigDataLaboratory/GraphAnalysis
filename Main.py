import argparse
import logging
import os
from collections import namedtuple
from datetime import datetime
from pathlib import Path

from Utils.logging_config import setup_logging
from algorithms.GraphAnalysis import GraphAnalysis
from Utils.memory_monitor import setup_memory_logging, memory_tracker, log_memory


os.chdir(Path(__file__).parent)

def get_properties(file_path="properties/prop.json"):
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


    setup_logging(prop["log"]["filepath"])
    logger = logging.getLogger(__name__)
    setup_memory_logging(prop["log"]["filepath"])

    do_graph_generation = prop["graph_generation"]["to_execute"]
    do_community_detection_leiden = prop["community_detection"]["leiden"]["to_execute"]
    do_community_detection_combo = prop["community_detection"]["combo"]["to_execute"]
    do_get_text = prop["get_users_text"]["to_execute"]
    do_topic_builder = prop["topic_builder"]["to_execute"]

    config = prop["graph_generation"]["parameters"]["input"]["conf"]
    source_input_type = prop["graph_generation"]["parameters"]["input"]["type"]
    source_uri = config["uri"]
    source_username = config["username"]
    source_password = config["password"]
    source_auth_source = config["authName"]
    source_auth_mechanism = config["authMechanism"]
    source_db_name = config["db_name"]
    source_collection = config["collection"]
    source_chunk_start_date = datetime.strptime(config["chunk_start_date"], '%d/%m/%Y')
    source_chunk_end_date = datetime.strptime(config["chunk_end_date"], '%d/%m/%Y')

    do_retweet_graph = prop["graph_generation"]["parameters"]["graph_type"]["retweet"]
    do_tweet_retweet_graph = prop["graph_generation"]["parameters"]["graph_type"]["tweet_retweet"]
    do_hashtag_graph = prop["graph_generation"]["parameters"]["graph_type"]["user_hashtag"]
    do_hashtag_cooccurrences_graph = prop["graph_generation"]["parameters"]["graph_type"]["hashtag_cooccurrences"]
    do_mention_graph = prop["graph_generation"]["parameters"]["graph_type"]["mention"]
    do_response_graph = prop["graph_generation"]["parameters"]["graph_type"]["response"]

    output_graph_path = prop["graph_generation"]["parameters"]["output"]["path"]
    output_retweet_graph_path = prop["graph_generation"]["parameters"]["output"]["graph_file_name"]["retweet"]
    output_response_graph_path = prop["graph_generation"]["parameters"]["output"]["graph_file_name"]["response"]
    output_mention_graph_path = prop["graph_generation"]["parameters"]["output"]["graph_file_name"]["mention"]
    output_hashtag_graph_path = prop["graph_generation"]["parameters"]["output"]["graph_file_name"]["hashtag"]
    output_hashtag_cooccurrences_graph_path = prop["graph_generation"]["parameters"]["output"]["graph_file_name"][
        "hashtag_cooccurrences"]
    output_multi_graph_path = prop["graph_generation"]["parameters"]["output"]["graph_file_name"]["multigraph"]
    output_map_prefix = prop["graph_generation"]["parameters"]["output"]["map_file_name_prefix"]

    community_config = prop["community_detection"]["parameters"]
    do_read_graph_from_file = community_config["read_from_file"]
    do_read_from_edge_list = community_config["read_from_edge_list"]
    graph_file_path = community_config["graph_file_path"]
    pickle_graph_path = community_config["pickle_graph_file_path"]
    community_combo_prop = prop["community_detection"]["combo"]["parameters"]
    community_leiden_prop = prop["community_detection"]["leiden"]["parameters"]

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

    topic_config = prop["topic_builder"]["parameters"]
    topics_file_path = topic_config["topics_file_path"]
    docs_file_path = topic_config["docs_file_path"]
    model_path = topic_config["model"]["model_path"]
    model_serialization = topic_config["model"]["serialization"]

    Parameters = namedtuple('Parameters', [
        "do_graph_generation",
        "do_community_detection_leiden",
        "do_community_detection_combo",
        "do_get_text",
        "do_topic_builder",
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
        "do_retweet_graph",
        "do_tweet_retweet_graph",
        "do_hashtag_graph",
        "do_hashtag_cooccurrences_graph",
        "do_mention_graph",
        "do_response_graph",
        "output_graph_path",
        "output_retweet_graph_path",
        "output_response_graph_path",
        "output_mention_graph_path",
        "output_hashtag_graph_path",
        "output_hashtag_cooccurrences_graph_path",
        "output_multi_graph_path",
        "output_map_prefix",
        "do_read_graph_from_file",
        "do_read_from_edge_list",
        "graph_file_path",
        "pickle_graph_path",
        "community_combo_prop",
        "community_leiden_prop",
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
        "topics_file_path",
        "docs_file_path",
        "model_path",
        "model_serialization"
    ])

    P = Parameters(do_graph_generation,
                   do_community_detection_leiden,
                   do_community_detection_combo,
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
                   do_read_from_edge_list,
                   do_read_graph_from_file,
                   graph_file_path,
                   pickle_graph_path,
                   community_combo_prop,
                   community_leiden_prop,
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
                   model_serialization
                   )
    graph_analysis = GraphAnalysis(P)
    try:
        graph_analysis.run()
    except Exception as e:
        raise

    """
    cluster_map = ClusterMap()
    clusters = cluster_map.read_cluster_file(prop["output"]["path"] + 'clusters_general.csv')
    maps = cluster_map.read_map_file(
        prop["output"]["path"] + prop["output"]["map_file_name_prefix"] + "hashtag_hash.csv")
    clusters_maps_df = cluster_map.from_hash_to_id(clusters, maps)
    clusters_grouped_df = cluster_map.group_and_count_by_cluster(clusters_maps_df)

    Utils.persist_to_file(clusters_maps_df, prop["output"]["path"] + 'cluster_wo_hash')
    Utils.persist_to_file(clusters_grouped_df, prop["output"]["path"] + 'cluster_grouped_count')


    cl = leiden_instance.hash_to_name(df, prop["output"]["path"] + prop["output"]["map_file_name_prefix"]
                                      + "user.csv", prop["output"]["path"] + prop["output"]["map_file_name_prefix"]
                                        + "hashtag_hash.csv", prop["output"]["path"] + prop["output"]["map_file_name_prefix"]
                                        + "user_screen_name_user_id.csv")
    Utils.persist_to_file(cl[0], prop["output"]["path"] + prop["output"]["clusters_file_name_prefix"] + "user")
    Utils.persist_to_file(cl[1], prop["output"]["path"] + prop["output"]["clusters_file_name_prefix"] + "hashtag")
    """
