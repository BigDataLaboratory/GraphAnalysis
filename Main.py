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
    # Parse command line arguments
    # This allows the user to specify a properties file path via command line.
    # If no path is provided, it defaults to 'properties/prop.json'.
    parser = argparse.ArgumentParser(description="Command line args")
    parser.add_argument('--properties', type=str, help='Properties file path')
    
    # GNN overrides
    parser.add_argument('--graph_type', type=str)
    parser.add_argument('--encoder', type=str)
    parser.add_argument('--decoder', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--neg_ratio', type=float)
    parser.add_argument('--batch_size', type=int)
    
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
    do_gnn_training = prop.get("gnn_training", {}).get("to_execute", False)

    config = prop["graph_generation"]["parameters"]["input"]["conf"]
    source_input_type = prop["graph_generation"]["parameters"]["input"]["type"]
    checkpoint_every = prop["graph_generation"]["parameters"].get("checkpoint_every", 5000)
    source_uri = config["uri"]
    source_username = config["username"]
    source_password = config["password"]
    source_auth_source = config["authName"]
    source_auth_mechanism = config["authMechanism"]
    source_db_name = config["db_name"]
    source_collection = config["collection"]
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
        "delete_tmp_after_merge",
        # Graph generation parameters
        "do_graph_generation",
        "do_community_detection_leiden",
        "do_community_detection_combo",
        "do_community_hierarchical",
        "do_get_text",
        "do_topic_builder",
        "checkpoint_every",
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

    P = Parameters(delete_tmp_after_merge,
                   do_graph_generation,
                   do_community_detection_leiden,
                   do_community_detection_combo,
                   do_community_hierarchical,
                   do_get_text,
                   do_topic_builder,
                   # Graph generation parameters
                   checkpoint_every,
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
    
    if do_gnn_training:
        print("\n" + "="*80)
        print("Starting GNN Link Prediction Task")
        print("="*80)
        
        import random
        import numpy as np
        import torch
        from gnn.preprocess.GraphBuilder import GraphBuilder
        from gnn.preprocess.GNNDataProcessor import GNNDataProcessor
        from gnn.train.GNNTraining import GNNTraining
        from gnn.train.BaselineTrainer import BaselineTrainer
        from gnn.model.RawMLPConcatPredictor import RawMLPConcatPredictor
        from gnn.evaluation.Evaluator import Evaluator
        from gnn.evaluation.GNNEvaluator import GNNEvaluator
        from gnn.evaluation.CosineBaselineScorer import CosineBaselineScorer
        from gnn.evaluation.MLPBaselineScorer import MLPBaselineScorer
        
        gnn_cfg = prop["gnn_training"]
        
        # ---- Reproducibility: set seed ----
        seed = gnn_cfg["model"].get("seed", 777)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"  Seed set to {seed}")
        
        # Override with CLI arguments if provided
        if args.graph_type:
            gnn_cfg["data"]["graph_type"] = {args.graph_type: True}
        if args.encoder:
            gnn_cfg["model"]["encoder"] = {args.encoder: True}
        if args.decoder:
            gnn_cfg["model"]["decoder"] = args.decoder
        if args.epochs:
            gnn_cfg["model"]["num_epochs"] = args.epochs
        if args.lr:
            gnn_cfg["model"]["learning_rate"] = args.lr
        if args.neg_ratio:
            gnn_cfg["model"]["neg_ratio"] = args.neg_ratio
        if args.batch_size:
            gnn_cfg["model"]["batch_size"] = args.batch_size
            
        enabled_graphs = [k for k, v in gnn_cfg["data"]["graph_type"].items() if v]
        enabled_encoders = [k for k, v in gnn_cfg["model"]["encoder"].items() if v]
        
        for graph_type in enabled_graphs:
            print(f"\n--- Processing Graph: {graph_type.upper()} ---")
            builder = GraphBuilder(data_dir=gnn_cfg["data"]["data_dir"], graph_type=graph_type)
            pt_path = builder.build_and_save()
            
            processor = GNNDataProcessor(
                pt_path, 
                train_ratio=gnn_cfg["data"]["train_ratio"],
                val_ratio=gnn_cfg["data"]["val_ratio"],
                test_ratio=gnn_cfg["data"]["test_ratio"],
                batch_size=gnn_cfg["model"]["batch_size"],
                num_neighbors=gnn_cfg["model"]["num_neighbors"]
            )
            train_data, train_loader, val_data, test_data, full_pos, in_ch, edge_dim = processor.prepare_data()
            
            # ---- 1. Baseline Evaluation ----
            print("\n>> Evaluating Baselines")
            base_evaluator = Evaluator()
            
            # Cosine Baseline (no training needed)
            cosine_scorer = CosineBaselineScorer()
            base_evaluator.evaluate_metrics_master(
                "Cosine", test_data, neg_ratio=1.0,
                scorer=cosine_scorer, threshold=0.5,
                full_pos_edges=full_pos)
            
            # MLP Baseline (trained on the real train_data, not reconstructed)
            mlp_model = RawMLPConcatPredictor(in_dim=in_ch, hidden=gnn_cfg["model"]["hidden_dim"])
            mlp_trainer = BaselineTrainer(mlp_model, lr=gnn_cfg["model"]["learning_rate"])
            mlp_model = mlp_trainer.train(train_data, val_data, epochs=gnn_cfg["model"]["num_epochs"])
            mlp_scorer = MLPBaselineScorer(mlp_model)
            base_evaluator.evaluate_metrics_master(
                "MLP", test_data, neg_ratio=1.0,
                scorer=mlp_scorer, threshold=0.5,
                full_pos_edges=full_pos)
            
            # ---- 2. GNN Training + Evaluation ----
            for encoder_name in enabled_encoders:
                print(f"\n>> Training GNN: {encoder_name.upper()}")
                actual_encoder = 'late-fuse' if graph_type == 'late_fuse' else encoder_name
                
                trainer = GNNTraining(
                    output_dir=gnn_cfg["output_dir"],
                    encoder=actual_encoder,
                    decoder=gnn_cfg["model"]["decoder"],
                    in_channels=in_ch,
                    hidden_dim=gnn_cfg["model"]["hidden_dim"],
                    embed_dim=gnn_cfg["model"]["embed_dim"],
                    edge_dim=edge_dim,
                    learning_rate=gnn_cfg["model"]["learning_rate"],
                    num_epochs=gnn_cfg["model"]["num_epochs"],
                    weight_decay=gnn_cfg["model"]["weight_decay"],
                    dropout=gnn_cfg["model"]["dropout"],
                    neg_ratio=gnn_cfg["model"]["neg_ratio"],
                    decoder_hidden=gnn_cfg["model"]["decoder_hidden"],
                    decoder_dropout=gnn_cfg["model"]["decoder_dropout"],
                    scheduler=gnn_cfg["model"]["scheduler"],
                    grad_clip=gnn_cfg["model"]["grad_clip"],
                    fusion=gnn_cfg["model"]["fusion"],
                    late_fuse_base_encoder=encoder_name if graph_type == 'late_fuse' else None,
                    edge_dim_retweet=getattr(train_data, 'edge_dim_retweet', 3),
                    edge_dim_reply=getattr(train_data, 'edge_dim_reply', 3),
                    edge_dim_mention=getattr(train_data, 'edge_dim_mention', 3),
                    early_stopping_patience=gnn_cfg["model"]["early_stopping_patience"]
                )
                
                model = trainer.train(train_loader, val_data)
                
                gnn_eval = GNNEvaluator(model=model)
                gnn_eval.evaluate(val_data, test_data, 
                                  neg_ratios=gnn_cfg["evaluation"]["neg_ratios"],
                                  full_pos_edges=full_pos)

    if do_graph_generation or do_community_detection_leiden or do_community_detection_combo or do_community_hierarchical or do_get_text or do_topic_builder:
        graph_analysis = GraphAnalysis(P)
        try:
            graph_analysis.run()
        except Exception as e:
            raise
