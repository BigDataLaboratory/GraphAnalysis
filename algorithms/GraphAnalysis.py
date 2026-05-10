import logging
import pandas as pd
import os
from collections import namedtuple

from Utils.Writer import Writer, SUPPORTED_FORMATS
from algorithms.CommunityText import CommunityText
from algorithms.GraphGeneration import GraphGeneration
from algorithms.GraphGenerationUser import GraphGenerationUser
from community.Combo import Combo
from community.Leiden import Leiden
from algorithms import mongoQueries


class GraphAnalysis:
    logger = logging.getLogger('GraphAnalysis')

    def __init__(self, parameters: namedtuple):
        self.parameters = parameters

    def run(self):
        import time
        start_time = time.time()

        # ── Resolve and validate file_format ──────────────────────────────────
        file_format = getattr(self.parameters, "output_file_format", "csv")
        if file_format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file_format '{file_format}'. "
                f"Choose from {SUPPORTED_FORMATS}."
            )

        # ── Graph Generation ──────────────────────────────────────────────────
        if self.parameters.do_graph_generation:
            p = self.parameters

            # User-user graph with node features and typed edges
            if p.do_user_user_graph:
                delete_tmp = getattr(p, "delete_tmp_after_merge", False)
                ggu = GraphGenerationUser(
                    uri=p.source_uri,
                    username=p.source_username,
                    password=p.source_password,
                    auth_source=p.source_auth_source,
                    auth_mechanism=p.source_auth_mechanism,
                    database_name=p.source_db_name,
                    collection=p.source_collection,
                    output_file_path=p.output_graph_path,
                    delete_tmp_after_merge=delete_tmp,
                    file_format=file_format,
                )
                ggu.run(checkpoint_every=p.checkpoint_every)

            # Typed edge extraction (retweet / mention / response / hashtag …)
            needs_graph_generation = any([
                p.do_retweet_graph,
                p.do_tweet_retweet_graph,
                p.do_hashtag_graph,
                p.do_hashtag_cooccurrences_graph,
                p.do_response_graph,
                p.do_mention_graph,
            ])

            if needs_graph_generation:
                gg = GraphGeneration(
                    uri=p.source_uri,
                    username=p.source_username,
                    password=p.source_password,
                    auth_source=p.source_auth_source,
                    auth_mechanism=p.source_auth_mechanism,
                    database_name=p.source_db_name,
                    collection=p.source_collection,
                    start_date=p.source_chunk_start_date,
                    end_date=p.source_chunk_end_date,
                    method=p.source_method,
                    input_type=p.source_input_type,
                    output_file_path=p.output_graph_path,
                    retweet=p.do_retweet_graph,
                    tweet_retweet=p.do_tweet_retweet_graph,
                    user_hashtag=p.do_hashtag_graph,
                    hashtag_cooccurrences=p.do_hashtag_cooccurrences_graph,
                    response=p.do_response_graph,
                    mention=p.do_mention_graph,
                    file_format=file_format,
                )
                w, s = mongoQueries.extract_tweets_if_contains_hashtags_or_is_retweet_or_reply()
                gg.query_data_in_chunks(w, s, method=p.source_method)
            else:
                self.logger.info("Skipping GraphGeneration: no graph type enabled.")

        # Community detection
        community_detection = self.parameters.do_community_detection_combo or self.parameters.do_community_detection_leiden or self.parameters.do_community_hierarchical
        if community_detection:
            if self.parameters.do_community_detection_leiden or self.parameters.do_community_hierarchical:
                w = Writer('igraph', self.parameters.temporal)
            elif self.parameters.do_community_detection_combo:
                w = Writer('nx')

            if self.parameters.do_read_from_edge_list:
                w.read_csv_in_batch(self.parameters.graph_file_path[0], self.parameters.pickle_graph_path, 300000)
            if self.parameters.do_read_graph_from_file:
                g = w.read_pickle_parallel_preserve_time(self.parameters.pickle_graph_path)

        if self.parameters.do_community_detection_combo:
            combo_instance = Combo()
            rps = combo_instance.compute_combo_in_parallel(g)
            for rp in rps:
                combo_instance.export_partition(g, rp,
                                                self.parameters.community_combo_prop["community_output_file_path"],
                                                ["type", "{}".format(rp)])
            combo_instance.export_graph(g, self.parameters.community_combo_prop["community_output_file_path"])

        if self.parameters.do_community_detection_leiden:
            leiden_instance = Leiden(g)
            if not self.parameters.temporal:
                rps = leiden_instance.compute_leiden((0.1, 1.0))
                for rp in rps:
                    leiden_instance.export_partition(leiden_instance.get_graph(), rp,
                                                    self.parameters.community_leiden_prop["community_output_file_path"],
                                                    ["name", "type", "{}".format(rp)])
                leiden_instance.export_graph(leiden_instance.get_graph(), self.parameters.community_leiden_prop["community_output_file_path"])
            else:
                rps = leiden_instance.compute_leiden_temporal_incremental(
                    method="CPM",
                    resolution_parameter_range=(0.1, 1.0),
                    lambda_temporal=0.1,
                    cap_bonus=1,
                    n_iterations=2,
                )   
                for rp in rps:
                    leiden_instance.export_partition(leiden_instance.get_graph(), rp,
                                                    self.parameters.community_leiden_prop["community_output_file_path"],
                                                    ["name", "type", "{}".format(rp)])
        
        if self.parameters.do_community_hierarchical:
            collapse_nodes = self.parameters.community_hierarchical_prop["collapse_nodes"]
            resolution_col = self.parameters.community_hierarchical_prop["community_col_name"]
            top_k = self.parameters.community_hierarchical_prop["top_k"]

            communities = pd.read_csv(self.parameters.community_hierarchical_prop["first_level_communities_file"], sep=',', header=0, low_memory=False)
            labels = communities[resolution_col].to_numpy()

            if collapse_nodes:
                col_to_extract_communities = "collapsed"
                self.logger.info("Collapsing nodes with min size {}".format(self.parameters.community_hierarchical_prop["min_size"]))
                self.logger.info("Number of communities before collapsing: {}".format(len(set(labels))))
                labels_c = w.collapse_nodes(labels, self.parameters.community_hierarchical_prop["min_size"])
                self.logger.info("Number of communities after collapsing: {}".format(len(set(labels_c))))
                # Update the communities DataFrame with collapsed labels
                communities[col_to_extract_communities] = labels_c
                vc = pd.Series(labels_c)
            else:
                col_to_extract_communities = resolution_col
                vc = pd.Series(labels)
            
            vc = vc[vc != -1].value_counts()
            top_communities = vc.head(top_k).index

            top_node_indices = communities[communities[col_to_extract_communities].isin(top_communities)]["id"].to_numpy()

            # Create a subgraph with only the top communities
            subgraph = g.subgraph(top_node_indices)
            self.logger.info("Number of nodes in subgraph: {}".format(len(subgraph.vs)))
            self.logger.info("Number of edges in subgraph: {}".format(len(subgraph.es)))

            # Export the subgraph
            filename = "top500_final.pkl"
            full_path = os.path.join("/ipazianas/pasquini/twitter_graph_dump", filename)
            with open(full_path, "wb") as f:
                subgraph.write_pickle(full_path)
            self.logger.info("Final graph saved.")
            
            leiden_instance = Leiden(subgraph)
            rps = leiden_instance.compute_leiden((0.1, 1.0), number_of_resolutions=10)
            for rp in rps:
                leiden_instance.export_partition(leiden_instance.get_graph(), rp,
                                                    self.parameters.community_hierarchical_prop["community_output_file_path"],
                                                    ["name", "type", "{}".format(rp)])
            leiden_instance.export_graph(leiden_instance.get_graph(), self.parameters.community_hierarchical_prop["community_output_file_path"])

        # Get text data from raw dataset
        if self.parameters.do_get_text:
            ct = CommunityText(self.parameters.community_indexes if self.parameters.community_indexes else [],
                               uri=self.parameters.td_uri,
                               username=self.parameters.td_username,
                               password=self.parameters.td_password,
                               auth_source=self.parameters.td_auth_source,
                               auth_mechanism=self.parameters.td_auth_mechanism,
                               collection=self.parameters.td_collection
                               )
            ct.connect(self.parameters.td_db_name)
            # read communities saved on external file
            if self.parameters.do_read_communities_from_file:
                ct.set_comms_file_path(self.parameters.community_file_path)
            else:
                raise AttributeError("It's not possible to use communities generated at runtime")
            # read map saved on external file
            if not self.parameters.do_read_maps_from_file:
                raise AttributeError("It's not possible to use maps generated at runtime")
            else:
                maps = [self.parameters.user_map, self.parameters.retweet_user_map]
                ct.set_maps(maps)

            """
            match = {'$match': {
                        'user.id': { '$in': u }  # Filter docs based on users list
                    }}
            """
            project = {'$project': {
                'text': {
                    '$cond': {
                        'if': {'$gt': ['$retweeted_status', None]},  # Check if it is a retweet
                        'then': '$retweeted_status.text',  # If it is a retweet, get the text field retweeted_status
                        'else': '$text'  # else, get the original text field
                    }
                },
                '_id': 0,
                'user.id': 1,
                'created_at': 1,
                'type': {
                    '$cond': {
                        'if': {'$gt': ['$retweeted_status', None]},  # Check if it is a retweet
                        'then': 'normal',  # If it is a retweet, set "normal" to type
                        'else': 'retweet'  # else set "retweet"
                    }
                }
            }}
            """
            use it when mongo is available again

            pipeline = [
                {
                    '$match': {
                        'user.id': { '$in': u }  # Filter docs based on users list
                    }
                },
                {
                    '$project': {
                        'text': {
                            '$cond': {
                                'if': { '$gt': ['$retweeted_status', None] },  # Check if it is a retweet
                                'then': '$retweeted_status.text',              # If it is a retweet, get the text field retweeted_status
                                'else': '$text'                                # else, get the original text field
                            }
                        },
                        '_id': 0,
                        'user.id': 1,
                        'created_at': 1,
                        'type': {
                            '$cond': {
                                'if': { '$gt': ['$retweeted_status', None] },  # Check if it is a retweet
                                'then': 'normal',                              # If it is a retweet, set "normal" to type
                                'else': 'retweet'                              # else set "retweet"
                            }
                        }
                    }
                }
            ]

            results = collection.aggregate(pipeline)
            """
            ct.get_users_tweet_text(0, project, col_comms=["id", "name", "type", "pagerank", "0.6"])
            # result = raw_data.query(None, ['text', 'user.id', 'created_at.$date'])
            self.logger.debug("Generated final intermediate result with text data")

        if self.parameters.do_topic_builder:
            # BERT Topic
            """
            topic_generator = TopicGenerator(result["text"].values.tolist())
            tm, t, p = topic_generator.topic_modeling()

            tp = pd.DataFrame(
                {'topic': t,
                 'prob': p
                 })

            text_with_topics = pd.concat([result, tp], axis=1)
            di = tm.get_document_info(result["text"].values.tolist())

            Utils.persist_to_file(text_with_topics, self.parameters.topics_file_path + "1234")
            Utils.persist_to_file(tm.get_topic_info(), self.parameters.topics_file_path)
            Utils.persist_to_file(di, self.parameters.docs_file_path)
            tm.save(self.parameters.model_path, serialization=self.parameters.model_serialization, save_ctfidf=True)
            """

        end_time = time.time()
        elapsed = end_time - start_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        self.logger.info(f"GraphAnalysis execution complete in {time_str}.")


