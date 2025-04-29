import logging
from Utils.logging_config import setup_logging
from collections import namedtuple

from Utils.Writer import Writer
from algorithms.CommunityText import CommunityText
from algorithms.EdgeToGraph import EdgeToGraph
from algorithms.GraphGeneration import GraphGeneration
from community.Combo import Combo
from community.Leiden import Leiden


class GraphAnalysis:
    logger = logging.getLogger('GraphAnalysis')

    def __init__(self, parameters: namedtuple):
        self.parameters = parameters

    def run(self):
        if self.parameters.do_graph_generation:
            gg = GraphGeneration(uri=self.parameters.source_uri,
                                 username=self.parameters.source_username,
                                 password=self.parameters.source_password,
                                 auth_source=self.parameters.source_auth_source,
                                 auth_mechanism=self.parameters.source_auth_mechanism,
                                 collection=self.parameters.source_collection,
                                 start_date=self.parameters.source_chunk_start_date,
                                 end_date=self.parameters.source_chunk_end_date,
                                 input_type=self.parameters.source_input_type,
                                 output_file_path=self.parameters.output_graph_path,
                                 retweet=self.parameters.do_retweet_graph,
                                 tweet_retweet=self.parameters.do_tweet_retweet_graph,
                                 user_hashtag=self.parameters.do_hashtag_graph,
                                 hashtag_cooccurrences=self.parameters.do_hashtag_cooccurrences_graph,
                                 response=self.parameters.do_response_graph,
                                 mention=self.parameters.do_mention_graph)

            gg.connect(self.parameters.source_db_name)

            """
            use it when mongo is available again
            """
            w = {'$or': [{'hashtagEntities': {'$exists': True}}, {'retweeted_status': {'$exists': True}},
                         {'in_reply_to_status_id': {'$exists': True}}]}
            s = {'_id': 0,
                 'id': 1,
                 'in_reply_to_status_id': 1,
                 'in_reply_to_user_id': 1,
                 'retweeted_status.created_at': 1,
                 'retweeted_status.id': 1,
                 'retweeted_status.user.id': 1,
                 'retweeted_status.user.screen_name': 1,
                 'user.id': 1,
                 'user.screen_name': 1,
                 'hashtagEntities': 1,
                 'created_at': 1,
                 'userMentionEntities': 1}

            """
            w = 'hashtagEntities.notnull() | `retweeted_status.id`.notnull() | in_reply_to_status_id.notnull()'
            s = ['id',
                 'in_reply_to_status_id',
                 'in_reply_to_user_id',
                 'retweeted_status.created_at.$date',
                 'retweeted_status.id',
                 'retweeted_status.user.id',
                 'retweeted_status.user.screen_name',
                 'user.id',
                 'user.screen_name',
                 'hashtagEntities',
                 'created_at.$date',
                 'userMentionEntities'
                 ]
            """
            gg.query_data_in_chunks(w, s)

        # Community detection
        community_detection = self.parameters.do_community_detection_combo or self.parameters.do_community_detection_leiden
        if community_detection:
            w = Writer()
            if self.parameters.do_read_graph_from_file:
                g = w.read_csv_in_batch(self.parameters.graph_file_path[0], 10000)


        if self.parameters.do_community_detection_combo:
            combo_instance = Combo()
            # todo remove?
            # g = combo_instance.csv_to_nx(graph)
            rps = combo_instance.compute_combo_in_parallel(g)
            for rp in rps:
                combo_instance.export_partition(g, rp,
                                                self.parameters.community_combo_prop["community_output_file_path"],
                                                ["type", "{}".format(rp)])
            combo_instance.export_graph(g, self.parameters.community_combo_prop["community_output_file_path"])

        if self.parameters.do_community_detection_leiden:
            leiden_instance = Leiden()
            g = leiden_instance.csv_to_igraph(g) #todo pay attention: switched graph with g
            # leiden_instance.compute_pagerank(g)
            rps = leiden_instance.compute_leiden_in_parallel(g)
            for rp in rps:
                leiden_instance.export_partition(g, rp,
                                                 self.parameters.community_leiden_prop["community_output_file_path"],
                                                 ["name", "type", "{}".format(rp)])
            leiden_instance.export_graph(g, self.parameters.community_leiden_prop["community_output_file_path"])

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


