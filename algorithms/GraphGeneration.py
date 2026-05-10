import logging
import multiprocessing
import os
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from datetime import timedelta, timezone, datetime
from enum import Enum
from zoneinfo import ZoneInfo

import pytz
from pymongo import ASCENDING

from Utils.Const import Const as c
from Utils.Utils import Utils
from Utils.Writer import Writer
from algorithms.MongoConnection import MongoConnection


class GraphType(Enum):
    retweet = 0
    tweet_retweet = 1
    user_hashtag = 2
    hashtag_cooccurrences = 3
    response = 4
    mention = 5


class MapType(Enum):
    user_id = 0
    user_retweeted_id = 1
    tweet_id = 2
    hashtag = 3


class GraphGeneration(MongoConnection):
    """
    GraphGeneration class for processing and generating graphs from MongoDB data.
    This class extends MongoConnection to connect to a MongoDB database and process tweet documents
    to extract relationships and generate graphs based on various criteria such as retweets, hashtags, responses,
    and mentions. It supports both full data processing and chunked processing based on time intervals.
    It also provides methods to save intermediate results to checkpoint files, merge and aggregate results,
    and generate date ranges for processing data in chunks.
    It uses a multiprocessing approach to handle large datasets efficiently, allowing for parallel processing
    of tweet documents. The results are saved in CSV files, organized by graph and map types
    for easy access and analysis.

    Attributes:
        uri (str): MongoDB connection URI.
        username (str): MongoDB username.
        password (str): MongoDB password.
        auth_source (str): Authentication source for MongoDB.
        auth_mechanism (str): Authentication mechanism for MongoDB.
        db (str): Database name.
        database_name (str): Name of the MongoDB database.
        collection (str): Collection name in the MongoDB database.
        start_date (datetime): Start date for processing data.
        end_date (datetime): End date for processing data.
        checkpoint_folder (str): Folder to store checkpoint files.
        sep (str): Separator used in file paths.
        type (str): Type of input data (e.g., "mongo").
        output_file_path (str): Path to save output files.
        id (str): Unique identifier for the graph generation process.
        retweet (bool): Flag to include retweet relationships.
        tweet_retweet (bool): Flag to include tweet retweet relationships.
        user_hashtag (bool): Flag to include user hashtag relationships.
        hashtag_cooccurrences (bool): Flag to include hashtag cooccurrences.
        response (bool): Flag to include response relationships.
        mention (bool): Flag to include mention relationships.
        w (Writer): Instance of Writer class for writing results to files.
        logger (logging.Logger): Logger instance for logging messages.
    """
    logger = logging.getLogger('GraphGeneration')

    def __init__(self, uri, username=None, password=None, auth_source=None, auth_mechanism=None, database_name=None, db=None,
                 collection=None, start_date=None, end_date=None, method="full", input_type="mongo", output_file_path=None, retweet=False, tweet_retweet=False,
                 user_hashtag=False, hashtag_cooccurrences=False, response=False, mention=False, file_format="csv"):
        super().__init__(uri, username, password, auth_source, auth_mechanism, db, database_name, collection, start_date, end_date)
        self.checkpoint_folder = "tmp"
        self.sep = "_"
        self.type = input_type
        self.output_file_path = output_file_path
        self.id = uuid.uuid1().hex
        self.retweet = retweet
        self.tweet_retweet = tweet_retweet
        self.user_hashtag = user_hashtag
        self.hashtag_cooccurrences = hashtag_cooccurrences
        self.response = response
        self.mention = mention
        self.file_format = file_format
        self.w = Writer()

    def process_document(self, d):
        """
        Processes a single document to extract edges and maps based on the specified relationships.
        This method extracts relationships such as retweets, tweet retweets, user hashtags, hashtag cooccurrences,
        responses, and mentions from the document and returns a list of edges and a set of maps.
        Each edge is represented as a tuple containing the source node, target node, weight, and relationship type.
        The maps are used to track unique nodes and relationships.

        :param d: Document to process, a tweet document from MongoDB.
        :return: A tuple containing a list of edges and a set of maps.
        """
        o = []
        m = set()
        n_user_id = Utils.hash(d['user']['id'])
        m.add((d['user']['id'], n_user_id, 0))

        weight = 1

        if d.get('retweeted_status', None) is not None and self.retweet:
            relationship_u_rt = 0
            n_rt_user_id = Utils.hash(d['retweeted_status']['user']['id'])
            e_rt = n_user_id, n_rt_user_id, weight, relationship_u_rt
            o.append(e_rt)
            m.add((d['retweeted_status']['user']['id'], n_rt_user_id, 1))

        if d.get('retweeted_status', None) is not None and self.tweet_retweet:
            relationship_t_rt = 1
            n_tweet_id = Utils.hash(d['id'])
            n_rt_tweet_id = Utils.hash(d['retweeted_status']['id'])
            a_created_at_tweet = d['created_at'].timestamp()
            a_created_at_rt = d['retweeted_status']['created_at'].timestamp()
            e_tweet_retweet = (
                n_tweet_id, n_rt_tweet_id, weight, (a_created_at_tweet, a_created_at_rt), relationship_t_rt)
            o.append(e_tweet_retweet)
            m.add((d['id'], n_tweet_id, 2))
            m.add((d['retweeted_status']['id'], n_rt_tweet_id, 2))

        if d.get('hashtagEntities', None) is not None and self.user_hashtag:
            relationship = 2
            n_ht = d['hashtagEntities'].lower().split('|') if isinstance(d['hashtagEntities'], str) else []
            ht = [(n_user_id, Utils.compute_hash(x), weight, relationship) for x in n_ht]
            o.extend(ht)
            for x in n_ht:
                m.add((x, Utils.compute_hash(x), 3))

        if d.get('hashtagEntities', None) is not None and self.hashtag_cooccurrences:
            relationship = 3
            ht_combinations = Utils.combinations_list(d['hashtagEntities'].lower().split('|')) if isinstance(
                d['hashtagEntities'], str) else []
            e_hts_natural = [(x[0], x[1], weight, relationship) for x in ht_combinations]
            e_hts_inverse = [(x[1], x[0], weight, relationship) for x in ht_combinations]
            o.extend(e_hts_natural)
            o.extend(e_hts_inverse)

        if d.get('in_reply_to_user_id', -1) != -1 and self.response:
            relationship = 4
            n_reply_user_id = Utils.hash(d['in_reply_to_user_id'])
            e_reply = n_user_id, n_reply_user_id, weight, relationship
            o.append(e_reply)

        if d.get('userMentionEntities', None) is not None and self.mention:
            relationship = 5
            n_mentions = d['userMentionEntities'].lower().split('|') if isinstance(d['userMentionEntities'],
                                                                                   str) else []
            e_mentions = [(n_user_id, Utils.compute_hash(x), weight, relationship) for x in n_mentions]
            o.extend(e_mentions)

        return o, m

    def process_bucket_document(self, d):
        """
        Processes a bucket document to extract edges and maps based on the specified relationships.
        This method extracts relationships such as retweets, tweet retweets, user hashtags, hashtag cooccurrences,
        responses, and mentions from the document and returns a list of edges and a set of maps.
        Each edge is represented as a tuple containing the source node, target node, date, weight, and relationship type.
        The maps are used to track unique nodes and relationships.

        :param d: Bucket document containing a list of tweets.
        :return: A tuple containing a list of edges and a set of maps.
        """
        o = []
        m = set()

        date = d["_id"]
        tz_rome = ZoneInfo("Europe/Rome")
        weight = 1

        for tweet in d['docs']:
            n_user_id = Utils.hash(tweet['user']['id'])
            m.add((tweet['user']['id'], n_user_id, 0))

            if tweet.get('retweeted_status', None) is not None and self.retweet:
                relationship_u_rt = 0
                n_rt_user_id = Utils.hash(tweet['retweeted_status']['user']['id'])
                e_rt = n_user_id, n_rt_user_id, date, weight, relationship_u_rt
                o.append(e_rt)
                m.add((tweet['retweeted_status']['user']['id'], n_rt_user_id, 1))

            if tweet.get('retweeted_status', None) is not None and self.tweet_retweet:
                relationship_t_rt = 1
                n_tweet_id = Utils.hash(tweet['id'])
                n_rt_tweet_id = Utils.hash(tweet['retweeted_status']['id'])
                a_created_at_tweet = tweet['created_at'].astimezone(tz_rome).timestamp()
                a_created_at_rt = tweet['retweeted_status']['created_at'].astimezone(tz_rome).timestamp()
                e_tweet_retweet = (
                    n_tweet_id, n_rt_tweet_id, date, weight, (a_created_at_tweet, a_created_at_rt), relationship_t_rt)
                o.append(e_tweet_retweet)
                m.add((tweet['id'], n_tweet_id, 2))
                m.add((tweet['retweeted_status']['id'], n_rt_tweet_id, 2))

            if tweet.get('hashtagEntities', None) is not None and self.user_hashtag:
                relationship = 2
                n_ht = tweet['hashtagEntities'].lower().split('|') if isinstance(tweet['hashtagEntities'], str) else []
                ht = [(n_user_id, Utils.compute_hash(x), date, weight, relationship) for x in n_ht]
                o.extend(ht)
                for x in n_ht:
                    m.add((x, Utils.compute_hash(x), 3))

            if tweet.get('hashtagEntities', None) is not None and self.hashtag_cooccurrences:
                relationship = 3
                ht_combinations = Utils.combinations_list(tweet['hashtagEntities'].lower().split('|')) if isinstance(
                    tweet['hashtagEntities'], str) else []
                e_hts_natural = [(x[0], x[1], date, weight, relationship) for x in ht_combinations]
                e_hts_inverse = [(x[1], x[0], date, weight, relationship) for x in ht_combinations]
                o.extend(e_hts_natural)
                o.extend(e_hts_inverse)

            if tweet.get('in_reply_to_user_id', -1) != -1 and self.response:
                relationship = 4
                n_reply_user_id = Utils.hash(tweet['in_reply_to_user_id'])
                e_reply = n_user_id, n_reply_user_id, date, weight, relationship
                o.append(e_reply)

            if tweet.get('userMentionEntities', None) is not None and self.mention:
                relationship = 5
                n_mentions = tweet['userMentionEntities'].lower().split('|') if isinstance(tweet['userMentionEntities'],
                                                                                       str) else []
                e_mentions = [(n_user_id, Utils.compute_hash(x), date, weight, relationship) for x in n_mentions]
                o.extend(e_mentions)

        return o, m


    def generate_date_chunks(self, start_date, end_date, delta):
        """
        Generates date ranges to divide the data into chunks based on the `created_at` field.
        This method yields tuples of start and end dates for each chunk, allowing for efficient processing of large datasets.

        :param start_date: Start date for the data range.
        :param end_date: End date for the data range.
        :param delta: Time delta to define the size of each chunk (e.g., timedelta(weeks=1) for weekly chunks).
        """
        current_date = start_date
        while current_date < end_date:
            next_date = min(current_date + delta, end_date)
            yield current_date, next_date
            current_date = next_date

    def merge_and_aggregate_checkpoints(self, method="full"):
        """
        Merges and aggregates checkpoint files for each graph type and map type.
        This method reads all checkpoint files, aggregates the results based on the graph and map types,
        and writes the final results to CSV files.
        The results are saved in a structured format based on the graph and map types.

        :param method: Processing method, either "full" or "day/week".
        """
        checkpoint_dir = os.sep.join([self.output_file_path, self.checkpoint_folder, self.id])
        # Iterate over all checkpoint files in the folder
        for graph_type in GraphType:
            if self.file_format == "pickle":
                checkpoint_files = self.sep.join([self.checkpoint_folder, self.id, graph_type.name, "*"]) + ".pkl"
            else:
                checkpoint_files = self.sep.join([self.checkpoint_folder, self.id, graph_type.name, "*"])
            list_checkpoint_files = Writer.list_checkpoint_files(os.sep.join([checkpoint_dir, checkpoint_files]))
            merged_file_path = os.sep.join([self.output_file_path, self.id, graph_type.name])

            if method != "full":
                aggregated_results = defaultdict(lambda: 0)  # Structure: { (key1, key2): sum_third }

                for file_path in list_checkpoint_files:
                    checkpoint_data = Writer.load_checkpoint_file(file_path)
                    if graph_type.name != GraphType(1).name:
                        for row in checkpoint_data:
                            key = (int(row[0]), int(row[1]), int(row[2]), float(row[3]))
                            aggregated_results[key] += int(row[4])
                    else:
                        for row in checkpoint_data:
                            key = (int(row[0]), int(row[1]), int(row[2]), float(row[3]))
                            aggregated_results[key] = eval(row[5])
                final_result_graph = []
                for k, v in aggregated_results.items():
                    if k[0] != 1:
                        final_result_graph.append((k[0], k[1], k[2], k[3], v))
                    else:
                        final_result_graph.append((k[0], k[1], k[2], k[3], v[0], v[1]))
                    
                if self.file_format == "pickle":
                    columns = ["relationship", "src", "dst", "date", "weight"] if graph_type.name != GraphType(1).name else ["relationship", "src", "dst", "date", "date_tweet", "date_rt"]
                    Writer.write_on_pickle(merged_file_path + ".pkl", final_result_graph, columns=columns)
                else:
                    Writer.write_on_csv(merged_file_path + ".csv", final_result_graph)
            elif method == "full":
                aggregated_results = defaultdict(lambda: 0)  # Structure: { (key1, key2): sum_third }
                for file_path in list_checkpoint_files:
                    checkpoint_data = Writer.load_checkpoint_file(file_path)
                    # Aggregate each row
                    if graph_type.name != GraphType(1).name:
                        for row in checkpoint_data:
                            key = (int(row[0]), int(row[1]), int(row[2]))
                            aggregated_results[key] += int(row[3])
                    else:
                        for row in checkpoint_data:
                            key = (int(row[0]), int(row[1]), int(row[2]))
                            aggregated_results[key] = eval(row[4])

                final_result_graph = []
                for k, v in aggregated_results.items():
                    if k[0] != 1:
                        final_result_graph.append((k[0], k[1], k[2], v))
                    else:
                        final_result_graph.append((k[0], k[1], k[2], v[0], v[1]))

                if self.file_format == "pickle":
                    columns = ["relationship", "src", "dst", "weight"] if graph_type.name != GraphType(1).name else ["relationship", "src", "dst", "date_tweet", "date_rt"]
                    Writer.write_on_pickle(merged_file_path + ".pkl", final_result_graph, columns=columns)
                else:
                    Writer.write_on_csv(merged_file_path + ".csv", final_result_graph)

        # Iterate over all checkpoint files in the folder
        for map_type in MapType:
            if self.file_format == "pickle":
                checkpoint_files = self.sep.join([self.checkpoint_folder, self.id, map_type.name, c.MAP, "*"]) + ".pkl"
            else:
                checkpoint_files = self.sep.join([self.checkpoint_folder, self.id, map_type.name, c.MAP, "*"])
            list_map_files = Writer.list_checkpoint_files(os.sep.join([checkpoint_dir, checkpoint_files]))
            merged_file_path = os.sep.join([self.output_file_path, str(self.id), map_type.name])

            all_data = []

            for file_path in list_map_files:
                checkpoint_data = Writer.load_checkpoint_file(file_path)
                all_data.extend(checkpoint_data)
            unique_data = list(set(tuple(row) for row in all_data))  # Remove duplicates by converting to set and back to list
            
            if self.file_format == "pickle":
                columns = ["original_id", "hash_id", "map_type"]
                Writer.write_on_pickle(merged_file_path + ".pkl", unique_data, columns=columns)
            else:
                Writer.write_on_csv(merged_file_path + ".csv", unique_data)

    def save_checkpoint(self, intermediate_results, intermediate_map, process_id):
        """
        Saves intermediate results and maps to checkpoint files.
        The results are saved in a structured format based on the graph and map types.
        Each type of graph and map is saved in its own file, organized by process ID.

        :param intermediate_results: Dictionary containing intermediate results.
        :param intermediate_map: Set containing intermediate map data.
        :param process_id: Unique identifier for the process saving the checkpoint.

        :return: None
        """
        result_graph = {GraphType(0).name: [], GraphType(1).name: [], GraphType(2).name: [], GraphType(3).name: [],
                        GraphType(4).name: [], GraphType(5).name: []}
        result_map = {MapType(0).name: [], MapType(1).name: [], MapType(2).name: [], MapType(3).name: []}

        for k, v in intermediate_results.items():
            result_graph[GraphType(k[2]).name].append((k[2], k[0], k[1], v)) if k[2] != 1 else result_graph[
                GraphType(k[2]).name].append((k[2], k[0], k[1], v[0], v[1]))

        for e in intermediate_map:
            result_map[MapType(e[2]).name].append(e)

        dir_path = os.sep.join([self.output_file_path, self.checkpoint_folder, str(self.id)])
        for k in result_graph:
            if self.file_format == "pickle":
                file_path = self.sep.join([self.checkpoint_folder, str(self.id), str(k), str(process_id)]) + ".pkl"
                path = os.sep.join([dir_path, file_path])
                columns = ["relationship", "src", "dst", "weight"] if k != GraphType(1).name else ["relationship", "src", "dst", "date_tweet", "date_rt"]
                Writer.write_on_pickle(path, result_graph[k], columns=columns)
            else:
                file_path = self.sep.join([self.checkpoint_folder, str(self.id), str(k), str(process_id)])
                path = os.sep.join([dir_path, file_path])
                Writer.write_on_csv(path, result_graph[k])
        for k in result_map:
            if self.file_format == "pickle":
                file_path = self.sep.join([self.checkpoint_folder, str(self.id), str(k), c.MAP, str(process_id)]) + ".pkl"
                path = os.sep.join([dir_path, file_path])
                columns = ["original_id", "hash_id", "map_type"]
                Writer.write_on_pickle(path, result_map[k], columns=columns)
            else:
                file_path = self.sep.join([self.checkpoint_folder, str(self.id), str(k), c.MAP, str(process_id)])
                path = os.sep.join([dir_path, file_path])
                Writer.write_on_csv(path, result_map[k])

    def save_bucket_checkpoint(self, intermediate_results, intermediate_map, process_id):
        """
        Saves intermediate results and maps to checkpoint files for bucket processing.
        The results are saved in a structured format based on the graph and map types.
        Each type of graph and map is saved in its own file, organized by process ID.
        This method is specifically designed for processing data in buckets, such as daily or weekly.

        :param intermediate_results: Dictionary containing intermediate results.
        :param intermediate_map: Set containing intermediate map data.
        :param process_id: Unique identifier for the process saving the checkpoint.
        """
        result_graph = {GraphType(0).name: [], GraphType(1).name: [], GraphType(2).name: [], GraphType(3).name: [],
                        GraphType(4).name: [], GraphType(5).name: []}
        result_map = {MapType(0).name: [], MapType(1).name: [], MapType(2).name: [], MapType(3).name: []}

        for k, v in intermediate_results.items():
            result_graph[GraphType(k[3]).name].append((k[3], k[0], k[1], k[2], v)) if k[3] != 1 else result_graph[
                GraphType(k[3]).name].append((k[3], k[0], k[1], k[2], v[0], v[1]))
        
        for e in intermediate_map:
            result_map[MapType(e[2]).name].append(e)

        dir_path = os.sep.join([self.output_file_path, self.checkpoint_folder, str(self.id)])
        for k in result_graph:
            if self.file_format == "pickle":
                file_path = self.sep.join([self.checkpoint_folder, str(self.id), str(k), str(process_id)]) + ".pkl"
                path = os.sep.join([dir_path, file_path])
                columns = ["relationship", "src", "dst", "date", "weight"] if k != GraphType(1).name else ["relationship", "src", "dst", "date", "date_tweet", "date_rt"]
                Writer.write_on_pickle(path, result_graph[k], columns=columns)
            else:
                file_path = self.sep.join([self.checkpoint_folder, str(self.id), str(k), str(process_id)])
                path = os.sep.join([dir_path, file_path])
                Writer.write_on_csv(path, result_graph[k])
        for k in result_map:
            if self.file_format == "pickle":
                file_path = self.sep.join([self.checkpoint_folder, str(self.id), str(k), c.MAP, str(process_id)]) + ".pkl"
                path = os.sep.join([dir_path, file_path])
                columns = ["original_id", "hash_id", "map_type"]
                Writer.write_on_pickle(path, result_map[k], columns=columns)
            else:
                file_path = self.sep.join([self.checkpoint_folder, str(self.id), str(k), c.MAP, str(process_id)])
                path = os.sep.join([dir_path, file_path])
                Writer.write_on_csv(path, result_map[k])

    def worker_process(self, where, project, method, chunk, batch_size, checkpoint_interval, process_id):
        """
        Worker function to process a chunk of data from MongoDB.
        This function connects to the MongoDB, retrieves documents in the specified chunk,
        processes them, and saves intermediate results to checkpoint files.

        :param where: MongoDB query filter.
        :param project: Fields to project in the MongoDB query.
        :param method: Processing method, either "full" or "day/week".
        :param chunk: Tuple containing start and end IDs for the chunk.
        :param batch_size: Number of documents to process in each batch.
        :param checkpoint_interval: Interval at which to save checkpoints.
        :param process_id: Unique identifier for the worker process.

        :return: A message indicating the completion of the worker process.
        """
        client = self.connect()

        db = self.get_db()
        c = db[self.get_collection()]

        start_id, end_id = chunk

        # Create the query filter for the chunk
        d = {"created_at": {"$gte": start_id, "$lt": end_id}}
        where_f = {'$and': [where, d]}

        if method != "full":
            cursor = c.find(where_f, project).sort('created_at', ASCENDING).batch_size(batch_size)
            
            # Use more generic variable names
            current_bucket_id = None
            bucket_docs = []
            intermediate_map = set()
            intermediate_result = {}
            tz_rome = ZoneInfo("Europe/Rome")
            for i, document in enumerate(cursor, 1):
                created_at_rome = document["created_at"].astimezone(tz_rome)
                
                if method == "day":
                    # Truncate to the beginning of the day
                    bucket_start_dt = created_at_rome.replace(hour=0, minute=0, second=0, microsecond=0)
                elif method == "week":
                    # Calculate the start of the week (Monday)
                    start_of_week_offset = created_at_rome.weekday()  # Monday is 0, Sunday is 6
                    bucket_start_dt = (created_at_rome - timedelta(days=start_of_week_offset)).replace(hour=0, minute=0, second=0, microsecond=0)
                else:
                    # Handle unsupported methods
                    raise ValueError(f"Unsupported aggregation method: '{method}'. Choose 'day' or 'week'.")
                
                bucket_id = bucket_start_dt.timestamp()

                if current_bucket_id is None:
                    current_bucket_id = bucket_id

                if bucket_id != current_bucket_id:
                    # Create and process the bucket for the previous period (day or week)
                    bucket = {"_id": current_bucket_id, "docs": bucket_docs}
                    edges, maps = self.process_bucket_document(bucket)

                    for item in edges:
                        key = (item[0], item[1], item[2], item[-1])
                        if item[-1] != 1:
                            if key not in intermediate_result:
                                intermediate_result[key] = 0
                            intermediate_result[key] += item[3]
                        else:
                            intermediate_result[key] = item[3:-1]
                    intermediate_map.update(maps)

                    # Save checkpoint
                    if i % checkpoint_interval == 0:
                        self.save_bucket_checkpoint(intermediate_result, intermediate_map, process_id)
                        intermediate_result = {}
                        intermediate_map = set()
                    
                    # Start a new bucket
                    bucket_docs = []
                    current_bucket_id = bucket_id

                bucket_docs.append(document)

            # Flush the final bucket after the loop
            if bucket_docs:
                bucket = {"_id": current_bucket_id, "docs": bucket_docs}
                edges, maps = self.process_bucket_document(bucket)

                for item in edges:
                    key = (item[0], item[1], item[2], item[-1])
                    if item[-1] != 1:
                        if key not in intermediate_result:
                            intermediate_result[key] = 0
                        intermediate_result[key] += item[3]
                    else:
                        intermediate_result[key] = item[2:-1]
                
                intermediate_map.update(maps)
            
            if intermediate_result:
                self.save_bucket_checkpoint(intermediate_result, intermediate_map, process_id)

        elif method == "full":
            # Process the full data without chunking by day/week
            # Create the query filter for the full data
            cursor = c.find(where_f, project).sort('created_at', ASCENDING).batch_size(batch_size)

            intermediate_result = {}
            intermediate_map = set()
            for i, document in enumerate(cursor, 1):
                edges, maps = self.process_document(document)

                for item in edges:
                    key = (item[0], item[1], item[-1])
                    if item[-1] != 1:
                        if key not in intermediate_result:
                            intermediate_result[key] = 0
                        intermediate_result[key] += item[2]  # Sum the third element
                    else:
                        intermediate_result[key] = item[2:-1]

                intermediate_map.update(maps)

                # Save checkpoint after every `checkpoint_interval` documents
                if i % checkpoint_interval == 0:
                    self.save_checkpoint(intermediate_result, intermediate_map, process_id)
                    intermediate_result = {}
                    intermediate_map = set()
            # Final save for any remaining results
            if intermediate_result:
                self.save_checkpoint(intermediate_result, intermediate_map, process_id)
        
        self.logger.info(f"[Worker {process_id}] Final checkpoint saved")
        client.close()
        return f"[Worker {process_id}] Done."

    def query_data_in_chunks(self, where, project, method="full", batch_size=100000, checkpoint_interval=10000):
        """
        Queries data from MongoDB in chunks based on the `created_at` field and processes it using multiple workers.
        This method divides the data into chunks based on the `created_at` field and processes each chunk in parallel using a pool of workers.
        Each worker processes a chunk of data, saves intermediate results to checkpoint files, and merges the results at the end.

        :param where: MongoDB query filter.
        :param project: Fields to project in the MongoDB query.
        :param method: Processing method, either "full" or "day/week".
        :param batch_size: Number of documents to process in each batch.
        :param checkpoint_interval: Interval at which to save checkpoints.

        :return: None
        """
        # Generate chunks based on the collection's create_at field
        # Ensure start_date and end_date are timezone-aware
        delta = timedelta(days=2)
        chunks = list(self.generate_date_chunks(self.start_date, self.end_date, delta))

        Writer.create_dirs(self.output_file_path, self.id)

        max_workers = min(4, len(chunks)) # Limit the number of workers to 30 or the number of chunks, whichever is smaller
        futures = []

        self.logger.info(f"Processing {len(chunks)} chunks with {max_workers} workers.")

        # Use ProcessPoolExecutor to process chunks in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i, chunk in enumerate(chunks):
                future = executor.submit(
                    self.worker_process, where, project, method, chunk, batch_size, checkpoint_interval, i)
                futures.append(future)
            for future in as_completed(futures):
                try:
                    self.logger.info(f"[Worker] {future.result()}")
                except Exception as e:
                    self.logger.error(f"[Worker] A worker failed: {e}")
    
        self.logger.info("All chunks processed. Merging and aggregating results.")
        self.merge_and_aggregate_checkpoints(method)
        self.logger.info("All data processed and intermediate results saved in checkpoint files.")
