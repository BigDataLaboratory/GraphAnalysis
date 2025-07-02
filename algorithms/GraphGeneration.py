import logging
import multiprocessing
import os
import uuid
from collections import defaultdict
from datetime import timedelta, timezone
from enum import Enum

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
    logger = logging.getLogger('GraphGeneration')

    def __init__(self, uri, username=None, password=None, auth_source=None, auth_mechanism=None, db=None,
                 collection=None, start_date=None, end_date=None, method="full", input_type="mongo", output_file_path=None, retweet=False, tweet_retweet=False,
                 user_hashtag=False, hashtag_cooccurrences=False, response=False, mention=False):
        super().__init__(uri, username, password, auth_source, auth_mechanism, db, collection, start_date, end_date)
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
        self.w = Writer()

    def process_document(self, d):
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
        o = []
        m = set()

        print(type(d["_id"]))
        date = d["_id"].replace(tzinfo=timezone.utc).timestamp()

        for tweet in d['docs']:
            n_user_id = Utils.hash(tweet['user']['id'])
            m.add((tweet['user']['id'], n_user_id, 0))

            if tweet.get('retweeted_status', None) is not None and self.retweet:
                relationship_u_rt = 0
                n_rt_user_id = Utils.hash(d['retweeted_status']['user']['id'])
                e_rt = n_user_id, n_rt_user_id, date, relationship_u_rt
                o.append(e_rt)
                m.add((d['retweeted_status']['user']['id'], n_rt_user_id, 1))

            if d.get('retweeted_status', None) is not None and self.tweet_retweet:
                relationship_t_rt = 1
                n_tweet_id = Utils.hash(tweet['id'])
                n_rt_tweet_id = Utils.hash(tweet['retweeted_status']['id'])
                a_created_at_tweet = tweet['created_at'].timestamp()
                a_created_at_rt = tweet['retweeted_status']['created_at'].timestamp()
                e_tweet_retweet = (
                    n_tweet_id, n_rt_tweet_id, date, (a_created_at_tweet, a_created_at_rt), relationship_t_rt)
                o.append(e_tweet_retweet)
                m.add((tweet['id'], n_tweet_id, 2))
                m.add((tweet['retweeted_status']['id'], n_rt_tweet_id, 2))

            if d.get('hashtagEntities', None) is not None and self.user_hashtag:
                relationship = 2
                n_ht = tweet['hashtagEntities'].lower().split('|') if isinstance(tweet['hashtagEntities'], str) else []
                ht = [(n_user_id, Utils.compute_hash(x), date, relationship) for x in n_ht]
                o.extend(ht)
                for x in n_ht:
                    m.add((x, Utils.compute_hash(x), 3))

            if d.get('hashtagEntities', None) is not None and self.hashtag_cooccurrences:
                relationship = 3
                ht_combinations = Utils.combinations_list(tweet['hashtagEntities'].lower().split('|')) if isinstance(
                    tweet['hashtagEntities'], str) else []
                e_hts_natural = [(x[0], x[1], date, relationship) for x in ht_combinations]
                e_hts_inverse = [(x[1], x[0], date, relationship) for x in ht_combinations]
                o.extend(e_hts_natural)
                o.extend(e_hts_inverse)

            if d.get('in_reply_to_user_id', -1) != -1 and self.response:
                relationship = 4
                n_reply_user_id = Utils.hash(tweet['in_reply_to_user_id'])
                e_reply = n_user_id, n_reply_user_id, date, relationship
                o.append(e_reply)

            if d.get('userMentionEntities', None) is not None and self.mention:
                relationship = 5
                n_mentions = tweet['userMentionEntities'].lower().split('|') if isinstance(tweet['userMentionEntities'],
                                                                                       str) else []
                e_mentions = [(n_user_id, Utils.compute_hash(x), date, relationship) for x in n_mentions]
                o.extend(e_mentions)

        return o, m


    def generate_date_chunks(self, start_date, end_date, delta):
        """
        Generates date ranges to divide the data into chunks based on the `created_at` field.
        """
        current_date = start_date
        while current_date < end_date:
            next_date = current_date + delta
            yield current_date, next_date
            current_date = next_date

    def merge_and_aggregate_checkpoints(self, method="full"):
        """
        Merges checkpoint files from a folder, aggregates by the first two elements,
        and writes the final result to a single output file. Deletes each checkpoint file after processing.
        """
        checkpoint_dir = os.sep.join([self.output_file_path, self.checkpoint_folder, self.id])
        # Iterate over all checkpoint files in the folder
        for graph_type in GraphType:
            checkpoint_files = self.sep.join([self.checkpoint_folder, self.id, graph_type.name, "*"])
            list_checkpoint_files = Writer.list_checkpoint_files(os.sep.join([checkpoint_dir, checkpoint_files]))
            merged_file_path = os.sep.join([self.output_file_path, self.id, graph_type.name])

            if method != "full":
                for file_path in list_checkpoint_files:
                    checkpoint_data = Writer.load_checkpoint_file(file_path)
                    Writer.write_on_csv(merged_file_path, checkpoint_data)
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

                Writer.write_on_csv(merged_file_path, final_result_graph)

        # Iterate over all checkpoint files in the folder
        for map_type in MapType:
            checkpoint_files = self.sep.join([self.checkpoint_folder, self.id, map_type.name, c.MAP, "*"])
            list_map_files = Writer.list_checkpoint_files(os.sep.join([checkpoint_dir, checkpoint_files]))
            merged_file_path = os.sep.join([self.output_file_path, str(self.id), map_type.name])
            for file_path in list_map_files:
                checkpoint_data = Writer.load_checkpoint_file(file_path)
                Writer.write_on_csv(merged_file_path, checkpoint_data)

    def save_checkpoint(self, intermediate_results, intermediate_map, process_id):
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
            file_path = self.sep.join([self.checkpoint_folder, str(self.id), str(k), str(process_id)])
            path = os.sep.join([dir_path, file_path])
            Writer.write_on_csv(path, result_graph[k])
        for k in result_map:
            file_path = self.sep.join([self.checkpoint_folder, str(self.id), str(k), c.MAP, str(process_id)])
            path = os.sep.join([dir_path, file_path])
            Writer.write_on_csv(path, result_map[k])

    def save_bucket_checkpoint(self, intermediate_results, intermediate_map, process_id):
        result_graph = {GraphType(0).name: [], GraphType(1).name: [], GraphType(2).name: [], GraphType(3).name: [],
                        GraphType(4).name: [], GraphType(5).name: []}
        result_map = {MapType(0).name: [], MapType(1).name: [], MapType(2).name: [], MapType(3).name: []}

        for e in intermediate_results:
            result_graph[GraphType(e[-1]).name].append(e[-1], e[0], e[1], e[2]) if e[-1] != 1 else result_graph[GraphType(e[-1]).name].append(e[-1], e[0], e[1], e[2], e[3], e[4])

        for e in intermediate_map:
            result_map[MapType(e[2]).name].append(e)

        dir_path = os.sep.join([self.output_file_path, self.checkpoint_folder, str(self.id)])
        for k in result_graph:
            file_path = self.sep.join([self.checkpoint_folder, str(self.id), str(k), str(process_id)])
            path = os.sep.join([dir_path, file_path])
            Writer.write_on_csv(path, result_graph[k])
        for k in result_map:
            file_path = self.sep.join([self.checkpoint_folder, str(self.id), str(k), c.MAP, str(process_id)])
            path = os.sep.join([dir_path, file_path])
            Writer.write_on_csv(path, result_map[k])

    def worker_process(self, where, project, method, chunk, batch_size, checkpoint_interval, process_id):
        """
        Worker function to process a chunk of data from MongoDB.
        """
        db = self.get_db()
        c = db[self.get_collection()]

        start_id, end_id = chunk

        # Create the query filter for the chunk
        d = {"created_at": {"$gte": start_id, "$lt": end_id}}
        where_f = {'$and': [where, d]}

        if method == "old":
            pipeline = [
                {"$match": where_f},
                {"$project": project},
                {"$group": {
                    "_id": {
                        "$dateTrunc": {
                            "date": "$created_at",
                            "unit": method,
                            "timezone": "Europe/Rome"
                        }
                    },
                    "docs": {"$push": "$$ROOT"}
                }},
                {"$sort": {"_id": 1}}
            ]

            cursor = c.aggregate(pipeline, allowDiskUse=True, batchSize=batch_size)
            intermediate_map = set()
            intermediate_result = {}

            for i, day_bucket in enumerate(cursor, 1):
                edges, maps = self.process_bucket_document(day_bucket)
                intermediate_result = edges
                for item in maps:
                    intermediate_map.add(item)
                # Save checkpoint after every `checkpoint_interval` documents
                if i % checkpoint_interval == 0:
                    self.save_bucket_checkpoint(intermediate_result, intermediate_map, process_id)
                    intermediate_result = {}
                    intermediate_map = set()
            if intermediate_result:
                self.save_bucket_checkpoint(intermediate_result, intermediate_map, process_id)
        elif method != "full":
            cursor = c.find(where_f, project).sort('created_at', ASCENDING).batch_size(batch_size)
            current_day = None
            bucket_docs = []
            intermediate_map = set()
            intermediate_result = {}
            for i, document in enumerate(cursor, 1):
                day = document["created_at"].astimezone(pytz.timezone("Europe/Rome")).date()
                if current_day is None:
                    current_day = day
                if day != current_day:
                    day_bucket = {"_id": current_day, "docs": bucket_docs}
                    # flush yesterday
                    edges, maps = self.process_bucket_document(day_bucket)
                    intermediate_result = edges
                    intermediate_map.update(maps)

                    # Save checkpoint after every `checkpoint_interval` documents
                    if i % checkpoint_interval == 0:
                        self.save_bucket_checkpoint(intermediate_result, intermediate_map, process_id)
                        intermediate_result = {}
                        intermediate_map = set()
                    bucket_docs = []  # start new bucket
                    current_day = day

                bucket_docs.append(document)

            # flush the last day
            if bucket_docs:
                day_bucket = {"_id": current_day, "docs": bucket_docs}
                edges, maps = self.process_bucket_document(day_bucket)
                intermediate_result = edges
                intermediate_map.update(maps)
            if intermediate_result:
                self.save_bucket_checkpoint(intermediate_result, intermediate_map, process_id)

        elif method == "full":
            # Retrieve documents in batches
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

                for item in maps:
                    intermediate_map.add(item)

                # Save checkpoint after every `checkpoint_interval` documents
                if i % checkpoint_interval == 0:
                    self.save_checkpoint(intermediate_result, intermediate_map, process_id)
                    intermediate_result = {}
                    intermediate_map = set()
            # Final save for any remaining results
            if intermediate_result:
                self.save_checkpoint(intermediate_result, intermediate_map, process_id)

    def query_data_in_chunks(self, where, project, method="full", batch_size=50000, checkpoint_interval=40000):
        """
        Distribute MongoDB query processing across multiple processes using chunked processing.
        """
        # Generate chunks based on the collection's create_at field
        delta = timedelta(weeks=1)
        chunks = list(self.generate_date_chunks(self.start_date, self.end_date, delta))
        processes = []

        Writer.create_dirs(self.output_file_path, self.id)

        # Define checkpoint file per worker
        for i, chunk in enumerate(chunks):
            process = multiprocessing.Process(target=self.worker_process,
                                              args=(where, project, method, chunk, batch_size, checkpoint_interval, i))
            processes.append(process)
            process.start()

        # Wait for all worker processes to complete
        for process in processes:
            process.join()

        self.merge_and_aggregate_checkpoints(method)
        print("All data processed and intermediate results saved in checkpoint files.")

    """
    def query(self, where=None, project=None, batch_size=100):
        if self.type == c.MONGO:
            for batch in self.collection.find(where, project, batch_size=batch_size):
                return pd.json_normalize(batch)
        elif self.type == c.JSON:
            with open(self.uri, 'r') as file:
                result = json.load(file)
            result = pd.json_normalize(result)
            result = result[project] if project is not None and len(project) > 0 else result
            result = result.query(where) if where is not None else result
            return result
        elif self.type == c.CSV:
            result = pd.read_csv(self.uri, sep=",", header=0, lineterminator='\n', usecols=project)
            result = result.query(where) if where is not None else result
            return result
    """
