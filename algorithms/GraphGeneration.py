import logging
import multiprocessing
import os
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from datetime import timedelta, timezone, datetime
from enum import Enum
from zoneinfo import ZoneInfo
from dateutil import parser
import json
import shutil

from pymongo import ASCENDING

from Utils.Const import Const as c
from Utils.Utils import Utils
from Utils.Writer import Writer, _ext, SUPPORTED_FORMATS
from algorithms.MongoConnection import MongoConnection


class GraphType(Enum):
    retweet               = 0
    tweet_retweet         = 1
    user_hashtag          = 2
    hashtag_cooccurrences = 3
    response              = 4
    mention               = 5


class MapType(Enum):
    user_id           = 0
    user_retweeted_id = 1
    tweet_id          = 2
    hashtag           = 3


class GraphGeneration(MongoConnection):
    logger = logging.getLogger('GraphGeneration')

    def __init__(self, uri, username=None, password=None,
                 auth_source=None, auth_mechanism=None,
                 database_name=None, db=None, collection=None,
                 start_date=None, end_date=None, method="full",
                 input_type="mongo", output_file_path=None,
                 retweet=False, tweet_retweet=False,
                 user_hashtag=False, hashtag_cooccurrences=False,
                 response=False, mention=False,
                 file_format="csv"):
        if file_format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file_format '{file_format}'. "
                             f"Choose from {SUPPORTED_FORMATS}.")
        super().__init__(uri, username, password, auth_source, auth_mechanism,
                         db, database_name, collection, start_date, end_date)
        self.checkpoint_folder      = "tmp"
        self.sep                    = "_"
        self.type                   = input_type
        self.output_file_path       = output_file_path
        self.id                     = uuid.uuid1().hex
        self.retweet                = retweet
        self.tweet_retweet          = tweet_retweet
        self.user_hashtag           = user_hashtag
        self.hashtag_cooccurrences  = hashtag_cooccurrences
        self.response               = response
        self.mention                = mention
        self.file_format            = file_format
        self.w                      = Writer()

    def process_document(self, d):
        o, m = [], set()
        n_user_id = Utils.hash(d['user']['id'])
        m.add((d['user']['id'], n_user_id, 0))
        weight = 1

        if d.get('retweeted_status') is not None and self.retweet:
            n_rt = Utils.hash(d['retweeted_status']['user']['id'])
            o.append((n_user_id, n_rt, weight, 0))
            m.add((d['retweeted_status']['user']['id'], n_rt, 1))

        if d.get('retweeted_status') is not None and self.tweet_retweet:
            n_t  = Utils.hash(d['id'])
            n_rt = Utils.hash(d['retweeted_status']['id'])
            ts_t  = Utils.to_datetime(d["created_at"]).timestamp()
            created_at_rt = d['retweeted_status'].get('created_at')
            if not created_at_rt:
                created_at_rt = d['created_at']
            ts_rt = Utils.to_datetime(created_at_rt).timestamp()
            o.append((n_t, n_rt, weight, (ts_t, ts_rt), 1))
            m.add((d['id'],                         n_t,  2))
            m.add((d['retweeted_status']['id'],     n_rt, 2))

        if d.get('hashtagEntities') is not None and self.user_hashtag:
            for ht in (d['hashtagEntities'].lower().split('|')
                       if isinstance(d['hashtagEntities'], str) else []):
                h = Utils.compute_hash(ht)
                o.append((n_user_id, h, weight, 2))
                m.add((ht, h, 3))

        if d.get('hashtagEntities') is not None and self.hashtag_cooccurrences:
            pairs = (Utils.combinations_list(d['hashtagEntities'].lower().split('|'))
                     if isinstance(d['hashtagEntities'], str) else [])
            for a, b in pairs:
                o.append((a, b, weight, 3))
                o.append((b, a, weight, 3))

        if d.get('in_reply_to_user_id', -1) != -1 and self.response:
            n_rep = Utils.hash(d['in_reply_to_user_id'])
            o.append((n_user_id, n_rep, weight, 4))

        if d.get('userMentionEntities') is not None and self.mention:
            for mn in (d['userMentionEntities'].lower().split('|')
                       if isinstance(d['userMentionEntities'], str) else []):
                o.append((n_user_id, Utils.compute_hash(mn), weight, 5))

        return o, m

    def process_bucket_document(self, d):
        o, m   = [], set()
        date   = d["_id"]
        tz_rome = ZoneInfo("Europe/Rome")
        weight = 1

        for tweet in d['docs']:
            n_uid = Utils.hash(tweet['user']['id'])
            m.add((tweet['user']['id'], n_uid, 0))

            if tweet.get('retweeted_status') is not None and self.retweet:
                n_rt = Utils.hash(tweet['retweeted_status']['user']['id'])
                o.append((n_uid, n_rt, date, weight, 0))
                m.add((tweet['retweeted_status']['user']['id'], n_rt, 1))

            if tweet.get('retweeted_status') is not None and self.tweet_retweet:
                n_t  = Utils.hash(tweet['id'])
                n_rt = Utils.hash(tweet['retweeted_status']['id'])
                ts_t  = Utils.to_datetime(tweet['created_at']).astimezone(tz_rome).timestamp()
                created_at_rt = tweet['retweeted_status'].get('created_at')
                if not created_at_rt:
                    created_at_rt = tweet['created_at']
                ts_rt = Utils.to_datetime(created_at_rt).astimezone(tz_rome).timestamp()
                o.append((n_t, n_rt, date, weight, (ts_t, ts_rt), 1))
                m.add((tweet['id'],                         n_t,  2))
                m.add((tweet['retweeted_status']['id'],     n_rt, 2))

            if tweet.get('hashtagEntities') is not None and self.user_hashtag:
                for ht in (tweet['hashtagEntities'].lower().split('|')
                           if isinstance(tweet['hashtagEntities'], str) else []):
                    h = Utils.compute_hash(ht)
                    o.append((n_uid, h, date, weight, 2))
                    m.add((ht, h, 3))

            if tweet.get('hashtagEntities') is not None and self.hashtag_cooccurrences:
                pairs = (Utils.combinations_list(tweet['hashtagEntities'].lower().split('|'))
                         if isinstance(tweet['hashtagEntities'], str) else [])
                for a, b in pairs:
                    o.append((a, b, date, weight, 3))
                    o.append((b, a, date, weight, 3))

            if tweet.get('in_reply_to_user_id', -1) != -1 and self.response:
                n_rep = Utils.hash(tweet['in_reply_to_user_id'])
                o.append((n_uid, n_rep, date, weight, 4))

            if tweet.get('userMentionEntities') is not None and self.mention:
                for mn in (tweet['userMentionEntities'].lower().split('|')
                           if isinstance(tweet['userMentionEntities'], str) else []):
                    o.append((n_uid, Utils.compute_hash(mn), date, weight, 5))

        return o, m

    def merge_and_aggregate_checkpoints(self, method="full"):
        checkpoint_dir = os.sep.join([self.output_file_path, self.checkpoint_folder, self.id])
        ext = _ext(self.file_format)

        for graph_type in GraphType:
            pattern = os.sep.join([checkpoint_dir, graph_type.name, "*"]) + ext
            list_checkpoint_files = Writer.list_checkpoint_files(pattern)
            merged_file_path = os.sep.join([self.output_file_path, self.id, graph_type.name])

            aggregated = defaultdict(int)
            if method == "full":
                for file_path in list_checkpoint_files:
                    checkpoint_data = Writer.load_checkpoint_file(file_path)
                    if graph_type.name != GraphType.tweet_retweet.name:
                        for row in checkpoint_data:
                            key = (int(row[0]), int(row[1]), int(row[2]))
                            aggregated[key] += int(float(row[3]))
                    else:
                        for row in checkpoint_data:
                            key = (int(row[0]), int(row[1]), int(row[2]))
                            # row[3] is date_tweet, row[4] is date_rt
                            aggregated[key] = (float(row[3]), float(row[4]))
                
                final = []
                for k, v in aggregated.items():
                    if k[0] != 1:
                        final.append((k[0], k[1], k[2], v))
                    else:
                        final.append((k[0], k[1], k[2], v[0], v[1]))
                cols = ["relationship", "src", "dst", "weight"] if graph_type.name != GraphType.tweet_retweet.name else ["relationship", "src", "dst", "date_tweet", "date_rt"]
                Writer.write_data(merged_file_path, final, columns=cols, file_format=self.file_format)

            else: # day/week
                aggregated = defaultdict(int)
                for file_path in list_checkpoint_files:
                    checkpoint_data = Writer.load_checkpoint_file(file_path)
                    if graph_type.name != GraphType.tweet_retweet.name:
                        for row in checkpoint_data:
                            key = (int(row[0]), int(row[1]), int(row[2]), float(row[3]))
                            aggregated[key] += int(float(row[4]))
                    else:
                        for row in checkpoint_data:
                            key = (int(row[0]), int(row[1]), int(row[2]), float(row[3]))
                            # row[4] is date_tweet, row[5] is date_rt
                            aggregated[key] = (float(row[4]), float(row[5]))
                final = []
                for k, v in aggregated.items():
                    if k[0] != 1:
                        final.append((k[0], k[1], k[2], k[3], v))
                    else:
                        final.append((k[0], k[1], k[2], k[3], v[0], v[1]))
                cols = ["relationship", "src", "dst", "date", "weight"] if graph_type.name != GraphType.tweet_retweet.name else ["relationship", "src", "dst", "date", "date_tweet", "date_rt"]
                Writer.write_data(merged_file_path, final, columns=cols, file_format=self.file_format)

        for map_type in MapType:
            pattern = os.sep.join([checkpoint_dir, map_type.name, c.MAP, "*"]) + ext
            list_map_files = Writer.list_checkpoint_files(pattern)
            merged_file_path = os.sep.join([self.output_file_path, str(self.id), map_type.name])
            all_data = []
            for file_path in list_map_files:
                all_data.extend(Writer.load_checkpoint_file(file_path))
            unique_data = list({tuple(row) for row in all_data})
            cols = ["original_id", "hash_id", "map_type"]
            Writer.write_data(merged_file_path, unique_data, columns=cols, file_format=self.file_format)

    def _checkpoint_base(self, subpath):
        return os.sep.join([self.output_file_path, subpath])

    def save_checkpoint(self, intermediate_results, intermediate_map, process_id):
        result_graph = {g.name: [] for g in GraphType}
        result_map = {m.name: [] for m in MapType}

        for k, v in intermediate_results.items():
            if k[2] != 1:
                result_graph[GraphType(k[2]).name].append((k[2], k[0], k[1], v))
            else:
                # v is (weight, (ts_t, ts_rt))
                result_graph[GraphType(k[2]).name].append((k[2], k[0], k[1], v[1][0], v[1][1]))

        for e in intermediate_map:
            result_map[MapType(e[2]).name].append(e)

        for gt_name, rows in result_graph.items():
            cols = (["relationship", "src", "dst", "weight"]
                    if gt_name != GraphType.tweet_retweet.name
                    else ["relationship", "src", "dst", "date_tweet", "date_rt"])
            path = self._checkpoint_base(os.sep.join([self.checkpoint_folder, str(self.id), gt_name, str(process_id)]))
            
            # --- CORRECTION ICI ---
            os.makedirs(os.path.dirname(path), exist_ok=True)
            Writer.write_data(path, rows, columns=cols, file_format=self.file_format)

        for mt_name, rows in result_map.items():
            cols = ["original_id", "hash_id", "map_type"]
            path = self._checkpoint_base(os.sep.join([self.checkpoint_folder, str(self.id), mt_name, c.MAP, str(process_id)]))
            
            # --- CORRECTION ICI ---
            os.makedirs(os.path.dirname(path), exist_ok=True)
            Writer.write_data(path, rows, columns=cols, file_format=self.file_format)

    def save_bucket_checkpoint(self, intermediate_results, intermediate_map, process_id):
        result_graph = {g.name: [] for g in GraphType}
        result_map = {m.name: [] for m in MapType}

        for k, v in intermediate_results.items():
            graph_type_val = k[3]
            date_val = k[2]
            gt_name = GraphType(graph_type_val).name

            if graph_type_val != 1:
                result_graph[gt_name].append((graph_type_val, k[0], k[1], date_val, v))
            else:
                # v is (ts_t, ts_rt)
                result_graph[gt_name].append((graph_type_val, k[0], k[1], date_val, v[0], v[1]))

        for e in intermediate_map:
            result_map[MapType(e[2]).name].append(e)

        for gt_name, rows in result_graph.items():
            cols = (["relationship", "src", "dst", "date", "weight"]
                    if gt_name != GraphType.tweet_retweet.name
                    else ["relationship", "src", "dst", "date", "date_tweet", "date_rt"])
            path = self._checkpoint_base(os.sep.join([self.checkpoint_folder, str(self.id), gt_name, str(process_id)]))
            
            # --- CORRECTION ICI ---
            os.makedirs(os.path.dirname(path), exist_ok=True)
            Writer.write_data(path, rows, columns=cols, file_format=self.file_format)

        for mt_name, rows in result_map.items():
            cols = ["original_id", "hash_id", "map_type"]
            path = self._checkpoint_base(os.sep.join([self.checkpoint_folder, str(self.id), mt_name, c.MAP, str(process_id)]))
            
            # --- CORRECTION ICI ---
            os.makedirs(os.path.dirname(path), exist_ok=True)
            Writer.write_data(path, rows, columns=cols, file_format=self.file_format)

    def generate_date_chunks(self, start_date, end_date, delta):
        current_date = start_date
        while current_date < end_date:
            next_date = min(current_date + delta, end_date)
            yield current_date, next_date
            current_date = next_date

    def worker_process(self, where, project, method, chunk,
                       batch_size, checkpoint_interval, process_id):
        client = self.connect()
        col    = self.get_db()[self.get_collection()]
        start_id, end_id = chunk
        date_filter = {"created_at": {"$gte": start_id, "$lt": end_id}}
        where_f = {'$and': [where, date_filter]}

        if method != "full":
            cursor = (col.find(where_f, project)
                        .sort('created_at', ASCENDING)
                        .batch_size(batch_size))
            tz_rome = ZoneInfo("Europe/Rome")
            current_bucket_id = None
            bucket_docs       = []
            intermediate_map  = set()
            intermediate_result = {}

            for i, doc in enumerate(cursor, 1):
                created_rome =  Utils.to_datetime(doc["created_at"]).astimezone(tz_rome)
                if method == "day":
                    bucket_dt = created_rome.replace(hour=0, minute=0, second=0, microsecond=0)
                elif method == "week":
                    offset    = created_rome.weekday()
                    bucket_dt = (created_rome - timedelta(days=offset)).replace(hour=0, minute=0, second=0, microsecond=0)
                else:
                    raise ValueError(f"Unsupported method '{method}'.")

                bucket_id = bucket_dt.timestamp()
                if current_bucket_id is None:
                    current_bucket_id = bucket_id

                if bucket_id != current_bucket_id:
                    edges, maps = self.process_bucket_document({"_id": current_bucket_id, "docs": bucket_docs})
                    for item in edges:
                        key = (item[0], item[1], item[2], item[-1])
                        if item[-1] != 1:
                            intermediate_result[key] = intermediate_result.get(key, 0) + item[3]
                        else:
                            intermediate_result[key] = item[4]
                    intermediate_map.update(maps)
                    if i % checkpoint_interval == 0:
                        self.save_bucket_checkpoint(intermediate_result, intermediate_map, process_id)
                        intermediate_result = {}
                        intermediate_map    = set()
                    bucket_docs       = []
                    current_bucket_id = bucket_id
                bucket_docs.append(doc)

            if bucket_docs:
                edges, maps = self.process_bucket_document({"_id": current_bucket_id, "docs": bucket_docs})
                for item in edges:
                    key = (item[0], item[1], item[2], item[-1])
                    if item[-1] != 1:
                        intermediate_result[key] = intermediate_result.get(key, 0) + item[3]
                    else:
                        intermediate_result[key] = item[4]
                intermediate_map.update(maps)

            if intermediate_result:
                self.save_bucket_checkpoint(intermediate_result, intermediate_map, process_id)

        else:  # method == "full"
            cursor = (col.find(where_f, project).sort('created_at', ASCENDING).batch_size(batch_size))
            intermediate_result = {}
            intermediate_map    = set()
            for i, doc in enumerate(cursor, 1):
                edges, maps = self.process_document(doc)
                for item in edges:
                    key = (item[0], item[1], item[-1])
                    if item[-1] != 1:
                        intermediate_result[key] = intermediate_result.get(key, 0) + item[2]
                    else:
                        intermediate_result[key] = item[3:5] # (weight, (ts_t, ts_rt))
                intermediate_map.update(maps)
                if i % checkpoint_interval == 0:
                    self.save_checkpoint(intermediate_result, intermediate_map, process_id)
                    intermediate_result = {}
                    intermediate_map    = set()
            if intermediate_result:
                self.save_checkpoint(intermediate_result, intermediate_map, process_id)

        self.logger.info(f"[Worker {process_id}] Done.")
        client.close()
        return f"[Worker {process_id}] Done."

    def query_data_in_chunks(self, where, project, method: str = "full",
                             batch_size: int = 100_000,
                             checkpoint_interval: int = 10_000):
        chunks      = list(self.generate_date_chunks(self.start_date, self.end_date, timedelta(days=2)))
        max_workers = min(4, len(chunks))
        Writer.create_dirs(self.output_file_path, self.id)
        self.logger.info(
            f"Processing {len(chunks)} chunks with {max_workers} workers. Format: {self.file_format}"
            f"[GraphGeneration] Starting extraction. "
            f"Run ID: {self.id}  Format: {self.file_format}"
        )
        futures = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i, chunk in enumerate(chunks):
                futures.append(executor.submit(self.worker_process, where, project, method, chunk, batch_size, checkpoint_interval, i))
            for future in as_completed(futures):
                try:
                    self.logger.info(f"[Worker] {future.result()}")
                except Exception as e:
                    raise e

        self.logger.info("All chunks done. Merging...")
        self.merge_and_aggregate_checkpoints(method)
        self.logger.info("GraphGeneration complete.")

        self.logger.info(
            f"Merge complete. Output in {os.path.join(self.output_file_path, self.id)}\n"
        )
