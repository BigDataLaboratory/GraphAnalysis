import json

import pandas as pd
import pymongo.errors
from pymongo import MongoClient
from pymongo import ASCENDING
import multiprocessing
from datetime import datetime, timedelta
from Utils.Const import Const as c
import os
import numpy as np
import csv
import uuid
import logging

from Utils.Utils import Utils

logger = logging.getLogger('GraphGeneration')


class GraphGeneration:
    checkpoint_folder = "checkpoint_tmp_"

    def __init__(self, uri, username=None, password=None, auth_source=None, auth_mechanism=None, db=None,
                 collection=None, start_date=None, end_date=None, input_type="mongo", output_file_path=None,
                 retweet=False, tweet_retweet=False, user_hashtag=False, hashtag_cooccurrences=False, response=False,
                 mention=False):
        self.uri = uri
        self.username = username
        self.password = password
        self.auth_source = auth_source
        self.auth_mechanism = auth_mechanism
        self.db = db
        self.collection = collection
        self.start_date = start_date
        self.end_date = end_date
        self.type = input_type
        self.output_file_path = output_file_path
        self.id = uuid.uuid1().hex
        self.retweet = retweet
        self.tweet_retweet = tweet_retweet
        self.user_hashtag = user_hashtag
        self.hashtag_cooccurrences = hashtag_cooccurrences
        self.response = response
        self.mention = mention

    def connect(self, database_name: str = None):
        if self.db is None and database_name is not None and self.type == c.MONGO:
            mongo_client = MongoClient(self.uri,
                                       username=self.username,
                                       password=self.password,
                                       authSource=self.auth_source,
                                       authMechanism=self.auth_mechanism)
            try:
                self.db = mongo_client[database_name]
            except pymongo.errors.ConnectionFailure as e:
                print(e)
        return self.get_db()

    def get_collection(self):
        return self.collection

    def set_collection(self, collection_name):
        if collection_name != self.collection:
            self.collection = collection_name

    def get_db(self):
        if self.db is not None:
            return self.db
        else:
            return None

    def process_document(self, d):
        o = []
        m = {}
        n_user_id = Utils.hash(d['user']['id'])
        m[(n_user_id, 0)] = d['user']['id']
        weight = 1

        if d.get('retweeted_status', None) is not None and self.retweet:
            relationship_u_rt = 0
            n_rt_user_id = Utils.hash(d['retweeted_status']['user']['id'])
            e_rt = n_user_id, n_rt_user_id, weight, relationship_u_rt
            o.append(e_rt)

            m[(n_rt_user_id, 1)] = d['retweeted_status']['user']['id']

        if d.get('retweeted_status', None) is not None and self.tweet_retweet:
            relationship_t_rt = 1
            n_tweet_id = Utils.hash(d['id'])
            n_rt_tweet_id = Utils.hash(d['retweeted_status']['id'])
            a_created_at_tweet = d['created_at'].timestamp()
            a_created_at_rt = d['retweeted_status']['created_at'].timestamp()
            e_tweet_retweet = (
                n_tweet_id, n_rt_tweet_id, weight, (a_created_at_tweet, a_created_at_rt), relationship_t_rt)
            o.append(e_tweet_retweet)

            m[(n_tweet_id, 2)] = d['id']
            m[(n_tweet_id, 2)] = d['retweeted_status']['id']

        if d.get('hashtagEntities', None) is not None and self.user_hashtag:
            relationship = 2
            n_ht = d['hashtagEntities'].lower().split('|') if isinstance(d['hashtagEntities'], str) else []
            ht = [(n_user_id, Utils.compute_hash(x), weight, relationship) for x in n_ht]
            ht = [(n_user_id, Utils.compute_hash(x), weight, relationship) for x in n_ht]
            o.extend(ht)

            for x in n_ht:
                m[(Utils.compute_hash(x), 3)] = x

        if d.get('hashtagEntities', None) is not None and self.hashtag_cooccurrences:
            relationship = 3
            ht_combinations = Utils.combinations_list(d['hashtagEntities'].lower().split('|')) if isinstance(
                d['hashtagEntities'], str) else []
            e_hts = [(x[0], x[1], weight, relationship) for x in ht_combinations]
            o.extend(e_hts)

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

    def generate_date_chunks(self, start_date, end_date, delta):
        """
        Generates date ranges to divide the data into chunks based on the `created_at` field.
        """
        current_date = start_date
        while current_date < end_date:
            next_date = current_date + delta
            yield current_date, next_date
            current_date = next_date

    """
    def worker_process(self, where, project, chunk, batch_size, process_id, shared_list):
        """"""
        #Worker function to process a chunk of data from MongoDB.
        """"""
        db = self.get_db()
        c = db[self.get_collection()]

        start_id, end_id = chunk

        # Create the query filter for the chunk
        d = {"created_at": {"$gte": start_id, "$lt": end_id}}
        where_f = {'$and': [where, d]}

        # Retrieve documents in batches
        cursor = c.find(where_f, project).sort('created_at', ASCENDING).batch_size(batch_size)

        for document in cursor:
            # Process the document here (you can modify this to suit your needs)
            edges = self.process_document(document)
            for item in edges:
                key = (item[0], item[1], item[3])  # key = (first, second, fourth)
                if key not in shared_list:
                    shared_list[key] = 0
                shared_list[key] += item[2]  # Sum the third element
    """

    def save_checkpoint(self, intermediate_results, intermediate_map, process_id):
        """
        Save intermediate results to a checkpoint file.
        """
        result = []
        for k, v in intermediate_results.items():
            if k[2] != 1:
                result.append((k[2], k[0], k[1], v))
            else:
                result.append((k[2], k[0], k[1], v[0], v[1]))
        with open("{}/{}{}_{}".format(self.output_file_path + self.checkpoint_folder + self.id, self.checkpoint_folder,
                                      self.id, process_id), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(result)
        # Convert and write JSON object to file
        with open("{}/{}{}_{}_{}".format(self.output_file_path + self.checkpoint_folder + self.id, self.checkpoint_folder,
                                      self.id, c.MAP, process_id), "a", newline='\n') as outfile:
            json.dump(intermediate_map, outfile)

    def worker_process(self, where, project, chunk, batch_size, checkpoint_interval, process_id):
        """
        Worker function to process a chunk of data from MongoDB.
        """
        db = self.get_db()
        c = db[self.get_collection()]

        start_id, end_id = chunk

        # Create the query filter for the chunk
        d = {"created_at": {"$gte": start_id, "$lt": end_id}}
        where_f = {'$and': [where, d]}

        # Retrieve documents in batches
        cursor = c.find(where_f, project).sort('created_at', ASCENDING).limit(100000).batch_size(batch_size)

        intermediate_result = {}
        intermediate_map = {}
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
            for key, value in maps.items():
                if key not in intermediate_map:
                    intermediate_map = value
            # Save checkpoint after every `checkpoint_interval` documents
            if i % checkpoint_interval == 0:
                self.save_checkpoint(intermediate_result, intermediate_map, process_id)
                intermediate_result = {}
                intermediate_map = {}
        # Final save for any remaining results
        if intermediate_result:
            self.save_checkpoint(intermediate_result, process_id)

    def check_dirs(self):
        if not os.path.exists(self.output_file_path):
            os.makedirs(self.output_file_path)
            logger.debug("Specified output file path doesn't exist. Created folder @ {}", self.output_file_path)
        else:
            logger.debug("Output file path exists")
            if not os.path.exists(self.output_file_path + self.checkpoint_folder + self.id):
                os.makedirs(self.output_file_path + self.checkpoint_folder + self.id)
                logger.debug("Checkpoints temporary dir doesn't exist. Created folder @ {}",
                             self.output_file_path + self.checkpoint_folder + self.id)

    def query_data_in_chunks(self, where, project, batch_size=500, checkpoint_interval=100):
        """
        Distribute MongoDB query processing across multiple processes using chunked processing.
        """
        # Generate chunks based on the collection's create_at field
        delta = timedelta(weeks=1)
        chunks = list(self.generate_date_chunks(self.start_date, self.end_date, delta))
        processes = []

        self.check_dirs()

        # Define checkpoint file per worker
        for i, chunk in enumerate(chunks):
            process = multiprocessing.Process(target=self.worker_process,
                                              args=(where, project, chunk, batch_size, checkpoint_interval, i))
            processes.append(process)
            process.start()

        # Wait for all worker processes to complete
        for process in processes:
            process.join()

        print("All data processed and intermediate results saved in checkpoint files.")

    """
    def query_data_in_chunks(self, where, project, num_processes=os.cpu_count(), batch_size=500):
        """"""
        Distribute MongoDB query processing across multiple processes using chunked processing.
        """"""
        # Generate chunks based on the collection's create_at field
        delta = timedelta(weeks=1)
        chunks = list(self.generate_date_chunks(self.start_date, self.end_date, delta))

        with multiprocessing.Manager() as manager:
            # Prepare to launch processes
            processes = []
            shared_list = manager.dict()

            for i, chunk in enumerate(chunks):
                process = multiprocessing.Process(target=self.worker_process, args=(where, project, chunk, batch_size, i, shared_list))
                processes.append(process)
                process.start()

                # If you've reached the max number of processes, wait for them to finish before continuing
                if len(processes) == num_processes:
                    for p in processes:
                        p.join()
                    processes = []

            # Ensure any remaining processes finish
            for process in processes:
                process.join()
            result = [(k[0], k[1], v, k[2]) for k, v in shared_list.items()]
        return result
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
