import json

import pandas as pd
import pymongo.errors
from pymongo import MongoClient
from pymongo import ASCENDING
import multiprocessing
from datetime import datetime, timedelta
from Utils.Const import Const as c
import os


class RawData:

    def __init__(self, uri, username=None, password=None, auth_source=None, auth_mechanism=None, db=None,
                 collection=None, start_date=None, end_date=None, input_type="mongo"):
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


    def generate_date_chunks(self, start_date, end_date, delta):
        """
        Generates date ranges to divide the data into chunks based on the `created_at` field.
        """
        current_date = start_date
        while current_date < end_date:
            next_date = current_date + delta
            yield (current_date, next_date)
            current_date = next_date

    def worker_process(self, where, project, chunk, batch_size, process_id):
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
        cursor = c.find(where_f, project).sort('created_at', ASCENDING).batch_size(batch_size)

        for document in cursor:
            # Process the document here (you can modify this to suit your needs)
            print(f"Process {process_id} processing document ID: {document['_id']}")
            return pd.json_normalize(document)

    import multiprocessing

    def query_data_in_chunks(self, where, project, num_processes, batch_size):
        """
        Distribute MongoDB query processing across multiple processes using chunked processing.
        """
        # Generate chunks based on the collection's create_at field
        delta = timedelta(weeks=1)
        chunks = list(self.generate_date_chunks(self.start_date, self.end_date, delta))

        # Prepare to launch processes
        processes = []

        for i, chunk in enumerate(chunks):
            process = multiprocessing.Process(target=self.worker_process, args=(where, project, chunk, batch_size, i))
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

    if __name__ == "__main__":
        query_data_in_chunks(where, project, num_processes=os.cpu_count(), batch_size=1000)

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
