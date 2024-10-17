import json

import pandas as pd
import pymongo.errors
from pymongo import MongoClient

from Utils.Const import Const as c
from typing import Any, Optional


class RawData:

    def __init__(self, uri, username=None, password=None, auth_source=None, auth_mechanism=None, db=None,
                 collection=None, input_type="mongo"):
        self.uri = uri
        self.username = username
        self.password = password
        self.auth_source = auth_source
        self.auth_mechanism = auth_mechanism
        self.db = db
        self.collection = collection
        self.type = input_type

    def connect(self, database_name: str = None, collection: str = None):
        if self.db is None and database_name is not None and self.type == c.MONGO:
            mongo_client = MongoClient(self.uri,
                                       username=self.username,
                                       password=self.password,
                                       authSource=self.auth_source,
                                       authMechanism=self.auth_mechanism)
            try:
                self.db = mongo_client[database_name]
                self.collection = self.db[collection]
            except(pymongo.errors.CollectionInvalid,
                   pymongo.errors.ConnectionFailure) as e:
                print(e)

    def get_collection(self):
        return self.collection

    def set_collection(self, collection_name):
        if collection_name != self.collection:
            self.collection = self.db[collection_name]

    def query(self, where=None, project=None, batch_size=100):
        if self.type == c.MONGO:
            result = self.collection.find(where, project, batch_size=batch_size)
            return pd.json_normalize(result)
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
