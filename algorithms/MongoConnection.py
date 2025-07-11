import pymongo.errors
from pymongo import MongoClient

import logging


class MongoConnection:
    
    logger = logging.getLogger('MongoConnection')

    def __init__(self, uri, username=None, password=None, auth_source=None, auth_mechanism=None, db=None,
                 database_name=None, collection=None, start_date=None, end_date=None):
        self.uri = uri
        self.username = username
        self.password = password
        self.auth_source = auth_source
        self.auth_mechanism = auth_mechanism
        self.db = db
        self.database_name = database_name
        self.collection = collection
        self.start_date = start_date
        self.end_date = end_date

    def connect(self):
        """
        Establishes a connection to the MongoDB database.
        If the database is not already set, it initializes the connection using the provided URI,
        username, password, auth source, and authentication mechanism.
        If the connection fails, it prints the error message.

        :raises pymongo.errors.ConnectionFailure: If the connection to MongoDB fails.
        :raises pymongo.errors.OperationFailure: If an operation on the database fails.
        :return: MongoClient instance connected to the specified database.
        """
        if self.db is None and self.database_name is not None:
            mongo_client = MongoClient(self.uri,
                                       username=self.username,
                                       password=self.password,
                                       authSource=self.auth_source,
                                       authMechanism=self.auth_mechanism)
            try:
                self.db = mongo_client[self.database_name]
            except pymongo.errors.ConnectionFailure as e:
                self.logger.error(f"Failed to connect to MongoDB: {e}")
                raise pymongo.errors.ConnectionFailure(f"Failed to connect to MongoDB: {e}")
            except pymongo.errors.OperationFailure as e:
                self.logger.error(f"Operation failed: {e}")
                raise pymongo.errors.OperationFailure(f"Operation failed: {e}")
        return mongo_client

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
