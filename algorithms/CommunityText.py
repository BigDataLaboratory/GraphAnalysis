import multiprocessing
import os
import uuid

import pandas as pd
from pymongo import ASCENDING

from Utils.Writer import Writer
from algorithms.MongoConnection import MongoConnection


class CommunityText(MongoConnection):

    def __init__(self, comms_index, uri, username=None, password=None, auth_source=None, auth_mechanism=None, db=None,
                 collection=None, start_date=None, end_date=None, output_file_path="/ipazianas/pasquini/output_graph_analysis"):
        super().__init__(uri, username, password, auth_source, auth_mechanism, db, collection, start_date, end_date)
        self.checkpoint_folder = "tmp"
        self.sep = "_"
        self.output_file_path = output_file_path
        self.community_file_path = None
        self.maps_file_path = None
        self.comms_index = comms_index
        self.id = uuid.uuid1().hex

    def set_comms_file_path(self, file_path):
        self.community_file_path = file_path

    def set_maps(self, maps_file_paths):
        self.maps_file_path = maps_file_paths

    def generate_date_chunks(self, start_date, end_date, delta):
        """
        Generates date ranges to divide the data into chunks based on the `created_at` field.
        """
        current_date = start_date
        while current_date < end_date:
            next_date = current_date + delta
            yield current_date, next_date
            current_date = next_date

    def generate_chunks(self, chunk_size):
        """
        Generates ranges of _id values to divide the data into chunks.
        """
        db = self.get_db()
        c = db[self.get_collection()]
        min_id = c.find_one(sort=[('user.id', ASCENDING)])['user.id']
        max_id = c.find_one(sort=[('user.id', -1)])['user.id']

        current_id = min_id
        while current_id < max_id:
            next_id = current_id + chunk_size
            yield (current_id, next_id)
            current_id = next_id

    def merge_and_aggregate_checkpoints(self):
        """
        Merges checkpoint files from a folder, aggregates by the first two elements,
        and writes the final result to a single output file. Deletes each checkpoint file after processing.
        """
        # Iterate over all checkpoint files in the folder
        checkpoint_dir = os.sep.join([self.output_file_path, self.checkpoint_folder, self.id])
        checkpoint_files = self.sep.join([self.checkpoint_folder, self.id, "*"])
        list_map_files = Writer.list_checkpoint_files(os.sep.join([checkpoint_dir, checkpoint_files]))
        merged_file_path = os.sep.join([self.output_file_path, str(self.id)])
        for file_path in list_map_files:
            checkpoint_data = Writer.load_checkpoint_file(file_path)
            Writer.write_on_csv(merged_file_path, checkpoint_data)

    def save_checkpoint(self, intermediate_results, process_id):
        dir_path = os.sep.join([self.output_file_path, self.checkpoint_folder, str(self.id)])
        file_path = self.sep.join([self.checkpoint_folder, str(self.id), str(process_id)])
        path = os.sep.join([dir_path, file_path])
        Writer.write_on_csv(path, intermediate_results)

    def process_document(self, d):
        return d["user"]["id"], d["created_at"], d["type"], d["text"]

    def worker_process(self, match, project, chunk, batch_size, checkpoint_interval, process_id):
        """
        Worker function to process a chunk of data from MongoDB.
        """
        db = self.get_db()
        c = db[self.get_collection()]

        start_id, end_id = chunk

        # Create the query filter for the chunk
        d = {"user.id": {"$gte": start_id, "$lt": end_id}}
        match["user.id"] = {"$and": [match["$match"]["user.id"], d]}

        pipeline = [match, project]
        # Retrieve documents in batches
        cursor = c.aggregate(pipeline).sort('user.id', ASCENDING).batch_size(batch_size)

        intermediate_result = []
        for i, document in enumerate(cursor, 1):
            e = self.process_document(document)
            intermediate_result.append(e)

            # Save checkpoint after every `checkpoint_interval` documents
            if i % checkpoint_interval == 0:
                self.save_checkpoint(intermediate_result, process_id)
                intermediate_result = []
        # Final save for any remaining results
        if intermediate_result:
            self.save_checkpoint(intermediate_result, process_id)

    def query_data_in_chunks(self, match, project, batch_size=500, checkpoint_interval=40000):
        """
        Distribute MongoDB query processing across multiple processes using chunked processing.
        """
        # Generate chunks based on the collection's create_at field

        # Generate chunks based on the collection's _id field
        chunks = list(self.generate_chunks(batch_size))
        processes = []

        Writer.create_dirs(self.output_file_path, self.id)

        # Define checkpoint file per worker
        for i, chunk in enumerate(chunks):
            process = multiprocessing.Process(target=self.worker_process,
                                              args=(match, project, chunk, batch_size, checkpoint_interval, i))
            processes.append(process)
            process.start()

        # Wait for all worker processes to complete
        for process in processes:
            process.join()

        self.merge_and_aggregate_checkpoints()
        print("All data processed and intermediate results saved in checkpoint files.")

    def get_users_tweet_text(self, column_cluster_position, project, cols_maps=None, col_comms=None):
        if col_comms is None:
            col_comms = ["node_hash", "pr", "community"]
        if cols_maps is None:
            cols_maps = ["original", "node_hash", "type"]
        w = Writer()
        communities = w.read_csv_files_in_folder_parallel(self.community_file_path, header=True)
        maps = w.read_csv_files_in_folder_parallel(self.maps_file_path)

        communities_filtered = [tup for tup in communities if tup[column_cluster_position] in self.comms_index]

        maps_df = pd.DataFrame(maps, columns=cols_maps)
        communities_df = pd.DataFrame(communities_filtered, columns=col_comms)

        merged = communities_df.merge(maps_df, left_on=col_comms[1],
                                   right_on="node_hash",
                                   how="inner")

        u = merged['original'].tolist()
        u = list(map(int, u))

        match = {'$match': {
            'user.id': {'$in': u}  # Filter docs based on users list
        }}

        self.query_data_in_chunks(match, project)
