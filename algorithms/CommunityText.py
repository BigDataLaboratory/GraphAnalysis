import multiprocessing
import uuid
from datetime import timedelta

import pandas as pd
from pymongo import ASCENDING

from Utils.Writer import Writer
from algorithms.MongoConnection import MongoConnection


class CommunityText(MongoConnection):

    def __init__(self, comms_index, uri):
        super().__init__(uri)
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

    """
    def merge_and_aggregate_checkpoints(self):
        
        #Merges checkpoint files from a folder, aggregates by the first two elements,
        #and writes the final result to a single output file. Deletes each checkpoint file after processing.
        
        checkpoint_dir = os.sep.join([self.output_file_path, self.checkpoint_folder, self.id])
        # Iterate over all checkpoint files in the folder
        for graph_type in GraphType:
            aggregated_results = defaultdict(lambda: 0)  # Structure: { (key1, key2): sum_third }
            checkpoint_files = self.sep.join([self.checkpoint_folder, self.id, graph_type.name, "*"])

            list_checkpoint_files = self.w.list_checkpoint_files(os.sep.join([checkpoint_dir, checkpoint_files]))
            for file_path in list_checkpoint_files:
                checkpoint_data = self.w.load_checkpoint_file(file_path)
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

            merged_file_path = os.sep.join([self.output_file_path, self.id, graph_type.name])
            self.w.write_on_csv(merged_file_path, final_result_graph)

        # Iterate over all checkpoint files in the folder
        for map_type in MapType:
            checkpoint_files = self.sep.join([self.checkpoint_folder, self.id, map_type.name, c.MAP, "*"])
            list_map_files = self.w.list_checkpoint_files(os.sep.join([checkpoint_dir, checkpoint_files]))
            merged_file_path = os.sep.join([self.output_file_path, str(self.id), map_type.name])
            for file_path in list_map_files:
                checkpoint_data = self.w.load_checkpoint_file(file_path)
                self.w.write_on_csv(merged_file_path, checkpoint_data)
    """

    """
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
            self.w.write_on_csv(path, result_graph[k])
        for k in result_map:
            file_path = self.sep.join([self.checkpoint_folder, str(self.id), str(k), c.MAP, str(process_id)])
            path = os.sep.join([dir_path, file_path])
            self.w.write_on_csv(path, result_map[k])

    """
    """
    def worker_process(self, where, project, chunk, batch_size, checkpoint_interval, process_id):
        
        #Worker function to process a chunk of data from MongoDB.
        
        db = self.get_db()
        c = db[self.get_collection()]

        start_id, end_id = chunk

        # Create the query filter for the chunk
        d = {"created_at": {"$gte": start_id, "$lt": end_id}}
        where_f = {'$and': [where, d]}

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
    """
    """
    def query_data_in_chunks(self, where, project, batch_size=500, checkpoint_interval=40000):
        
        #Distribute MongoDB query processing across multiple processes using chunked processing.
        
        # Generate chunks based on the collection's create_at field

        # Generate chunks based on the collection's _id field
        chunks = list(self.generate_chunks(batch_size))
        processes = []

        Writer.create_dirs(self.output_file_path, self.id)

        # Define checkpoint file per worker
        for i, chunk in enumerate(chunks):
            process = multiprocessing.Process(target=self.worker_process,
                                              args=(where, project, chunk, batch_size, checkpoint_interval, i))
            processes.append(process)
            process.start()

        # Wait for all worker processes to complete
        for process in processes:
            process.join()

        self.merge_and_aggregate_checkpoints()
        print("All data processed and intermediate results saved in checkpoint files.")
    """

    def get_users_tweet_text(self, column_cluster_position, cols_maps=None, col_comms=None):
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

        merged = communities_df.merge(maps_df, left_on="node_hash",
                                   right_on="node_hash",
                                   how="inner")

        u = merged['original'].tolist()
        u = list(map(int, u))