import logging
import multiprocessing
import os
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from datetime import timedelta, timezone, datetime
from enum import Enum
from zoneinfo import ZoneInfo
import ast
import json

import pytz
from pymongo import ASCENDING

import sys

# Queste due righe dicono a Python di guardare anche nella cartella principale
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
                 user_hashtag=False, hashtag_cooccurrences=False, response=False, mention=False):
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
        ### 1. Inizializziamo il dizionario delle statistiche per questo tweet
        stats = {}

        ### 2. Identificazione del tipo
        is_retweet = d.get('retweeted_status', None) is not None
        is_reply = d.get('in_reply_to_status_id', None) is not None

        n_user_id = Utils.hash(d['user']['id'])
        m.add((d['user']['id'], n_user_id, 0))

        # --- VERIFIED STATUS ---
        # verified è un booleano (True/False). Lo trasformiamo in 1/0
        is_verified = 1 if d['user'].get('verified', False) else 0

        # --- ACCOUNT AGE ---
        user_created_at = d['user'].get('created_at')

        # NON calcoliamo l'age qui
        account_age_days = None

        # PRENDIAMO I LIKE
        current_likes = d['user'].get('favourites_count', 0)
        current_followers = d['user'].get('followers_count', 0)
        current_following = d['user'].get('friends_count', 0)

        # AGGIUNTA HASHTAG: Estraiamo gli hashtag unici di questo tweet
        raw_ht = d.get('hashtagEntities', "")
        if isinstance(raw_ht, str) and raw_ht.strip():
            # Dividiamo la stringa e puliamo gli spazi
            hashtags_set = set(tag.strip().lower() for tag in raw_ht.split('|') if tag.strip())
        else:
            hashtags_set = set()

        # GESTIONE MENZIONI (Aggiungi questo qui sotto) ---
        raw_mentions = d.get('userMentionEntities', "")
        if isinstance(raw_mentions, str) and raw_mentions.strip():
            # Dividiamo la stringa delle menzioni e puliamo gli spazi
            mentions_set = set(m.strip().lower() for m in raw_mentions.split('|') if m.strip())
        else:
            mentions_set = set()

        # Prepariamo i dati per l'utente corrente
        # Contiamo 1 tweet totale e 0 retweet di base
        stats[n_user_id] = {'total': 1,
                            'retweets': 1 if is_retweet else 0,
                            'reply': 1 if is_reply else 0,
                            'original': 1 if (not is_retweet and not is_reply) else 0,
                            'likes': current_likes,
                            'followers': current_followers,
                            'following': current_following,
                            'verified': is_verified,
                            'account_age_days': account_age_days,
                            'created_at_user': user_created_at,
                            'timestamp': d['created_at'].timestamp(),
                            'hashtags': hashtags_set,
                            'mentions': mentions_set
                            }

        weight = 1

        if is_retweet:
            # Se entriamo qui, il tweet corrente è un retweet. Incrementiamo il contatore.
            stats[n_user_id]['retweets'] = 1
            if self.retweet:
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
            # 1. Questa è la tua aggiunta per il conteggio
            stats[n_user_id]['reply'] = 1
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
        
        if not is_retweet and d.get('in_reply_to_user_id', -1) == -1:
            stats[n_user_id]['original'] = 1

        # Cambiamo il return per restituire anche le stats
        return o, m, stats

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
                            aggregated_results[key] = (float(row[4]), float(row[5]))
                final_result_graph = []
                for k, v in aggregated_results.items():
                    if graph_type.value != 1:
                        final_result_graph.append((k[0], k[1], k[2], k[3], v))
                    else:
                        final_result_graph.append((k[0], k[1], k[2], k[3], v[0], v[1]))
                    
                Writer.write_on_csv(merged_file_path, final_result_graph)
            else:
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
                            aggregated_results[key] = (float(row[3]), float(row[4]))

                final_result_graph = []
                for k, v in aggregated_results.items():
                    if graph_type.value != 1:
                        final_result_graph.append((k[0], k[1], k[2], v))
                    else:
                        final_result_graph.append((k[0], k[1], k[2], v[0], v[1]))

                Writer.write_on_csv(merged_file_path, final_result_graph)

        # Iterate over all checkpoint files in the folder
        for map_type in MapType:
            checkpoint_files = self.sep.join([self.checkpoint_folder, self.id, map_type.name, c.MAP, "*"])
            list_map_files = Writer.list_checkpoint_files(os.sep.join([checkpoint_dir, checkpoint_files]))
            merged_file_path = os.sep.join([self.output_file_path, str(self.id), map_type.name])

            all_data = []

            for file_path in list_map_files:
                checkpoint_data = Writer.load_checkpoint_file(file_path)
                all_data.extend(checkpoint_data)
            unique_data = list(set(tuple(row) for row in all_data))  # Remove duplicates by converting to set and back to list
            Writer.write_on_csv(merged_file_path, unique_data)

            # --- AGGREGAZIONE FINALE DELLE USER STATS ---
        self.logger.info("Merging user statistics...")
        
        # Cerchiamo i file di statistiche creati dai worker
        stats_dir = os.sep.join([checkpoint_dir, "USER_STATS"])
        if os.path.exists(stats_dir):
            all_stats_files = Writer.list_checkpoint_files(os.sep.join([stats_dir, "*"]))
            
            final_user_metrics = {} # Qui uniremo tutto

            for file_path in all_stats_files:
                worker_stats = Writer.load_checkpoint_file(file_path)
                
                for row in worker_stats:
    # Indici coerenti con save_stats_checkpoint
                    uid = row[0]
                    t_tweets = int(row[1]) if row[1] != "" else 0
                    t_rt = int(row[2]) if row[2] != "" else 0
                    t_rep = int(row[3]) if row[3] != "" else 0
                    t_orig = int(row[4]) if row[4] != "" else 0
                    t_likes = int(row[5]) if len(row) > 5 and row[5] != "" else 0
                    t_followers = int(row[6]) if len(row) > 6 and row[6] != "" else 0
                    t_following = int(row[7]) if len(row) > 7 and row[7] != "" else 0
                    t_verified = int(row[8]) if len(row) > 8 and row[8] != "" else 0
                    t_time = float(row[9]) if len(row) > 9 and row[9] != "" else 0.0

                    user_created_iso = row[10] if len(row) > 10 else ""
                    hashtags_list = []
                    mentions_list = []

                    if len(row) > 11 and row[11]:
                        try:
                            hashtags_list = json.loads(row[11])
                        except Exception:
                            hashtags_list = []
                        if len(row) > 12 and row[12]:
                            try:
                                mentions_list = json.loads(row[12])
                            except Exception:
                                mentions_list = []

                    # Convert ISO string to datetime safely
                    user_created_dt = None
                    if user_created_iso:
                        try:
                            user_created_dt = datetime.fromisoformat(user_created_iso)
                        except Exception:
                            user_created_dt = None

                    # Populate/merge final_user_metrics
                    if uid not in final_user_metrics:
                        final_user_metrics[uid] = {
                            'total': t_tweets,
                            'retweets': t_rt,
                            'reply': t_rep,
                            'original': t_orig,
                            'likes': t_likes,
                            'followers': t_followers,
                            'following': t_following,
                            'verified': t_verified,
                            'timestamp': t_time,
                            'created_at_user': user_created_dt,
                            'hashtags': set(hashtags_list),
                            'mentions': set(mentions_list)
                        }
                    else:
                        final_user_metrics[uid]['total'] += t_tweets
                        final_user_metrics[uid]['retweets'] += t_rt
                        final_user_metrics[uid]['reply'] += t_rep
                        final_user_metrics[uid]['original'] += t_orig

                        final_user_metrics[uid]['hashtags'].update(hashtags_list)
                        final_user_metrics[uid]['mentions'].update(mentions_list)

                        if t_time > final_user_metrics[uid]['timestamp']:
                            final_user_metrics[uid]['likes'] = t_likes
                            final_user_metrics[uid]['followers'] = t_followers
                            final_user_metrics[uid]['following'] = t_following
                            final_user_metrics[uid]['verified'] = t_verified
                            final_user_metrics[uid]['timestamp'] = t_time
                            if user_created_dt:
                                final_user_metrics[uid]['created_at_user'] = user_created_dt

            # Transform into list for CSV with requested fields:
            # uid, total, retweets, reply, original, likes (latest), followers (latest), following (latest),
            # verified (latest), account_age_days (latest), unique_hashtags_count, unique_mentions_count
            final_stats_list = []
            for uid, data in final_user_metrics.items():
                user_created = data.get('created_at_user', None)
                latest_ts = data.get('timestamp', 0)

                if user_created and latest_ts:
                    try:
                        account_age_days = int((datetime.fromtimestamp(latest_ts) - user_created).days)
                    except Exception:
                        account_age_days = 0
                else:
                    account_age_days = 0

                final_stats_list.append((
                    uid,
                    data['total'],
                    data['retweets'],
                    data['reply'],
                    data['original'],
                    data.get('likes', 0),
                    data.get('followers', 0),
                    data.get('following', 0),
                    data.get('verified', 0),
                    account_age_days,
                    len(data.get('hashtags', set())),
                    len(data.get('mentions', set()))
                ))


            stats_output_path = os.sep.join([self.output_file_path, self.id, "final_user_stats"])
            Writer.write_on_csv(stats_output_path, final_stats_list)
            self.logger.info(f"User statistics saved to {stats_output_path}.csv")

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
            # k is a tuple whose position of graph type depends on bucket/full mode
            # For full mode we used (src, dst, graph_type)
            # For bucket mode we used (src, dst, date, graph_type)
            if len(k) == 3:
                # full mode
                result_graph[GraphType(k[2]).name].append((k[2], k[0], k[1], v)) if k[2] != 1 else result_graph[
                GraphType(k[2]).name].append((k[2], k[0], k[1], v[0], v[1]))
            else:
                # bucket mode
                result_graph[GraphType(k[3]).name].append((k[3], k[0], k[1], k[2], v)) if k[3] != 1 else result_graph[
                    GraphType(k[3]).name].append((k[3], k[0], k[1], k[2], v[0], v[1]))

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
    

    def save_stats_checkpoint(self, intermediate_stats, process_id):
    
            rows = []
            for uid, s in intermediate_stats.items():
                # Serializziamo i set in JSON per evitare eval()
                created_at_user = s.get('created_at_user', None)
                created_iso = created_at_user.isoformat() if isinstance(created_at_user, datetime) else ""

                h_json = json.dumps(list(s.get('hashtags', [])))
                m_json = json.dumps(list(s.get('mentions', [])))
            
                # Scriviamo tutti i campi richiesti
                # Usiamo il tabulatore \t per evitare problemi con virgole nei testi
                row = [
                    str(uid),
                    str(s.get('total', 0)),
                    str(s.get('retweets', 0)),
                    str(s.get('reply', 0)),
                    str(s.get('original', 0)),
                    str(s.get('likes', 0)),
                    str(s.get('followers', 0)),
                    str(s.get('following', 0)),
                    str(s.get('verified', 0)),
                    str(s.get('timestamp', 0)),
                    created_iso,
                    h_json, # Hashtag serializzati
                    m_json  # Menzioni serializzate
                ]
                rows.append(row)

            # Ensure directory exists and write using Writer
            dir_path = os.sep.join([self.output_file_path, self.checkpoint_folder, str(self.id), "USER_STATS"])
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.sep.join([dir_path, str(process_id)])
            Writer.write_on_csv(file_path, rows)

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
            intermediate_stats = {}

            for i, document in enumerate(cursor, 1):
                edges, maps, tweet_stats = self.process_document(document)

                for uid, s in tweet_stats.items():
                    if uid not in intermediate_stats:
                        intermediate_stats[uid] = {
                            'total': s['total'],
                            'retweets': s['retweets'],
                            'reply': s['reply'],
                            'original': s['original'],
                            'likes': s['likes'],
                            'followers': s['followers'],
                            'following': s['following'],
                            'verified': s['verified'],
                            'account_age_days': s['account_age_days'],
                            'timestamp': s['timestamp'],
                            'hashtags': set(s.get('hashtags', set())),
                            'mentions': set(s.get('mentions', set()))
                        }
                    else:
                        intermediate_stats[uid]['total'] += s['total']
                        intermediate_stats[uid]['retweets'] += s['retweets']
                        intermediate_stats[uid]['reply'] += s['reply']
                        intermediate_stats[uid]['original'] += s['original']

                        # Merge sets
                        intermediate_stats[uid]['hashtags'].update(s.get('hashtags', set()))
                        intermediate_stats[uid]['mentions'].update(s.get('mentions', set()))

                        # Update snapshot fields if this tweet is more recent
                        if s['timestamp'] > intermediate_stats[uid]['timestamp']:
                            intermediate_stats[uid]['likes'] = s['likes']
                            intermediate_stats[uid]['followers'] = s['followers']
                            intermediate_stats[uid]['following'] = s['following']
                            intermediate_stats[uid]['verified'] = s['verified']
                            intermediate_stats[uid]['account_age_days'] = s['account_age_days']
                            intermediate_stats[uid]['created_at_user'] = s['created_at_user']
                            intermediate_stats[uid]['timestamp'] = s['timestamp']

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
                    # AGGIUNGI IL SALVATAGGIO STATS (vedi sotto come fare)
                    self.save_stats_checkpoint(intermediate_stats, process_id)
                    intermediate_result = {}
                    intermediate_map = set()
                    intermediate_stats = {} # Svuota
            # Final save for any remaining results
            if intermediate_result:
                self.save_checkpoint(intermediate_result, intermediate_map, process_id)
            if intermediate_stats:
                self.save_stats_checkpoint(intermediate_stats, process_id)

        self.logger.info(f"[Worker {process_id}] Final checkpoint saved")
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
        self.logger.info(f"Processing {len(chunks)} chunks with up to {max_workers} workers (sequential execution).")
        # Sequential processing to avoid pickling self. If you need parallelism, refactor worker_process.
        for i, chunk in enumerate(chunks):
            try:
                self.logger.info(f"[Worker {i}] Starting chunk {i} ({chunk[0]} -> {chunk[1]})")
                result = self.worker_process(where, project, method, chunk, batch_size, checkpoint_interval, i)
                self.logger.info(f"[Worker {i}] {result}")
            except Exception as e:
                self.logger.error(f"[Worker {i}] Failed: {e}")
    
        self.logger.info("All chunks processed. Merging and aggregating results.")
        self.merge_and_aggregate_checkpoints(method)
        self.logger.info("All data processed and intermediate results saved in checkpoint files.")