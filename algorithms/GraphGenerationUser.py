import itertools
import json
import logging
import math
import os
import threading
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional
from pymongo import ASCENDING, DESCENDING, MongoClient
from dateutil import parser  # Import to parse ISO dates
import re
import shutil
import time

import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.Utils import Utils
from Utils.Writer import Writer, _ext, SUPPORTED_FORMATS
from algorithms.MongoConnection import MongoConnection
import algorithms.mongoQueries as mongoQueries

_HTML_TAG_RE = re.compile(r'<.*?>')


def tweet_regularity_score_from_timestamps(timestamps):
    """
    Computes the regularity of posting intervals using the coefficient of variation (CV)
    of Inter-Tweet Intervals (ITI = time between consecutive tweets, in seconds).

    CV = std(ITI) / mean(ITI). A perfectly regular poster (bot) has CV ≈ 0 → score ≈ 1.
    A chaotic poster has high CV → score near 0.

    Requires at least 2 timestamps. O(N log N) due to sort.
    """
    if len(timestamps) < 2:
        return {'regularity_score': 0.0, 'iti_std': 0.0, 'iti_mean': 0.0, 'n_intervals': 0}

    sorted_ts = np.sort(np.array(timestamps, dtype=float))
    intervals = np.diff(sorted_ts)

    iti_mean = float(intervals.mean())
    iti_std = float(intervals.std(ddof=0))
    cv = iti_std / iti_mean if iti_mean > 0 else float('inf')
    regularity_score = 1.0 / (1.0 + cv)

    return {
        'regularity_score': float(regularity_score),
        'iti_std': iti_std,
        'iti_mean': iti_mean,
        'n_intervals': len(intervals),
    }


def daily_posting_consistency(timestamps, window_days=90, min_days=7, cv_clip=5.0):
    """
    Measures consistency of day-over-day posting volume within a trailing window.

    Uses the CV of log-transformed daily tweet counts. Lower CV = more consistent
    daily volume. Combined with activity_ratio (fraction of days with any activity)
    into a single score in [0, 1].
    """
    if not timestamps:
        return {'daily_score': 0.0, 'daily_cv_log': None, 'median_daily_count': 0, 'n_days': 0}

    ts = np.sort(np.array(timestamps, dtype=float))
    start = int(ts[-1]) - int(window_days) * 86400

    days = [datetime.fromtimestamp(t, tz=timezone.utc).date() for t in ts if t >= start]
    if not days:
        return {'daily_score': 0.0, 'daily_cv_log': None, 'median_daily_count': 0, 'n_days': 0}

    day_counts = Counter(days)
    unique_days = sorted(set(days))
    n_days = len(unique_days)
    counts = [day_counts[d] for d in unique_days]

    if n_days < min_days:
        median_daily = float(np.median(counts)) if counts else 0.0
        return {'daily_score': 0.0, 'daily_cv_log': None, 'median_daily_count': median_daily, 'n_days': n_days}

    counts_arr = np.array(counts, dtype=float)
    log_counts = np.log1p(counts_arr)
    mean_log = float(log_counts.mean())
    std_log = float(log_counts.std(ddof=0))
    daily_cv_log = std_log / mean_log if mean_log > 0 else float('inf')

    score_from_cv = max(0.0, 1.0 - min(daily_cv_log, cv_clip) / cv_clip)
    activity_bonus = min(1.0, (n_days / float(window_days)) * 2.0)
    daily_score = float(max(0.0, min(1.0, 0.75 * score_from_cv + 0.25 * activity_bonus)))

    return {
        'daily_score': daily_score,
        'daily_cv_log': None if not np.isfinite(daily_cv_log) else float(daily_cv_log),
        'median_daily_count': float(np.median(counts_arr)),
        'n_days': n_days,
    }


def internal_tweet_density(timestamps, window_days=90):
    """
    Measures the spread of posting activity across the 24-hour cycle using
    Shannon entropy, normalized to [0, 1] by dividing by log2(24).

    H = 1 → perfectly uniform (one tweet per hour, human-like).
    H = 0 → all tweets at the same hour (bot-like, no sleep pattern).
    """
    if not timestamps:
        return 0.0

    ts = np.array(timestamps, dtype=float)
    start = float(ts.max()) - int(window_days) * 86400
    hours = [datetime.fromtimestamp(t, tz=timezone.utc).hour for t in ts if t >= start]

    if len(hours) < 2:
        return 0.0

    hour_counts = Counter(hours)
    total_h = len(hours)
    probs = [c / total_h for c in hour_counts.values()]
    raw_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return raw_entropy / math.log2(24)


def classify_source(src_raw: str) -> str:
    """
    Classifies the tweet source into one of: 'mobile', 'web', 'news_manager', 'bot_api'.
    Categories based on the most frequent sources in the dataset.
    """
    if not isinstance(src_raw, str):
        return 'bot_api'

    m = _HTML_TAG_RE.sub('', src_raw).strip().lower()

    # these values are based on this query: db.collection.distinct('source')
    MOBILE = ['iphone', 'android', 'ipad', 'mobile', 'twitter for mac']
    WEB = ['web app', 'web client', 'twitter web']
    NEWS_MANAGER = [
        'postpickr', 'hootsuite', 'wordpress', 'blog2social',
        'dlvr.it', 'dlvrit', 'ifttt', 'instagram'
    ]

    if any(k in m for k in MOBILE):
        return 'mobile'
    if any(k in m for k in WEB):
        return 'web'
    if any(k in m for k in NEWS_MANAGER):
        return 'news_manager'
    return 'bot_api'  # TweetDeck, API, unknown → bot/api by default


# ─────────────────────────────────────────────────────────────────────────────
# Column definitions (single source of truth)
# ─────────────────────────────────────────────────────────────────────────────
USER_FEATURES_COLUMNS = [
    'user_node_id', 'total', 'retweets', 'replies', 'original',
    'likes', 'followers', 'following', 'verified', 'account_date',
    'listed_count', 'favourites_count', 'reputation_score',
    'n_unique_hashtags', 'n_unique_mentions',
    'n_hashtags_total', 'hashtag_entropy',
    'activation_age', 'tweet_regularity_score', 'regularity_reliable',
    'tweet_avg_interval_seconds', 'daily_score', 'daily_cv_log',
    'internal_tweet_density', 'profile_has_url', 'geo_enabled_flag',
    'sensitive_rate', 'mobile_ratio', 'web_ratio',
    'news_manager_ratio', 'bot_api_ratio', 'source_entropy', 'community',
]
USER_FEATURES_COLUMNS_FINAL = USER_FEATURES_COLUMNS + [
    'received_retweets', 'received_replies', 'received_mentions',
]
EDGE_RETWEET_COLUMNS  = ['src', 'dst', 'weight', 'lifespan', 'fast_rt_ratio', 'rt_cadence', 'rt_temporal_jitter', 'rt_topic_consistency']
EDGE_REPLY_COLUMNS    = ['src', 'dst', 'weight', 'lifespan', 'avg_reply_latency_seconds', 'reply_regularity', 'reply_burstiness', 'reply_diurnal_sync']
EDGE_MENTION_COLUMNS  = ['src', 'dst', 'weight', 'mention_lifespan', 'mention_regularity', 'mention_burstiness', 'mention_in_reply_ratio', 'mention_solo_ratio']
SCREEN_NAME_COLUMNS   = ['screen_name', 'user_id']



from abc import ABC, abstractmethod

class ExtractionStrategy(ABC):
    @abstractmethod
    def partition_work(self, collection, n_workers, max_users, logger):
        pass

    @abstractmethod
    def iter_users(self, collection, work_item):
        pass

    @abstractmethod
    def get_community_id(self, user_id):
        pass

class TweetsSortedByUserScanStrategy(ExtractionStrategy):
    def partition_work(self, collection, n_workers, max_users, logger):
        logger.info("Scanning collection to determine user.id range...")
        bounds = list(collection.aggregate([
            {"$group": {
                "_id": None,
                "min_uid": {"$min": "$user.id"},
                "max_uid": {"$max": "$user.id"},
            }}
        ]))
        if not bounds:
            return []
        
        global_min = bounds[0]["min_uid"]
        global_max = bounds[0]["max_uid"] + 1
        
        step = max(1, (global_max - global_min + n_workers - 1) // n_workers)
        ranges = []
        for i in range(n_workers):
            uid_start = global_min + i * step
            uid_end   = min(global_min + (i + 1) * step, global_max)
            if uid_start >= global_max:
                break
            ranges.append((uid_start, uid_end))
        return ranges

    def iter_users(self, collection, work_item):
        uid_start, uid_end = work_item
        _, proj = mongoQueries.extract_tweets_for_user_graph()
        query = {"user.id": {"$gte": uid_start, "$lt": uid_end}}
        cursor = collection.find(query, proj).sort("user.id", 1)
        for user_id, user_tweets in itertools.groupby(cursor, key=lambda t: t['user']['id']):
            yield user_id, user_tweets

    def get_community_id(self, user_id):
        return -1

class CommunityUserBatchStrategy(ExtractionStrategy):
    def __init__(self, user_community_map, batch_size=10000):
        self.user_community_map = user_community_map
        self.batch_size = batch_size

    def partition_work(self, collection, n_workers, max_users, logger):
        user_ids = sorted(self.user_community_map.keys())
        if max_users is not None:
            user_ids = user_ids[:max_users]

        n_users = len(user_ids)
        min_batches = n_workers * 2
        batch_size = min(self.batch_size, max(1, n_users // min_batches))

        logger.info(
            f"CommunityUserBatchStrategy: {n_users:,} users -> "
            f"batch_size={batch_size} -> ~{math.ceil(n_users / batch_size)} batches "
            f"for {n_workers} workers"
        )

        def chunked(iterable, n):
            it = iter(iterable)
            while batch := list(itertools.islice(it, n)):
                yield batch

        return list(chunked(user_ids, batch_size))

    def iter_users(self, collection, work_item):
        user_ids = work_item
        _, proj = mongoQueries.extract_tweets_for_user_graph()
        query = {"user.id": {"$in": user_ids}}
        cursor = collection.find(query, proj).sort("user.id", 1)
        for user_id, user_tweets in itertools.groupby(cursor, key=lambda t: t['user']['id']):
            yield user_id, user_tweets

    def get_community_id(self, user_id):
        return self.user_community_map.get(user_id, -1)

class CommunityAwareShardStrategy(ExtractionStrategy):
    """
    Partitions work BY COMMUNITY instead of by ID range.
    Each worker takes N complete communities → no coordination,
    and checkpoints are naturally grouped by community.
    """
    def __init__(self, user_community_map: dict, batch_size: int = 500):
        self.user_community_map = user_community_map
        self.batch_size = batch_size
        # Inverse map : community_id -> [user_ids]
        self.community_users: dict = defaultdict(list)
        for uid, cid in user_community_map.items():
            self.community_users[cid].append(uid)

    def partition_work(self, collection, n_workers, max_users, logger):
        # Sort communities by size (desc) for better load balancing
        communities = sorted(
            self.community_users.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        # Round-robin on workers to balance the load
        worker_loads = [[] for _ in range(n_workers)]
        worker_sizes = [0] * n_workers
        for cid, uids in communities:
            lightest = min(range(n_workers), key=lambda i: worker_sizes[i])
            worker_loads[lightest].append((cid, sorted(uids)))
            worker_sizes[lightest] += len(uids)

        logger.info(
            f"CommunityAwareShard: {len(communities)} communities | "
            f"{sum(worker_sizes):,} users | "
            f"load per worker: {worker_sizes}"
        )
        return worker_loads  # work_item = list of (cid, [uids])

    def iter_users(self, collection, work_item):
        _, proj = mongoQueries.extract_tweets_for_user_graph()
        for cid, user_ids in work_item:
            for i in range(0, len(user_ids), self.batch_size):
                batch = user_ids[i : i + self.batch_size]
                print(f"[CommunityAwareShardStrategy] Query:{{'user.id': {{'$in': {batch}}}}}\n")
                cursor = collection.find(
                    {"user.id": {"$in": batch}}, proj
                ).sort("user.id", 1)
                for user_id, user_tweets in itertools.groupby(
                    cursor, key=lambda t: t['user']['id']
                ):
                    yield user_id, user_tweets

    def get_community_id(self, user_id):
        return self.user_community_map.get(user_id, -1)

class CommunitySortedLinearScanStrategy(ExtractionStrategy):
    """
    Reads the collection LINEARLY via the ID ranges (Range Scan).
    Filters in Python to keep ONLY users present in the map.
    This is the fastest strategy for very large volumes (Sequential I/O).
    """
    def __init__(self, user_community_map, batch_size=5000):
        self.user_community_map = user_community_map
        self.batch_size = batch_size
        # On pré-calcule un set pour une recherche O(1)
        self.target_users_set = set(user_community_map.keys())

    def partition_work(self, collection, n_workers, max_users, logger):
        """
        Cuts the collection into equal ID ranges (Min/Max).
        each worker scans its segment linearly.
        """
        logger.info("Calculating UID ranges for linear scan...")
        bounds = list(collection.aggregate([
            {"$group": {
                "_id": None,
                "min_uid": {"$min": "$user.id"},
                "max_uid": {"$max": "$user.id"},
            }}
        ]))
        if not bounds: return []
        
        g_min, g_max = bounds[0]["min_uid"], bounds[0]["max_uid"] + 1
        step = (g_max - g_min + n_workers - 1) // n_workers
        
        ranges = []
        for i in range(n_workers):
            s = g_min + i * step
            e = min(s + step, g_max)
            ranges.append((s, e))
        return ranges

    def iter_users(self, collection, work_item):
        uid_start, uid_end = work_item
        _, proj = mongoQueries.extract_tweets_for_user_graph()
        
        # Filtre de plage pour le worker
        query = {"user.id": {"$gte": uid_start, "$lt": uid_end}}
        
        # Le curseur ne charge pas tout en RAM, il 'stream' les données
        cursor = collection.find(query, proj).sort("user.id", 1).batch_size(self.batch_size)
        
        # Groupby par utilisateur
        for user_id, user_tweets in itertools.groupby(cursor, key=lambda t: t['user']['id']):
            # LA MAGIE : On ignore instantanément l'utilisateur s'il n'est pas dans Leiden
            if user_id not in self.target_users_set:
                continue
            
            yield user_id, user_tweets

    def get_community_id(self, user_id):
        return self.user_community_map.get(user_id, -1)


class GraphGenerationUser(MongoConnection):
    """
    User-centric graph generation for the GNN pipeline.

    Extracts node features (user statistics) and edges (retweet, reply, mention)
    from MongoDB tweets, sorted by user.id to guarantee complete user histories
    per batch.

    Outputs (under output_file_path/<run_id>/):
        user_features.<ext>   : node feature matrix (one row per user)
        edges_retweet.<ext>   : (src, dst, weight, lifespan, fast_rt_ratio)
        edges_reply.<ext>     : (src, dst, weight, lifespan, avg_reply_latency)
        edges_mention.<ext>   : (src, dst, weight)
        screen_name_map.<ext> : (screen_name, user_id)
        metadata.json         : run metadata

    Mention edges whose target screen_name cannot be resolved to a known
    user in the dataset are silently dropped.
    """

    logger = logging.getLogger('GraphGenerationUser')

    def __init__(self, uri, database_name, collection, output_file_path,
                 username=None, password=None, auth_source=None, auth_mechanism=None,
                 delete_tmp_after_merge=False,
                 intermediate_file_format="feather", final_file_format="parquet",
                 file_format=None, fast_rt_threshold=60, strategy=None, is_community_run=False, community_file_path=None):
        """
        :param uri: MongoDB connection URI.
        :param database_name: Name of the MongoDB database.
        :param collection: Collection name.
        :param output_file_path: Root directory for all output files.
        :param username: MongoDB username (optional).
        :param password: MongoDB password (optional).
        :param auth_source: Authentication source (optional).
        :param auth_mechanism: Authentication mechanism (optional).
        :param delete_tmp_after_merge: Whether to delete the tmp folder after merging.
        :param intermediate_file_format: Format for checkpoint files ('feather' recommended).
        :param final_file_format: Format for final merged output ('parquet' recommended).
        :param file_format: Legacy alias — sets both formats if provided.
        :raises ValueError: If any format is not supported.
        """
        # Legacy compatibility: if only file_format is passed, use it for both
        if file_format is not None:
            intermediate_file_format = file_format
            final_file_format = file_format
        for fmt in (intermediate_file_format, final_file_format):
            if fmt not in SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported file_format '{fmt}'. "
                                 f"Choose from {SUPPORTED_FORMATS}.")
        super().__init__(uri, username, password, auth_source, auth_mechanism,
                         db=None, database_name=database_name, collection=collection)
        self.output_file_path = output_file_path
        self.id = uuid.uuid1().hex
        self.checkpoint_folder = "tmp"
        self.delete_tmp_after_merge = delete_tmp_after_merge
        self.file_format = intermediate_file_format        # used by _write (checkpoints)
        self.final_file_format = final_file_format         # used by _write_final (merge output)
        self.fast_rt_threshold = fast_rt_threshold
        self.strategy = strategy or TweetsSortedByUserScanStrategy()
        self.is_community_run = is_community_run
        self.community_file_path = community_file_path

        # Stored for worker threads — each creates its own MongoClient
        self._uri = uri
        self._database_name = database_name
        self._collection_name = collection
        self._mongo_kwargs: dict = {}
        if username:
            self._mongo_kwargs["username"] = username
            self._mongo_kwargs["password"] = password or ""
        if auth_source:
            self._mongo_kwargs["authSource"] = auth_source
        if auth_mechanism:
            self._mongo_kwargs["authMechanism"] = auth_mechanism

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _checkpoint_path(self, batch_id: int, name: str) -> str:
        """Return the base path (without extension) for a checkpoint file."""
        return os.sep.join([
            self.output_file_path, self.checkpoint_folder,
            self.id, str(batch_id), name,
        ])

    def _output_path(self, name: str) -> str:
        """Return the base path (without extension) for a final output file."""
        return os.sep.join([self.output_file_path, self.id, name])

    def _write(self, base_path: str, rows: list, columns: list) -> None:
        """Write checkpoint rows using the intermediate file format."""
        Writer.write_data(base_path, rows,
                          columns=columns, file_format=self.file_format)

    def _write_final(self, base_path: str, rows: list, columns: list) -> None:
        """Write final merged output using the final file format."""
        Writer.write_data(base_path, rows,
                          columns=columns, file_format=self.final_file_format)

    def _glob_checkpoints(self, batch_path: str, name: str) -> str:
        """Return the full path (with extension) of a checkpoint file if it exists."""
        path = os.path.join(batch_path, name + _ext(self.file_format))
        return path if os.path.exists(path) else None

    # ─────────────────────────────────────────────────────────────────────────
    # CORE PROCESSING
    # ─────────────────────────────────────────────────────────────────────────

    def process_user_tweets(self, user_id, tweets):
        """
        Process all tweets of a single user and extract node features and edges.
        """
        #########################################
        ### NODE FEATURES VARIABLES
        #########################################
        n_total = n_retweets = n_replies = n_original = 0
        n_sensitive = n_mobile = n_web = n_news_manager = n_bot_api = 0
        latest_tweet = None
        hashtags = set()
        hashtag_counter = Counter()
        timestamps = []

        ##################################################
        ### EDGE FEATURES VARIABLES
        ##################################################
        retweet_targets   = defaultdict(int)
        reply_targets     = defaultdict(int)
        mention_targets   = defaultdict(int)
        mention_ts = defaultdict(list)   # dst_screen_name -> list of timestamps
        mention_in_reply_count = defaultdict(int)   # screen_name -> count of mentions inside replies
        mention_solo_count = defaultdict(int)   # screen_name -> count of tweets where B is the ONLY mention

        reply_first_ts    = {}
        reply_last_ts     = {}
        reply_latency_sum = defaultdict(float)
        reply_latency_count = defaultdict(int)
        reply_ts_per_edge = defaultdict(list)   # uid -> list of reply timestamps (for regularity/burstiness)
        reply_hours_per_edge = defaultdict(list)  # uid -> list of reply hours (for diurnal synchronicity)

        retweet_first_ts  = {}
        retweet_last_ts   = {}
        retweet_fast_count = defaultdict(int)
        retweet_ts_per_edge = defaultdict(list)  # uid -> list of retweet timestamps (for cadence/jitter)
        retweet_hashtags_per_edge = defaultdict(lambda: Counter())  # uid -> Counter of hashtags in RTs
        FAST_RT_THRESHOLD = 30  # seconds

        seen_tweet_ids = set()

        for tweet in tweets:
            # ---- Node features ----
            tweet_id = tweet.get('id')
            if tweet_id in seen_tweet_ids:
                continue
            seen_tweet_ids.add(tweet_id)

            # Ensure created_at is a datetime object
            tweet['created_at'] = Utils.to_datetime(tweet['created_at'])

            n_total += 1
            ts = tweet['created_at'].timestamp()
            timestamps.append(ts)

            # Source classification
            source_class = classify_source(tweet.get('source', ''))
            if source_class == 'mobile':
                n_mobile += 1
            elif source_class == 'web':
                n_web += 1
            elif source_class == 'news_manager':
                n_news_manager += 1
            else:
                n_bot_api += 1

            # Sensitive flag
            if tweet.get('possibly_sensitive') is True:
                n_sensitive += 1

            is_retweet = tweet.get('retweeted_status') is not None
            is_reply   = tweet.get('in_reply_to_status_id') not in (None, -1)

            # Hashtags
            raw_ht = tweet.get('hashtagEntities', '')
            if isinstance(raw_ht, str) and raw_ht.strip():
                for ht in raw_ht.split('|'):
                    ht = ht.strip().lower()
                    if ht:
                        hashtags.add(ht)
                        hashtag_counter[ht] += 1

            # Keep latest tweet for snapshot fields (followers, verified, etc.)
            if latest_tweet is None or tweet['created_at'] > latest_tweet['created_at']:
                latest_tweet = tweet

            # ---- Edge features ----
            if is_retweet:
                n_retweets += 1
                rt_uid = tweet['retweeted_status']['user']['id']
                retweet_targets[rt_uid] += 1
                rs = tweet.get('retweeted_status') or {}
                
                retweet_first_ts[rt_uid] = min(retweet_first_ts.get(rt_uid, ts), ts)
                retweet_last_ts[rt_uid]  = max(retweet_last_ts.get(rt_uid, ts), ts)
                retweet_ts_per_edge[rt_uid].append(ts)
                
                # Topic Consistency: collect hashtags from the tweet (root usually has them even for RTs)
                # Fallback to retweeted_status if root is empty
                rt_ht_raw = tweet.get('hashtagEntities') or rs.get('hashtagEntities', '')
                if isinstance(rt_ht_raw, str) and rt_ht_raw.strip():
                    for ht in rt_ht_raw.split('|'):
                        ht = ht.strip().lower()
                        if ht:
                            retweet_hashtags_per_edge[rt_uid][ht] += 1
                # Fast-RT detection
                try:
                    orig_created = rs.get('created_at')
                    if orig_created:
                        orig_dt = Utils.to_datetime(orig_created)
                        if orig_dt:
                            latency = ts - orig_dt.timestamp()
                            if 0 <= latency <= self.fast_rt_threshold:
                                retweet_fast_count[rt_uid] += 1
                except Exception:
                    pass

            elif is_reply:
                n_replies += 1
                reply_uid = tweet.get('in_reply_to_user_id')
                if reply_uid and reply_uid != -1:
                    reply_targets[reply_uid] += 1
                    reply_first_ts[reply_uid] = min(reply_first_ts.get(reply_uid, ts), ts)
                    reply_last_ts[reply_uid]  = max(reply_last_ts.get(reply_uid, ts), ts)
                    reply_ts_per_edge[reply_uid].append(ts)
                    reply_hours_per_edge[reply_uid].append(datetime.fromtimestamp(ts, tz=timezone.utc).hour)
                    parent_created = tweet.get('in_reply_to_status_created_at')
                    if parent_created:
                        try:
                            parent_dt = Utils.to_datetime(parent_created)
                            if parent_dt:
                                lat = ts - parent_dt.timestamp()
                                if lat >= 0:
                                    reply_latency_sum[reply_uid]   += lat
                                    reply_latency_count[reply_uid] += 1
                        except Exception:
                            pass
            else:
                n_original += 1

            # Mentions — stored as screen_names (resolved later)
            if not is_retweet:
                raw_mentions = tweet.get('userMentionEntities', '')
                if isinstance(raw_mentions, str) and raw_mentions.strip():
                    mentions = [mn.strip().lower() for mn in raw_mentions.split('|') if mn.strip()]
                    if mentions:
                        # if there is only ONE mention → solo mention
                        if len(mentions) == 1:
                            mention_solo_count[mentions[0]] += 1

                        for mn in mentions:
                            mention_targets[mn] += 1
                            mention_ts[mn].append(ts)

                            # mention inside reply
                            if is_reply:
                                mention_in_reply_count[mn] += 1


        if latest_tweet is None:
            return None, [], [], []

        # ---- Node feature computation ----
        user_id_int  = int(user_id)
        src_node_id  = user_id_int
        latest_user  = latest_tweet['user']

        followers_raw = latest_user.get('followers_count', 0) or 0
        friends_raw   = latest_user.get('friends_count', 0) or 0
        den = followers_raw + friends_raw
        social_influence_ratio = followers_raw / den if den > 0 else 0.0

        # Account date: timestamp of account creation
        user_created_at = Utils.to_datetime(latest_user.get('created_at'))
        TWITTER_EPOCH = datetime(2006, 3, 21, tzinfo=timezone.utc).timestamp()
        # Instead of keeping the full timestamp (wasting space for unused dates), we start counting the account age from the TWITTER_EPOCH (March 21, 2006).
        account_date = round(user_created_at.timestamp() - TWITTER_EPOCH) if user_created_at else 0

        # profile url extraction
        user_profile_has_url = latest_user.get('url') is not None

        first_tweet_ts = min(timestamps) if timestamps else 0.0
        activation_age = 0
        if user_created_at and first_tweet_ts:
            try:
                first_dt = datetime.fromtimestamp(first_tweet_ts, tz=timezone.utc)
                uca = (user_created_at if user_created_at.tzinfo
                       else user_created_at.replace(tzinfo=timezone.utc))
                activation_age = max(0, (first_dt - uca).total_seconds())
            except Exception:
                activation_age = 0

        reg_iti = tweet_regularity_score_from_timestamps(timestamps)
        tweet_regularity_score    = reg_iti['regularity_score']
        tweet_avg_interval_seconds = reg_iti['iti_mean']

        reg_daily   = daily_posting_consistency(timestamps, window_days=90, min_days=14)
        daily_score = reg_daily['daily_score']
        daily_cv_log = (reg_daily['daily_cv_log']
                        if reg_daily['daily_cv_log'] is not None else 0.0)

        internal_density = internal_tweet_density(timestamps, window_days=90)

        profile_geo_enabled = int(latest_user.get('geo_enabled', False))

        sensitive_rate = float(n_sensitive) / n_total if n_total > 0 else 0.0
        mobile_ratio   = n_mobile   / n_total if n_total > 0 else 0.0
        web_ratio      = n_web      / n_total if n_total > 0 else 0.0
        nm_ratio       = n_news_manager / n_total if n_total > 0 else 0.0
        bot_api_ratio  = n_bot_api  / n_total if n_total > 0 else 0.0

        # Source entropy (Shannon) normalized to [0,1] using log2(4)
        try:
            counts = [n_mobile, n_web, n_news_manager, n_bot_api]
            total_c = sum(counts)
            if total_c > 0:
                probs = [c / total_c for c in counts if c > 0]
                source_entropy = (-sum(p * math.log2(p) for p in probs)
                                  / math.log2(4))
            else:
                source_entropy = 0.0
        except Exception:
            source_entropy = 0.0

        # Hashtag entropy
        try:
            total_hashtags  = sum(hashtag_counter.values())
            unique_hashtags = len(hashtag_counter)
            if total_hashtags > 0 and unique_hashtags > 1:
                raw_h = (total_hashtags * math.log2(total_hashtags)
                         - sum(c * math.log2(c)
                               for c in hashtag_counter.values() if c > 0))
                hashtag_entropy = (raw_h / total_hashtags) / math.log2(unique_hashtags)
            else:
                hashtag_entropy = 0.0
        except Exception:
            total_hashtags = 0
            unique_hashtags = len(hashtags)
            hashtag_entropy = 0.0

        def log1p(x):
            # Clamp to avoid math domain error if x <= -1
            safe_x = max(x, -0.9999)
            return round(math.log1p(safe_x), 2)

        features = {
            'user_id':      user_id_int,
            'user_node_id': src_node_id,
            # Activity
            'total':    log1p(n_total),
            'retweets': log1p(n_retweets),
            'replies':  log1p(n_replies),
            'original': log1p(n_original),
            # Profile
            'likes':          log1p(latest_user.get('favourites_count', 0)),
            'followers':      log1p(latest_user.get('followers_count', 0)),
            'following':      log1p(latest_user.get('friends_count', 0)),
            'verified':       1 if latest_user.get('verified', False) else 0,
            'account_date':   account_date,
            'listed_count':   log1p(latest_user.get('listed_count', 0)),
            'favourites_count': log1p(latest_user.get('favourites_count', 0)),
            'reputation_score': round(social_influence_ratio, 2),
            # Content
            'n_unique_hashtags': len(hashtags),
            'n_unique_mentions': len(mention_targets),
            'n_hashtags_total':  log1p(total_hashtags),
            'hashtag_entropy':   round(hashtag_entropy, 4),
            # Automation
            'activation_age':             log1p(activation_age),
            'tweet_regularity_score':     log1p(tweet_regularity_score),
            'regularity_reliable':        1 if n_total >= 20 else 0,
            'tweet_avg_interval_seconds': log1p(tweet_avg_interval_seconds),
            'daily_score':                round(daily_score, 2),
            'daily_cv_log':               round(daily_cv_log, 2),
            'internal_tweet_density':     round(internal_density, 2),
            'profile_has_url':            int(user_profile_has_url),
            'geo_enabled_flag':           profile_geo_enabled,
            # Sensitive / Source
            'sensitive_rate':    sensitive_rate,
            'mobile_ratio':      round(mobile_ratio, 4),
            'web_ratio':         round(web_ratio, 4),
            'news_manager_ratio': round(nm_ratio, 4),
            'bot_api_ratio':     round(bot_api_ratio, 4),
            'source_entropy':    round(source_entropy, 4),
            # Metadata (not a GNN feature)
            'screen_name': latest_user.get('screen_name', ''),
        }

        # ---- Edge objects ----
        def _fast_rt_ratio(dst_uid):
            count = retweet_fast_count.get(dst_uid, 0)
            total = retweet_targets.get(dst_uid, 0)
            return round(count / total if total > 0 else 0.0, 4)

        def _rt_lifespan(dst_uid):
            f = retweet_first_ts.get(dst_uid)
            l = retweet_last_ts.get(dst_uid)
            return log1p(l - f) if f is not None and l is not None and l >= f else 0.0

        def _rt_cadence_jitter(dst_uid):
            """Retweet Cadence (mean interval) and Temporal Jitter (SD of intervals)."""
            ts_list = retweet_ts_per_edge.get(dst_uid, [])
            if len(ts_list) < 2:
                return 0.0, 0.0
            intervals = np.diff(sorted(ts_list))
            cadence = log1p(float(np.mean(intervals)))
            jitter  = log1p(float(np.std(intervals)))
            return round(cadence, 4), round(jitter, 4)

        def _rt_topic_consistency(dst_uid):
            """Entropy of hashtags in retweeted content (Topic Consistency)."""
            counter = retweet_hashtags_per_edge.get(dst_uid)
            if not counter or len(counter) == 0:
                return 0.0
            total = sum(counter.values())
            probs = np.array([c / total for c in counter.values()])
            entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))
            return round(entropy, 4)

        def _avg_reply_latency(dst_uid):
            s = reply_latency_sum.get(dst_uid, 0)
            c = reply_latency_count.get(dst_uid, 1)
            return log1p(max(0, s / c if c > 0 else 0))

        def _reply_lifespan(dst_uid):
            f = reply_first_ts.get(dst_uid)
            l = reply_last_ts.get(dst_uid)
            return log1p(l - f) if f is not None and l is not None and l >= f else 0.0

        def _reply_regularity_burstiness(dst_uid):
            """Reply Regularity (SD of intervals) and Reply Burstiness Index."""
            ts_list = reply_ts_per_edge.get(dst_uid, [])
            if len(ts_list) < 2:
                return 0.0, 0.0
            intervals = np.diff(sorted(ts_list))
            if len(intervals) < 1:
                return 0.0, 0.0
            sd_i = float(np.std(intervals))
            mean_i = float(np.mean(intervals))
            regularity = log1p(sd_i)
            raw_burstiness = (sd_i - mean_i) / (sd_i + mean_i) if (sd_i + mean_i) > 0 else 0.0
            # Burstiness range is [-1, 1]. Clamp for log1p safety.
            burstiness = log1p(max(raw_burstiness, -0.9999))
            return round(regularity, 4), round(burstiness, 4)

        def _reply_diurnal_sync(dst_uid):
            """Diurnal Synchronicity: entropy of hourly distribution of replies."""
            hours = reply_hours_per_edge.get(dst_uid, [])
            if len(hours) < 2:
                return 0.0
            counts = np.bincount(hours, minlength=24).astype(float)
            total = counts.sum()
            if total == 0:
                return 0.0
            probs = counts / total
            probs = probs[probs > 0]  # filter zeros for log
            entropy = -float(np.sum(probs * np.log2(probs)))
            return round(log1p(entropy), 4)

        def _mention_lifespan(screen_name):
            """Mention Lifespan: duration between first and last mention."""
            ts_list = mention_ts.get(screen_name, [])
            if len(ts_list) < 2:
                return 0.0
            return log1p(max(ts_list) - min(ts_list))

        def _mention_metrics(screen_name):
            ts_list = mention_ts.get(screen_name, [])
            if len(ts_list) < 2:
                return 0.0, 0.0
            ts_sorted = sorted(ts_list)
            intervals = np.diff(ts_sorted)
            if len(intervals) < 1:
                return 0.0, 0.0
            
            sd_i = float(np.std(intervals))
            mean_i = float(np.mean(intervals))
            
            # Mention Regularity: log1p(SD)
            regularity = log1p(sd_i)
            
            # Burstiness Index: log1p((SD - Mean) / (SD + Mean))
            raw_burstiness = (sd_i - mean_i) / (sd_i + mean_i) if (sd_i + mean_i) > 0 else 0.0
            burstiness = log1p(max(raw_burstiness, -0.9999))
            
            return round(regularity, 4), round(burstiness, 4)

        def _mention_in_reply_ratio(screen_name):
            total = mention_targets.get(screen_name, 0)
            inside = mention_in_reply_count.get(screen_name, 0)
            return log1p(inside / total) if total > 0 else 0.0

        def _mention_solo_ratio(screen_name):
            total = mention_targets.get(screen_name, 0)
            solo  = mention_solo_count.get(screen_name, 0)
            return log1p(solo / total) if total > 0 else 0.0

        # Retweet edges
        edges_retweet = []
        for dst, weight in retweet_targets.items():
            cadence, jitter = _rt_cadence_jitter(dst)
            edges_retweet.append((
                src_node_id,
                int(dst),
                weight,
                _rt_lifespan(dst),
                _fast_rt_ratio(dst),
                cadence,
                jitter,
                _rt_topic_consistency(dst),
            ))

        # Reply edges
        edges_reply = []
        for dst, weight in reply_targets.items():
            reg, burst = _reply_regularity_burstiness(dst)
            edges_reply.append((
                src_node_id,
                int(dst),
                weight,
                _reply_lifespan(dst),
                _avg_reply_latency(dst),
                reg,
                burst,
                _reply_diurnal_sync(dst),
            ))

        # Mention edges
        mention_edges_raw = []
        for screen_name, weight in mention_targets.items():
            reg, burst = _mention_metrics(screen_name)
            mention_edges_raw.append((
                src_node_id,
                screen_name,
                weight,
                _mention_lifespan(screen_name),
                reg,
                burst,
                _mention_in_reply_ratio(screen_name),
                _mention_solo_ratio(screen_name),
            ))

        return features, edges_retweet, edges_reply, mention_edges_raw

    # ─────────────────────────────────────────────────────────────────────────
    # CHECKPOINT I/O
    # ─────────────────────────────────────────────────────────────────────────

    def save_checkpoint(self, user_features_list, edges_rt, edges_reply,
                        mention_edges_raw, screen_names_list, batch_id):
        """
        Persist intermediate results for one batch to disk.
        The file format (csv / pickle / parquet) is determined by self.file_format.
        """
        dir_path = os.path.join(
            self.output_file_path, self.checkpoint_folder,
            self.id, str(batch_id)
        )
        os.makedirs(dir_path, exist_ok=True)

        def _base(name):
            return os.path.join(dir_path, name)

        features_rows = [
            [
                f['user_node_id'], f['total'], f['retweets'], f['replies'],
                f['original'], f['likes'], f['followers'], f['following'],
                f['verified'], f['account_date'], f['listed_count'],
                f['favourites_count'], f['reputation_score'],
                f['n_unique_hashtags'], f['n_unique_mentions'],
                f['n_hashtags_total'], f['hashtag_entropy'],
                f['activation_age'], f['tweet_regularity_score'],
                f['regularity_reliable'], f['tweet_avg_interval_seconds'],
                f['daily_score'], f['daily_cv_log'], f['internal_tweet_density'],
                f['profile_has_url'], f['geo_enabled_flag'], f['sensitive_rate'],
                f['mobile_ratio'], f['web_ratio'], f['news_manager_ratio'],
                f['bot_api_ratio'], f['source_entropy'], f.get('community', -1)
            ]
            for f in user_features_list
        ]

        self._write(_base("user_features"),    features_rows,     USER_FEATURES_COLUMNS)
        self._write(_base("edges_retweet"),    edges_rt,          EDGE_RETWEET_COLUMNS)
        self._write(_base("edges_reply"),      edges_reply,       EDGE_REPLY_COLUMNS)
        self._write(_base("edges_mention_raw"),
                    mention_edges_raw,
                    EDGE_MENTION_COLUMNS)
        self._write(_base("screen_name_map"),  screen_names_list, SCREEN_NAME_COLUMNS)

    # ─────────────────────────────────────────────────────────────────────────
    # MERGE
    # ─────────────────────────────────────────────────────────────────────────

    def merge_checkpoints(self):
        """
        Merge all batch checkpoints into final output files.
        Resolves mention screen_names to user_ids and injects in-degree counts.
        """
        checkpoint_dir = os.path.join(
            self.output_file_path, self.checkpoint_folder, self.id
        )
        out_dir = os.path.join(self.output_file_path, self.id)
        os.makedirs(out_dir, exist_ok=True)

        ext = _ext(self.file_format)

        if not os.path.exists(checkpoint_dir):
            self.logger.warning(f"Checkpoint directory {checkpoint_dir} not found. Nothing to merge.")
            return

        self.logger.info("Merging checkpoints...")

        all_features, all_rt, all_reply, all_mention = [], [], [], []
        screen_name_map    = {}
        valid_user_node_ids = set()

        # ── Pass 1: build screen_name map ────────────────────────────────────
        self.logger.info("Building screen_name map...")
        for batch_dir in sorted(os.listdir(checkpoint_dir)):
            batch_path = os.sep.join([checkpoint_dir, batch_dir])
            if not os.path.isdir(batch_path):
                continue
            map_file = os.path.join(batch_path, "screen_name_map" + ext)
            if os.path.exists(map_file):
                for row in Writer.load_checkpoint_file(map_file):
                    sn  = row[0].lower()
                    uid = int(row[1])
                    screen_name_map[sn] = uid
                    valid_user_node_ids.add(uid)

        resolved_mention = dropped_mention = dropped_rt = dropped_reply = 0
        received_retweets  = defaultdict(int)
        received_replies   = defaultdict(int)
        received_mentions  = defaultdict(int)

        # ── Pass 2: load, filter, resolve ────────────────────────────────────
        self.logger.info("Filtering and resolving edges...")
        for batch_dir in sorted(os.listdir(checkpoint_dir)):
            batch_path = os.sep.join([checkpoint_dir, batch_dir])
            if not os.path.isdir(batch_path):
                continue

            feat_file = os.path.join(batch_path, "user_features" + ext)
            if os.path.exists(feat_file):
                all_features.extend(Writer.load_checkpoint_file(feat_file))

            rt_file = os.path.join(batch_path, "edges_retweet" + ext)
            if os.path.exists(rt_file):
                for row in Writer.load_checkpoint_file(rt_file):
                    dst = int(row[1])
                    if dst in valid_user_node_ids:
                        all_rt.append(row)
                        received_retweets[dst] += int(float(row[2]))
                    else:
                        dropped_rt += 1

            rep_file = os.path.join(batch_path, "edges_reply" + ext)
            if os.path.exists(rep_file):
                for row in Writer.load_checkpoint_file(rep_file):
                    dst = int(row[1])
                    if dst in valid_user_node_ids:
                        all_reply.append(row)
                        received_replies[dst] += int(float(row[2]))
                    else:
                        dropped_reply += 1

            men_file = os.path.join(batch_path, "edges_mention_raw" + ext)
            if os.path.exists(men_file):
                for row in Writer.load_checkpoint_file(men_file):
                    # row: [src, screen_name, weight, lifespan, regularity, burstiness, in_reply_ratio, solo_ratio]
                    src, sn, weight = row[0], row[1], row[2]
                    metrics = row[3:] 
                    
                    dst = screen_name_map.get(sn.lower())
                    if dst is not None:
                        all_mention.append((src, dst, weight, *metrics))
                        received_mentions[dst] += int(float(weight))
                        resolved_mention += 1
                    else:
                        dropped_mention += 1

        # ── Inject in-degree counts into feature rows ─────────────────────────
        for row in all_features:
            uid = row[0]  # user_node_id is first column
            row.append(received_retweets.get(uid, 0))
            row.append(received_replies.get(uid, 0))
            row.append(received_mentions.get(uid, 0))

        # ── Write final files (using final_file_format) ───────────────────────
        def _out(name):
            return os.path.join(out_dir, name)

        # For CSV: write header first, then data (append mode)
        if self.final_file_format == "csv":
            Writer.write_on_csv(_out("user_features.csv"),
                                [USER_FEATURES_COLUMNS_FINAL])
            Writer.write_on_csv(_out("edges_retweet.csv"),  [EDGE_RETWEET_COLUMNS])
            Writer.write_on_csv(_out("edges_reply.csv"),    [EDGE_REPLY_COLUMNS])
            Writer.write_on_csv(_out("edges_mention.csv"),  [EDGE_MENTION_COLUMNS])
            Writer.write_on_csv(_out("screen_name_map.csv"), [SCREEN_NAME_COLUMNS])
            Writer.write_on_csv(_out("user_features.csv"),  all_features)
            Writer.write_on_csv(_out("edges_retweet.csv"),  all_rt)
            Writer.write_on_csv(_out("edges_reply.csv"),    all_reply)
            Writer.write_on_csv(_out("edges_mention.csv"),  all_mention)
            map_rows = list(screen_name_map.items())
            Writer.write_on_csv(_out("screen_name_map.csv"), map_rows)
        else:
            # pickle / parquet: columns are stored in the file metadata
            self._write_final(_out("user_features"),
                              all_features, USER_FEATURES_COLUMNS_FINAL)
            self._write_final(_out("edges_retweet"),  all_rt,     EDGE_RETWEET_COLUMNS)
            self._write_final(_out("edges_reply"),    all_reply,  EDGE_REPLY_COLUMNS)
            self._write_final(_out("edges_mention"),  all_mention, EDGE_MENTION_COLUMNS)
            map_rows = list(screen_name_map.items())
            self._write_final(_out("screen_name_map"), map_rows,  SCREEN_NAME_COLUMNS)

        # ── Metadata ─────────────────────────────────────────────────────────
        filename = os.path.basename(self.community_file_path) if self.is_community_run else None
        metadata = {
            "collection":           self.get_collection(),
            "date":                 datetime.now(timezone.utc).isoformat(),
            "run_id":               self.id,
            "intermediate_format":  self.file_format,
            "final_format":         self.final_file_format,
            "users_processed":      len(valid_user_node_ids),
            "communities":          filename,
        }
        with open(os.path.join(out_dir, "metadata.json"), "w",
                  encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        if self.delete_tmp_after_merge:
            self.logger.info(f"Deleting tmp: {checkpoint_dir}")
            try:
                shutil.rmtree(checkpoint_dir)
            except Exception as e:
                self.logger.warning(f"Failed to delete {checkpoint_dir}: {e}")

        self.logger.info(
            f"Merge complete. Output in {out_dir}\n"
            f"  Mentions : {resolved_mention} resolved, {dropped_mention} dropped.\n"
            f"  Retweets : {len(all_rt)} kept, {dropped_rt} dropped.\n"
            f"  Replies  : {len(all_reply)} kept, {dropped_reply} dropped."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PARALLEL WORKER
    # ─────────────────────────────────────────────────────────────────────────

    def _process_work_item(self, worker_id: int, work_item,
                           checkpoint_every: int, max_users: Optional[int],
                           total_users_counter: Optional[list] = None,
                           total_users_lock: Optional[threading.Lock] = None) -> int:
        client      = MongoClient(self._uri, **self._mongo_kwargs)
        collection  = client[self._database_name][self._collection_name]

        batch_offset = worker_id * 1_000_000
        local_batch  = 0
        features_list, edges_rt, edges_reply, mention_raw, screen_names_list = \
            [], [], [], [], []
        user_count = 0

        try:
            for user_id, user_tweets in self.strategy.iter_users(collection, work_item):
                features, rt, rep, men = self.process_user_tweets(user_id, user_tweets)
                if features is None:
                    continue
                features['community'] = self.strategy.get_community_id(features['user_id'])

                sn = features.get('screen_name', '')
                if sn:
                    screen_names_list.append((sn.lower(), features['user_id']))

                features_list.append(features)
                edges_rt.extend(rt)
                edges_reply.extend(rep)
                mention_raw.extend(men)
                user_count += 1

                if user_count % checkpoint_every == 0:
                    bid = batch_offset + local_batch
                    # Update and read the global counter
                    global_total = user_count  # fallback if no shared counter
                    if total_users_counter is not None and total_users_lock is not None:
                        with total_users_lock:
                            total_users_counter[0] += checkpoint_every
                            global_total = total_users_counter[0]
                    self.logger.info(
                        f"[Worker {worker_id} | Batch {bid}] {user_count} users | "
                        f"{len(edges_rt)} RT | {len(edges_reply)} reply | "
                        f"{len(mention_raw)} mention "
                        f"[total: {global_total:,} users]"
                    )
                    self.save_checkpoint(features_list, edges_rt, edges_reply,
                                         mention_raw, screen_names_list, bid)
                    features_list, edges_rt, edges_reply, mention_raw, screen_names_list = \
                        [], [], [], [], []
                    local_batch += 1

                if max_users is not None and user_count >= max_users:
                    self.logger.info(f"[Worker {worker_id}] Reached max_users={max_users}. Stopping.")
                    break

            if features_list:
                self.save_checkpoint(features_list, edges_rt, edges_reply,
                                     mention_raw, screen_names_list, batch_offset + local_batch)
        finally:
            client.close()

        return user_count

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, checkpoint_every: int = 500, max_users: Optional[int] = None,
            n_workers: int = 4):
        self.logger.info(
            f"[GraphGenerationUser] Starting extraction. "
            f"Run ID: {self.id}  Format: {self.file_format}"
        )
        run_start = time.time()

        client = MongoClient(self._uri, **self._mongo_kwargs)
        collection = client[self._database_name][self._collection_name]
        
        work_items = self.strategy.partition_work(collection, n_workers, max_users, self.logger)
        client.close()

        if not work_items:
            self.logger.warning("No work to process.")
            return

        per_worker_max = (
            (max_users + len(work_items) - 1) // len(work_items)
            if max_users is not None and len(work_items) > 0 else None
        )

        self.logger.info(
            f"Dispatching {len(work_items)} task(s) to {n_workers} thread(s) "
            f"(checkpoint every {checkpoint_every} users per task)..."
        )

        total_users = 0
        _total_users_counter = [0]              # shared mutable counter
        _total_users_lock    = threading.Lock()
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    self._process_work_item,
                    task_id, work_item,
                    checkpoint_every, per_worker_max,
                    _total_users_counter, _total_users_lock,
                ): task_id
                for task_id, work_item in enumerate(work_items)
            }
            for future in as_completed(futures):
                worker_id = futures[future]
                try:
                    count = future.result()
                    total_users += count
                    self.logger.info(
                        f"[Worker {worker_id}] finished — {count} users processed."
                    )
                except Exception as exc:
                    self.logger.error(
                        f"[Worker {worker_id}] failed with exception: {exc}",
                        exc_info=True,
                    )

        # ── Step 4: merge all checkpoints into final output ───────────────────
        extraction_elapsed = time.time() - run_start
        extr_h, extr_rem = divmod(extraction_elapsed, 3600)
        extr_m, extr_s = divmod(extr_rem, 60)
        self.logger.info(
            f"{total_users} users processed in total in "
            f"{int(extr_h):02}:{int(extr_m):02}:{int(extr_s):02}. "
            f"Starting merge (checkpoints: {self.file_format} -> final: {self.final_file_format})..."
        )
        self.merge_checkpoints()
        self.logger.info("Extraction complete.")