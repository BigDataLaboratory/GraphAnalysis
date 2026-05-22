# algorithms/graph/orchestrator.py
# Core orchestrator class for graph generation.
# Preserves all features (strategies, snapshot load, community runs) in a modular way.
# Keep comments in English as requested.

import os
import time
import math
import uuid
import json
import logging
import threading
from collections import defaultdict, Counter
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any, Optional

from pymongo import MongoClient

from Utils.Utils import Utils
from Utils.Writer import Writer, _ext
from algorithms.MongoConnection import MongoConnection

from algorithms.graph.graph_constants import (
    SUPPORTED_FORMATS,
    USER_FEATURES_COLUMNS,
    USER_FEATURES_COLUMNS_FINAL,
    EDGE_RETWEET_COLUMNS,
    EDGE_REPLY_COLUMNS,
    EDGE_MENTION_COLUMNS,
    SCREEN_NAME_COLUMNS
)
from algorithms.graph import node_feature_extractors as feat
from algorithms.graph import retweet_reply_edge_extractors as rm
from algorithms.graph import mention_edge_extractors as mm
from algorithms.graph import checkpoint_io as cp_io


class GraphGenerationUser(MongoConnection):
    """
    Orchestrates user-level feature extraction and checkpoint writing.
    Delegates metrics calculations and I/O tasks to specialized modules.
    """

    logger = logging.getLogger("GraphGenerationUser")

    def __init__(self, uri: str, database_name: str, collection: str, output_file_path: str,
                 username: Optional[str] = None, password: Optional[str] = None,
                 auth_source: Optional[str] = None, auth_mechanism: Optional[str] = None,
                 delete_tmp_after_merge: bool = False, intermediate_file_format: str = "feather",
                 final_file_format: str = "parquet", file_format: Optional[str] = None,
                 fast_rt_threshold: Optional[int] = None, strategy: Any = None,
                 is_community_run: bool = False, community_file_path: Optional[str] = None,
                 load_snapshot_status: bool = False, load_snapshot_tmp_path: str = ""):

        # Legacy compatibility: if file_format provided, use it for both
        if file_format is not None:
            intermediate_file_format = file_format
            final_file_format = file_format

        # Validate formats
        for fmt in (intermediate_file_format, final_file_format):
            if fmt not in SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported file_format '{fmt}'. Choose from {SUPPORTED_FORMATS}."
                )

        super().__init__(uri, username, password, auth_source, auth_mechanism,
                         db=None, database_name=database_name, collection=collection)

        self.output_file_path = output_file_path
        self.id = uuid.uuid1().hex
        self.checkpoint_folder = "tmp"
        self.delete_tmp_after_merge = delete_tmp_after_merge
        self.file_format = intermediate_file_format
        self.final_file_format = final_file_format
        self.fast_rt_threshold = fast_rt_threshold or 30

        self.strategy = strategy
        self.is_community_run = is_community_run
        self.community_file_path = community_file_path
        self.load_snapshot_status = load_snapshot_status
        self.load_snapshot_tmp_path = load_snapshot_tmp_path

        # store for worker threads
        self._uri = uri
        self._database_name = database_name
        self._collection_name = collection
        self._mongo_kwargs = {}
        if username:
            self._mongo_kwargs["username"] = username
            self._mongo_kwargs["password"] = password or ""
        if auth_source:
            self._mongo_kwargs["authSource"] = auth_source
        if auth_mechanism:
            self._mongo_kwargs["authMechanism"] = auth_mechanism

    def process_user_tweets(self, user_id: Any, tweets: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[Tuple], List[Tuple], List[Tuple]]:
        """
        Process all tweets of a single user and extract node features and edges.
        """
        if not tweets:
            return None, [], [], []

        user_id_int = int(user_id)

        # 1. State Variables / Accumulators
        n_total = n_retweets = n_replies = n_original = 0
        n_sensitive = 0
        n_mobile = n_web = n_news_manager = n_bot_api = 0

        latest_tweet = None
        hashtags = set()
        hashtag_counter = Counter()
        timestamps = []

        retweet_targets = defaultdict(int)
        reply_targets = defaultdict(int)
        mention_targets = defaultdict(int)
        mention_ts = defaultdict(list)
        mention_in_reply_count = defaultdict(int)
        mention_solo_count = defaultdict(int)

        reply_first_ts = {}
        reply_last_ts = {}
        reply_latency_sum = defaultdict(float)
        reply_latency_count = defaultdict(int)
        reply_ts_per_edge = defaultdict(list)
        reply_hours_per_edge = defaultdict(list)

        retweet_first_ts = {}
        retweet_last_ts = {}
        retweet_fast_count = defaultdict(int)
        retweet_ts_per_edge = defaultdict(list)
        retweet_hashtags_per_edge = defaultdict(lambda: Counter())

        seen_tweet_ids = set()

        # 2. Iterative aggregation over tweets
        for tweet in tweets:
            tweet_id = tweet.get('id')
            if tweet_id in seen_tweet_ids:
                continue
            seen_tweet_ids.add(tweet_id)

            tweet['created_at'] = Utils.to_datetime(tweet['created_at'])
            if not tweet['created_at']:
                continue

            n_total += 1
            ts = tweet['created_at'].timestamp()
            timestamps.append(ts)

            # Source classification
            source_class = feat.classify_source(tweet.get('source', ''))
            if source_class == 'mobile':
                n_mobile += 1
            elif source_class == 'web':
                n_web += 1
            elif source_class == 'news_manager':
                n_news_manager += 1
            else:
                n_bot_api += 1

            # Sensitive rate
            if tweet.get('possibly_sensitive') is True:
                n_sensitive += 1

            # Hashtags
            raw_ht = tweet.get('hashtagEntities', '')
            if isinstance(raw_ht, str) and raw_ht.strip():
                for ht in raw_ht.split('|'):
                    h = ht.strip().lower()
                    if h:
                        hashtags.add(h)
                        hashtag_counter[h] += 1

            # Track latest tweet
            if not latest_tweet or tweet['created_at'] > latest_tweet['created_at']:
                latest_tweet = tweet

            is_retweet = tweet.get('retweeted_status') is not None
            is_reply = tweet.get('in_reply_to_status_id') not in (None, -1)

            if is_retweet:
                n_retweets += 1
                rs = tweet.get('retweeted_status') or {}
                dst_uid = rs.get('user', {}).get('id') or rs.get('user', {}).get('id_str')
                if dst_uid is not None:
                    dst_uid = int(dst_uid)
                    retweet_targets[dst_uid] += 1
                    retweet_first_ts[dst_uid] = min(retweet_first_ts.get(dst_uid, ts), ts)
                    retweet_last_ts[dst_uid] = max(retweet_last_ts.get(dst_uid, ts), ts)
                    retweet_ts_per_edge[dst_uid].append(ts)

                    # topic consistency
                    rt_ht_raw = tweet.get('hashtagEntities') or rs.get('hashtagEntities', '')
                    if isinstance(rt_ht_raw, str) and rt_ht_raw.strip():
                        for ht in rt_ht_raw.split('|'):
                            h = ht.strip().lower()
                            if h:
                                retweet_hashtags_per_edge[dst_uid][h] += 1

                    # fast RT detection
                    try:
                        orig_created = rs.get('created_at')
                        if orig_created:
                            orig_dt = Utils.to_datetime(orig_created)
                            if orig_dt:
                                latency = ts - orig_dt.timestamp()
                                if 0 <= latency <= self.fast_rt_threshold:
                                    retweet_fast_count[dst_uid] += 1
                    except Exception:
                        pass
            elif is_reply:
                n_replies += 1
                reply_uid = tweet.get('in_reply_to_user_id')
                if reply_uid and reply_uid != -1:
                    reply_uid = int(reply_uid)
                    reply_targets[reply_uid] += 1
                    reply_first_ts[reply_uid] = min(reply_first_ts.get(reply_uid, ts), ts)
                    reply_last_ts[reply_uid] = max(reply_last_ts.get(reply_uid, ts), ts)
                    reply_ts_per_edge[reply_uid].append(ts)
                    reply_hours_per_edge[reply_uid].append(tweet['created_at'].hour)

                    parent_created = tweet.get('in_reply_to_status_created_at')
                    if parent_created:
                        try:
                            parent_dt = Utils.to_datetime(parent_created)
                            if parent_dt:
                                lat = ts - parent_dt.timestamp()
                                if lat >= 0:
                                    reply_latency_sum[reply_uid] += lat
                                    reply_latency_count[reply_uid] += 1
                        except Exception:
                            pass
            else:
                n_original += 1

            # Mentions (excluding retweets)
            if not is_retweet:
                raw_mentions = tweet.get('userMentionEntities', '')
                if isinstance(raw_mentions, str) and raw_mentions.strip():
                    mentions = [mn.strip().lower() for mn in raw_mentions.split('|') if mn.strip()]
                    if mentions:
                        if len(mentions) == 1:
                            mention_solo_count[mentions[0]] += 1
                        for mn in mentions:
                            mention_targets[mn] += 1
                            mention_ts[mn].append(ts)
                            if is_reply:
                                mention_in_reply_count[mn] += 1

        if not latest_tweet:
            return None, [], [], []

        src_node_id = user_id_int
        latest_user = latest_tweet['user']

        followers_raw = latest_user.get('followers_count', 0) or 0
        friends_raw = latest_user.get('friends_count', 0) or 0
        den = followers_raw + friends_raw
        social_influence_ratio = followers_raw / den if den > 0 else 0.0

        user_created_at = Utils.to_datetime(latest_user.get('created_at'))
        TWITTER_EPOCH = datetime(2006, 3, 21, tzinfo=timezone.utc).timestamp()
        account_date = round(user_created_at.timestamp() - TWITTER_EPOCH) if user_created_at else 0

        user_profile_has_url = latest_user.get('url') is not None

        first_tweet_ts = min(timestamps) if timestamps else 0.0
        activation_age = 0
        if user_created_at and first_tweet_ts:
            try:
                first_dt = datetime.fromtimestamp(first_tweet_ts, tz=timezone.utc)
                uca = user_created_at if user_created_at.tzinfo else user_created_at.replace(tzinfo=timezone.utc)
                activation_age = max(0, (first_dt - uca).total_seconds())
            except Exception:
                activation_age = 0

        reg_iti = feat.tweet_regularity_score_from_timestamps(timestamps)
        tweet_regularity_score = reg_iti['regularity_score']
        tweet_avg_interval_seconds = reg_iti['iti_mean']

        reg_daily = feat.daily_posting_consistency(timestamps, window_days=90, min_days=14)
        daily_score = reg_daily['daily_score']
        daily_cv_log = reg_daily['daily_cv_log'] if reg_daily['daily_cv_log'] is not None else 0.0

        internal_density = feat.internal_tweet_density(timestamps, window_days=90)
        profile_geo_enabled = int(latest_user.get('geo_enabled', False))

        sensitive_rate = float(n_sensitive) / n_total if n_total > 0 else 0.0
        mobile_ratio = n_mobile / n_total if n_total > 0 else 0.0
        web_ratio = n_web / n_total if n_total > 0 else 0.0
        nm_ratio = n_news_manager / n_total if n_total > 0 else 0.0
        bot_api_ratio = n_bot_api / n_total if n_total > 0 else 0.0

        # Source entropy
        try:
            counts = [n_mobile, n_web, n_news_manager, n_bot_api]
            total_c = sum(counts)
            if total_c > 0:
                probs = [c / total_c for c in counts if c > 0]
                source_entropy = -sum(p * math.log2(p) for p in probs) / math.log2(4)
            else:
                source_entropy = 0.0
        except Exception:
            source_entropy = 0.0

        # Hashtag entropy
        try:
            total_hashtags = sum(hashtag_counter.values())
            unique_hashtags = len(hashtag_counter)
            if total_hashtags > 0 and unique_hashtags > 1:
                raw_h = total_hashtags * math.log2(total_hashtags) - sum(c * math.log2(c) for c in hashtag_counter.values() if c > 0)
                hashtag_entropy = (raw_h / total_hashtags) / math.log2(unique_hashtags)
            else:
                hashtag_entropy = 0.0
        except Exception:
            total_hashtags = 0
            hashtag_entropy = 0.0

        def log1p(x):
            return round(math.log1p(max(x, -0.9999)), 2)

        features = {
            'user_id': user_id_int,
            'user_node_id': src_node_id,
            # Activity
            'total': log1p(n_total),
            'retweets': log1p(n_retweets),
            'replies': log1p(n_replies),
            'original': log1p(n_original),
            # Profile
            'likes': log1p(latest_user.get('favourites_count', 0)),
            'followers': log1p(latest_user.get('followers_count', 0)),
            'following': log1p(latest_user.get('friends_count', 0)),
            'verified': 1 if latest_user.get('verified', False) else 0,
            'account_date': account_date,
            'listed_count': log1p(latest_user.get('listed_count', 0)),
            'favourites_count': log1p(latest_user.get('favourites_count', 0)),
            'reputation_score': round(social_influence_ratio, 2),
            # Content
            'n_unique_hashtags': len(hashtags),
            'n_unique_mentions': len(mention_targets),
            'n_hashtags_total': log1p(total_hashtags),
            'hashtag_entropy': round(hashtag_entropy, 4),
            # Automation
            'activation_age': log1p(activation_age),
            'tweet_regularity_score': log1p(tweet_regularity_score),
            'regularity_reliable': 1 if n_total >= 20 else 0,
            'tweet_avg_interval_seconds': log1p(tweet_avg_interval_seconds),
            'daily_score': round(daily_score, 2),
            'daily_cv_log': round(daily_cv_log, 2),
            'internal_tweet_density': round(internal_density, 2),
            'profile_has_url': int(user_profile_has_url),
            'geo_enabled_flag': profile_geo_enabled,
            # Sensitive / Source
            'sensitive_rate': sensitive_rate,
            'mobile_ratio': round(mobile_ratio, 4),
            'web_ratio': round(web_ratio, 4),
            'news_manager_ratio': round(nm_ratio, 4),
            'bot_api_ratio': round(bot_api_ratio, 4),
            'source_entropy': round(source_entropy, 4),
            # Metadata
            'screen_name': latest_user.get('screen_name', ''),
        }

        # 3. Build edges using edge metrics modules
        edges_retweet = []
        for dst, weight in retweet_targets.items():
            ts_list = retweet_ts_per_edge.get(dst, [])
            cadence, jitter = rm.rt_cadence_jitter(ts_list)
            lifespan = rm.lifespan_from_ts(ts_list)
            fast_ratio = rm.fast_rt_ratio(retweet_fast_count.get(dst, 0), weight)
            topic_cons = rm.rt_topic_consistency(retweet_hashtags_per_edge.get(dst, {}))
            edges_retweet.append((
                src_node_id,
                int(dst),
                weight,
                lifespan,
                fast_ratio,
                cadence,
                jitter,
                topic_cons,
            ))

        edges_reply = []
        for dst, weight in reply_targets.items():
            ts_list = reply_ts_per_edge.get(dst, [])
            reg, burst = rm.reply_regularity_burstiness(ts_list)
            lifespan = rm.lifespan_from_ts(ts_list)
            avg_lat = rm.avg_reply_latency(reply_latency_sum.get(dst, 0.0), reply_latency_count.get(dst, 0))
            diurnal = rm.reply_diurnal_sync(reply_hours_per_edge.get(dst, []))
            edges_reply.append((
                src_node_id,
                int(dst),
                weight,
                lifespan,
                avg_lat,
                reg,
                burst,
                diurnal,
            ))

        mention_state = {
            "mention_targets": mention_targets,
            "mention_ts": mention_ts,
            "mention_in_reply_count": mention_in_reply_count,
            "mention_solo_count": mention_solo_count,
        }
        mention_edges_raw = mm.build_mention_edges(mention_state, src_node_id)

        return features, edges_retweet, edges_reply, mention_edges_raw

    def save_checkpoint(self, user_features_list: List[Dict], edges_rt: List[Tuple],
                        edges_reply: List[Tuple], mention_edges_raw: List[Tuple],
                        screen_names_list: List[Tuple], batch_id: int) -> None:
        cp_io.save_checkpoint(
            output_root=self.output_file_path,
            checkpoint_folder=self.checkpoint_folder,
            run_id=self.id,
            file_format=self.file_format,
            user_features_list=user_features_list,
            edges_rt=edges_rt,
            edges_reply=edges_reply,
            mention_edges_raw=mention_edges_raw,
            screen_names_list=screen_names_list,
            batch_id=batch_id,
            load_snapshot_status=self.load_snapshot_status,
            load_snapshot_tmp_path=self.load_snapshot_tmp_path
        )

    def merge_checkpoints(self) -> None:
        cp_io.merge_checkpoints(
            output_root=self.output_file_path,
            checkpoint_folder=self.checkpoint_folder,
            run_id=self.id,
            file_format=self.file_format,
            final_file_format=self.final_file_format,
            delete_tmp_after_merge=self.delete_tmp_after_merge,
            is_community_run=self.is_community_run,
            community_file_path=self.community_file_path,
            load_snapshot_status=self.load_snapshot_status,
            load_snapshot_tmp_path=self.load_snapshot_tmp_path,
            logger=self.logger
        )

    def _process_work_item(self, worker_id: int, work_item: Any,
                            checkpoint_every: int, max_users: Optional[int],
                            total_users_counter: Optional[list] = None,
                            total_users_lock: Optional[threading.Lock] = None) -> int:
        client = MongoClient(self._uri, **self._mongo_kwargs)
        collection = client[self._database_name][self._collection_name]

        batch_offset = worker_id * 1_000_000 + int(time.time() % 10000) * 1000
        local_batch = 0
        features_list, edges_rt, edges_reply, mention_raw, screen_names_list = [], [], [], [], []
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
                    global_total = user_count
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
                    self.save_checkpoint(features_list, edges_rt, edges_reply, mention_raw, screen_names_list, bid)
                    features_list, edges_rt, edges_reply, mention_raw, screen_names_list = [], [], [], [], []
                    local_batch += 1

                if max_users is not None and user_count >= max_users:
                    self.logger.info(f"[Worker {worker_id}] Reached max_users={max_users}. Stopping.")
                    break

            if features_list:
                self.save_checkpoint(features_list, edges_rt, edges_reply, mention_raw, screen_names_list, batch_offset + local_batch)
        finally:
            client.close()

        return user_count

    def _get_processed_users_from_snapshot(self) -> set:
        processed_users = set()
        ext = _ext(self.file_format)
        if not os.path.exists(self.load_snapshot_tmp_path):
            return processed_users

        for batch_dir in os.listdir(self.load_snapshot_tmp_path):
            batch_path = os.sep.join([self.load_snapshot_tmp_path, batch_dir])
            if not os.path.isdir(batch_path):
                continue

            feat_file = os.path.join(batch_path, "user_features" + ext)
            if os.path.exists(feat_file):
                for row in Writer.load_checkpoint_file(feat_file):
                    processed_users.add(int(row[0]))
        return processed_users

    def run(self, checkpoint_every: int = 500, max_users: Optional[int] = None, n_workers: int = 4) -> None:
        self.logger.info(
            f"[GraphGenerationUser] Starting extraction. "
            f"Run ID: {self.id}  Format: {self.file_format}"
        )
        run_start = time.time()

        client = MongoClient(self._uri, **self._mongo_kwargs)
        collection = client[self._database_name][self._collection_name]

        if self.load_snapshot_status and self.load_snapshot_tmp_path:
            self.logger.info(f"Loading snapshot from {self.load_snapshot_tmp_path} to resume extraction.")
            processed_users = self._get_processed_users_from_snapshot()
            self.logger.info(f"Found {len(processed_users)} already processed users in snapshot.")
            skipped = self.strategy.exclude_users(processed_users)
            self.logger.info(f"Excluded {skipped} users from strategy processing queue.")
            snapshot_basename = os.path.basename(self.load_snapshot_tmp_path.rstrip(os.sep))
            if snapshot_basename:
                self.id = snapshot_basename

        work_items = self.strategy.partition_work(collection, n_workers, max_users, self.logger)
        client.close()

        if not work_items:
            self.logger.warning("No work to process. Proceeding to merge (if any).")
            total_users = 0
        else:
            per_worker_max = (
                (max_users + len(work_items) - 1) // len(work_items)
                if max_users is not None and len(work_items) > 0 else None
            )

            self.logger.info(
                f"Dispatching {len(work_items)} task(s) to {n_workers} thread(s) "
                f"(checkpoint every {checkpoint_every} users per task)..."
            )

            total_users = 0
            _total_users_counter = [0]
            _total_users_lock = threading.Lock()
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