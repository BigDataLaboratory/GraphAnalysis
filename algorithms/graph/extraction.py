# algorithms/graph/extraction.py
# GraphGenerationUser — thin orchestrator class.
#
# Responsibilities:
#   - Configure MongoDB connections and file format options
#   - Delegate tweet parsing to tweet_parser
#   - Delegate feature computation to node_features / edge_features
#   - Delegate I/O to checkpoint
#   - Manage the parallel worker pool and the run() entry point

import math
import os
import time
import uuid
import logging
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from pymongo import MongoClient

from Utils.Utils import Utils
from Utils.Writer import _ext
from algorithms.MongoConnection import MongoConnection

from algorithms.graph.output_structure import SUPPORTED_FORMATS
from algorithms.graph.tweet_parser import init_accumulators, parse_tweet
from algorithms.graph import node_features as nf
from algorithms.graph import edge_features as ef
from algorithms.graph import checkpoint as cp


class GraphGenerationUser(MongoConnection):
    """
    Orchestrates user-level graph feature extraction from MongoDB tweets.

    The heavy lifting is delegated to:
      - tweet_parser   → per-tweet accumulation
      - node_features  → user-level metrics
      - edge_features  → edge-level metrics
      - checkpoint     → file I/O and merge
    """

    logger = logging.getLogger("GraphGenerationUser")

    def __init__(self, uri: str, database_name: str, collection: str,
                 output_file_path: str,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 auth_source: Optional[str] = None,
                 auth_mechanism: Optional[str] = None,
                 delete_tmp_after_merge: bool = False,
                 intermediate_file_format: str = "feather",
                 final_file_format: str = "parquet",
                 file_format: Optional[str] = None,
                 fast_rt_threshold: Optional[int] = None,
                 strategy: Any = None,
                 is_community_run: bool = False,
                 community_file_path: Optional[str] = None,
                 load_snapshot_status: bool = False,
                 load_snapshot_tmp_path: str = ""):

        # Legacy compat: single file_format overrides both
        if file_format is not None:
            intermediate_file_format = file_format
            final_file_format = file_format

        for fmt in (intermediate_file_format, final_file_format):
            if fmt not in SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported file_format '{fmt}'. "
                    f"Choose from {SUPPORTED_FORMATS}.")

        super().__init__(uri, username, password, auth_source, auth_mechanism,
                         db=None, database_name=database_name,
                         collection=collection)

        self.output_file_path      = output_file_path
        self.id                    = uuid.uuid1().hex
        self.checkpoint_folder     = "tmp"
        self.delete_tmp_after_merge = delete_tmp_after_merge
        self.file_format           = intermediate_file_format
        self.final_file_format     = final_file_format
        self.fast_rt_threshold     = fast_rt_threshold or 30

        self.strategy              = strategy
        self.is_community_run      = is_community_run
        self.community_file_path   = community_file_path
        self.load_snapshot_status   = load_snapshot_status
        self.load_snapshot_tmp_path = load_snapshot_tmp_path

        # Stored for spawning per-worker MongoDB connections
        self._uri             = uri
        self._database_name   = database_name
        self._collection_name = collection
        self._mongo_kwargs    = {}
        if username:
            self._mongo_kwargs["username"] = username
            self._mongo_kwargs["password"] = password or ""
        if auth_source:
            self._mongo_kwargs["authSource"] = auth_source
        if auth_mechanism:
            self._mongo_kwargs["authMechanism"] = auth_mechanism

    # ─────────────────────────────────────────────────────────────────────
    # Core processing: tweets → (features_dict, edges_rt, edges_rep, edges_men)
    # ─────────────────────────────────────────────────────────────────────

    def process_user_tweets(self, user_id, tweets):
        """Process all tweets of a single user and extract features + edges."""
        if not tweets:
            return None, [], [], []

        user_id_int = int(user_id)

        # 1. Accumulate per-tweet state
        acc = init_accumulators()
        for tweet in tweets:
            parse_tweet(acc, tweet, self.fast_rt_threshold)

        if not acc["latest_tweet"]:
            return None, [], [], []

        # 2. Build node features from accumulated state
        features = self._build_node_features(user_id_int, acc)

        # 3. Build edge tuples
        src = user_id_int
        edges_rt  = self._build_retweet_edges(src, acc)
        edges_rep = self._build_reply_edges(src, acc)
        edges_men = ef.build_mention_edges(acc, src)

        return features, edges_rt, edges_rep, edges_men

    # ─────────────────────────────────────────────────────────────────────
    # Feature assembly (private)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_node_features(user_id_int: int, acc: dict) -> dict:
        latest_user = acc["latest_tweet"]["user"]
        timestamps  = acc["timestamps"]
        n_total     = acc["n_total"]

        # Profile ratios
        followers_raw = latest_user.get("followers_count", 0) or 0
        friends_raw   = latest_user.get("friends_count", 0) or 0
        den = followers_raw + friends_raw
        social_influence = followers_raw / den if den > 0 else 0.0

        # Account date (seconds since Twitter epoch)
        user_created_at = Utils.to_datetime(latest_user.get("created_at"))
        TWITTER_EPOCH = datetime(2006, 3, 21, tzinfo=timezone.utc).timestamp()
        account_date = (round(user_created_at.timestamp() - TWITTER_EPOCH)
                        if user_created_at else 0)

        # Activation age
        first_ts = min(timestamps) if timestamps else 0.0
        activation_age = 0
        if user_created_at and first_ts:
            try:
                first_dt = datetime.fromtimestamp(first_ts, tz=timezone.utc)
                uca = (user_created_at if user_created_at.tzinfo
                       else user_created_at.replace(tzinfo=timezone.utc))
                activation_age = max(0, (first_dt - uca).total_seconds())
            except Exception:
                activation_age = 0

        # Temporal metrics
        reg_iti = nf.tweet_regularity_score_from_timestamps(timestamps)
        reg_daily = nf.daily_posting_consistency(timestamps,
                                                 window_days=90, min_days=14)
        daily_cv = (reg_daily["daily_cv_log"]
                    if reg_daily["daily_cv_log"] is not None else 0.0)
        internal_density = nf.internal_tweet_density(timestamps, window_days=90)

        # Source ratios
        sensitive_rate = float(acc["n_sensitive"]) / n_total if n_total > 0 else 0.0
        mobile_ratio   = acc["n_mobile"]       / n_total if n_total > 0 else 0.0
        web_ratio      = acc["n_web"]           / n_total if n_total > 0 else 0.0
        nm_ratio       = acc["n_news_manager"]  / n_total if n_total > 0 else 0.0
        bot_api_ratio  = acc["n_bot_api"]       / n_total if n_total > 0 else 0.0

        source_entropy  = nf.compute_source_entropy(
            acc["n_mobile"], acc["n_web"], acc["n_news_manager"], acc["n_bot_api"])
        hashtag_entropy = nf.compute_hashtag_entropy(acc["hashtag_counter"])
        total_hashtags  = sum(acc["hashtag_counter"].values())

        def log1p(x):
            return round(math.log1p(max(x, -0.9999)), 2)

        return {
            "user_id":      user_id_int,
            "user_node_id": user_id_int,
            # Activity
            "total":    log1p(n_total),
            "retweets": log1p(acc["n_retweets"]),
            "replies":  log1p(acc["n_replies"]),
            "original": log1p(acc["n_original"]),
            # Profile
            "likes":             log1p(latest_user.get("favourites_count", 0)),
            "followers":         log1p(latest_user.get("followers_count", 0)),
            "following":         log1p(latest_user.get("friends_count", 0)),
            "verified":          1 if latest_user.get("verified", False) else 0,
            "account_date":      account_date,
            "listed_count":      log1p(latest_user.get("listed_count", 0)),
            "favourites_count":  log1p(latest_user.get("favourites_count", 0)),
            "reputation_score":  round(social_influence, 2),
            # Content
            "n_unique_hashtags": len(acc["hashtags"]),
            "n_unique_mentions": len(acc["mention_targets"]),
            "n_hashtags_total":  log1p(total_hashtags),
            "hashtag_entropy":   round(hashtag_entropy, 4),
            # Automation
            "activation_age":             log1p(activation_age),
            "tweet_regularity_score":     log1p(reg_iti["regularity_score"]),
            "regularity_reliable":        1 if n_total >= 20 else 0,
            "tweet_avg_interval_seconds": log1p(reg_iti["iti_mean"]),
            "daily_score":                round(reg_daily["daily_score"], 2),
            "daily_cv_log":               round(daily_cv, 2),
            "internal_tweet_density":     round(internal_density, 2),
            "profile_has_url":  int(latest_user.get("url") is not None),
            "geo_enabled_flag": int(latest_user.get("geo_enabled", False)),
            # Sensitive / Source
            "sensitive_rate":    sensitive_rate,
            "mobile_ratio":     round(mobile_ratio, 4),
            "web_ratio":        round(web_ratio, 4),
            "news_manager_ratio": round(nm_ratio, 4),
            "bot_api_ratio":    round(bot_api_ratio, 4),
            "source_entropy":   round(source_entropy, 4),
            # Metadata (not persisted in final schema)
            "screen_name": latest_user.get("screen_name", ""),
        }

    @staticmethod
    def _build_retweet_edges(src: int, acc: dict) -> list:
        edges = []
        for dst, weight in acc["retweet_targets"].items():
            ts_list = acc["retweet_ts_per_edge"].get(dst, [])
            cadence, jitter = ef.rt_cadence_jitter(ts_list)
            edges.append((
                src, int(dst), weight,
                ef.lifespan(ts_list),
                ef.fast_rt_ratio(acc["retweet_fast_count"].get(dst, 0), weight),
                cadence, jitter,
                ef.rt_topic_consistency(
                    acc["retweet_hashtags_per_edge"].get(dst, {})),
            ))
        return edges

    @staticmethod
    def _build_reply_edges(src: int, acc: dict) -> list:
        edges = []
        for dst, weight in acc["reply_targets"].items():
            ts_list = acc["reply_ts_per_edge"].get(dst, [])
            reg, burst = ef.reply_regularity_burstiness(ts_list)
            edges.append((
                src, int(dst), weight,
                ef.lifespan(ts_list),
                ef.avg_reply_latency(
                    acc["reply_latency_sum"].get(dst, 0.0),
                    acc["reply_latency_count"].get(dst, 0)),
                reg, burst,
                ef.reply_diurnal_sync(
                    acc["reply_hours_per_edge"].get(dst, [])),
            ))
        return edges

    # ─────────────────────────────────────────────────────────────────────
    # Checkpoint I/O (delegates to checkpoint module)
    # ─────────────────────────────────────────────────────────────────────

    def save_checkpoint(self, user_features_list, edges_rt, edges_reply,
                        mention_edges_raw, screen_names_list, batch_id):
        cp.save_checkpoint(
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
            load_snapshot_tmp_path=self.load_snapshot_tmp_path,
        )

    def merge_checkpoints(self):
        cp.merge_checkpoints(
            output_root=self.output_file_path,
            checkpoint_folder=self.checkpoint_folder,
            run_id=self.id,
            file_format=self.file_format,
            final_file_format=self.final_file_format,
            delete_tmp_after_merge=self.delete_tmp_after_merge,
            is_community_run=self.is_community_run,
            community_file_path=self.community_file_path,
            collection_name=self.get_collection(),
            load_snapshot_status=self.load_snapshot_status,
            load_snapshot_tmp_path=self.load_snapshot_tmp_path,
            logger=self.logger,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Parallel worker
    # ─────────────────────────────────────────────────────────────────────

    def _process_work_item(self, worker_id: int, work_item,
                           checkpoint_every: int,
                           max_users: Optional[int],
                           total_users_counter: Optional[list] = None,
                           total_users_lock: Optional[threading.Lock] = None
                           ) -> int:
        client     = MongoClient(self._uri, **self._mongo_kwargs)
        collection = client[self._database_name][self._collection_name]

        batch_offset = worker_id * 1_000_000 + int(time.time() % 10000) * 1000
        local_batch  = 0
        features_list, edges_rt, edges_reply = [], [], []
        mention_raw, screen_names_list       = [], []
        user_count = 0

        try:
            for user_id, user_tweets in self.strategy.iter_users(
                    collection, work_item):
                features, rt, rep, men = self.process_user_tweets(
                    user_id, user_tweets)
                if features is None:
                    continue

                features["community"] = self.strategy.get_community_id(
                    features["user_id"])

                sn = features.get("screen_name", "")
                if sn:
                    screen_names_list.append((sn.lower(), features["user_id"]))

                features_list.append(features)
                edges_rt.extend(rt)
                edges_reply.extend(rep)
                mention_raw.extend(men)
                user_count += 1

                if user_count % checkpoint_every == 0:
                    bid = batch_offset + local_batch
                    global_total = user_count
                    if (total_users_counter is not None
                            and total_users_lock is not None):
                        with total_users_lock:
                            total_users_counter[0] += checkpoint_every
                            global_total = total_users_counter[0]
                    self.logger.info(
                        f"[Worker {worker_id} | Batch {bid}] "
                        f"{user_count} users | "
                        f"{len(edges_rt)} RT | {len(edges_reply)} reply | "
                        f"{len(mention_raw)} mention "
                        f"[total: {global_total:,} users]")
                    self.save_checkpoint(features_list, edges_rt, edges_reply,
                                         mention_raw, screen_names_list, bid)
                    features_list, edges_rt, edges_reply = [], [], []
                    mention_raw, screen_names_list       = [], []
                    local_batch += 1

                if max_users is not None and user_count >= max_users:
                    self.logger.info(
                        f"[Worker {worker_id}] Reached max_users="
                        f"{max_users}. Stopping.")
                    break

            if features_list:
                self.save_checkpoint(
                    features_list, edges_rt, edges_reply,
                    mention_raw, screen_names_list,
                    batch_offset + local_batch)
        finally:
            client.close()

        return user_count

    # ─────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────

    def run(self, checkpoint_every: int = 500,
            max_users: Optional[int] = None, n_workers: int = 4) -> None:
        self.logger.info(
            f"[GraphGenerationUser] Starting extraction. "
            f"Run ID: {self.id}  Format: {self.file_format}")
        run_start = time.time()

        client     = MongoClient(self._uri, **self._mongo_kwargs)
        collection = client[self._database_name][self._collection_name]

        if self.load_snapshot_status and self.load_snapshot_tmp_path:
            self.logger.info(
                f"Loading snapshot from {self.load_snapshot_tmp_path} "
                "to resume extraction.")
            processed = cp.load_snapshot_users(
                self.load_snapshot_tmp_path, self.file_format)
            self.logger.info(
                f"Found {len(processed)} already processed users in snapshot.")
            skipped = self.strategy.exclude_users(processed)
            self.logger.info(
                f"Excluded {skipped} users from strategy processing queue.")
            snapshot_basename = os.path.basename(
                self.load_snapshot_tmp_path.rstrip(os.sep))
            if snapshot_basename:
                self.id = snapshot_basename

        work_items = self.strategy.partition_work(
            collection, n_workers, max_users, self.logger)
        client.close()

        if not work_items:
            self.logger.warning(
                "No work to process. Proceeding to merge (if any).")
            total_users = 0
        else:
            per_worker_max = (
                (max_users + len(work_items) - 1) // len(work_items)
                if max_users is not None and len(work_items) > 0
                else None)

            self.logger.info(
                f"Dispatching {len(work_items)} task(s) to {n_workers} "
                f"thread(s) (checkpoint every {checkpoint_every} users "
                f"per task)...")

            total_users = 0
            _counter = [0]
            _lock    = threading.Lock()
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        self._process_work_item,
                        task_id, work_item,
                        checkpoint_every, per_worker_max,
                        _counter, _lock,
                    ): task_id
                    for task_id, work_item in enumerate(work_items)
                }
                for future in as_completed(futures):
                    wid = futures[future]
                    try:
                        count = future.result()
                        total_users += count
                        self.logger.info(
                            f"[Worker {wid}] finished — "
                            f"{count} users processed.")
                    except Exception as exc:
                        self.logger.error(
                            f"[Worker {wid}] failed with exception: {exc}",
                            exc_info=True)

        elapsed = time.time() - run_start
        h, rem = divmod(elapsed, 3600)
        m, s   = divmod(rem, 60)
        self.logger.info(
            f"{total_users} users processed in total in "
            f"{int(h):02}:{int(m):02}:{int(s):02}. "
            f"Starting merge (checkpoints: {self.file_format} "
            f"-> final: {self.final_file_format})...")
        self.merge_checkpoints()
        self.logger.info("Extraction complete.")
