# algorithms/GraphGenerationUser.py
# GraphGenerationUser — Orchestrator class for user-level graph features extraction.
#
# Responsibilities:
#   - Configure MongoDB connections and output format configurations.
#   - Coordinate parsing of individual tweets via tweet_parser.
#   - Construct user node feature dictionaries using node_features.
#   - Compute retweet, reply, and mention interaction edges using edge_features.
#   - Persist temporary chunks and merge them via checkpoint.
#   - Manage thread pools for parallelized processing of the database query.

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

# Sub-package imports for specific extraction logic
from algorithms.graph.output_structure import SUPPORTED_FORMATS
from algorithms.graph.tweet_parser import init_accumulators, parse_tweet
from algorithms.graph import node_features as nf
from algorithms.graph import edge_features as ef
from algorithms.graph import checkpoint as cp


class GraphGenerationUser(MongoConnection):
    """
    Orchestrates user-level graph feature extraction from MongoDB tweets.

    The actual computation and I/O tasks are delegated to sub-modules:
      - tweet_parser   → Aggregating tweets into state accumulators.
      - node_features  → Compiling user profile and regularity statistics.
      - edge_features  → Building interaction metrics for retweet/reply/mention links.
      - checkpoint     → Saving partial batches and performing the final file merge.
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
        """
        Initialize the GraphGenerationUser extraction harness.

        :param uri: MongoDB connection URI string.
        :param database_name: Target database to query.
        :param collection: Collection containing raw tweet documents.
        :param output_file_path: Base folder where extraction runs will be saved.
        :param username: DB auth username.
        :param password: DB auth password.
        :param auth_source: DB auth database source.
        :param auth_mechanism: DB auth mechanism (e.g. SCRAM-SHA-1).
        :param delete_tmp_after_merge: If True, intermediate checkpoint folders are cleaned up.
        :param intermediate_file_format: File format for checkpoint chunks (feather/parquet).
        :param final_file_format: Final merged table format (feather/parquet/csv).
        :param file_format: Legacy override to specify both formats at once.
        :param fast_rt_threshold: Seconds threshold under which a retweet is flagged as "fast".
        :param strategy: An instance of UserFetchStrategy dictating how database documents are traversed.
        :param is_community_run: If True, users are mapped to specific communities.
        :param community_file_path: Path to the CSV mapping users to their community IDs.
        :param load_snapshot_status: If True, tries to resume from a previous checkpoint run.
        :param load_snapshot_tmp_path: Temp path containing the checkpoint run to resume.
        """

        # Legacy backward compatibility: override both formats if file_format is passed directly
        if file_format is not None:
            intermediate_file_format = file_format
            final_file_format = file_format

        # Validate formats against supported storage engines
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

        # Stored attributes for spinning up independent MongoDB client connections in workers
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

    def process_user_tweets(self, user_id: Any, tweets: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[Tuple], List[Tuple], List[Tuple]]:
        """
        Process all tweets of a single user and extract their node features + edge links.

        :param user_id: Unique identifier for the Twitter user.
        :param tweets: List of tweet dictionaries associated with this user.
        :return: A tuple containing:
                 - User features dictionary (or None if no valid tweets could be parsed)
                 - Retweet edges list
                 - Reply edges list
                 - Mention edges list
        """
        if not tweets:
            return None, [], [], []

        user_id_int = int(user_id)

        # 1. Accumulate timeline metrics per tweet (timestamps, target users, hashtags)
        acc = init_accumulators()
        for tweet in tweets:
            parse_tweet(acc, tweet, self.fast_rt_threshold)

        if not acc["latest_tweet"]:
            return None, [], [], []

        # 2. Compile aggregated node features
        features = self._build_node_features(user_id_int, acc)

        # 3. Compile interaction edge tuples
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
        """
        Compile user profile attributes, temporal activity, and linguistic metrics into a feature dictionary.
        """
        latest_user = acc["latest_tweet"]["user"]
        timestamps  = acc["timestamps"]
        n_total     = acc["n_total"]

        # Calculate social reputation ratio (Influence = followers / (followers + following))
        followers_raw = latest_user.get("followers_count", 0) or 0
        friends_raw   = latest_user.get("friends_count", 0) or 0
        den = followers_raw + friends_raw
        social_influence = followers_raw / den if den > 0 else 0.0

        # Calculate user account age relative to Twitter's epoch (March 21, 2006)
        user_created_at = Utils.to_datetime(latest_user.get("created_at"))
        TWITTER_EPOCH = datetime(2006, 3, 21, tzinfo=timezone.utc).timestamp()
        account_date = (round(user_created_at.timestamp() - TWITTER_EPOCH)
                        if user_created_at else 0)

        # Calculate activation age (duration in seconds between account creation and the user's first observed tweet)
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

        # Compute posting regularity and posting consistency metrics
        reg_iti = nf.tweet_regularity_score_from_timestamps(timestamps)
        reg_daily = nf.daily_posting_consistency(timestamps,
                                                 window_days=90, min_days=14)
        daily_cv = (reg_daily["daily_cv_log"]
                    if reg_daily["daily_cv_log"] is not None else 0.0)
        internal_density = nf.internal_tweet_density(timestamps, window_days=90)

        # Compute ratio of tweets coming from mobile/web/bot-API platforms and compute entropy
        sensitive_rate = float(acc["n_sensitive"]) / n_total if n_total > 0 else 0.0
        mobile_ratio   = acc["n_mobile"]       / n_total if n_total > 0 else 0.0
        web_ratio      = acc["n_web"]           / n_total if n_total > 0 else 0.0
        nm_ratio       = acc["n_news_manager"]  / n_total if n_total > 0 else 0.0
        bot_api_ratio  = acc["n_bot_api"]       / n_total if n_total > 0 else 0.0

        source_entropy  = nf.compute_source_entropy(
            acc["n_mobile"], acc["n_web"], acc["n_news_manager"], acc["n_bot_api"])
        hashtag_entropy = nf.compute_hashtag_entropy(acc["hashtag_counter"])
        total_hashtags  = sum(acc["hashtag_counter"].values())

        # Log1p helper to scale highly skewed metrics (e.g. total follower counts or tweet counts)
        def log1p(x):
            return round(math.log1p(max(x, -0.9999)), 2)

        return {
            "user_id":      user_id_int,
            "user_node_id": user_id_int,
            # Activity statistics
            "total":    log1p(n_total),
            "retweets": log1p(acc["n_retweets"]),
            "replies":  log1p(acc["n_replies"]),
            "original": log1p(acc["n_original"]),
            # User profile data
            "likes":             log1p(latest_user.get("favourites_count", 0)),
            "followers":         log1p(latest_user.get("followers_count", 0)),
            "following":         log1p(latest_user.get("friends_count", 0)),
            "verified":          1 if latest_user.get("verified", False) else 0,
            "account_date":      account_date,
            "listed_count":      log1p(latest_user.get("listed_count", 0)),
            "favourites_count":  log1p(latest_user.get("favourites_count", 0)),
            "reputation_score":  round(social_influence, 2),
            # Content statistics
            "n_unique_hashtags": len(acc["hashtags"]),
            "n_unique_mentions": len(acc["mention_targets"]),
            "n_hashtags_total":  log1p(total_hashtags),
            "hashtag_entropy":   round(hashtag_entropy, 4),
            # Posting patterns / automation heuristics
            "activation_age":             log1p(activation_age),
            "tweet_regularity_score":     log1p(reg_iti["regularity_score"]),
            "regularity_reliable":        1 if n_total >= 20 else 0,
            "tweet_avg_interval_seconds": log1p(reg_iti["iti_mean"]),
            "daily_score":                round(reg_daily["daily_score"], 2),
            "daily_cv_log":               round(daily_cv, 2),
            "internal_tweet_density":     round(internal_density, 2),
            "profile_has_url":  int(latest_user.get("url") is not None),
            "geo_enabled_flag": int(latest_user.get("geo_enabled", False)),
            # Platform sensitivity and client source ratios
            "sensitive_rate":    sensitive_rate,
            "mobile_ratio":     round(mobile_ratio, 4),
            "web_ratio":        round(web_ratio, 4),
            "news_manager_ratio": round(nm_ratio, 4),
            "bot_api_ratio":    round(bot_api_ratio, 4),
            "source_entropy":   round(source_entropy, 4),
            # Screen name mapping (dropped from final schemas but used for checks)
            "screen_name": latest_user.get("screen_name", ""),
        }

    @staticmethod
    def _build_retweet_edges(src: int, acc: dict) -> list:
        """
        Build edge list for retweet interactions including weight, lifespan, and cadence metadata.
        """
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
        """
        Build edge list for reply interactions including latency, burstiness, and diurnal synchronization attributes.
        """
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

    def save_checkpoint(self, user_features_list: List[dict], edges_rt: List[Tuple], edges_reply: List[Tuple],
                        mention_edges_raw: List[Tuple], screen_names_list: List[Tuple], batch_id: int) -> None:
        """
        Save intermediate extraction tables (checkpoint) to the temporary directory.
        """
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

    def merge_checkpoints(self) -> None:
        """
        Merge all processed chunk partition tables into the final consolidated output schemas.
        """
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

    def _process_work_item(self, worker_id: int, work_item: Any,
                           checkpoint_every: int,
                           max_users: Optional[int],
                           total_users_counter: Optional[list] = None,
                           total_users_lock: Optional[threading.Lock] = None
                           ) -> int:
        """
        Individual thread worker routine. Processes users mapped to its partition segment.
        """
        client     = MongoClient(self._uri, **self._mongo_kwargs)
        collection = client[self._database_name][self._collection_name]

        # Calculate a thread-safe batch offset to avoid collision in intermediate filenames
        batch_offset = worker_id * 1_000_000 + int(time.time() % 10000) * 1000
        local_batch  = 0
        features_list, edges_rt, edges_reply = [], [], []
        mention_raw, screen_names_list       = [], []
        user_count = 0

        try:
            # Iterate through the partition queue defined by the strategy
            for user_id, user_tweets in self.strategy.iter_users(
                    collection, work_item):
                features, rt, rep, men = self.process_user_tweets(
                    user_id, user_tweets)
                if features is None:
                    continue

                # Add community identifier if executing a community-aware query run
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

                # Flush local state accumulator to a checkpoint file periodically
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

            # Flush final remaining users in worker buffer
            if features_list:
                self.save_checkpoint(
                    features_list, edges_rt, edges_reply,
                    mention_raw, screen_names_list,
                    batch_offset + local_batch)
        finally:
            client.close()

        return user_count

    # ─────────────────────────────────────────────────────────────────────
    # Main entry point stages
    # ─────────────────────────────────────────────────────────────────────

    def _should_skip_run(self) -> bool:
        """
        Stage 1: Pre-execution checks & snapshot run ID setup.
        Checks if the metadata output file already exists. If yes, the run is complete.
        """
        if self.load_snapshot_status and self.load_snapshot_tmp_path:
            snapshot_basename = os.path.basename(
                self.load_snapshot_tmp_path.rstrip(os.sep))
            if snapshot_basename:
                self.id = snapshot_basename

        # Check if complete
        final_dir = os.path.join(self.output_file_path, self.id)
        metadata_file = os.path.join(final_dir, "metadata.json")
        if self.load_snapshot_status and os.path.exists(metadata_file):
            self.logger.info(
                f"Final output metadata.json found at '{metadata_file}'. "
                "Extraction and merge have already completed for this snapshot. Skipping.")
            return True
        return False

    def _load_and_apply_snapshot(self) -> None:
        """
        Stage 2: Resume extraction from snapshot state.
        Retrieves users already processed in the snapshot files and excludes them from strategies queue.
        """
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

    def _run_parallel_workers(self, collection: Any, checkpoint_every: int,
                              max_users: Optional[int], n_workers: int) -> int:
        """
        Stage 3: Parallelized worker dispatch.
        Partitions the query workload across the requested number of worker threads.
        """
        work_items = self.strategy.partition_work(
            collection, n_workers, max_users, self.logger)

        if not work_items:
            self.logger.warning(
                "No work to process. Proceeding to merge (if any).")
            return 0

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

        return total_users

    def _finalize_run(self, total_users: int, run_start_time: float) -> None:
        """
        Stage 4: Post-execution log and final file merge.
        """
        elapsed = time.time() - run_start_time
        h, rem = divmod(elapsed, 3600)
        m, s   = divmod(rem, 60)
        self.logger.info(
            f"{total_users} users processed in total in "
            f"{int(h):02}:{int(m):02}:{int(s):02}. "
            f"Starting merge (checkpoints: {self.file_format} "
            f"-> final: {self.final_file_format})...")
        self.merge_checkpoints()
        self.logger.info("Extraction complete.")

    def run(self, checkpoint_every: int = 500,
            max_users: Optional[int] = None, n_workers: int = 4) -> None:
        """
        High-level entry point orchestrating the user extraction pipeline.
        Decomposed into 4 sequential stages.
        """
        # Stage 1: Pre-execution and completeness verification
        if self._should_skip_run():
            return

        self.logger.info(
            f"[GraphGenerationUser] Starting extraction. "
            f"Run ID: {self.id}  Format: {self.file_format}")
        run_start = time.time()

        # Connect to MongoDB client instance
        client     = MongoClient(self._uri, **self._mongo_kwargs)
        collection = client[self._database_name][self._collection_name]

        try:
            # Stage 2: Resume from snapshot if enabled
            if self.load_snapshot_status and self.load_snapshot_tmp_path:
                self._load_and_apply_snapshot()

            # Stage 3: Parallelized worker execution
            total_users = self._run_parallel_workers(
                collection, checkpoint_every, max_users, n_workers
            )
        finally:
            client.close()

        # Stage 4: Run completion stats and checkpoint merge
        self._finalize_run(total_users, run_start)