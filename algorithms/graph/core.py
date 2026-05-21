# algorithms/graph/core.py
# Core orchestrator for graph generation.
# Keep comments in English as requested.

import os
import uuid
import logging
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any

# Import modular helpers
from algorithms.graph import features as feat
from algorithms.graph import mention_metrics as mm
from algorithms.graph import retweet_metrics as rm
from algorithms.graph import io as io_mod
from algorithms.graph import constants as consts

# Import Writer-related utilities and MongoConnection
from Utils.Writer import Writer
from algorithms.MongoConnection import MongoConnection
from Utils.Utils import Utils

logger = logging.getLogger("algorithms.graph.core")

# -------------------------
# Module-level helpers
# -------------------------

def _init_state() -> Dict[str, Any]:
    """
    Initialize the per-user accumulation state.
    This keeps process_user_tweets concise and testable.
    """
    return {
        # node-level accumulators
        "n_total": 0,
        "n_retweets": 0,
        "n_replies": 0,
        "n_original": 0,
        "n_sensitive": 0,
        "n_mobile": 0,
        "n_web": 0,
        "n_news_manager": 0,
        "n_bot_api": 0,
        "latest_tweet": None,
        "hashtags": set(),
        "hashtag_counter": Counter(),
        "timestamps": [],

        # edge-level accumulators
        "retweet_targets": defaultdict(int),
        "retweet_first_ts": {},
        "retweet_last_ts": {},
        "retweet_fast_count": defaultdict(int),
        "retweet_ts_per_edge": defaultdict(list),
        "retweet_hashtags_per_edge": defaultdict(Counter),

        "reply_targets": defaultdict(int),
        "reply_first_ts": {},
        "reply_last_ts": {},
        "reply_latency_sum": defaultdict(float),
        "reply_latency_count": defaultdict(int),
        "reply_ts_per_edge": defaultdict(list),
        "reply_hours_per_edge": defaultdict(list),

        "mention_targets": defaultdict(int),
        "mention_ts": defaultdict(list),
        "mention_in_reply_count": defaultdict(int),
        "mention_solo_count": defaultdict(int),

        "seen_tweet_ids": set(),
    }

def _accumulate_from_tweet(state: Dict[str, Any], tweet: Dict[str, Any], fast_rt_threshold: int) -> None:
    """
    Parse a single tweet and update the state.
    This function is intentionally small and side-effecting on `state`.
    """
    tid = tweet.get("id")
    if tid in state["seen_tweet_ids"]:
        return
    state["seen_tweet_ids"].add(tid)

    # normalize created_at
    tweet["created_at"] = Utils.to_datetime(tweet.get("created_at"))
    if tweet["created_at"] is None:
        return

    ts = tweet["created_at"].timestamp()
    state["timestamps"].append(ts)
    state["n_total"] += 1

    # classify source
    src_class = feat.classify_source(tweet.get("source", ""))
    if src_class == "mobile":
        state["n_mobile"] += 1
    elif src_class == "web":
        state["n_web"] += 1
    elif src_class == "news_manager":
        state["n_news_manager"] += 1
    else:
        state["n_bot_api"] += 1

    # sensitive
    if tweet.get("possibly_sensitive") is True:
        state["n_sensitive"] += 1

    # hashtags
    raw_ht = tweet.get("hashtagEntities", "")
    if isinstance(raw_ht, str) and raw_ht.strip():
        for ht in raw_ht.split("|"):
            h = ht.strip().lower()
            if h:
                state["hashtags"].add(h)
                state["hashtag_counter"][h] += 1

    # latest tweet snapshot
    if state["latest_tweet"] is None or tweet["created_at"] > state["latest_tweet"]["created_at"]:
        state["latest_tweet"] = tweet

    # retweet / reply / original
    is_retweet = tweet.get("retweeted_status") is not None
    is_reply = tweet.get("in_reply_to_status_id") not in (None, -1)

    if is_retweet:
        state["n_retweets"] += 1
        rs = tweet.get("retweeted_status") or {}
        dst_uid = rs.get("user", {}).get("id") or rs.get("user", {}).get("id_str")
        if dst_uid is not None:
            state["retweet_targets"][dst_uid] += 1
            state["retweet_first_ts"][dst_uid] = min(state["retweet_first_ts"].get(dst_uid, ts), ts)
            state["retweet_last_ts"][dst_uid] = max(state["retweet_last_ts"].get(dst_uid, ts), ts)
            state["retweet_ts_per_edge"][dst_uid].append(ts)
            # collect hashtags for topic consistency
            rt_ht_raw = tweet.get("hashtagEntities") or rs.get("hashtagEntities", "")
            if isinstance(rt_ht_raw, str) and rt_ht_raw.strip():
                for ht in rt_ht_raw.split("|"):
                    h = ht.strip().lower()
                    if h:
                        state["retweet_hashtags_per_edge"][dst_uid][h] += 1
            # fast RT detection
            try:
                orig_created = rs.get("created_at")
                if orig_created:
                    orig_dt = Utils.to_datetime(orig_created)
                    if orig_dt:
                        latency = ts - orig_dt.timestamp()
                        if 0 <= latency <= fast_rt_threshold:
                            state["retweet_fast_count"][dst_uid] += 1
            except Exception:
                pass

    elif is_reply:
        state["n_replies"] += 1
        reply_uid = tweet.get("in_reply_to_user_id")
        if reply_uid and reply_uid != -1:
            state["reply_targets"][reply_uid] += 1
            state["reply_first_ts"][reply_uid] = min(state["reply_first_ts"].get(reply_uid, ts), ts)
            state["reply_last_ts"][reply_uid] = max(state["reply_last_ts"].get(reply_uid, ts), ts)
            state["reply_ts_per_edge"][reply_uid].append(ts)
            state["reply_hours_per_edge"][reply_uid].append(tweet["created_at"].hour)
            parent_created = tweet.get("in_reply_to_status_created_at")
            if parent_created:
                try:
                    parent_dt = Utils.to_datetime(parent_created)
                    if parent_dt:
                        lat = ts - parent_dt.timestamp()
                        if lat >= 0:
                            state["reply_latency_sum"][reply_uid] += lat
                            state["reply_latency_count"][reply_uid] += 1
                except Exception:
                    pass
    else:
        state["n_original"] += 1

    # mentions (only for non-retweets)
    if not is_retweet:
        raw_mentions = tweet.get("userMentionEntities", "")
        if isinstance(raw_mentions, str) and raw_mentions.strip():
            mentions = [mn.strip().lower() for mn in raw_mentions.split("|") if mn.strip()]
            if mentions:
                if len(mentions) == 1:
                    state["mention_solo_count"][mentions[0]] += 1
                for mn in mentions:
                    state["mention_targets"][mn] += 1
                    state["mention_ts"][mn].append(ts)
                    if is_reply:
                        state["mention_in_reply_count"][mn] += 1

# -------------------------
# GraphGenerationUser class
# -------------------------

class GraphGenerationUser(MongoConnection):
    """
    Orchestrates user-level feature extraction and checkpoint writing.
    The heavy computations are delegated to modules in algorithms.graph.
    """

    logger = logging.getLogger("GraphGenerationUser")

    def __init__(self, uri, database_name, collection, output_file_path,
                 username=None, password=None, auth_source=None, auth_mechanism=None,
                 delete_tmp_after_merge=False, intermediate_file_format="feather",
                 final_file_format="parquet", file_format=None, fast_rt_threshold=None):
        # Legacy compatibility: if file_format provided, use it for both
        if file_format is not None:
            intermediate_file_format = file_format
            final_file_format = file_format

        # Validate formats
        for fmt in (intermediate_file_format, final_file_format):
            if fmt not in consts.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported file_format '{fmt}'. Choose from {consts.SUPPORTED_FORMATS}.")

        super().__init__(uri, username, password, auth_source, auth_mechanism,
                         db=None, database_name=database_name, collection=collection)

        self.output_file_path = output_file_path
        self.id = uuid.uuid1().hex
        self.checkpoint_folder = "tmp"
        self.delete_tmp_after_merge = delete_tmp_after_merge
        self.file_format = intermediate_file_format
        self.final_file_format = final_file_format
        self.fast_rt_threshold = fast_rt_threshold or consts.FAST_RT_THRESHOLD

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

    # -------------------------
    # Core processing
    # -------------------------

    def process_user_tweets(self, user_id, tweets):
        """
        Process all tweets of a single user and extract node features and edges.
        This function orchestrates accumulation and delegates computations.
        Returns: (features_dict, edges_retweet, edges_reply, mention_edges_raw)
        """
        state = _init_state()

        # accumulate per-tweet state
        for tweet in tweets:
            _accumulate_from_tweet(state, tweet, self.fast_rt_threshold)

        if state["latest_tweet"] is None:
            return None, [], [], []

        # compute node-level features using features module
        latest_user = state["latest_tweet"]["user"]
        # convert timestamps list to floats
        timestamps = state["timestamps"]
        reg = feat.tweet_regularity_score_from_timestamps(timestamps)
        daily = feat.daily_posting_consistency(timestamps, window_days=consts.DEFAULT_WINDOW_DAYS, min_days=14)
        internal_density = feat.internal_tweet_density(timestamps, window_days=consts.DEFAULT_WINDOW_DAYS)

        # profile and counts
        followers_raw = latest_user.get("followers_count", 0) or 0
        friends_raw = latest_user.get("friends_count", 0) or 0
        den = followers_raw + friends_raw
        social_influence_ratio = followers_raw / den if den > 0 else 0.0

        user_created_at = Utils.to_datetime(latest_user.get("created_at"))
        TWITTER_EPOCH = feat.datetime(2006, 3, 21, tzinfo=feat.timezone.utc).timestamp() if hasattr(feat, "datetime") else 0
        account_date = round(user_created_at.timestamp() - TWITTER_EPOCH) if user_created_at else 0

        def _log1p_round(x):
            try:
                return round(feat.math.log1p(max(x, -0.9999)), 2)
            except Exception:
                return 0.0

        total_hashtags = sum(state["hashtag_counter"].values())
        hashtag_entropy = feat.compute_hashtag_entropy(state["hashtag_counter"]) if hasattr(feat, "compute_hashtag_entropy") else 0.0

        features = {
            "user_id": int(user_id),
            "user_node_id": int(user_id),
            "total": _log1p_round(state["n_total"]),
            "retweets": _log1p_round(state["n_retweets"]),
            "replies": _log1p_round(state["n_replies"]),
            "original": _log1p_round(state["n_original"]),
            "likes": _log1p_round(latest_user.get("favourites_count", 0)),
            "followers": _log1p_round(latest_user.get("followers_count", 0)),
            "following": _log1p_round(latest_user.get("friends_count", 0)),
            "verified": 1 if latest_user.get("verified", False) else 0,
            "account_date": account_date,
            "listed_count": _log1p_round(latest_user.get("listed_count", 0)),
            "favourites_count": _log1p_round(latest_user.get("favourites_count", 0)),
            "reputation_score": round(social_influence_ratio, 2),
            "n_unique_hashtags": len(state["hashtags"]),
            "n_unique_mentions": len(state["mention_targets"]),
            "n_hashtags_total": _log1p_round(total_hashtags),
            "hashtag_entropy": round(hashtag_entropy, consts.ROUND_DECIMALS),
            "activation_age": _log1p_round(0),  # keep original activation_age logic if needed
            "tweet_regularity_score": _log1p_round(reg.get("regularity_score", 0.0)),
            "regularity_reliable": 1 if state["n_total"] >= 20 else 0,
            "tweet_avg_interval_seconds": _log1p_round(reg.get("iti_mean", 0.0)),
            "daily_score": round(daily.get("daily_score", 0.0), 2),
            "daily_cv_log": round(daily.get("daily_cv_log", 0.0) or 0.0, 2),
            "internal_tweet_density": round(internal_density, 2),
            "profile_has_url": int(latest_user.get("url") is not None),
            "geo_enabled_flag": int(latest_user.get("geo_enabled", False)),
            "sensitive_rate": float(state["n_sensitive"]) / state["n_total"] if state["n_total"] > 0 else 0.0,
            "mobile_ratio": round(state["n_mobile"] / state["n_total"], 4) if state["n_total"] > 0 else 0.0,
            "web_ratio": round(state["n_web"] / state["n_total"], 4) if state["n_total"] > 0 else 0.0,
            "news_manager_ratio": round(state["n_news_manager"] / state["n_total"], 4) if state["n_total"] > 0 else 0.0,
            "bot_api_ratio": round(state["n_bot_api"] / state["n_total"], 4) if state["n_total"] > 0 else 0.0,
            "source_entropy": round(feat.compute_source_entropy(state) if hasattr(feat, "compute_source_entropy") else 0.0, consts.ROUND_DECIMALS),
            "screen_name": latest_user.get("screen_name", ""),
        }

        # -------------------------
        # Build edges using modular helpers
        # -------------------------

        # Retweet edges
        edges_retweet = []
        for dst, weight in state["retweet_targets"].items():
            ts_list = state["retweet_ts_per_edge"].get(dst, [])
            cadence, jitter = rm.rt_cadence_jitter(ts_list)
            lifespan = rm.lifespan_from_ts(ts_list)
            fast_ratio = rm.fast_rt_ratio(state["retweet_fast_count"].get(dst, 0), state["retweet_targets"].get(dst, 0))
            topic_cons = rm.rt_topic_consistency(state["retweet_hashtags_per_edge"].get(dst, {}))
            edges_retweet.append((
                features["user_node_id"],
                int(dst),
                weight,
                lifespan,
                fast_ratio,
                cadence,
                jitter,
                topic_cons,
            ))

        # Reply edges
        edges_reply = []
        for dst, weight in state["reply_targets"].items():
            ts_list = state["reply_ts_per_edge"].get(dst, [])
            reg, burst = rm.reply_regularity_burstiness(ts_list)
            lifespan = rm.lifespan_from_ts(ts_list)
            avg_lat = rm.avg_reply_latency(state["reply_latency_sum"].get(dst, 0), state["reply_latency_count"].get(dst, 0))
            diurnal = rm.reply_diurnal_sync(state["reply_hours_per_edge"].get(dst, []))
            edges_reply.append((
                features["user_node_id"],
                int(dst),
                weight,
                lifespan,
                avg_lat,
                reg,
                burst,
                diurnal,
            ))

        # Mention edges (use mention_metrics builder)
        mention_state = {
            "mention_targets": state["mention_targets"],
            "mention_ts": state["mention_ts"],
            "mention_in_reply_count": state["mention_in_reply_count"],
            "mention_solo_count": state["mention_solo_count"],
        }
        mention_edges_raw = mm.build_mention_edges(mention_state, features["user_node_id"])

        return features, edges_retweet, edges_reply, mention_edges_raw

    # -------------------------
    # Checkpoint I/O wrappers (delegates to io module)
    # -------------------------

    def save_checkpoint(self, user_features_list: List[Dict], edges_rt: List[Tuple],
                        edges_reply: List[Tuple], mention_edges_raw: List[Tuple],
                        screen_names_list: List[Tuple], batch_id: int) -> None:
        """
        Persist intermediate results for one batch using io module.
        """
        io_mod.save_checkpoint_files(
            output_root=self.output_file_path,
            checkpoint_folder=self.checkpoint_folder,
            run_id=self.id,
            batch_id=batch_id,
            user_features_rows=[[f[col] for col in consts.USER_FEATURES_COLUMNS] for f in user_features_list],
            user_features_columns=consts.USER_FEATURES_COLUMNS,
            edges_rt_rows=edges_rt,
            edges_rt_columns=consts.EDGE_RETWEET_COLUMNS,
            edges_reply_rows=edges_reply,
            edges_reply_columns=consts.EDGE_REPLY_COLUMNS,
            edges_mention_rows=mention_edges_raw,
            edges_mention_columns=consts.EDGE_MENTION_COLUMNS,
            screen_name_rows=screen_names_list,
            screen_name_columns=consts.SCREEN_NAME_COLUMNS,
            file_format=self.file_format,
        )

    def merge_checkpoints(self) -> None:
        """
        Merge intermediate checkpoints into final outputs using io module.
        """
        io_mod.merge_checkpoints_and_write_final(
            output_root=self.output_file_path,
            run_id=self.id,
            checkpoint_folder=self.checkpoint_folder,
            final_user_features_name=consts.FILE_USER_FEATURES,
            final_edges_rt_name=consts.FILE_EDGES_RETWEET,
            final_edges_reply_name=consts.FILE_EDGES_REPLY,
            final_edges_mention_name=consts.FILE_EDGES_MENTION,
            final_screen_name_name=consts.FILE_SCREEN_NAME_MAP,
            intermediate_format=self.file_format,
            final_format=self.final_file_format,
        )