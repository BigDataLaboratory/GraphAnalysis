# algorithms/graph/tweet_parser.py
# Parses individual tweets and accumulates per-user state.
#
# The main function `parse_tweet()` is called once per tweet inside
# `process_user_tweets()`. Extracting it here keeps the orchestrator
# thin and makes the parsing logic independently testable.

from collections import Counter, defaultdict
from typing import Any, Dict

from Utils.Utils import Utils
from algorithms.graph.node_features import classify_source
from algorithms.graph.output_structure import FAST_RT_THRESHOLD


def init_accumulators() -> Dict[str, Any]:
    """Return a fresh set of per-user accumulators."""
    return {
        # ── Node-level counters ──────────────────────────────────────────
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

        # ── Edge-level accumulators (retweet) ────────────────────────────
        "retweet_targets": defaultdict(int),
        "retweet_first_ts": {},
        "retweet_last_ts": {},
        "retweet_fast_count": defaultdict(int),
        "retweet_ts_per_edge": defaultdict(list),
        "retweet_hashtags_per_edge": defaultdict(Counter),

        # ── Edge-level accumulators (reply) ──────────────────────────────
        "reply_targets": defaultdict(int),
        "reply_first_ts": {},
        "reply_last_ts": {},
        "reply_latency_sum": defaultdict(float),
        "reply_latency_count": defaultdict(int),
        "reply_ts_per_edge": defaultdict(list),
        "reply_hours_per_edge": defaultdict(list),

        # ── Edge-level accumulators (mention) ────────────────────────────
        "mention_targets": defaultdict(int),
        "mention_ts": defaultdict(list),
        "mention_in_reply_count": defaultdict(int),
        "mention_solo_count": defaultdict(int),

        # ── Deduplication ────────────────────────────────────────────────
        "seen_tweet_ids": set(),
    }


def parse_tweet(acc: Dict[str, Any], tweet: Dict[str, Any],
                fast_rt_threshold: int = FAST_RT_THRESHOLD) -> None:
    """
    Parse a single tweet and update the accumulators *in place*.

    This function owns all branching logic:
      - retweet vs reply vs original
      - hashtag / mention extraction
      - source classification
      - fast-RT detection

    Parameters
    ----------
    acc : dict
        Mutable accumulator dict returned by `init_accumulators()`.
    tweet : dict
        Raw tweet document from MongoDB.
    fast_rt_threshold : int
        Seconds — retweet latency below this is flagged as "fast".
    """
    # ── Dedup ────────────────────────────────────────────────────────────
    tweet_id = tweet.get("id")
    if tweet_id in acc["seen_tweet_ids"]:
        return
    acc["seen_tweet_ids"].add(tweet_id)

    # ── Normalise created_at ─────────────────────────────────────────────
    tweet["created_at"] = Utils.to_datetime(tweet["created_at"])
    if not tweet["created_at"]:
        return

    ts = tweet["created_at"].timestamp()
    acc["timestamps"].append(ts)
    acc["n_total"] += 1

    # ── Source classification ────────────────────────────────────────────
    src_class = classify_source(tweet.get("source", ""))
    if src_class == "mobile":
        acc["n_mobile"] += 1
    elif src_class == "web":
        acc["n_web"] += 1
    elif src_class == "news_manager":
        acc["n_news_manager"] += 1
    else:
        acc["n_bot_api"] += 1

    # ── Sensitive flag ───────────────────────────────────────────────────
    if tweet.get("possibly_sensitive") is True:
        acc["n_sensitive"] += 1

    # ── Hashtags ─────────────────────────────────────────────────────────
    raw_ht = tweet.get("hashtagEntities", "")
    if isinstance(raw_ht, str) and raw_ht.strip():
        for ht in raw_ht.split("|"):
            h = ht.strip().lower()
            if h:
                acc["hashtags"].add(h)
                acc["hashtag_counter"][h] += 1

    # ── Track latest tweet (for user-profile snapshot) ───────────────────
    if not acc["latest_tweet"] or tweet["created_at"] > acc["latest_tweet"]["created_at"]:
        acc["latest_tweet"] = tweet

    # ── Classify interaction type ────────────────────────────────────────
    is_retweet = tweet.get("retweeted_status") is not None
    is_reply   = tweet.get("in_reply_to_status_id") not in (None, -1)

    if is_retweet:
        _handle_retweet(acc, tweet, ts, fast_rt_threshold)
    elif is_reply:
        _handle_reply(acc, tweet, ts)
    else:
        acc["n_original"] += 1

    # ── Mentions (excluded for retweets to avoid double-counting) ────────
    if not is_retweet:
        _handle_mentions(acc, tweet, ts, is_reply)


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers — one per interaction type
# ─────────────────────────────────────────────────────────────────────────────

def _handle_retweet(acc: dict, tweet: dict, ts: float,
                    fast_rt_threshold: int) -> None:
    acc["n_retweets"] += 1
    rs = tweet.get("retweeted_status") or {}
    dst_uid = rs.get("user", {}).get("id") or rs.get("user", {}).get("id_str")
    if dst_uid is None:
        return

    dst_uid = int(dst_uid)
    acc["retweet_targets"][dst_uid] += 1
    acc["retweet_first_ts"][dst_uid] = min(acc["retweet_first_ts"].get(dst_uid, ts), ts)
    acc["retweet_last_ts"][dst_uid]  = max(acc["retweet_last_ts"].get(dst_uid, ts), ts)
    acc["retweet_ts_per_edge"][dst_uid].append(ts)

    # Hashtags for topic consistency
    rt_ht_raw = tweet.get("hashtagEntities") or rs.get("hashtagEntities", "")
    if isinstance(rt_ht_raw, str) and rt_ht_raw.strip():
        for ht in rt_ht_raw.split("|"):
            h = ht.strip().lower()
            if h:
                acc["retweet_hashtags_per_edge"][dst_uid][h] += 1

    # Fast-RT detection
    try:
        orig_created = rs.get("created_at")
        if orig_created:
            orig_dt = Utils.to_datetime(orig_created)
            if orig_dt:
                latency = ts - orig_dt.timestamp()
                if 0 <= latency <= fast_rt_threshold:
                    acc["retweet_fast_count"][dst_uid] += 1
    except Exception:
        pass


def _handle_reply(acc: dict, tweet: dict, ts: float) -> None:
    acc["n_replies"] += 1
    reply_uid = tweet.get("in_reply_to_user_id")
    if not reply_uid or reply_uid == -1:
        return

    reply_uid = int(reply_uid)
    acc["reply_targets"][reply_uid] += 1
    acc["reply_first_ts"][reply_uid] = min(acc["reply_first_ts"].get(reply_uid, ts), ts)
    acc["reply_last_ts"][reply_uid]  = max(acc["reply_last_ts"].get(reply_uid, ts), ts)
    acc["reply_ts_per_edge"][reply_uid].append(ts)
    acc["reply_hours_per_edge"][reply_uid].append(tweet["created_at"].hour)

    # Reply latency
    parent_created = tweet.get("in_reply_to_status_created_at")
    if parent_created:
        try:
            parent_dt = Utils.to_datetime(parent_created)
            if parent_dt:
                lat = ts - parent_dt.timestamp()
                if lat >= 0:
                    acc["reply_latency_sum"][reply_uid] += lat
                    acc["reply_latency_count"][reply_uid] += 1
        except Exception:
            pass


def _handle_mentions(acc: dict, tweet: dict, ts: float,
                     is_reply: bool) -> None:
    raw = tweet.get("userMentionEntities", "")
    if not isinstance(raw, str) or not raw.strip():
        return

    mentions = [mn.strip().lower() for mn in raw.split("|") if mn.strip()]
    if not mentions:
        return

    if len(mentions) == 1:
        acc["mention_solo_count"][mentions[0]] += 1

    for mn in mentions:
        acc["mention_targets"][mn] += 1
        acc["mention_ts"][mn].append(ts)
        if is_reply:
            acc["mention_in_reply_count"][mn] += 1
