import itertools
import logging
import math
import os
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pymongo import ASCENDING
from dateutil import parser  # Import to parse ISO dates
import re

import numpy as np
import sys
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.Utils import Utils
from Utils.Writer import Writer
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
    return 'bot_api'  # TweetDeck, API, inconnus → bot/api par défaut

def convert_to_datetime(value):
    if isinstance(value, datetime):
        return value
    elif isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except Exception:
            return None
    elif isinstance(value, str):
        try:
            return parser.parse(value)
        except Exception:
            return None
    else:
        return None

class GraphGenerationUser(MongoConnection):
    """
    User-centric graph generation for the GNN pipeline.

    Extracts node features (user statistics) and edges (retweet, reply, mention)
    from MongoDB tweets, sorted by user.id to guarantee complete user histories
    per batch. Uses the reject-last-user-id pagination strategy.

    Outputs (under output_file_path/<run_id>/):
        user_features.csv     : node feature matrix (one row per user)
        edges_retweet.csv     : (src, dst, weight)
        edges_reply.csv       : (src, dst, weight)
        edges_mention.csv     : (src, dst, weight) — resolved via map
        screen_name_map.csv   : (screen_name, user_id) for reference

    Mention edges whose target screen_name cannot be resolved to a known
    user in the dataset are silently dropped.
    """

    logger = logging.getLogger('GraphGenerationUser')

    def __init__(self, uri, database_name, collection, output_file_path,
                 username=None, password=None, auth_source=None, auth_mechanism=None,
                 delete_tmp_after_merge=False, file_format="csv"):
        """
        :param uri: MongoDB connection URI.
        :param database_name: Name of the MongoDB database.
        :param collection: Collection name.
        :param output_file_path: Root directory for all output files.
        :param username: MongoDB username (optional).
        :param password: MongoDB password (optional).
        :param auth_source: Authentication source (optional).
        :param auth_mechanism: Authentication mechanism (optional).
        :param delete_tmp_after_merge: Whether to delete the temporary checkpoint folder after merging (default: False).
        :param file_format: Output file format ("csv" or "pickle").
        """
        super().__init__(uri, username, password, auth_source, auth_mechanism,
                         db=None, database_name=database_name, collection=collection)
        self.output_file_path = output_file_path
        self.id = uuid.uuid1().hex
        self.checkpoint_folder = "tmp"
        self.delete_tmp_after_merge = delete_tmp_after_merge
        self.file_format = file_format

    # ─────────────────────────────────────────────────────────────────────────
    # CORE PROCESSING
    # ─────────────────────────────────────────────────────────────────────────

    def process_user_tweets(self, user_id, tweets):
        """
        Process all tweets of a single user and extract node features and edges.

        Iterates once over the tweet list accumulating:
        - Counters (total, retweets, replies, original tweets)
        - Snapshot fields (followers, likes, etc.) — kept from the most recent tweet
        - Sets (unique hashtags, unique mentions)
        - Edge targets grouped by interaction type

        Mention edges are returned as raw (src, screen_name, weight) tuples
        because screen_names must be resolved to user_ids at merge time.

        :param user_id: The Twitter user ID (integer) of the author.
        :param tweets: Iterable of tweet documents for this user.
        :return: (features_dict, edges_retweet, edges_reply, mention_edges_raw)
                 Returns (None, [], [], []) if the tweet list is empty.
        """

        ##############################################################################
        ### NODE FEATURES VARIABLES ##################################################
        ##############################################################################
        n_total = 0
        n_retweets = 0
        n_replies = 0
        n_original = 0
        n_sensitive = 0
        n_mobile = 0
        n_web = 0
        n_news_manager = 0
        n_bot_api = 0
        latest_tweet = None
        hashtags = set()
        timestamps = []
        coords_list = []
        places_list = []
        hashtag_counter = Counter()
        reply_first_ts = {}   # dst_user_id -> first reply timestamp (float)
        reply_last_ts = {}    # dst_user_id -> last reply timestamp (float)

        # reply latency accumulators (per couple A->B)
        reply_latency_sum = defaultdict(float)   # dst_user_id -> sum of latencies (seconds)
        reply_latency_count = defaultdict(int)   # dst_user_id -> count of reply events

        # retweet lifespan tracking
        retweet_first_ts = {}   # dst_user_id -> first retweet timestamp (float)
        retweet_last_ts = {}    # dst_user_id -> last retweet timestamp (float)

        last_seen_ts_by_user = {}   # user_id -> last seen tweet timestamp (float)

        ##############################################################################
        ### EDGE FEATURES VARIABLES ##################################################
        ##############################################################################
        retweet_targets = defaultdict(int)   # dst_user_id (int) → interaction count
        reply_targets = defaultdict(int)     # dst_user_id (int) → interaction count
        mention_targets = defaultdict(int)   # screen_name (str) → interaction count

        # Fast retweet tracking
        retweet_fast_count = defaultdict(int)   # dst_user_id -> number of retweets within threshold
        FAST_RT_THRESHOLD = 30                  # threshold in seconds; default 30s.



        seen_tweet_ids = set()

        for tweet in tweets:
            ##########################################################################
            ### NODE FEATURES COMPUTATION ############################################
            ##########################################################################
            tweet_id = tweet.get('id')
            if tweet_id in seen_tweet_ids:
                continue
            seen_tweet_ids.add(tweet_id)

            # Ensure created_at is a datetime object
            if isinstance(tweet['created_at'], str):
                tweet['created_at'] = parser.parse(tweet['created_at'])

            n_total += 1
            timestamps.append(tweet['created_at'].timestamp())

            # --- source classification (Mobile / Web / News Manager / Bot/API) ---
            source_class = classify_source(tweet.get('source', ''))
            if source_class == 'mobile':
                n_mobile += 1
            elif source_class == 'web':
                n_web += 1
            elif source_class == 'news_manager':
                n_news_manager += 1
            else:
                n_bot_api += 1

            # Sensitive content flag
            if tweet.get('possibly_sensitive') is True:
                n_sensitive += 1

            # --- geotag extraction ---
            # coordinates: Twitter may store as tweet['coordinates']['coordinates'] = [lon, lat]
            if tweet.get('coordinates'):
                c = tweet['coordinates'].get('coordinates')
                if c and len(c) >= 2:
                    coords_list.append((c[1], c[0]))  # store as (lat, lon)
            elif tweet.get('geo'):
                g = tweet['geo'].get('coordinates')
                if g and len(g) >= 2:
                    coords_list.append((g[0], g[1]))  # geo may be [lat, lon]

            # place: prefer id or full_name
            place = tweet.get('place')
            if place:
                pname = place.get('id') or place.get('full_name') or place.get('name')
                if pname:
                    places_list.append(pname)
            # --- end geotag extraction ---

            is_retweet = tweet.get('retweeted_status') is not None
            # In the dataset, -1 indicates the absence of a reply.
            is_reply = tweet.get('in_reply_to_status_id') not in (None, -1)


            # Unique hashtags
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

            # se il tweet è di un utente (tweet['user']['id']), aggiorna last_seen
            try:
                author_id = tweet.get('user', {}).get('id')
                if author_id:
                    last_seen_ts_by_user[author_id] = tweet['created_at'].timestamp()
            except Exception:
                pass

            ##########################################################################
            ### EDGE FEATURES COMPUTATION ############################################
            ##########################################################################
            if is_retweet:
                n_retweets += 1
                rt_uid = tweet['retweeted_status']['user']['id']
                retweet_targets[rt_uid] += 1
                ts = tweet['created_at'].timestamp()
                if rt_uid not in retweet_first_ts or ts < retweet_first_ts[rt_uid]:
                    retweet_first_ts[rt_uid] = ts
                if rt_uid not in retweet_last_ts or ts > retweet_last_ts[rt_uid]:
                    retweet_last_ts[rt_uid] = ts

                # --- Fast-RT detection: latency between retweet and original tweet creation ---
                try:
                    orig_created = None
                    # prefer retweeted_status.created_at if present
                    rs = tweet.get('retweeted_status') or {}
                    orig_created = rs.get('created_at') or rs.get('timestamp_ms') or rs.get('created_at_str')
                    if orig_created:
                        orig_dt = convert_to_datetime(orig_created)
                        if orig_dt:
                            latency = ts - orig_dt.timestamp()
                            if latency >= 0 and latency <= FAST_RT_THRESHOLD:
                                retweet_fast_count[rt_uid] += 1
                except Exception:
                    # non blocchiamo l'estrazione per formati inattesi
                    pass

            elif is_reply:
                n_replies += 1
                reply_uid = tweet.get('in_reply_to_user_id')
                if reply_uid and reply_uid != -1:
                    reply_targets[reply_uid] += 1
                    ts = tweet['created_at'].timestamp()
                    # first timestamp
                    if reply_uid not in reply_first_ts or ts < reply_first_ts[reply_uid]:
                        reply_first_ts[reply_uid] = ts
                    # last timestamp
                    if reply_uid not in reply_last_ts or ts > reply_last_ts[reply_uid]:
                        reply_last_ts[reply_uid] = ts

                    # --- Avg Reply Latency: prefer parent tweet creation time if disponibile ---
                    parent_created = tweet.get('in_reply_to_status_created_at') or tweet.get('in_reply_to_status_created_at_str')
                    if parent_created:
                        try:
                            parent_dt = convert_to_datetime(parent_created)
                            if parent_dt:
                                latency = ts - parent_dt.timestamp()
                                if latency >= 0:
                                    reply_latency_sum[reply_uid] += latency
                                    reply_latency_count[reply_uid] += 1
                        except Exception:
                            pass
            else:
                n_original += 1


            # Mentions — stored as screen_names (resolved later)
            if not is_retweet:
                raw_mentions = tweet.get('userMentionEntities', '')
                if isinstance(raw_mentions, str) and raw_mentions.strip():
                    for mn in raw_mentions.split('|'):
                        mn = mn.strip().lower()
                        
                        # If the tweet is a retweet, ignore the mentions.
                        # By ignoring the mentions, we avoid counting false interactions.
                        if mn:
                            mention_targets[mn] += 1

        if latest_tweet is None:
            return None, [], [], []

        ##########################################################################
        ### NODE FEATURES OBJECT CREATION ########################################
        ##########################################################################
        user_id_int = int(user_id)                 # int — used internally as dict key
        src_node_id = Utils.to_node_id(user_id_int) if self.file_format != 'pickle' else user_id_int
        latest_user = latest_tweet['user']

        # --- Social Influence Ratio (followers / (followers + friends)) ---
        followers_raw = latest_user.get('followers_count', 0) or 0
        friends_raw = latest_user.get('friends_count', 0) or 0
        den = followers_raw + friends_raw
        social_influence_ratio = followers_raw / den if den > 0 else 0.0

        # Account date: timestamp of account creation
        user_created_at = convert_to_datetime(latest_user.get('created_at'))
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
                uca = user_created_at if user_created_at.tzinfo else user_created_at.replace(tzinfo=timezone.utc)
                activation_age = max(0, (first_dt - uca).total_seconds())
            except Exception:
                activation_age = 0

        reg_iti = tweet_regularity_score_from_timestamps(timestamps)
        tweet_regularity_score = reg_iti['regularity_score']
        tweet_avg_interval_seconds = reg_iti['iti_mean']

        reg_daily = daily_posting_consistency(timestamps, window_days=90, min_days=14)
        daily_score = reg_daily['daily_score']
        daily_cv_log = reg_daily['daily_cv_log'] if reg_daily['daily_cv_log'] is not None else 0.0

        internal_density = internal_tweet_density(timestamps, window_days=90)

        profile_geo_enabled = int(latest_user.get('geo_enabled', False))

        # Sensitive content metrics
        sensitive_count = int(n_sensitive)
        sensitive_rate = float(sensitive_count) / n_total if n_total > 0 else 0.0

        # Source ratios
        mobile_ratio = n_mobile / n_total if n_total > 0 else 0.0
        web_ratio = n_web / n_total if n_total > 0 else 0.0
        news_manager_ratio = n_news_manager / n_total if n_total > 0 else 0.0
        bot_api_ratio = n_bot_api / n_total if n_total > 0 else 0.0

        # Source entropy (Shannon) normalized to [0,1] using log2(4)
        try:
            counts = [n_mobile, n_web, n_news_manager, n_bot_api]
            total_counts = sum(counts)
            if total_counts > 0:
                probs = [c / total_counts for c in counts if c > 0]
                raw_entropy = -sum(p * math.log2(p) for p in probs)
                source_entropy = raw_entropy / math.log2(4)  # normalize by log2(4)
            else:
                source_entropy = 0.0
        except Exception:
            source_entropy = 0.0

        # --- Hashtag entropy (Shannon) ---
        try:
            # total hashtag occurrences (non unici)
            hashtag_counts = Counter()
            # se nel loop sopra hai raccolto solo l'insieme `hashtags`, allora
            # devi invece contare le occorrenze: se non le hai, puoi ricostruirle
            # dal campo hashtagEntities per ogni tweet; qui assumiamo che
            # `hashtags` sia l'insieme e che tu abbia anche raccolto i conteggi
            # durante il loop in una struttura `hashtag_counter` (preferibile).
            # Se non esiste, fallback: treat unique only (entropy=0).
            if 'hashtag_counter' in locals():
                hashtag_counts = hashtag_counter
            else:
                # fallback: uniform distribution over unique hashtags (no entropy)
                hashtag_counts = Counter({h: 1 for h in hashtags})

            total_hashtags = sum(hashtag_counts.values())
            unique_hashtags = len(hashtag_counts)

            if total_hashtags > 0 and unique_hashtags > 1:
                probs = [c / total_hashtags for c in hashtag_counts.values() if c > 0]
                raw_hashtag_entropy = -sum(p * math.log2(p) for p in probs)
                hashtag_entropy = raw_hashtag_entropy / math.log2(unique_hashtags)
            else:
                hashtag_entropy = 0.0
        except Exception:
            hashtag_entropy = 0.0

        # also expose simple counts for downstream use
        n_hashtags_total = int(total_hashtags) if 'total_hashtags' in locals() else 0
        n_unique_hashtags = unique_hashtags if 'unique_hashtags' in locals() else len(hashtags)


        def log1p(x):
            return round(math.log1p(x), 2)

        features = {
            'user_id':           user_id_int,     # int — kept for internal joins (do not modify)
            'user_node_id':      src_node_id,  # hex — written to CSV as the final node ID
            
            # --- Activity Features ---
            'total':             log1p(n_total),      # Total number of tweets by this user
            'retweets':          log1p(n_retweets),
            'replies':           log1p(n_replies),
            'original':          log1p(n_original),

            # --- Profile Features (from their most recent tweet) ---
            'likes':             log1p(latest_user.get('favourites_count', 0)),
            'followers':         log1p(latest_user.get('followers_count', 0)),
            'following':         log1p(latest_user.get('friends_count', 0)),
            'verified':          1 if latest_user.get('verified', False) else 0,
            'account_date':      account_date,
            'listed_count':      log1p(latest_user.get('listed_count', 0)),
            'favourites_count':  log1p(latest_user.get('favourites_count', 0)),
            'reputation_score':  round(social_influence_ratio, 2),

            # --- Network/Content Features ---
            'n_unique_hashtags': len(hashtags),
            'n_unique_mentions': len(mention_targets),
            'n_hashtags_total':  log1p(n_hashtags_total),
            'hashtag_entropy':   round(hashtag_entropy, 4),

            # --- Automation / Regularity Features ---
            'activation_age':             log1p(activation_age),
            'tweet_regularity_score':     log1p(tweet_regularity_score),
            'regularity_reliable':        1 if n_total >= 20 else 0,
            'tweet_avg_interval_seconds': log1p(tweet_avg_interval_seconds),
            'daily_score':                round(daily_score, 2),
            'daily_cv_log':               round(daily_cv_log, 2),
            'internal_tweet_density':     round(internal_density, 2),
            'profile_has_url':            int(user_profile_has_url),
            'geo_enabled_flag':           profile_geo_enabled,

            # --- Metadata (not features) ---
            'screen_name':                latest_user.get('screen_name', ''), # Used to resolve mentions
            
            # Sensitive content metrics
            'sensitive_count':            log1p(sensitive_count),
            'sensitive_rate':             sensitive_rate,

            # Source counts and ratios
            'mobile_ratio':            round(mobile_ratio, 4),
            'web_ratio':               round(web_ratio, 4),
            'news_manager_ratio':      round(news_manager_ratio, 4),
            'bot_api_ratio':           round(bot_api_ratio, 4),
            'source_entropy':          round(source_entropy, 4),
        }

        ##########################################################################
        ### EDGE FEATURES OBJECTS CREATION #######################################
        ##########################################################################
        def fast_retweet_ratio(dst_uid):
            count = retweet_fast_count.get(dst_uid, 0)
            total = retweet_targets.get(dst_uid, 0)
            return round((count / float(total)) if total > 0 else 0.0, 4)

        def retweet_lifespan(dst_uid):
            first = retweet_first_ts.get(dst_uid)
            last = retweet_last_ts.get(dst_uid)
            if first is not None and last is not None and last >= first:
                return log1p(last - first)
            return 0.0

        def avg_reply_latency(dst_uid):
            dst_reply_latency = reply_latency_sum.get(dst_uid, 0)
            dst_reply_count = reply_latency_count.get(dst_uid, 1)
            avg_latency = dst_reply_latency / dst_reply_count if dst_reply_count > 0 else 0.0
            return log1p(max(0, avg_latency))

        def reply_lifespan(dst_uid):
            first = reply_first_ts.get(dst_uid)
            last = reply_last_ts.get(dst_uid)
            if first is not None and last is not None and last >= first:
                return log1p(last - first)
            return 0.0


        # Retweet and reply edges — node IDs in hex for compact CSV output
        edges_retweet_features = [
            (
                src_node_id,                        # source node ID (hex string)
                Utils.to_node_id(int(dst_uid)) if self.file_format != 'pickle' else int(dst_uid),     # destination node ID (hex string)
                weight,                             # retweet count (weight)
                retweet_lifespan(dst_uid),          # lifespan of the retweet interaction
                fast_retweet_ratio(dst_uid)         # ratio of fast retweets
            )
            for dst_uid, weight in retweet_targets.items()
        ]
        edges_reply_features = [
            (
                src_node_id,                        # source node ID (hex string)
                Utils.to_node_id(int(dst_uid)) if self.file_format != 'pickle' else int(dst_uid),     # destination node ID (hex string)
                weight,                             # reply count (weight)
                reply_lifespan(dst_uid),            # lifespan of the reply interaction
                avg_reply_latency(dst_uid)          # average reply latency
            )
            for dst_uid, weight in reply_targets.items()
        ]

        # Mention edges — src in hex, screen_name kept as-is (resolved at merge time)
        mention_edges_raw_features = [
            (
                src_node_id,        # source node ID (hex string)
                screen_name,        # destination screen_name (to be resolved later)
                weight              # mention count (weight)
            )
            for screen_name, weight in mention_targets.items()
        ]

        return features, edges_retweet_features, edges_reply_features, mention_edges_raw_features

    # ─────────────────────────────────────────────────────────────────────────
    # CHECKPOINT I/O
    # ─────────────────────────────────────────────────────────────────────────

    def save_checkpoint(self, user_features_list, edges_rt, edges_reply,
                        mention_edges_raw, screen_names_list, batch_id):
        """
        Persist intermediate results for one batch to disk.

        Files written under tmp/<run_id>/<batch_id>/:
            user_features.csv       columns: user_id, total, retweets, replies, ...
            edges_retweet.csv       columns: src, dst, weight
            edges_reply.csv         columns: src, dst, weight
            edges_mention_raw.csv   columns: src, screen_name, weight  (unresolved)
            screen_name_map.csv     columns: screen_name, user_id

        :param user_features_list: List of feature dicts.
        :param edges_rt: List of (src, dst, weight) for retweet edges.
        :param edges_reply: List of (src, dst, weight) for reply edges.
        :param mention_edges_raw: List of (src, screen_name, weight).
        :param screen_names_list: List of (screen_name, user_id).
        :param batch_id: Integer identifier for this batch.
        """
        dir_path = os.sep.join([
            self.output_file_path, self.checkpoint_folder, self.id, str(batch_id)
        ])
        os.makedirs(dir_path, exist_ok=True)

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
                f['mobile_ratio'],f['web_ratio'],f['news_manager_ratio'],
                f['bot_api_ratio'],f['source_entropy']
            ]
            for f in user_features_list
        ]

        if self.file_format == "pickle":
            user_features_header_checkpoint = [
                'user_node_id', 'total', 'retweets', 'replies', 'original', 'likes', 'followers', 'following',
                'verified', 'account_date', 'listed_count', 'favourites_count', 'reputation_score',
                'n_unique_hashtags', 'n_unique_mentions', 'n_hashtags_total', 'hashtag_entropy',
                'activation_age', 'tweet_regularity_score', 'regularity_reliable',
                'tweet_avg_interval_seconds', 'daily_score', 'daily_cv_log', 'internal_tweet_density',
                'profile_has_url', 'geo_enabled_flag', 'sensitive_rate', 'mobile_ratio', 'web_ratio',
                'news_manager_ratio', 'bot_api_ratio', 'source_entropy'
            ]
            Writer.write_on_pickle(os.sep.join([dir_path, "user_features.pkl"]), features_rows, columns=user_features_header_checkpoint)
            Writer.write_on_pickle(os.sep.join([dir_path, "edges_retweet.pkl"]), edges_rt, columns=['src', 'dst', 'weight', 'lifespan', 'fast_rt_ratio'])
            Writer.write_on_pickle(os.sep.join([dir_path, "edges_reply.pkl"]), edges_reply, columns=['src', 'dst', 'weight', 'lifespan', 'avg_reply_latency_seconds'])
            Writer.write_on_pickle(os.sep.join([dir_path, "edges_mention_raw.pkl"]), [(src, sn, w) for src, sn, w in mention_edges_raw], columns=['src', 'screen_name', 'weight'])
            Writer.write_on_pickle(os.sep.join([dir_path, "screen_name_map.pkl"]), screen_names_list, columns=['screen_name', 'user_id'])
        else:
            Writer.write_on_csv(os.sep.join([dir_path, "user_features"]), features_rows)
            Writer.write_on_csv(os.sep.join([dir_path, "edges_retweet"]), edges_rt)
            Writer.write_on_csv(os.sep.join([dir_path, "edges_reply"]), edges_reply)
            Writer.write_on_csv(
                os.sep.join([dir_path, "edges_mention_raw"]),
                [(src, sn, w) for src, sn, w in mention_edges_raw]
            )
            Writer.write_on_csv(os.sep.join([dir_path, "screen_name_map"]), screen_names_list)

    # ─────────────────────────────────────────────────────────────────────────
    # MERGE
    # ─────────────────────────────────────────────────────────────────────────

    def merge_checkpoints(self):
        """
        Merge all batch checkpoints into final output CSVs.

        For mention edges: reads all intermediate screen_name_map files to
        build a dictionary, then resolves screen_names to user_id hashes.
        Edges whose target screen_name is not found in the map
        (external users not present in the dataset) are silently dropped.
        """
        checkpoint_dir = os.sep.join([
            self.output_file_path, self.checkpoint_folder, self.id
        ])
        out_dir = os.sep.join([self.output_file_path, self.id])
        os.makedirs(out_dir, exist_ok=True)

        self.logger.info("Merging checkpoints...")

        all_features = []
        all_rt = []
        all_reply = []
        all_mention = []
        
        screen_name_map = {}
        valid_user_node_ids = set()

        # First pass: build the complete screen_name_map from all batches
        self.logger.info("Building complete screen_name map from checkpoints...")
        for batch_dir in sorted(os.listdir(checkpoint_dir)):
            batch_path = os.sep.join([checkpoint_dir, batch_dir])
            if not os.path.isdir(batch_path):
                continue
                
            map_file = os.sep.join([batch_path, "screen_name_map" + (".pkl" if self.file_format == "pickle" else "")])
            if os.path.exists(map_file):
                for row in Writer.load_checkpoint_file(map_file):
                    screen_name = row[0].lower()
                    uid = int(row[1])
                    screen_name_map[screen_name] = uid
                    valid_user_node_ids.add(Utils.to_node_id(uid) if self.file_format != 'pickle' else uid)

        resolved_mention = 0
        dropped_mention = 0
        dropped_rt = 0
        dropped_reply = 0

        # --- In-degree tracking ---
        received_retweets = defaultdict(int)
        received_replies = defaultdict(int)
        received_mentions = defaultdict(int)

        # Second pass: read all data and resolve/filter edges
        self.logger.info("Filtering and resolving edges...")
        for batch_dir in sorted(os.listdir(checkpoint_dir)):
            batch_path = os.sep.join([checkpoint_dir, batch_dir])
            if not os.path.isdir(batch_path):
                continue

            feat_file = os.sep.join([batch_path, "user_features" + (".pkl" if self.file_format == "pickle" else "")])
            if os.path.exists(feat_file):
                all_features.extend(Writer.load_checkpoint_file(feat_file))

            rt_file = os.sep.join([batch_path, "edges_retweet" + (".pkl" if self.file_format == "pickle" else "")])
            if os.path.exists(rt_file):
                for row in Writer.load_checkpoint_file(rt_file):
                    if row[1] in valid_user_node_ids:
                        all_rt.append(row)
                        received_retweets[row[1]] += int(float(row[2]))
                    else:
                        dropped_rt += 1

            rep_file = os.sep.join([batch_path, "edges_reply" + (".pkl" if self.file_format == "pickle" else "")])
            if os.path.exists(rep_file):
                for row in Writer.load_checkpoint_file(rep_file):
                    if row[1] in valid_user_node_ids:
                        all_reply.append(row)
                        received_replies[row[1]] += int(float(row[2]))
                    else:
                        dropped_reply += 1

            men_file = os.sep.join([batch_path, "edges_mention_raw" + (".pkl" if self.file_format == "pickle" else "")])
            if os.path.exists(men_file):
                for row in Writer.load_checkpoint_file(men_file):
                    src, screen_name, weight = row[0], row[1], row[2]
                    dst = screen_name_map.get(screen_name.lower())
                    if dst is not None:
                        dst_node_id = Utils.to_node_id(dst) if self.file_format != 'pickle' else dst
                        all_mention.append((src, dst_node_id, weight))
                        received_mentions[dst_node_id] += int(float(weight))
                        resolved_mention += 1
                    else:
                        dropped_mention += 1  # External user — link dropped

        # Inject computed incoming degrees into feature rows
        for row in all_features:
            uid = row[0]
            rt_count = received_retweets.get(uid, 0)
            rep_count = received_replies.get(uid, 0)
            men_count = received_mentions.get(uid, 0)

            row.insert(15, rt_count)
            row.insert(16, rep_count)
            row.insert(17, men_count)

        # Write CSV headers
        user_features_header = [
                'user_node_id', 'total', 'retweets', 'replies',
                'original', 'likes', 'followers', 'following',
                'verified', 'account_date', 'listed_count',
                'favourites_count', 'reputation_score',
                'n_unique_hashtags', 'n_unique_mentions',
                'n_hashtags_total', 'hashtag_entropy',
                'activation_age', 'tweet_regularity_score',
                'regularity_reliable', 'tweet_avg_interval_seconds', 
                'daily_score', 'daily_cv_log', 'internal_tweet_density', 
                'profile_has_url', 'geo_enabled_flag', 'sensitive_rate',
                'mobile_ratio', 'web_ratio', 'news_manager_ratio',
                'bot_api_ratio', 'source_entropy',
                # Additionnal features computed at merge time:
                'received_retweets', 'received_replies', 'received_mentions',
        ]
        retweet_header = ['src', 'dst', 'weight', 'lifespan', 'avg_reply_latency_seconds']
        reply_header = ['src', 'dst', 'weight', 'lifespan', 'fast_rt_ratio']
        mention_header = ['src', 'dst', 'weight']
        screen_name_map_header = ['screen_name', 'user_id']

        if self.file_format == "pickle":
            Writer.write_on_pickle(os.sep.join([out_dir, "user_features.pkl"]), all_features, columns=user_features_header)
            Writer.write_on_pickle(os.sep.join([out_dir, "edges_retweet.pkl"]), all_rt, columns=retweet_header)
            Writer.write_on_pickle(os.sep.join([out_dir, "edges_reply.pkl"]), all_reply, columns=reply_header)
            Writer.write_on_pickle(os.sep.join([out_dir, "edges_mention.pkl"]), all_mention, columns=mention_header)
            map_rows = [(sn, (Utils.to_node_id(uid) if self.file_format != 'pickle' else uid)) for sn, uid in screen_name_map.items()]
            Writer.write_on_pickle(os.sep.join([out_dir, "screen_name_map.pkl"]), map_rows, columns=screen_name_map_header)
        else:
            Writer.write_on_csv(os.sep.join([out_dir, "user_features.csv"]), [user_features_header])
            Writer.write_on_csv(os.sep.join([out_dir, "edges_retweet.csv"]), [retweet_header])
            Writer.write_on_csv(os.sep.join([out_dir, "edges_reply.csv"]), [reply_header])
            Writer.write_on_csv(os.sep.join([out_dir, "edges_mention.csv"]), [mention_header])
            Writer.write_on_csv(os.sep.join([out_dir, "screen_name_map.csv"]), [screen_name_map_header])

            # Append data
            Writer.write_on_csv(os.sep.join([out_dir, "user_features.csv"]), all_features)
            Writer.write_on_csv(os.sep.join([out_dir, "edges_retweet.csv"]), all_rt)
            Writer.write_on_csv(os.sep.join([out_dir, "edges_reply.csv"]), all_reply)
            Writer.write_on_csv(os.sep.join([out_dir, "edges_mention.csv"]), all_mention)

            map_rows = [(sn, (Utils.to_node_id(uid) if self.file_format != 'pickle' else uid)) for sn, uid in screen_name_map.items()]
            Writer.write_on_csv(os.sep.join([out_dir, "screen_name_map.csv"]), map_rows)

        # Metadata file useful for MLFlow, to get the name of the used collection
        import json
        metadata = {
            "collection": self.get_collection(),
            "date": datetime.now(timezone.utc).isoformat(),
            "run_id": self.id,
            "users_processed": len(valid_user_node_ids)
        }
        with open(os.sep.join([out_dir, "metadata.json"]), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        if self.delete_tmp_after_merge:
            self.logger.info(f"Deleting temporary checkpoint directory: {checkpoint_dir}")
            try:
                shutil.rmtree(checkpoint_dir)
            except Exception as e:
                self.logger.warning(f"Failed to delete {checkpoint_dir}: {e}")

        self.logger.info(
            f"Merge complete. Output in {out_dir}\n"
            f"  Mentions: {resolved_mention} resolved, {dropped_mention} dropped (external users).\n"
            f"  Retweets: {len(all_rt)} kept, {dropped_rt} dropped (external users).\n"
            f"  Replies:  {len(all_reply)} kept, {dropped_reply} dropped (external users)."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, checkpoint_every=500, max_users=None):
        """
        Main entry point. Connects to MongoDB and streams all tweets sorted
        by user.id through a cursor, processing one user at a time.

        The MongoDB cursor is a lazy iterator — tweets are fetched on demand,
        so only one user's tweets are held in memory at a time.
        Checkpoints are written to disk every `checkpoint_every` users.

        :param checkpoint_every: Number of users to process before writing
                                 a checkpoint to disk (default: 500).
        :param max_users: Optional hard limit on the number of users to extract.
                          If set (e.g., 500), extraction stops early.
        """
        client = self.connect()
        col = self.get_db()[self.get_collection()]
        _, proj = mongoQueries.extract_tweets_for_user_graph()

        self.logger.info(f"[GraphGenerationUser] Starting extraction. Run ID: {self.id}")

        # Stream all tweets sorted by user.id — cursor stays lazy (no list())
        cursor = col.find({}, proj).sort("user.id", ASCENDING)

        batch_id = 0
        user_count = 0
        features_list = []
        edges_rt = []
        edges_reply = []
        mention_raw = []
        screen_names_list = []

        for user_id, user_tweets in itertools.groupby(
            cursor, key=lambda t: t['user']['id']
        ):
            # Pass the iterator directly instead of list() to stream tweets one by one
            features, rt, rep, men = self.process_user_tweets(user_id, user_tweets)

            if features is None:
                continue

            # Register screen_name -> user_id
            screen_name = features.get('screen_name', '')
            if screen_name:
                screen_names_list.append((screen_name.lower(), features['user_id']))

            features_list.append(features)
            edges_rt.extend(rt)
            edges_reply.extend(rep)
            mention_raw.extend(men)
            user_count += 1

            # Periodic checkpoint
            if user_count % checkpoint_every == 0:
                self.logger.info(
                    f"[Batch {batch_id}] Checkpoint at {user_count} users. "
                    f"{len(features_list)} features, "
                    f"{len(edges_rt)} RT, {len(edges_reply)} reply, "
                    f"{len(mention_raw)} mention edges."
                )
                self.save_checkpoint(
                    features_list, edges_rt, edges_reply, 
                    mention_raw, screen_names_list, batch_id
                )
                features_list, edges_rt, edges_reply, mention_raw, screen_names_list = [], [], [], [], []
                batch_id += 1
            if max_users is not None and user_count >= max_users:
                self.logger.info(f"Reached max_users limit ({max_users}). Stopping extraction early.")
                break
        # Save remaining data
        if features_list:
            self.save_checkpoint(
                features_list, edges_rt, edges_reply, 
                mention_raw, screen_names_list, batch_id
            )
            batch_id += 1

        self.logger.info(
            f"All {user_count} users processed ({batch_id} checkpoints). Merging..."
        )
        self.merge_checkpoints()
        self.logger.info("Extraction complete.")

        client.close()