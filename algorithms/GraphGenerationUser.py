import itertools
import logging
import math
import os
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pymongo import ASCENDING
from dateutil import parser  # Import to parse ISO dates

import numpy as np
import sys
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.Utils import Utils
from Utils.Writer import Writer
from algorithms.MongoConnection import MongoConnection
import algorithms.mongoQueries as mongoQueries


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
        edges_retweet.csv     : (src_hash, dst_hash, weight)
        edges_reply.csv       : (src_hash, dst_hash, weight)
        edges_mention.csv     : (src_hash, dst_hash, weight) — resolved via map
        screen_name_map.csv   : (screen_name, user_id_hash) for reference

    Mention edges whose target screen_name cannot be resolved to a known
    user in the dataset are silently dropped.
    """

    logger = logging.getLogger('GraphGenerationUser')

    def __init__(self, uri, database_name, collection, output_file_path,
                 username=None, password=None, auth_source=None, auth_mechanism=None,
                 delete_tmp_after_merge=False):
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
        """
        super().__init__(uri, username, password, auth_source, auth_mechanism,
                         db=None, database_name=database_name, collection=collection)
        self.output_file_path = output_file_path
        self.id = uuid.uuid1().hex
        self.checkpoint_folder = "tmp"
        self.delete_tmp_after_merge = delete_tmp_after_merge

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

        Mention edges are returned as raw (src_hash, screen_name, weight) tuples
        because screen_names must be resolved to user_ids at merge time.

        :param user_id: The Twitter user ID (integer) of the author.
        :param tweets: Iterable of tweet documents for this user.
        :return: (features_dict, edges_retweet, edges_reply, mention_edges_raw)
                 Returns (None, [], [], []) if the tweet list is empty.
        """
        n_total = 0
        n_retweets = 0
        n_replies = 0
        n_original = 0
        latest_tweet = None
        hashtags = set()
        timestamps = []
        coords_list = []
        places_list = []

        retweet_targets = defaultdict(int)   # dst_user_id (int) → interaction count
        reply_targets = defaultdict(int)     # dst_user_id (int) → interaction count
        mention_targets = defaultdict(int)   # screen_name (str) → interaction count

        seen_tweet_ids = set()


        for tweet in tweets:
            tweet_id = tweet.get('id')
            if tweet_id in seen_tweet_ids:
                continue
            seen_tweet_ids.add(tweet_id)

            # Ensure created_at is a datetime object
            if isinstance(tweet['created_at'], str):
                tweet['created_at'] = parser.parse(tweet['created_at'])

            n_total += 1
            timestamps.append(tweet['created_at'].timestamp())

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

            if is_retweet:
                n_retweets += 1
                rt_uid = tweet['retweeted_status']['user']['id']
                retweet_targets[rt_uid] += 1
            elif is_reply:
                n_replies += 1
                reply_uid = tweet.get('in_reply_to_user_id')
                if reply_uid and reply_uid != -1:
                    reply_targets[reply_uid] += 1
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

            # Unique hashtags
            raw_ht = tweet.get('hashtagEntities', '')
            if isinstance(raw_ht, str) and raw_ht.strip():
                for ht in raw_ht.split('|'):
                    ht = ht.strip().lower()
                    if ht:
                        hashtags.add(ht)

            # Keep latest tweet for snapshot fields (followers, verified, etc.)
            if latest_tweet is None or tweet['created_at'] > latest_tweet['created_at']:
                latest_tweet = tweet

        if latest_tweet is None:
            return None, [], [], []

        src_hash = int(user_id)                 # int — used internally as dict key
        src_node_id = Utils.to_node_id(src_hash) # hex str — used only for CSV output
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
        profile_url_raw = latest_user.get('url') or ''
        entities_urls = []
        try:
            entities_urls = latest_user.get('entities', {}).get('url', {}).get('urls', []) or []
        except Exception:
            entities_urls = []

        # prefer expanded_url if available
        profile_urls = []
        if profile_url_raw:
            profile_urls.append(profile_url_raw)
        for u in entities_urls:
            if isinstance(u, dict):
                expanded = u.get('expanded_url') or u.get('url')
                if expanded:
                    profile_urls.append(expanded)


        # normalize and dedupe
        profile_urls = list(dict.fromkeys([str(x).strip() for x in profile_urls if x]))
        profile_has_url = 1 if profile_urls else 0
        profile_primary_domain = ''
        if profile_has_url:
            # extract domain simply
            try:
                from urllib.parse import urlparse
                parsed = urlparse(profile_urls[0])
                profile_primary_domain = parsed.netloc.lower()
            except Exception:
                profile_primary_domain = ''

        first_tweet_ts = min(timestamps) if timestamps else 0.0

        activation_age_days = 0
        if user_created_at and first_tweet_ts:
            try:
                first_dt = datetime.fromtimestamp(first_tweet_ts, tz=timezone.utc)
                uca = user_created_at if user_created_at.tzinfo else user_created_at.replace(tzinfo=timezone.utc)
                activation_age_days = max(0, (first_dt - uca).days)
            except Exception:
                activation_age_days = 0

        reg_iti = tweet_regularity_score_from_timestamps(timestamps)
        tweet_regularity_score = reg_iti['regularity_score']
        tweet_avg_interval_seconds = reg_iti['iti_mean']

        reg_daily = daily_posting_consistency(timestamps, window_days=90, min_days=14)
        daily_score = reg_daily['daily_score']
        daily_cv_log = reg_daily['daily_cv_log'] if reg_daily['daily_cv_log'] is not None else 0.0

        international_density = internal_tweet_density(timestamps, window_days=90)

        profile_geo_enabled = int(latest_user.get('geo_enabled', False))

        # ---------------------------------------------------------------------
        # [STUDENTS] NODE FEATURES DEFINITION
        # ---------------------------------------------------------------------
        # This dictionary defines the attributes (features) that will be 
        # assigned to each user node in the final graph. 
        #
        # If you want to compute new features for your GNN 
        # (e.g., average tweet length, sentiment score, activity frequency), 
        # you need to:
        # 1. Compute the value in the loop above (lines 108-144).
        # 2. Add the new feature as a key-value pair in this dictionary.
        # 3. Update the CSV column headers in `save_checkpoint` (around line 265).
        # ---------------------------------------------------------------------
        def log1p(x):
            return round(math.log1p(x), 2)

        features = {
            'user_id':           src_hash,     # int — kept for internal joins (do not modify)
            'user_node_id':      src_node_id,  # hex — written to CSV as the final node ID
            
            # --- Activity Features ---
            'total':             n_total,      # Total number of tweets by this user
            'retweets':          n_retweets,
            'replies':           n_replies,
            'original':          n_original,
            
            # --- Profile Features (from their most recent tweet) ---
            'likes':             log1p(latest_user.get('favourites_count', 0)),
            'followers':         log1p(latest_user.get('followers_count', 0)),
            'following':         log1p(latest_user.get('friends_count', 0)),
            'verified':          1 if latest_user.get('verified', False) else 0,
            'account_date':      account_date,
            'listed_count':      log1p(latest_user.get('listed_count', 0)),
            'favourites_count':  log1p(latest_user.get('favourites_count', 0)),
            'reputation_score': round(social_influence_ratio, 4),

            # --- Network/Content Features ---
            'n_unique_hashtags': len(hashtags),
            'n_unique_mentions': len(mention_targets),

            # --- Automation / Regularity Features ---
            'activation_age_days':        round(math.log1p(activation_age_days), 2),
            'tweet_regularity_score':     log1p(tweet_regularity_score),
            'regularity_reliable':        1 if n_total >= 20 else 0,
            'tweet_avg_interval_seconds': log1p(tweet_avg_interval_seconds),
            'daily_score':                round(daily_score, 2),
            'daily_cv_log':               round(daily_cv_log, 2),
            'internal_tweet_density':     round(international_density, 2),
            'profile_has_url':            profile_has_url,
            'geo_enabled_flag':           profile_geo_enabled,

            # --- Metadata (not features) ---
            'screen_name':                latest_user.get('screen_name', ''), # Used to resolve mentions
        }


        # Retweet and reply edges — node IDs in hex for compact CSV output
        edges_retweet = [
            (src_node_id, Utils.to_node_id(int(dst_uid)), weight)
            for dst_uid, weight in retweet_targets.items()
        ]
        edges_reply = [
            (src_node_id, Utils.to_node_id(int(dst_uid)), weight)
            for dst_uid, weight in reply_targets.items()
        ]

        # Mention edges — src in hex, screen_name kept as-is (resolved at merge time)
        mention_edges_raw = [
            (src_node_id, screen_name, weight)
            for screen_name, weight in mention_targets.items()
        ]

        return features, edges_retweet, edges_reply, mention_edges_raw

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
                f['activation_age_days'], f['tweet_regularity_score'], 
                f['regularity_reliable'], f['tweet_avg_interval_seconds'], 
                f['daily_score'], f['daily_cv_log'], f['internal_tweet_density'], 
                f['profile_has_url'], f['geo_enabled_flag'],
            ]
            for f in user_features_list
        ]

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
                
            map_file = os.sep.join([batch_path, "screen_name_map"])
            if os.path.exists(map_file):
                for row in Writer.load_checkpoint_file(map_file):
                    screen_name = row[0].lower()
                    uid_hash = int(row[1])
                    screen_name_map[screen_name] = uid_hash
                    valid_user_node_ids.add(Utils.to_node_id(uid_hash))

        resolved_mention = 0
        dropped_mention = 0
        dropped_rt = 0
        dropped_reply = 0

        # Second pass: read all data and resolve/filter edges
        self.logger.info("Filtering and resolving edges...")
        for batch_dir in sorted(os.listdir(checkpoint_dir)):
            batch_path = os.sep.join([checkpoint_dir, batch_dir])
            if not os.path.isdir(batch_path):
                continue

            feat_file = os.sep.join([batch_path, "user_features"])
            if os.path.exists(feat_file):
                all_features.extend(Writer.load_checkpoint_file(feat_file))

            rt_file = os.sep.join([batch_path, "edges_retweet"])
            if os.path.exists(rt_file):
                for row in Writer.load_checkpoint_file(rt_file):
                    if row[1] in valid_user_node_ids:
                        all_rt.append(row)
                    else:
                        dropped_rt += 1

            rep_file = os.sep.join([batch_path, "edges_reply"])
            if os.path.exists(rep_file):
                for row in Writer.load_checkpoint_file(rep_file):
                    if row[1] in valid_user_node_ids:
                        all_reply.append(row)
                    else:
                        dropped_reply += 1

            men_file = os.sep.join([batch_path, "edges_mention_raw"])
            if os.path.exists(men_file):
                for row in Writer.load_checkpoint_file(men_file):
                    src, screen_name, weight = row[0], row[1], row[2]
                    dst = screen_name_map.get(screen_name.lower())
                    if dst is not None:
                        all_mention.append((src, Utils.to_node_id(dst), weight))
                        resolved_mention += 1
                    else:
                        dropped_mention += 1  # External user — link dropped

        # Write CSV headers
        user_features_header = [
            'user_node_id', 'total', 'retweets', 'replies', 'original', 'likes', 'followers', 'following',
            'verified', 'account_date', 'listed_count', 'favourites_count', 'reputation_score',
            'n_unique_hashtags', 'n_unique_mentions','activation_age_days', 'tweet_regularity_score', 
            'regularity_reliable', 'tweet_avg_interval_seconds', 'daily_score', 'daily_cv_log', 
            'internal_tweet_density', 'profile_has_url', 'geo_enabled_flag'
        ]
        edge_header = ['src', 'dst', 'weight']
        mention_header = ['src', 'dst', 'weight']
        screen_name_map_header = ['screen_name', 'user_id']

        Writer.write_on_csv(os.sep.join([out_dir, "user_features.csv"]), [user_features_header])
        Writer.write_on_csv(os.sep.join([out_dir, "edges_retweet.csv"]), [edge_header])
        Writer.write_on_csv(os.sep.join([out_dir, "edges_reply.csv"]), [edge_header])
        Writer.write_on_csv(os.sep.join([out_dir, "edges_mention.csv"]), [mention_header])
        Writer.write_on_csv(os.sep.join([out_dir, "screen_name_map.csv"]), [screen_name_map_header])

        # Append data
        Writer.write_on_csv(os.sep.join([out_dir, "user_features.csv"]), all_features)
        Writer.write_on_csv(os.sep.join([out_dir, "edges_retweet.csv"]), all_rt)
        Writer.write_on_csv(os.sep.join([out_dir, "edges_reply.csv"]), all_reply)
        Writer.write_on_csv(os.sep.join([out_dir, "edges_mention.csv"]), all_mention)

        map_rows = [(sn, Utils.to_node_id(uid)) for sn, uid in screen_name_map.items()]
        Writer.write_on_csv(os.sep.join([out_dir, "screen_name_map.csv"]), map_rows)

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
                    features_list, edges_rt, edges_reply, mention_raw, screen_names_list, batch_id
                )
                features_list, edges_rt, edges_reply, mention_raw, screen_names_list = [], [], [], [], []
                batch_id += 1
            if max_users is not None and user_count >= max_users:
                self.logger.info(f"Reached max_users limit ({max_users}). Stopping extraction early.")
                break
        # Save remaining data
        if features_list:
            self.save_checkpoint(
                features_list, edges_rt, edges_reply, mention_raw, screen_names_list, batch_id
            )
            batch_id += 1

        self.logger.info(
            f"All {user_count} users processed ({batch_id} checkpoints). Merging..."
        )
        self.merge_checkpoints()
        self.logger.info("Extraction complete.")

        client.close()