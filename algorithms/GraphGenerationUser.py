import itertools
import logging
import os
import uuid
from collections import defaultdict
from pymongo import ASCENDING

import sys
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.Utils import Utils
from Utils.Writer import Writer
from algorithms.MongoConnection import MongoConnection
import algorithms.mongoQueries as mongoQueries


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

        retweet_targets = defaultdict(int)   # dst_user_id (int) → interaction count
        reply_targets = defaultdict(int)     # dst_user_id (int) → interaction count
        mention_targets = defaultdict(int)   # screen_name (str) → interaction count

        seen_tweet_ids = set()

        for tweet in tweets:
            tweet_id = tweet.get('id')
            if tweet_id in seen_tweet_ids:
                continue
            seen_tweet_ids.add(tweet_id)

            n_total += 1
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
        user_created_at = latest_user.get('created_at')

        # Account age: days between account creation and last observed tweet
        account_age_days = 0
        if user_created_at:
            try:
                account_age_days = (latest_tweet['created_at'] - user_created_at).days
            except Exception:
                account_age_days = 0

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
        features = {
            'user_id':           src_hash,     # int — kept for internal joins (do not modify)
            'user_node_id':      src_node_id,  # hex — written to CSV as the final node ID
            
            # --- Activity Features ---
            'total':             n_total,      # Total number of tweets by this user
            'retweets':          n_retweets,
            'replies':           n_replies,
            'original':          n_original,
            
            # --- Profile Features (from their most recent tweet) ---
            'likes':             latest_user.get('favourites_count', 0),
            'followers':         latest_user.get('followers_count', 0),
            'following':         latest_user.get('friends_count', 0),
            'verified':          1 if latest_user.get('verified', False) else 0,
            'account_age_days':  account_age_days,
            
            # --- Network/Content Features ---
            'n_unique_hashtags': len(hashtags),
            'n_unique_mentions': len(mention_targets),
            
            # --- Metadata ---
            'screen_name':       latest_user.get('screen_name', ''), # Used to resolve mentions
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
                f['verified'], f['account_age_days'],
                f['n_unique_hashtags'], f['n_unique_mentions'],
                f['screen_name']
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

        # First pass: build the complete screen_name_map from all batches
        self.logger.info("Building complete screen_name map from checkpoints...")
        for batch_dir in sorted(os.listdir(checkpoint_dir)):
            batch_path = os.sep.join([checkpoint_dir, batch_dir])
            if not os.path.isdir(batch_path):
                continue
                
            map_file = os.sep.join([batch_path, "screen_name_map"])
            if os.path.exists(map_file):
                for row in Writer.load_checkpoint_file(map_file):
                    screen_name_map[row[0].lower()] = int(row[1])

        resolved = 0
        dropped = 0

        # Second pass: read all data and resolve mentions
        for batch_dir in sorted(os.listdir(checkpoint_dir)):
            batch_path = os.sep.join([checkpoint_dir, batch_dir])
            if not os.path.isdir(batch_path):
                continue

            feat_file = os.sep.join([batch_path, "user_features"])
            if os.path.exists(feat_file):
                all_features.extend(Writer.load_checkpoint_file(feat_file))

            rt_file = os.sep.join([batch_path, "edges_retweet"])
            if os.path.exists(rt_file):
                all_rt.extend(Writer.load_checkpoint_file(rt_file))

            rep_file = os.sep.join([batch_path, "edges_reply"])
            if os.path.exists(rep_file):
                all_reply.extend(Writer.load_checkpoint_file(rep_file))

            men_file = os.sep.join([batch_path, "edges_mention_raw"])
            if os.path.exists(men_file):
                for row in Writer.load_checkpoint_file(men_file):
                    src, screen_name, weight = row[0], row[1], row[2]
                    dst = screen_name_map.get(screen_name.lower())
                    if dst is not None:
                        all_mention.append((src, Utils.to_node_id(dst), weight))
                        resolved += 1
                    else:
                        dropped += 1  # External user — link dropped

        # Write final output files
        Writer.write_on_csv(os.sep.join([out_dir, "user_features"]), all_features)
        Writer.write_on_csv(os.sep.join([out_dir, "edges_retweet"]), all_rt)
        Writer.write_on_csv(os.sep.join([out_dir, "edges_reply"]), all_reply)
        Writer.write_on_csv(os.sep.join([out_dir, "edges_mention"]), all_mention)

        map_rows = [(sn, Utils.to_node_id(uid)) for sn, uid in screen_name_map.items()]
        Writer.write_on_csv(os.sep.join([out_dir, "screen_name_map"]), map_rows)

        if self.delete_tmp_after_merge:
            self.logger.info(f"Deleting temporary checkpoint directory: {checkpoint_dir}")
            try:
                shutil.rmtree(checkpoint_dir)
            except Exception as e:
                self.logger.warning(f"Failed to delete {checkpoint_dir}: {e}")

        self.logger.info(
            f"Merge complete. "
            f"Mention edges: {resolved} resolved, {dropped} dropped (external users). "
            f"Output in {out_dir}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, checkpoint_every=500):
        """
        Main entry point. Connects to MongoDB and streams all tweets sorted
        by user.id through a cursor, processing one user at a time.

        The MongoDB cursor is a lazy iterator — tweets are fetched on demand,
        so only one user's tweets are held in memory at a time.
        Checkpoints are written to disk every `checkpoint_every` users.

        :param checkpoint_every: Number of users to process before writing
                                 a checkpoint to disk (default: 5000).
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