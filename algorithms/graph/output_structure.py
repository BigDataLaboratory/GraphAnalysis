# algorithms/graph/output_structure.py
# Column definitions, file name constants, and extraction thresholds.
#
# This is the single source of truth for the output schema of the
# user-user graph extraction pipeline.

# ─────────────────────────────────────────────────────────────────────────────
# Node (user) feature columns — written in each checkpoint batch
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

# Final user features include in-degree counts computed during merge
USER_FEATURES_COLUMNS_FINAL = USER_FEATURES_COLUMNS + [
    'received_retweets', 'received_replies', 'received_mentions',
]

# ─────────────────────────────────────────────────────────────────────────────
# Edge columns
# ─────────────────────────────────────────────────────────────────────────────
EDGE_RETWEET_COLUMNS = [
    'src', 'dst', 'weight', 'lifespan', 'fast_rt_ratio',
    'rt_cadence', 'rt_temporal_jitter', 'rt_topic_consistency',
]

EDGE_REPLY_COLUMNS = [
    'src', 'dst', 'weight', 'lifespan', 'avg_reply_latency_seconds',
    'reply_regularity', 'reply_burstiness', 'reply_diurnal_sync',
]

EDGE_MENTION_COLUMNS = [
    'src', 'dst', 'weight', 'mention_lifespan', 'mention_regularity',
    'mention_burstiness', 'mention_in_reply_ratio', 'mention_solo_ratio',
]

SCREEN_NAME_COLUMNS = ['screen_name', 'user_id']

# ─────────────────────────────────────────────────────────────────────────────
# Output file base names (without extension)
# ─────────────────────────────────────────────────────────────────────────────
FILE_USER_FEATURES   = "user_features"
FILE_EDGES_RETWEET   = "edges_retweet"
FILE_EDGES_REPLY     = "edges_reply"
FILE_EDGES_MENTION   = "edges_mention"
FILE_EDGES_MENTION_RAW = "edges_mention_raw"
FILE_SCREEN_NAME_MAP = "screen_name_map"
FILE_METADATA        = "metadata"

# ─────────────────────────────────────────────────────────────────────────────
# Extraction thresholds & defaults
# ─────────────────────────────────────────────────────────────────────────────
FAST_RT_THRESHOLD     = 30     # seconds — retweet latency below this is "fast"
DEFAULT_WINDOW_DAYS   = 90     # trailing window for daily / diurnal metrics
MIN_REGULARITY_TWEETS = 2      # minimum timestamps for interval-based metrics

# ─────────────────────────────────────────────────────────────────────────────
# Writer / file format
# ─────────────────────────────────────────────────────────────────────────────
SUPPORTED_FORMATS = ["csv", "pickle", "parquet", "feather"]
ROUND_DECIMALS    = 4
