# algorithms/graph/constants.py
# Shared constants and column definitions for the graph generation package.
# Keep comments in English as requested.

# Columns for user features checkpoint files and final output
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
    'news_manager_ratio', 'bot_api_ratio', 'source_entropy',
]

# Final user features may include received interactions aggregated later
USER_FEATURES_COLUMNS_FINAL = USER_FEATURES_COLUMNS + [
    'received_retweets', 'received_replies', 'received_mentions',
]

# Edge column definitions
EDGE_RETWEET_COLUMNS = [
    'src', 'dst', 'weight', 'lifespan', 'fast_rt_ratio',
    'rt_cadence', 'rt_jitter', 'rt_topic_consistency'
]

EDGE_REPLY_COLUMNS = [
    'src', 'dst', 'weight', 'lifespan', 'avg_reply_latency_seconds',
    'reply_regularity', 'reply_burstiness', 'reply_diurnal_sync'
]

EDGE_MENTION_COLUMNS = [
    'src', 'dst', 'weight', 'mention_lifespan', 'mention_regularity',
    'mention_burstiness', 'mention_in_reply_ratio', 'mention_solo_ratio'
]

SCREEN_NAME_COLUMNS = ['screen_name', 'user_id']

# File name templates used by io module (base name without extension)
FILE_USER_FEATURES = "user_features"
FILE_EDGES_RETWEET = "edges_retweet"
FILE_EDGES_REPLY = "edges_reply"
FILE_EDGES_MENTION = "edges_mention_raw"
FILE_SCREEN_NAME_MAP = "screen_name_map"
FILE_METADATA = "metadata"

# Default thresholds and parameters
FAST_RT_THRESHOLD = 30            # seconds used to detect "fast" retweets
MIN_REGULARITY_TWEETS = 2         # minimum timestamps to compute interval-based metrics
DEFAULT_WINDOW_DAYS = 90          # default trailing window for daily/diurnal metrics

# Supported file formats for Writer
SUPPORTED_FORMATS = ["csv", "pickle", "parquet", "feather"]

# Rounding precision used across modules
ROUND_DECIMALS = 4