# algorithms/graph/features.py
# Pure feature computation helpers.
# These functions are stateless and easy to unit test.

import numpy as np
from collections import Counter
from datetime import datetime, timezone
import math
import re

_HTML_TAG_RE = re.compile(r'<.*?>')

def to_timestamp(dt):
    # Convert various datetime-like inputs to POSIX timestamp (float).
    if dt is None:
        return 0.0
    if isinstance(dt, (int, float)):
        return float(dt)
    try:
        # assume dt is a datetime or ISO string
        if hasattr(dt, "timestamp"):
            return float(dt.timestamp())
        return float(datetime.fromisoformat(str(dt)).timestamp())
    except Exception:
        return 0.0

def tweet_regularity_score_from_timestamps(timestamps):
    # Compute posting regularity using coefficient of variation of intervals.
    if len(timestamps) < 2:
        return {'regularity_score': 0.0, 'iti_std': 0.0, 'iti_mean': 0.0, 'n_intervals': 0}
    sorted_ts = np.sort(np.array(timestamps, dtype=float))
    intervals = np.diff(sorted_ts)
    iti_mean = float(intervals.mean())
    iti_std = float(intervals.std(ddof=0))
    cv = iti_std / iti_mean if iti_mean > 0 else float('inf')
    regularity_score = 1.0 / (1.0 + cv)
    return {'regularity_score': float(regularity_score), 'iti_std': iti_std, 'iti_mean': iti_mean, 'n_intervals': len(intervals)}

def daily_posting_consistency(timestamps, window_days=90, min_days=7, cv_clip=5.0):
    # Measure day-to-day posting consistency using CV of log daily counts.
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
    return {'daily_score': daily_score, 'daily_cv_log': None if not np.isfinite(daily_cv_log) else float(daily_cv_log), 'median_daily_count': float(np.median(counts_arr)), 'n_days': n_days}

def internal_tweet_density(timestamps, window_days=90):
    # Compute entropy of posting hours normalized to [0,1].
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
    # Classify tweet source into mobile, web, news_manager, bot_api.
    if not isinstance(src_raw, str):
        return 'bot_api'
    m = _HTML_TAG_RE.sub('', src_raw).strip().lower()
    MOBILE = ['iphone', 'android', 'ipad', 'mobile', 'twitter for mac']
    WEB = ['web app', 'web client', 'twitter web']
    NEWS_MANAGER = ['postpickr', 'hootsuite', 'wordpress', 'blog2social', 'dlvr.it', 'dlvrit', 'ifttt', 'instagram']
    if any(k in m for k in MOBILE):
        return 'mobile'
    if any(k in m for k in WEB):
        return 'web'
    if any(k in m for k in NEWS_MANAGER):
        return 'news_manager'
    return 'bot_api'