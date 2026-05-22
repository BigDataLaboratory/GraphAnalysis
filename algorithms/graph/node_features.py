# algorithms/graph/node_features.py
# Pure, stateless functions that compute user-level (node) features
# from aggregated tweet data.
#
# Every function here takes simple inputs (lists, counters, scalars)
# and returns simple outputs — no side-effects, no I/O.

import math
import re
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from algorithms.graph.output_structure import DEFAULT_WINDOW_DAYS

_HTML_TAG_RE = re.compile(r'<.*?>')


# ─────────────────────────────────────────────────────────────────────────────
# Source classification
# ─────────────────────────────────────────────────────────────────────────────

_MOBILE       = ['iphone', 'android', 'ipad', 'mobile', 'twitter for mac']
_WEB          = ['web app', 'web client', 'twitter web']
_NEWS_MANAGER = ['postpickr', 'hootsuite', 'wordpress', 'blog2social',
                 'dlvr.it', 'dlvrit', 'ifttt', 'instagram']


def classify_source(src_raw: str) -> str:
    """Classify a tweet source string into mobile / web / news_manager / bot_api."""
    if not isinstance(src_raw, str):
        return 'bot_api'
    m = _HTML_TAG_RE.sub('', src_raw).strip().lower()
    if any(k in m for k in _MOBILE):
        return 'mobile'
    if any(k in m for k in _WEB):
        return 'web'
    if any(k in m for k in _NEWS_MANAGER):
        return 'news_manager'
    return 'bot_api'


# ─────────────────────────────────────────────────────────────────────────────
# Temporal regularity
# ─────────────────────────────────────────────────────────────────────────────

def tweet_regularity_score_from_timestamps(timestamps: list) -> dict:
    """
    Compute posting regularity using the coefficient of variation (CV)
    of Inter-Tweet Intervals.

    CV = std / mean.  A perfectly regular poster has CV ≈ 0 → score ≈ 1.
    Requires ≥ 2 timestamps.
    """
    if len(timestamps) < 2:
        return {'regularity_score': 0.0, 'iti_std': 0.0,
                'iti_mean': 0.0, 'n_intervals': 0}

    sorted_ts = np.sort(np.array(timestamps, dtype=float))
    intervals = np.diff(sorted_ts)
    iti_mean = float(intervals.mean())
    iti_std  = float(intervals.std(ddof=0))
    cv = iti_std / iti_mean if iti_mean > 0 else float('inf')

    return {
        'regularity_score': float(1.0 / (1.0 + cv)),
        'iti_std':          iti_std,
        'iti_mean':         iti_mean,
        'n_intervals':      len(intervals),
    }


def daily_posting_consistency(timestamps: list, window_days: int = DEFAULT_WINDOW_DAYS,
                              min_days: int = 7, cv_clip: float = 5.0) -> dict:
    """
    Measure day-over-day posting consistency within a trailing window.

    Uses the CV of log-transformed daily tweet counts, combined with
    an activity_ratio bonus into a single score in [0, 1].
    """
    if not timestamps:
        return {'daily_score': 0.0, 'daily_cv_log': None,
                'median_daily_count': 0, 'n_days': 0}

    ts = np.sort(np.array(timestamps, dtype=float))
    start = int(ts[-1]) - int(window_days) * 86400
    days = [datetime.fromtimestamp(t, tz=timezone.utc).date()
            for t in ts if t >= start]
    if not days:
        return {'daily_score': 0.0, 'daily_cv_log': None,
                'median_daily_count': 0, 'n_days': 0}

    day_counts  = Counter(days)
    unique_days = sorted(set(days))
    n_days      = len(unique_days)
    counts      = [day_counts[d] for d in unique_days]

    if n_days < min_days:
        return {'daily_score': 0.0, 'daily_cv_log': None,
                'median_daily_count': float(np.median(counts)) if counts else 0.0,
                'n_days': n_days}

    counts_arr   = np.array(counts, dtype=float)
    log_counts   = np.log1p(counts_arr)
    mean_log     = float(log_counts.mean())
    std_log      = float(log_counts.std(ddof=0))
    daily_cv_log = std_log / mean_log if mean_log > 0 else float('inf')

    score_from_cv  = max(0.0, 1.0 - min(daily_cv_log, cv_clip) / cv_clip)
    activity_bonus = min(1.0, (n_days / float(window_days)) * 2.0)
    daily_score    = float(max(0.0, min(1.0,
                        0.75 * score_from_cv + 0.25 * activity_bonus)))

    return {
        'daily_score':        daily_score,
        'daily_cv_log':       None if not np.isfinite(daily_cv_log) else float(daily_cv_log),
        'median_daily_count': float(np.median(counts_arr)),
        'n_days':             n_days,
    }


def internal_tweet_density(timestamps: list,
                           window_days: int = DEFAULT_WINDOW_DAYS) -> float:
    """
    Shannon entropy of posting hours, normalised to [0, 1] by log2(24).

    H = 1 → uniform across hours (human-like).
    H = 0 → all tweets in one hour (bot-like).
    """
    if not timestamps:
        return 0.0
    ts = np.array(timestamps, dtype=float)
    start = float(ts.max()) - int(window_days) * 86400
    hours = [datetime.fromtimestamp(t, tz=timezone.utc).hour
             for t in ts if t >= start]
    if len(hours) < 2:
        return 0.0
    hour_counts = Counter(hours)
    total_h = len(hours)
    probs = [c / total_h for c in hour_counts.values()]
    raw_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return raw_entropy / math.log2(24)


# ─────────────────────────────────────────────────────────────────────────────
# Entropy helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_source_entropy(n_mobile: int, n_web: int,
                           n_news_manager: int, n_bot_api: int) -> float:
    """Shannon entropy of source-category distribution, normalised to [0, 1]."""
    counts = [n_mobile, n_web, n_news_manager, n_bot_api]
    total = sum(counts)
    if total <= 0:
        return 0.0
    try:
        probs = [c / total for c in counts if c > 0]
        return -sum(p * math.log2(p) for p in probs) / math.log2(4)
    except Exception:
        return 0.0


def compute_hashtag_entropy(hashtag_counter: Counter) -> float:
    """Normalised Shannon entropy of hashtag usage distribution."""
    total   = sum(hashtag_counter.values())
    n_unique = len(hashtag_counter)
    if total <= 0 or n_unique <= 1:
        return 0.0
    try:
        raw_h = (total * math.log2(total)
                 - sum(c * math.log2(c)
                       for c in hashtag_counter.values() if c > 0))
        return (raw_h / total) / math.log2(n_unique)
    except Exception:
        return 0.0
