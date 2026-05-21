# algorithms/graph/retweet_metrics.py
# Retweet and reply metrics helpers.
# Keep comments in English as requested.

import numpy as np
import math
from typing import List, Dict, Tuple

def lifespan_from_ts(ts_list: List[float]) -> float:
    """
    Lifespan: log1p(max_ts - min_ts). Returns 0.0 if fewer than 2 timestamps.
    """
    if not ts_list or len(ts_list) < 2:
        return 0.0
    try:
        lifespan = max(ts_list) - min(ts_list)
        return round(math.log1p(max(lifespan, 0.0)), 4)
    except Exception:
        return 0.0

def rt_cadence_jitter(ts_list: List[float]) -> Tuple[float, float]:
    """
    Return (cadence_mean_log, jitter_log) for retweet timestamps.
    Cadence = log1p(mean(intervals)), Jitter = log1p(std(intervals)).
    """
    if not ts_list or len(ts_list) < 2:
        return 0.0, 0.0
    intervals = np.diff(sorted(ts_list))
    cadence = math.log1p(float(np.mean(intervals)))
    jitter = math.log1p(float(np.std(intervals)))
    return round(cadence, 4), round(jitter, 4)

def rt_topic_consistency(counter: Dict[str, int]) -> float:
    """
    Entropy of hashtags used in retweets (topic consistency).
    Returns entropy (base 2) rounded to 4 decimals, 0.0 if no data.
    """
    if not counter:
        return 0.0
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counter.values()])
    entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))
    return round(entropy, 4)

def fast_rt_ratio(count_fast: int, total: int) -> float:
    """
    Fast retweet ratio: proportion of fast retweets (no log).
    """
    return round(count_fast / total if total > 0 else 0.0, 4)

def avg_reply_latency(sum_latency: float, count: int) -> float:
    """
    Average reply latency, log1p scaled. Returns 0.0 if no replies.
    """
    if count <= 0:
        return 0.0
    try:
        avg = max(0.0, sum_latency / count)
        return round(math.log1p(avg), 4)
    except Exception:
        return 0.0

def reply_regularity_burstiness(ts_list: List[float]) -> Tuple[float, float]:
    """
    Compute reply regularity and burstiness.
    - regularity: log1p(sd(intervals))
    - burstiness: raw (sd - mean)/(sd + mean) clamped and rounded
    """
    if not ts_list or len(ts_list) < 2:
        return 0.0, 0.0
    intervals = np.diff(sorted(ts_list))
    if len(intervals) < 1:
        return 0.0, 0.0
    sd = float(np.std(intervals))
    mean = float(np.mean(intervals))
    regularity = round(math.log1p(max(sd, 0.0)), 4)
    raw_burstiness = (sd - mean) / (sd + mean) if (sd + mean) > 0 else 0.0
    clamped = max(raw_burstiness, -0.9999)
    return regularity, round(clamped, 4)

def reply_diurnal_sync(hours_list: List[int]) -> float:
    """
    Diurnal synchronicity: entropy of hourly distribution of replies.
    Returns log1p(entropy) rounded to 4 decimals, 0.0 if insufficient data.
    """
    if not hours_list or len(hours_list) < 2:
        return 0.0
    counts = np.bincount(hours_list, minlength=24).astype(float)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log2(probs)))
    try:
        return round(math.log1p(entropy), 4)
    except Exception:
        return 0.0