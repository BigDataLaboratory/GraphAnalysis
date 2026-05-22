# algorithms/graph/edge_features.py
# Pure functions that compute edge-level metrics for retweet, reply,
# and mention edges.
#
# All functions follow the same pattern:
#   take timestamps / counters  →  return log1p-scaled metric(s)

import math
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def lifespan(ts_list: List[float]) -> float:
    """log1p(max_ts - min_ts).  Returns 0.0 if < 2 timestamps."""
    if not ts_list or len(ts_list) < 2:
        return 0.0
    try:
        return round(math.log1p(max(max(ts_list) - min(ts_list), 0.0)), 4)
    except Exception:
        return 0.0


def _regularity_burstiness(ts_list: List[float]) -> Tuple[float, float]:
    """
    Common regularity / burstiness computation for any edge type.

    regularity = log1p(std(intervals))
    burstiness = (std - mean) / (std + mean), clamped to [-0.9999, +∞)
    """
    if not ts_list or len(ts_list) < 2:
        return 0.0, 0.0
    intervals = np.diff(sorted(ts_list))
    if len(intervals) < 1:
        return 0.0, 0.0
    sd   = float(np.std(intervals))
    mean = float(np.mean(intervals))
    regularity = round(math.log1p(max(sd, 0.0)), 4)
    raw_burst  = (sd - mean) / (sd + mean) if (sd + mean) > 0 else 0.0
    burstiness = round(max(raw_burst, -0.9999), 4)
    return regularity, burstiness


# ─────────────────────────────────────────────────────────────────────────────
# Retweet edge metrics
# ─────────────────────────────────────────────────────────────────────────────

def rt_cadence_jitter(ts_list: List[float]) -> Tuple[float, float]:
    """Cadence = log1p(mean(intervals)),  Jitter = log1p(std(intervals))."""
    if not ts_list or len(ts_list) < 2:
        return 0.0, 0.0
    intervals = np.diff(sorted(ts_list))
    cadence = round(math.log1p(float(np.mean(intervals))), 4)
    jitter  = round(math.log1p(float(np.std(intervals))), 4)
    return cadence, jitter


def rt_topic_consistency(counter: Dict[str, int]) -> float:
    """Entropy of hashtags in retweeted content (base-2)."""
    if not counter:
        return 0.0
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counter.values()])
    return round(-float(np.sum(probs * np.log2(probs + 1e-12))), 4)


def fast_rt_ratio(count_fast: int, total: int) -> float:
    """Proportion of retweets whose latency was below the threshold."""
    return round(count_fast / total if total > 0 else 0.0, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Reply edge metrics
# ─────────────────────────────────────────────────────────────────────────────

def avg_reply_latency(sum_latency: float, count: int) -> float:
    """Average reply latency, log1p-scaled."""
    if count <= 0:
        return 0.0
    try:
        return round(math.log1p(max(0.0, sum_latency / count)), 4)
    except Exception:
        return 0.0


def reply_regularity_burstiness(ts_list: List[float]) -> Tuple[float, float]:
    """Regularity and burstiness of reply intervals."""
    return _regularity_burstiness(ts_list)


def reply_diurnal_sync(hours_list: List[int]) -> float:
    """
    Diurnal synchronicity: log1p of the Shannon entropy of the
    hourly distribution of replies.
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


# ─────────────────────────────────────────────────────────────────────────────
# Mention edge metrics
# ─────────────────────────────────────────────────────────────────────────────

def mention_regularity_burstiness(ts_list: List[float]) -> Tuple[float, float]:
    """Regularity and burstiness of mention intervals (log1p on burstiness)."""
    reg, burst_raw = _regularity_burstiness(ts_list)
    try:
        burst = round(math.log1p(burst_raw), 4)
    except Exception:
        burst = 0.0
    return reg, burst


def mention_in_reply_ratio(total_count: int, in_reply_count: int) -> float:
    """log1p(in_reply_count / total_count)."""
    if total_count <= 0:
        return 0.0
    try:
        return round(math.log1p(in_reply_count / total_count), 4)
    except Exception:
        return 0.0


def mention_solo_ratio(total_count: int, solo_count: int) -> float:
    """log1p(solo_count / total_count)."""
    if total_count <= 0:
        return 0.0
    try:
        return round(math.log1p(solo_count / total_count), 4)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Edge builder (mention)
# ─────────────────────────────────────────────────────────────────────────────

def build_mention_edges(acc: dict, src_node_id: int) -> List[Tuple]:
    """
    Build mention edge tuples from the accumulator state.

    Returns list of (src, screen_name, weight, lifespan, regularity,
                     burstiness, in_reply_ratio, solo_ratio).
    """
    rows = []
    targets    = acc.get("mention_targets", {})
    ts_map     = acc.get("mention_ts", {})
    reply_cnt  = acc.get("mention_in_reply_count", {})
    solo_cnt   = acc.get("mention_solo_count", {})

    for screen_name, weight in targets.items():
        ts_list = ts_map.get(screen_name, [])
        reg, burst = mention_regularity_burstiness(ts_list)
        rows.append((
            src_node_id,
            screen_name,
            weight,
            lifespan(ts_list),
            reg,
            burst,
            mention_in_reply_ratio(weight, reply_cnt.get(screen_name, 0)),
            mention_solo_ratio(weight, solo_cnt.get(screen_name, 0)),
        ))
    return rows
