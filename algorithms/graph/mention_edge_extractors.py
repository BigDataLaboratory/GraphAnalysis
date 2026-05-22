# algorithms/graph/mention_metrics.py
# Mention-specific metrics builders and a helper to build mention edges.
# Keep comments in English as requested.

import numpy as np
import math
from typing import Dict, List, Tuple

def mention_lifespan(ts_list: List[float]) -> float:
    """
    Mention Lifespan: duration between first and last mention, log1p scaled.
    Returns 0.0 if fewer than 2 timestamps.
    """
    if not ts_list or len(ts_list) < 2:
        return 0.0
    try:
        lifespan = max(ts_list) - min(ts_list)
        return round(math.log1p(max(lifespan, 0)), 4)
    except Exception:
        return 0.0

def mention_regularity_and_burstiness(ts_list: List[float]) -> Tuple[float, float]:
    """
    Compute mention regularity and burstiness.
    - regularity: log1p(sd(intervals))
    - burstiness: log1p(clamped_raw_burstiness) where raw_burstiness = (sd - mean) / (sd + mean)
    Returns (regularity, burstiness) both rounded to 4 decimals.
    """
    if not ts_list or len(ts_list) < 2:
        return 0.0, 0.0

    ts_sorted = sorted(ts_list)
    intervals = np.diff(ts_sorted)
    if len(intervals) < 1:
        return 0.0, 0.0

    sd_i = float(np.std(intervals))
    mean_i = float(np.mean(intervals))

    # Mention Regularity: log1p(SD)
    try:
        regularity = math.log1p(max(sd_i, 0.0))
    except Exception:
        regularity = 0.0

    # Burstiness Index: (sd - mean) / (sd + mean), clamp to avoid <= -1
    raw_burstiness = (sd_i - mean_i) / (sd_i + mean_i) if (sd_i + mean_i) > 0 else 0.0
    clamped = max(raw_burstiness, -0.9999)
    try:
        burstiness = math.log1p(clamped)
    except Exception:
        burstiness = 0.0

    return round(regularity, 4), round(burstiness, 4)

def mention_in_reply_ratio_log1p(total_count: int, in_reply_count: int) -> float:
    """
    Return log1p(inside/total) if total>0 else 0.0.
    This mirrors your previous implementation that used log1p on the ratio.
    """
    if total_count <= 0:
        return 0.0
    ratio = float(in_reply_count) / float(total_count)
    try:
        return round(math.log1p(ratio), 4)
    except Exception:
        return 0.0

def mention_solo_ratio_log1p(total_count: int, solo_count: int) -> float:
    """
    Return log1p(solo/total) if total>0 else 0.0.
    """
    if total_count <= 0:
        return 0.0
    ratio = float(solo_count) / float(total_count)
    try:
        return round(math.log1p(ratio), 4)
    except Exception:
        return 0.0

def build_mention_edges(state: Dict, src_node_id: int) -> List[Tuple]:
    """
    Build mention edges rows from the state dictionaries.
    Expected state keys:
      - 'mention_targets' : dict screen_name -> count
      - 'mention_ts' : dict screen_name -> list of timestamps
      - 'mention_in_reply_count' : dict screen_name -> count
      - 'mention_solo_count' : dict screen_name -> count

    Returns list of tuples:
      (src_node_id, screen_name, weight, mention_lifespan, mention_regularity, mention_burstiness, mention_in_reply_ratio, mention_solo_ratio)
    """
    rows = []
    mention_targets = state.get('mention_targets', {})
    mention_ts = state.get('mention_ts', {})
    mention_in_reply_count = state.get('mention_in_reply_count', {})
    mention_solo_count = state.get('mention_solo_count', {})

    for screen_name, weight in mention_targets.items():
        ts_list = mention_ts.get(screen_name, [])
        lifespan = mention_lifespan(ts_list)
        reg, burst = mention_regularity_and_burstiness(ts_list)
        in_reply = mention_in_reply_ratio_log1p(weight, mention_in_reply_count.get(screen_name, 0))
        solo = mention_solo_ratio_log1p(weight, mention_solo_count.get(screen_name, 0))
        rows.append((src_node_id, screen_name, weight, lifespan, reg, burst, in_reply, solo))
    return rows