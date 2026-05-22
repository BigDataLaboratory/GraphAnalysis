# algorithms/graph/checkpoint_io.py
# Robust I/O helper functions for saving checkpoints and merging them.
# Keep comments in English as requested.

import os
import time
import json
import shutil
from datetime import datetime, timezone
from collections import defaultdict
from typing import List, Sequence, Dict, Optional, Tuple

from Utils.Writer import Writer, _ext
from algorithms.graph.graph_constants import (
    USER_FEATURES_COLUMNS,
    USER_FEATURES_COLUMNS_FINAL,
    EDGE_RETWEET_COLUMNS,
    EDGE_REPLY_COLUMNS,
    EDGE_MENTION_COLUMNS,
    SCREEN_NAME_COLUMNS
)

def checkpoint_base_path(output_root: str, checkpoint_folder: str, run_id: str, batch_id: int, name: str,
                         load_snapshot_status: bool = False, load_snapshot_tmp_path: Optional[str] = None) -> str:
    """Return the base path (without extension) for a checkpoint file."""
    if load_snapshot_status and load_snapshot_tmp_path:
        return os.path.join(load_snapshot_tmp_path, str(batch_id), name)
    return os.path.join(output_root, checkpoint_folder, run_id, str(batch_id), name)

def output_base_path(output_root: str, run_id: str, name: str) -> str:
    """Return the base path (without extension) for a final output file."""
    return os.path.join(output_root, run_id, name)

def save_checkpoint(output_root: str, checkpoint_folder: str, run_id: str, file_format: str,
                    user_features_list: List[Dict], edges_rt: List[Tuple], edges_reply: List[Tuple],
                    mention_edges_raw: List[Tuple], screen_names_list: List[Tuple], batch_id: int,
                    load_snapshot_status: bool = False, load_snapshot_tmp_path: Optional[str] = None) -> None:
    """Persist intermediate results for one batch to disk."""
    base_dummy = checkpoint_base_path(output_root, checkpoint_folder, run_id, batch_id, "dummy",
                                      load_snapshot_status, load_snapshot_tmp_path)
    dir_path = os.path.dirname(base_dummy)
    os.makedirs(dir_path, exist_ok=True)

    def _base(name: str) -> str:
        return checkpoint_base_path(output_root, checkpoint_folder, run_id, batch_id, name,
                                    load_snapshot_status, load_snapshot_tmp_path)

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
            f['mobile_ratio'], f['web_ratio'], f['news_manager_ratio'],
            f['bot_api_ratio'], f['source_entropy'], f.get('community', -1)
        ]
        for f in user_features_list
    ]

    Writer.write_data(_base("user_features"), features_rows, columns=USER_FEATURES_COLUMNS, file_format=file_format)
    Writer.write_data(_base("edges_retweet"), edges_rt, columns=EDGE_RETWEET_COLUMNS, file_format=file_format)
    Writer.write_data(_base("edges_reply"), edges_reply, columns=EDGE_REPLY_COLUMNS, file_format=file_format)
    Writer.write_data(_base("edges_mention_raw"), mention_edges_raw, columns=EDGE_MENTION_COLUMNS, file_format=file_format)
    Writer.write_data(_base("screen_name_map"), screen_names_list, columns=SCREEN_NAME_COLUMNS, file_format=file_format)

def merge_checkpoints(output_root: str, checkpoint_folder: str, run_id: str, file_format: str,
                      final_file_format: str, delete_tmp_after_merge: bool, is_community_run: bool,
                      community_file_path: Optional[str], load_snapshot_status: bool = False,
                      load_snapshot_tmp_path: Optional[str] = None, logger = None) -> None:
    """Merge all batch checkpoints into final output files."""
    if load_snapshot_status and load_snapshot_tmp_path:
        checkpoint_dir = load_snapshot_tmp_path
    else:
        checkpoint_dir = os.path.join(output_root, checkpoint_folder, run_id)
    
    out_dir = os.path.join(output_root, run_id)
    os.makedirs(out_dir, exist_ok=True)

    ext = _ext(file_format)

    if not os.path.exists(checkpoint_dir):
        if logger:
            logger.warning(f"Checkpoint directory {checkpoint_dir} not found. Nothing to merge.")
        return

    if logger:
        logger.info("Merging checkpoints...")

    all_features, all_rt, all_reply, all_mention = [], [], [], []
    screen_name_map = {}
    valid_user_node_ids = set()

    # ── Pass 1: build screen_name map ────────────────────────────────────
    if logger:
        logger.info("Building screen_name map...")
    for batch_dir in sorted(os.listdir(checkpoint_dir)):
        batch_path = os.path.join(checkpoint_dir, batch_dir)
        if not os.path.isdir(batch_path):
            continue
        map_file = os.path.join(batch_path, "screen_name_map" + ext)
        if os.path.exists(map_file):
            for row in Writer.load_checkpoint_file(map_file):
                sn = row[0].lower()
                uid = int(row[1])
                screen_name_map[sn] = uid
                valid_user_node_ids.add(uid)

    resolved_mention = dropped_mention = dropped_rt = dropped_reply = 0
    received_retweets = defaultdict(int)
    received_replies = defaultdict(int)
    received_mentions = defaultdict(int)

    # ── Pass 2: load, filter, resolve ────────────────────────────────────
    if logger:
        logger.info("Filtering and resolving edges...")
    for batch_dir in sorted(os.listdir(checkpoint_dir)):
        batch_path = os.path.join(checkpoint_dir, batch_dir)
        if not os.path.isdir(batch_path):
            continue

        feat_file = os.path.join(batch_path, "user_features" + ext)
        if os.path.exists(feat_file):
            all_features.extend(Writer.load_checkpoint_file(feat_file))

        rt_file = os.path.join(batch_path, "edges_retweet" + ext)
        if os.path.exists(rt_file):
            for row in Writer.load_checkpoint_file(rt_file):
                dst = int(row[1])
                if dst in valid_user_node_ids:
                    all_rt.append(row)
                    received_retweets[dst] += int(float(row[2]))
                else:
                    dropped_rt += 1

        rep_file = os.path.join(batch_path, "edges_reply" + ext)
        if os.path.exists(rep_file):
            for row in Writer.load_checkpoint_file(rep_file):
                dst = int(row[1])
                if dst in valid_user_node_ids:
                    all_reply.append(row)
                    received_replies[dst] += int(float(row[2]))
                else:
                    dropped_reply += 1

        men_file = os.path.join(batch_path, "edges_mention_raw" + ext)
        if os.path.exists(men_file):
            for row in Writer.load_checkpoint_file(men_file):
                src, sn, weight = row[0], row[1], row[2]
                metrics = row[3:]
                dst = screen_name_map.get(sn.lower())
                if dst is not None:
                    all_mention.append((src, dst, weight, *metrics))
                    received_mentions[dst] += int(float(weight))
                    resolved_mention += 1
                else:
                    dropped_mention += 1

    # ── Inject in-degree counts into feature rows ─────────────────────────
    for row in all_features:
        uid = row[0]  # user_node_id is first column
        row.append(received_retweets.get(uid, 0))
        row.append(received_replies.get(uid, 0))
        row.append(received_mentions.get(uid, 0))

    # ── Write final files ────────────────────────────────────────────────
    def _out(name: str) -> str:
        return os.path.join(out_dir, name)

    if final_file_format == "csv":
        Writer.write_on_csv(_out("user_features.csv"), [USER_FEATURES_COLUMNS_FINAL])
        Writer.write_on_csv(_out("edges_retweet.csv"), [EDGE_RETWEET_COLUMNS])
        Writer.write_on_csv(_out("edges_reply.csv"), [EDGE_REPLY_COLUMNS])
        Writer.write_on_csv(_out("edges_mention.csv"), [EDGE_MENTION_COLUMNS])
        Writer.write_on_csv(_out("screen_name_map.csv"), [SCREEN_NAME_COLUMNS])
        Writer.write_on_csv(_out("user_features.csv"), all_features)
        Writer.write_on_csv(_out("edges_retweet.csv"), all_rt)
        Writer.write_on_csv(_out("edges_reply.csv"), all_reply)
        Writer.write_on_csv(_out("edges_mention.csv"), all_mention)
        map_rows = list(screen_name_map.items())
        Writer.write_on_csv(_out("screen_name_map.csv"), map_rows)
    else:
        Writer.write_data(_out("user_features"), all_features, columns=USER_FEATURES_COLUMNS_FINAL, file_format=final_file_format)
        Writer.write_data(_out("edges_retweet"), all_rt, columns=EDGE_RETWEET_COLUMNS, file_format=final_file_format)
        Writer.write_data(_out("edges_reply"), all_reply, columns=EDGE_REPLY_COLUMNS, file_format=final_file_format)
        Writer.write_data(_out("edges_mention"), all_mention, columns=EDGE_MENTION_COLUMNS, file_format=final_file_format)
        map_rows = list(screen_name_map.items())
        Writer.write_data(_out("screen_name_map"), map_rows, columns=SCREEN_NAME_COLUMNS, file_format=final_file_format)

    # ── Metadata ─────────────────────────────────────────────────────────
    filename = os.path.basename(community_file_path) if is_community_run and community_file_path else None
    metadata = {
        "collection": self_get_collection_fallback(checkpoint_dir),
        "date": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "intermediate_format": file_format,
        "final_format": final_file_format,
        "users_processed": len(valid_user_node_ids),
        "communities": filename,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    if delete_tmp_after_merge:
        if logger:
            logger.info(f"Deleting tmp: {checkpoint_dir}")
        try:
            shutil.rmtree(checkpoint_dir)
        except Exception as e:
            if logger:
                logger.warning(f"Failed to delete {checkpoint_dir}: {e}")

    if logger:
        logger.info(
            f"Merge complete. Output in {out_dir}\n"
            f"  Mentions : {resolved_mention} resolved, {dropped_mention} dropped.\n"
            f"  Retweets : {len(all_rt)} kept, {dropped_rt} dropped.\n"
            f"  Replies  : {len(all_reply)} kept, {dropped_reply} dropped."
        )

def self_get_collection_fallback(path: str) -> str:
    # A simple helper to get collection name if logger context isn't available
    return "tweets"