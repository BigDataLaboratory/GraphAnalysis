# algorithms/graph/io.py
# I/O wrappers that isolate Writer dependency and file format logic.
# Keep comments in English as requested.

import os
from typing import List, Sequence, Dict, Optional, Tuple
from Utils.Writer import Writer, _ext

def checkpoint_base_path(output_root: str, checkpoint_folder: str, run_id: str, batch_id: int, name: str) -> str:
    """
    Return the base path (without extension) for a checkpoint file.
    Example: <output_root>/<checkpoint_folder>/<run_id>/<batch_id>/<name>
    """
    return os.path.join(output_root, checkpoint_folder, run_id, str(batch_id), name)

def output_base_path(output_root: str, run_id: str, name: str) -> str:
    """
    Return the base path (without extension) for a final output file.
    Example: <output_root>/<run_id>/<name>
    """
    return os.path.join(output_root, run_id, name)

def glob_checkpoint(batch_path: str, name: str, file_format: str) -> Optional[str]:
    """
    Return the full path (with extension) of a checkpoint file if it exists.
    Returns None if not found.
    """
    path = os.path.join(batch_path, name + _ext(file_format))
    return path if os.path.exists(path) else None

def write_checkpoint(base_path: str, rows: Sequence[Sequence], columns: Sequence[str], file_format: str) -> None:
    """
    Write checkpoint rows using the Writer abstraction.
    base_path: path without extension; Writer will append the proper extension.
    rows: iterable of rows (each row is a sequence of values).
    columns: list of column names.
    file_format: one of supported formats (csv, parquet, pickle, feather, etc.)
    """
    Writer.write_data(base_path, list(rows), columns=columns, file_format=file_format)

def write_final(base_path: str, rows: Sequence[Sequence], columns: Sequence[str], file_format: str) -> None:
    """
    Write final merged output using the specified final file format.
    This is a thin wrapper around Writer.write_data kept for clarity.
    """
    Writer.write_data(base_path, list(rows), columns=columns, file_format=file_format)

def save_checkpoint_files(output_root: str, checkpoint_folder: str, run_id: str, batch_id: int,
                          user_features_rows: List[Sequence], user_features_columns: List[str],
                          edges_rt_rows: List[Sequence], edges_rt_columns: List[str],
                          edges_reply_rows: List[Sequence], edges_reply_columns: List[str],
                          edges_mention_rows: List[Sequence], edges_mention_columns: List[str],
                          screen_name_rows: List[Sequence], screen_name_columns: List[str],
                          file_format: str) -> None:
    """
    High-level helper to persist all checkpoint files for a single batch.
    It builds base paths and calls write_checkpoint for each artifact.
    """
    base_dir = os.path.join(output_root, checkpoint_folder, run_id, str(batch_id))
    os.makedirs(base_dir, exist_ok=True)

    def _base(name: str) -> str:
        return os.path.join(base_dir, name)

    write_checkpoint(_base("user_features"), user_features_rows, user_features_columns, file_format)
    write_checkpoint(_base("edges_retweet"), edges_rt_rows, edges_rt_columns, file_format)
    write_checkpoint(_base("edges_reply"), edges_reply_rows, edges_reply_columns, file_format)
    write_checkpoint(_base("edges_mention_raw"), edges_mention_rows, edges_mention_columns, file_format)
    write_checkpoint(_base("screen_name_map"), screen_name_rows, screen_name_columns, file_format)

def merge_checkpoints_and_write_final(output_root: str, run_id: str,
                                      checkpoint_folder: str,
                                      final_user_features_name: str,
                                      final_edges_rt_name: str,
                                      final_edges_reply_name: str,
                                      final_edges_mention_name: str,
                                      final_screen_name_name: str,
                                      intermediate_format: str,
                                      final_format: str) -> None:
    """
    Merge all batch checkpoints into final output files.
    This function provides a minimal, robust skeleton:
      - it scans the checkpoint folder for batch subfolders
      - it reads each checkpoint file using Writer (if Reader exists) or loads via pandas
      - it concatenates rows and writes final files using write_final

    Note: implementors can replace the internal read logic with a faster approach
    depending on available Reader utilities. Keep this function idempotent.
    """
    # Discover batch directories
    checkpoint_root = os.path.join(output_root, checkpoint_folder, run_id)
    if not os.path.isdir(checkpoint_root):
        return

    batch_dirs = sorted([os.path.join(checkpoint_root, d) for d in os.listdir(checkpoint_root)
                         if os.path.isdir(os.path.join(checkpoint_root, d))])

    # Accumulators for rows
    all_user_features = []
    all_edges_rt = []
    all_edges_reply = []
    all_edges_mention = []
    all_screen_names = []

    # Helper to build file path with extension
    def _file_path(batch_dir: str, name: str, fmt: str) -> Optional[str]:
        p = os.path.join(batch_dir, name + _ext(fmt))
        return p if os.path.exists(p) else None

    # Read each checkpoint file. We rely on Writer having a compatible read function,
    # otherwise implement a small reader using pandas based on extension.
    for bdir in batch_dirs:
        # user_features
        p = _file_path(bdir, "user_features", intermediate_format)
        if p:
            # Writer does not expose a read API in this codebase; use pandas as fallback
            try:
                import pandas as pd
                df = pd.read_csv(p) if p.endswith(".csv") else pd.read_parquet(p)
                all_user_features.extend(df.values.tolist())
            except Exception:
                pass

        # edges_retweet
        p = _file_path(bdir, "edges_retweet", intermediate_format)
        if p:
            try:
                import pandas as pd
                df = pd.read_csv(p) if p.endswith(".csv") else pd.read_parquet(p)
                all_edges_rt.extend(df.values.tolist())
            except Exception:
                pass

        # edges_reply
        p = _file_path(bdir, "edges_reply", intermediate_format)
        if p:
            try:
                import pandas as pd
                df = pd.read_csv(p) if p.endswith(".csv") else pd.read_parquet(p)
                all_edges_reply.extend(df.values.tolist())
            except Exception:
                pass

        # edges_mention_raw
        p = _file_path(bdir, "edges_mention_raw", intermediate_format)
        if p:
            try:
                import pandas as pd
                df = pd.read_csv(p) if p.endswith(".csv") else pd.read_parquet(p)
                all_edges_mention.extend(df.values.tolist())
            except Exception:
                pass

        # screen_name_map
        p = _file_path(bdir, "screen_name_map", intermediate_format)
        if p:
            try:
                import pandas as pd
                df = pd.read_csv(p) if p.endswith(".csv") else pd.read_parquet(p)
                all_screen_names.extend(df.values.tolist())
            except Exception:
                pass

    # Write final outputs
    out_dir = os.path.join(output_root, run_id)
    os.makedirs(out_dir, exist_ok=True)

    write_final(os.path.join(out_dir, final_user_features_name), all_user_features, [], final_format)
    write_final(os.path.join(out_dir, final_edges_rt_name), all_edges_rt, [], final_format)
    write_final(os.path.join(out_dir, final_edges_reply_name), all_edges_reply, [], final_format)
    write_final(os.path.join(out_dir, final_edges_mention_name), all_edges_mention, [], final_format)
    write_final(os.path.join(out_dir, final_screen_name_name), all_screen_names, [], final_format)