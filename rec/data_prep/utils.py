from __future__ import annotations

from typing import Tuple

import pandas as pd


def split_interactions_by_time(
    interactions: pd.DataFrame,
    timestamp_col: str = "timestamp",
    val_t: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions into train/val based on a time threshold.

    The threshold is defined as:
    min_timestamp + (max_timestamp - min_timestamp) * (1 - val_t)

    Args:
        interactions: Input interactions with a timestamp column.
        timestamp_col: Name of the timestamp column.
        val_t: Fraction of the time range to allocate to validation.

    Returns:
        (train_df, val_df) split by timestamp.
    """
    if not 0.0 < val_t < 1.0:
        raise ValueError("val_t must be between 0 and 1 (exclusive)")

    if timestamp_col not in interactions.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")

    ts = pd.to_datetime(interactions[timestamp_col], errors="coerce")
    if ts.isna().any():
        raise ValueError("Found invalid timestamps in interactions")

    min_ts = ts.min()
    max_ts = ts.max()
    assert min_ts != max_ts, "All timestamps are the same"

    threshold = min_ts + (max_ts - min_ts) * (1 - val_t)
    train_mask = ts <= threshold
    val_mask = ts > threshold

    train_df = interactions.loc[train_mask].reset_index(drop=True)
    val_df = interactions.loc[val_mask].reset_index(drop=True)
    return train_df, val_df
