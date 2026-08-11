"""Past-only historical features for timestamped AML transactions.

All statistics for an edge at time ``t`` are computed from edges with a
strictly smaller timestamp. Edges that share the same timestamp are therefore
not allowed to observe one another.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


HISTORY_FEATURE_NAMES = (
    "hist_log_seconds_since_prev_out",
    "hist_log_seconds_since_prev_in",
    "hist_has_prev_out",
    "hist_has_prev_in",
    "hist_log_prior_out_count",
    "hist_log_prior_in_count",
    "hist_log_prior_pair_count",
    "hist_log_amount_over_prior_out_mean",
)

# Binary indicators stay in {0, 1}; the other columns are standardized.
CONTINUOUS_HISTORY_COLUMNS = (0, 1, 4, 5, 6, 7)


def _previous_distinct_timestamp(
    frame: pd.DataFrame, key_columns: list[str]
) -> np.ndarray:
    """Return the previous strictly earlier timestamp for each row."""
    keys = key_columns + ["Timestamp"]
    unique_times = frame.loc[:, keys].drop_duplicates()
    unique_times = unique_times.sort_values(keys, kind="stable")
    unique_times["_prev_timestamp"] = unique_times.groupby(
        key_columns, sort=False
    )["Timestamp"].shift(1)
    row_order = frame.loc[:, keys].copy()
    row_order["_row_order"] = np.arange(len(frame), dtype=np.int64)
    merged = row_order.merge(unique_times, on=keys, how="left", sort=False)
    merged = merged.sort_values("_row_order", kind="stable")
    return merged["_prev_timestamp"].to_numpy(dtype=np.float64)


def compute_past_only_history_features_raw(df_edges: pd.DataFrame) -> np.ndarray:
    """Compute eight leakage-safe history features in original row order.

    Required columns are ``from_id``, ``to_id``, ``Timestamp`` and
    ``Amount Received``. The implementation is vectorized for the multi-million
    edge AML files used by the project.
    """
    required = {"from_id", "to_id", "Timestamp", "Amount Received"}
    missing = required.difference(df_edges.columns)
    if missing:
        raise ValueError(f"Missing columns for history features: {sorted(missing)}")

    frame = df_edges.loc[:, sorted(required)].copy()
    frame["_row_order"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.sort_values(
        ["Timestamp", "_row_order"], kind="stable"
    ).reset_index(drop=True)

    # Counts before the whole (entity, timestamp) group exclude every edge at
    # the current timestamp, not just the current row.
    out_position = frame.groupby("from_id", sort=False).cumcount()
    out_same_time_position = frame.groupby(
        ["from_id", "Timestamp"], sort=False
    ).cumcount()
    prior_out_count = (out_position - out_same_time_position).to_numpy(np.float64)

    in_position = frame.groupby("to_id", sort=False).cumcount()
    in_same_time_position = frame.groupby(
        ["to_id", "Timestamp"], sort=False
    ).cumcount()
    prior_in_count = (in_position - in_same_time_position).to_numpy(np.float64)

    pair_position = frame.groupby(["from_id", "to_id"], sort=False).cumcount()
    pair_same_time_position = frame.groupby(
        ["from_id", "to_id", "Timestamp"], sort=False
    ).cumcount()
    prior_pair_count = (pair_position - pair_same_time_position).to_numpy(
        np.float64
    )

    amounts = frame["Amount Received"].astype(np.float64)
    out_cumulative_amount = amounts.groupby(frame["from_id"], sort=False).cumsum()
    out_same_time_amount = amounts.groupby(
        [frame["from_id"], frame["Timestamp"]], sort=False
    ).cumsum()
    prior_out_amount = (out_cumulative_amount - out_same_time_amount).to_numpy()

    prev_out_time = _previous_distinct_timestamp(frame, ["from_id"])
    prev_in_time = _previous_distinct_timestamp(frame, ["to_id"])
    timestamps = frame["Timestamp"].to_numpy(dtype=np.float64)
    has_prev_out = ~np.isnan(prev_out_time)
    has_prev_in = ~np.isnan(prev_in_time)
    out_gap = np.where(has_prev_out, timestamps - prev_out_time, 0.0)
    in_gap = np.where(has_prev_in, timestamps - prev_in_time, 0.0)

    prior_out_mean = np.divide(
        prior_out_amount,
        prior_out_count,
        out=np.zeros_like(prior_out_amount),
        where=prior_out_count > 0,
    )
    amount_ratio = np.divide(
        amounts.to_numpy(),
        prior_out_mean,
        out=np.zeros(len(frame), dtype=np.float64),
        where=prior_out_mean > 0,
    )
    amount_ratio = np.clip(amount_ratio, 0.0, 1e6)

    features = np.column_stack(
        (
            np.log1p(np.maximum(out_gap, 0.0)),
            np.log1p(np.maximum(in_gap, 0.0)),
            has_prev_out.astype(np.float64),
            has_prev_in.astype(np.float64),
            np.log1p(prior_out_count),
            np.log1p(prior_in_count),
            np.log1p(prior_pair_count),
            np.log1p(amount_ratio),
        )
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Restore the exact order supplied by the caller.
    inverse_order = np.argsort(frame["_row_order"].to_numpy(), kind="stable")
    return features[inverse_order].astype(np.float32, copy=False)


def normalize_history_features_train_only(
    features: np.ndarray, train_indices: Iterable[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize continuous history columns using training edges only."""
    features = np.asarray(features, dtype=np.float32)
    if hasattr(train_indices, "detach"):
        train_indices = train_indices.detach().cpu().numpy()
    train_indices = np.asarray(train_indices, dtype=np.int64).reshape(-1)
    if features.ndim != 2 or features.shape[1] != len(HISTORY_FEATURE_NAMES):
        raise ValueError(
            f"Expected history feature shape [N, {len(HISTORY_FEATURE_NAMES)}], "
            f"got {features.shape}"
        )
    if train_indices.size == 0:
        raise ValueError("train_indices must not be empty")

    continuous = np.asarray(CONTINUOUS_HISTORY_COLUMNS, dtype=np.int64)
    train_values = features[train_indices[:, None], continuous]
    means = train_values.mean(axis=0, dtype=np.float64).astype(np.float32)
    stds = train_values.std(axis=0, dtype=np.float64).astype(np.float32)
    stds = np.where(stds > 0, stds, np.float32(1.0))

    normalized = features.copy()
    normalized[:, continuous] = (
        normalized[:, continuous] - means[None, :]
    ) / stds[None, :]
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized.astype(np.float32, copy=False), means, stds
