"""Leakage-safe threshold selection utilities."""

from typing import Iterable

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_thresholds(y_true, y_score, thresholds: Iterable[float]):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    rows = []
    for threshold in thresholds:
        prediction = (y_score > threshold).astype(np.int64)
        rows.append({
            'threshold': float(threshold),
            'precision': precision_score(
                y_true, prediction, zero_division=0),
            'recall': recall_score(
                y_true, prediction, zero_division=0),
            'f1': f1_score(y_true, prediction, zero_division=0),
        })
    return rows


def select_on_validation(y_true, y_score, thresholds: Iterable[float]):
    """Return the threshold with maximum validation F1.

    Ties prefer the higher threshold, which produces fewer false alerts.
    """
    rows = evaluate_thresholds(y_true, y_score, thresholds)
    return max(rows, key=lambda row: (row['f1'], row['threshold']))
