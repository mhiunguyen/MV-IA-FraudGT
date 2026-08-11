import numpy as np
import pandas as pd

from fraudGT.datasets.history_features import (
    compute_past_only_history_features_raw,
    normalize_history_features_train_only,
)


def test_same_timestamp_edges_do_not_observe_each_other():
    frame = pd.DataFrame({
        'from_id': [0, 0, 0, 3],
        'to_id': [1, 2, 1, 1],
        'Timestamp': [10, 10, 20, 20],
        'Amount Received': [10.0, 20.0, 30.0, 5.0],
    })
    result = compute_past_only_history_features_raw(frame)

    # The first two edges share t=10 and both see empty history.
    np.testing.assert_allclose(result[0, :], np.zeros(8), atol=1e-7)
    np.testing.assert_allclose(result[1, :], np.zeros(8), atol=1e-7)

    # At t=20, source 0 has exactly two earlier outgoing edges, while the
    # simultaneous edge from source 3 is not visible.
    assert np.isclose(result[2, 4], np.log1p(2))
    assert np.isclose(result[2, 6], np.log1p(1))
    assert np.isclose(result[3, 5], np.log1p(1))


def test_normalization_uses_train_indices_only_and_preserves_flags():
    frame = pd.DataFrame({
        'from_id': [0, 0, 0, 0],
        'to_id': [1, 2, 1, 2],
        'Timestamp': [10, 20, 30, 40],
        'Amount Received': [10.0, 20.0, 30.0, 1000000.0],
    })
    raw = compute_past_only_history_features_raw(frame)
    normalized, means, stds = normalize_history_features_train_only(raw, [0, 1, 2])

    assert means.shape == (6,)
    assert stds.shape == (6,)
    assert np.isfinite(normalized).all()
    np.testing.assert_array_equal(normalized[:, 2], raw[:, 2])
    np.testing.assert_array_equal(normalized[:, 3], raw[:, 3])
