"""Shared reciprocal-rank fusion. Scores from unlike models are never added directly."""
import numpy as np
import pandas as pd


def reciprocal_rank_fusion(*channels, k: int = 60) -> np.ndarray:
    """Aligned arrays; NaN marks absence in a channel. Preserve average ranks for ties."""
    if k <= 0:
        raise ValueError('RRF k must be positive')
    if not channels:
        return np.zeros(0)
    arrays = [np.asarray(values, dtype=float) for values in channels]
    if any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError('RRF channels must be aligned one-dimensional arrays')
    result = np.zeros(len(arrays[0]))
    for values in arrays:
        valid = np.isfinite(values)
        ranks = pd.Series(values[valid]).rank(ascending=False).to_numpy()
        result[valid] += 1 / (k + ranks)
    return result


def top_channel(scores, limit: int, *, positive: bool = True) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    out = np.full(scores.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(scores) & ((scores > 0) if positive else True))
    order = valid[np.argsort(-scores[valid], kind='stable')[:max(0, limit)]]
    out[order] = scores[order]
    return out
