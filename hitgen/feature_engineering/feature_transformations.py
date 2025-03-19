import numpy as np
import pandas as pd


def detemporalize(
    windows: np.ndarray,
    metadata: np.ndarray,
    T: int,
    N: int,
) -> np.ndarray:
    """
    Reconstruct a [T, N] array from:
      windows: [B, window_size, 1]
      metadata: [B, 2] => (series_idx, start_idx)
    If multiple windows overlap a position => average them.
    Returns shape [T, N].
    """
    out = np.zeros((T, N), dtype=np.float32)
    count = np.zeros((T, N), dtype=np.float32)

    B, window_size, _ = windows.shape
    for b in range(B):
        s_idx, st = metadata[b]
        end = min(st + window_size, T)
        w_slice = windows[b, : end - st, 0]  # shape [slice_len]
        out[st:end, s_idx] += w_slice
        count[st:end, s_idx] += 1

    valid = count > 0
    out[valid] /= count[valid]
    return out
