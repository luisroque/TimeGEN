import numpy as np
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape_value = 100 * np.mean(
        np.where(denominator == 0, 0, 2 * np.abs(y_true - y_pred) / denominator)
    )
    return smape_value


def mase(y_true, y_pred, h, m: int = 1):
    """
    Mean Absolute Scaled Error (MASE).
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)

    if y_true.size <= m:
        return np.nan

    y_true_insample = y_true[:-h]
    y_true_h = y_true[-h:]

    y_pred_h = y_pred[-h:]

    mask = ~np.isnan(y_pred_h)
    if mask.sum() == 0:
        return np.nan

    scale = np.mean(np.abs(y_true_insample[m:] - y_true_insample[:-m]))
    if scale == 0.0 or np.isnan(scale):
        return np.nan

    mase_value = np.mean(np.abs(y_true_h - y_pred_h)) / scale
    return float(mase_value)
