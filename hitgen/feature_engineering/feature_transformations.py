import numpy as np
import pandas as pd


def temporalize(data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    Transforming the data using a rolling window
    """
    X = []
    step = stride
    for i in range(0, len(data) - window_size + 1, step):
        row = data[i : i + window_size]
        X.append(row)
    return np.array(X)


def detemporalize(input_data, stride=1):
    """
    Convert a 3D time series array into a 2D array
    """
    inp = np.array(input_data)
    final_list = []

    for i in range(len(inp) - 1):
        # take the first stride timesteps from each window
        sel = inp[i, :stride, :]
        final_list.append(sel)

    # include all timesteps of the last window
    final_list.append(inp[-1])

    # concatenate into a single 2D array
    final = np.concatenate(final_list, axis=0)
    return final
