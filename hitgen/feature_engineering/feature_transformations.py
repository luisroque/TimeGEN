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


def combine_inputs_to_model(
    X_train: np.ndarray,
    dynamic_features: pd.DataFrame,
    static_features: dict,
    window_size: int,
    stride: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Combining the input features to the model: dynamic features, raw time series data and static features

    :param X_train: raw time series data
    :param dynamic_features: dynamic features already processed
    :param static_features: static features already processed
    :param window_size: rolling window

    :return: dynamic features ready to be inputed by the model
    :return: raw time series features ready to be inputed by the model
    :return: static features ready to be inputed by the model

    """

    X_dyn = temporalize(dynamic_features.to_numpy(), window_size, stride)
    n_samples = X_train.shape[0]

    dynamic_features_inp, X_inp, static_features_inp = (
        [X_dyn[:, :, i] for i in range(len(dynamic_features.columns))],
        [X_train],
        [
            np.tile(group_array, (1, n_samples)).T
            for group, group_array in static_features.items()
        ],
    )

    return dynamic_features_inp, X_inp, static_features_inp
