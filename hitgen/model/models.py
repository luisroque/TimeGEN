import logging
import os
from typing import List, Tuple, Union
import numpy as np
import pandas as pd
import hashlib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from keras import layers
import tensorflow as tf
from keras import regularizers
from tensorflow import keras

K = keras.backend


def build_tf_dataset(
    data: pd.DataFrame,  # [T, N]
    mask: pd.DataFrame,  # [T, N]
    dyn_features: pd.DataFrame,  # [T, dyn_dim]
    cache_split: str,
    cache_dataset_name: str,
    cache_dataset_group: str,
    window_size: int = 16,
    stride: int = None,
    coverage_mode: str = "systematic",
    coverage_fraction: float = 0.1,
    prediction_mode: str = "one_step_ahead",
    future_steps: int = 1,
    batch_size: int = 8,
    windows_batch_size: int = 8,
    shuffle_buffer_size: int = 1000,
    reshuffle_each_epoch: bool = True,
    store_dataset: bool = True,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset that yields ((x_data, x_mask, x_dyn, pred_mask),
                                         (recon_target, recon_mask, pred_target, pred_mask)).

    Steps:
    1) Enumerate all (series_idx, start_time) pairs for valid windows of length `window_size`.
    2) If coverage_mode='partial', sample `coverage_fraction` of these windows.
    3) Build the arrays for inputs/targets: reconstruction + forecast.
    4) Create a Dataset from these arrays.
    5) Shuffle, then batch, then prefetch.

    Args:
      data:            shape [T, N]. T = time length, N = number of series.
      mask:            shape [T, N], or None.
      dyn_features:    shape [T, dyn_dim], or None.
      window_size:     Number of time steps in each window.
      stride:          Step size between consecutive windows (default=window_size).
      coverage_mode:   'systematic' => use all windows,
                       'partial' => randomly sample coverage_fraction of windows.
      coverage_fraction: Fraction of windows to keep if coverage_mode='partial'.
      prediction_mode: 'one_step_ahead' or 'multi_step_ahead' for the forecast target.
      future_steps:    Number of steps to forecast if multi_step_ahead.
      batch_size:      Batch size for final dataset.
      windows_batch_size: Number of windows per series for each batch.
      shuffle_buffer_size:  Buffer size used in `dataset.shuffle()`.
      reshuffle_each_epoch: Whether tf.data should reshuffle at each epoch.

    Returns:
      A `tf.data.Dataset` of length = (num_windows // batch_size). Each element is:
        Inputs: (x_data, x_mask, x_dyn, pred_mask)
        Targets: (recon_target, recon_mask, pred_target, pred_mask)
      where shapes are:
        - x_data: [batch_size, window_size, 1]
        - x_mask: [batch_size, window_size, 1]
        - x_dyn:  [batch_size, window_size, dyn_dim]
        - pred_mask: either [batch_size, 1, 1] (one_step) or [batch_size, future_steps, 1]
        - recon_target, recon_mask: same shape as (x_data, x_mask)
        - pred_target, pred_mask: same shape as (pred_mask).
    """
    param_str = (
        f"win{window_size}-str{stride}-covmode{coverage_mode}-covfrac{coverage_fraction}"
        f"-predmode{prediction_mode}-fut{future_steps}-batch{batch_size}-wbatch{windows_batch_size}"
        f"-shufbuf{shuffle_buffer_size}-reshuff{reshuffle_each_epoch}"
    )

    data_shape_str = f"T{data.shape[0]}_N{data.shape[1]}"
    if dyn_features is not None:
        data_shape_str += f"_dyn{dyn_features.shape[1]}"
    mask_shape_str = "mask" if mask is not None else "nomask"

    full_param_str = f"{data_shape_str}-{mask_shape_str}-{param_str}"

    # hashing
    hash_val = hashlib.md5(full_param_str.encode("utf-8")).hexdigest()
    cache_dir = (
        f"assets/tf_datasets/{cache_dataset_name}_{cache_dataset_group}_"
        f"{cache_split}_dataset_{hash_val}"
    )

    if os.path.isdir(cache_dir) and os.listdir(cache_dir):
        print(f"[build_tf_dataset] Loading dataset from {cache_dir} ...")
        dataset = tf.data.Dataset.load(cache_dir)
        return dataset
    else:
        print(f"[build_tf_dataset] No cached dataset found; building from scratch.\n")

    if stride is None:
        stride = window_size

    T = data.shape[0]
    N = data.shape[1]

    data = tf.convert_to_tensor(data, dtype=tf.float32)
    mask = tf.convert_to_tensor(mask, dtype=tf.float32) if mask is not None else None
    if dyn_features is not None:
        dyn_features = tf.convert_to_tensor(dyn_features, dtype=tf.float32)
        dyn_dim = dyn_features.shape[-1]
    else:
        dyn_dim = 0

    # all valid (series, start_index) window pairs
    big_meta = []
    for s_idx in range(N):
        n_possible = T - window_size + 1
        if n_possible < 1:
            continue
        for st in range(0, n_possible, stride):
            big_meta.append((s_idx, st))
    big_meta = np.array(big_meta, dtype=object)
    num_windows_total = len(big_meta)

    block_size = batch_size * windows_batch_size

    if coverage_mode == "partial":
        windows_needed = int(num_windows_total * coverage_fraction)
        windows_needed = max(block_size, windows_needed)  # at least block_size windows
        sampled_indices = np.random.choice(
            num_windows_total, size=windows_needed, replace=False
        )
        big_meta = big_meta[sampled_indices]

    x_data_list = []
    x_mask_list = []
    x_dyn_list = []
    recon_list = []
    recon_mask_list = []
    pred_list = []
    pred_mask_list = []

    def build_targets(s_idx, st):
        """
        Returns recon_target, recon_mask, pred_target, pred_mask
        for a single window start, following your old logic.
        """
        # reconstruction
        recon_target = data[st : st + window_size, s_idx]  # shape [window_size]
        recon_mask = (
            mask[st : st + window_size, s_idx]
            if mask is not None
            else tf.ones_like(recon_target)
        )

        # forecast
        if prediction_mode == "one_step_ahead":
            next_idx = st + window_size
            if next_idx < T:
                val_data = data[next_idx, s_idx]
                val_mask = mask[next_idx, s_idx] if mask is not None else 1.0
            else:
                val_data = 0.0
                val_mask = 0.0
            pred_target = tf.reshape(val_data, [1])  # shape [1]
            pred_mask = tf.reshape(val_mask, [1])  # shape [1]
        else:
            start_pred = st + window_size
            end_pred = start_pred + future_steps
            if start_pred >= T:
                # out of range => all zeros
                seg_data = tf.zeros([future_steps], dtype=tf.float32)
                seg_mask = tf.zeros([future_steps], dtype=tf.float32)
            else:
                seg_data = data[start_pred : min(end_pred, T), s_idx]
                seg_mask = (
                    mask[start_pred : min(end_pred, T), s_idx]
                    if mask is not None
                    else tf.ones_like(seg_data)
                )
                valid_len = tf.shape(seg_data)[0]
                needed = future_steps - valid_len
                if needed > 0:
                    seg_data = tf.concat([seg_data, tf.zeros([needed])], axis=0)
                    seg_mask = tf.concat([seg_mask, tf.zeros([needed])], axis=0)
            pred_target = seg_data  # [future_steps]
            pred_mask = seg_mask  # [future_steps]

        return recon_target, recon_mask, pred_target, pred_mask

    for s_idx, st in big_meta:
        w_data = data[st : st + window_size, s_idx]  # [window_size]
        w_mask = (
            mask[st : st + window_size, s_idx]
            if mask is not None
            else tf.ones_like(w_data)
        )
        if dyn_dim > 0:
            w_dyn = dyn_features[st : st + window_size, :]  # [window_size, dyn_dim]
        else:
            w_dyn = tf.zeros((window_size, 0), dtype=tf.float32)

        recon_t, recon_m, pred_t, pred_m = build_targets(s_idx, st)

        # reshape so all are rank-3 or rank-2 consistently
        x_data_list.append(tf.reshape(w_data, [window_size, 1]))
        x_mask_list.append(tf.reshape(w_mask, [window_size, 1]))
        x_dyn_list.append(w_dyn)  # shape [window_size, dyn_dim]

        recon_list.append(tf.reshape(recon_t, [-1, 1]))  # [window_size,1] or [1,1]
        recon_mask_list.append(tf.reshape(recon_m, [-1, 1]))
        pred_list.append(tf.reshape(pred_t, [-1, 1]))  # [future_steps,1] or [1,1]
        pred_mask_list.append(tf.reshape(pred_m, [-1, 1]))

    # convert all lists to a single big tensor
    x_data = tf.stack(x_data_list, axis=0)  # [num_windows, window_size, 1]
    x_mask = tf.stack(x_mask_list, axis=0)  # [num_windows, window_size, 1]
    x_dyn = tf.stack(x_dyn_list, axis=0)  # [num_windows, window_size, dyn_dim]

    recon = tf.stack(
        recon_list, axis=0
    )  # shape [num_windows, window_size, 1] if one_step => window_size= ...
    recon_m = tf.stack(recon_mask_list, axis=0)
    pred = tf.stack(pred_list, axis=0)
    pred_mask = tf.stack(pred_mask_list, axis=0)

    big_meta = np.array(big_meta, dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            (x_data, x_mask, x_dyn, pred_mask),
            (recon, recon_m, pred, pred_mask),
            big_meta,  # needed for detemporalize [num_windows, 2]
        )
    )

    if not (coverage_mode == "systematic"):
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer_size,
            reshuffle_each_iteration=reshuffle_each_epoch,
        )

    dataset = dataset.batch(block_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    if store_dataset:
        print(f"[build_tf_dataset] Saving dataset to {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)
        dataset.save(cache_dir)

    return dataset


def build_tf_dataset_multivariate(
    data: pd.DataFrame,  # shape [T, N]
    mask: pd.DataFrame,  # shape [T, N]
    dyn_features: pd.DataFrame,  # shape [T, dyn_dim]
    cache_split: str,
    cache_dataset_name: str,
    cache_dataset_group: str,
    window_size: int = 16,
    stride: int = None,
    coverage_mode: str = "systematic",
    coverage_fraction: float = 0.1,
    prediction_mode: str = "one_step_ahead",
    future_steps: int = 1,
    batch_size: int = 8,
    windows_batch_size: int = 8,
    shuffle_buffer_size: int = 1000,
    reshuffle_each_epoch: bool = True,
    store_dataset: bool = True,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset for a multivariate approach, where each window
    includes *all* series in the last dimension. i.e. x_data => shape [window_size, N].
    """

    param_str = (
        f"multivar-win{window_size}-str{stride}-covmode{coverage_mode}-covfrac{coverage_fraction}"
        f"-predmode{prediction_mode}-fut{future_steps}-batch{batch_size}-wbatch{windows_batch_size}"
        f"-shufbuf{shuffle_buffer_size}-reshuff{reshuffle_each_epoch}"
    )

    data_shape_str = f"T{data.shape[0]}_N{data.shape[1]}"
    if dyn_features is not None:
        data_shape_str += f"_dyn{dyn_features.shape[1]}"
    mask_shape_str = "mask" if mask is not None else "nomask"

    full_param_str = f"{data_shape_str}-{mask_shape_str}-{param_str}"

    import hashlib, os

    hash_val = hashlib.md5(full_param_str.encode("utf-8")).hexdigest()
    cache_dir = (
        f"assets/tf_datasets/{cache_dataset_name}_{cache_dataset_group}_"
        f"{cache_split}_dataset_multivar_{hash_val}"
    )

    if os.path.isdir(cache_dir) and os.listdir(cache_dir):
        print(f"[build_tf_dataset_multivariate] Loading dataset from {cache_dir} ...")
        dataset = tf.data.Dataset.load(cache_dir)
        return dataset
    else:
        print(
            "[build_tf_dataset_multivariate] No cached dataset found; building from scratch.\n"
        )

    if stride is None:
        stride = window_size

    T = data.shape[0]
    N = data.shape[1]

    data = tf.convert_to_tensor(data, dtype=tf.float32)
    if mask is not None:
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    if dyn_features is not None:
        dyn_features = tf.convert_to_tensor(dyn_features, dtype=tf.float32)
        dyn_dim = dyn_features.shape[-1]
    else:
        dyn_dim = 0

    big_meta = []
    n_possible = T - window_size + 1
    for st in range(0, n_possible, stride):
        big_meta.append(st)

    big_meta = np.array(big_meta, dtype=np.int32)
    num_windows_total = len(big_meta)

    block_size = batch_size * windows_batch_size

    if coverage_mode == "partial":
        windows_needed = int(num_windows_total * coverage_fraction)
        windows_needed = max(block_size, windows_needed)
        sampled_indices = np.random.choice(
            num_windows_total, size=windows_needed, replace=False
        )
        big_meta = big_meta[sampled_indices]

    x_data_list = []
    x_mask_list = []
    x_dyn_list = []
    recon_list = []
    recon_mask_list = []
    pred_list = []
    pred_mask_list = []

    def build_targets(st):
        """
        Returns recon_target, recon_mask, pred_target, pred_mask
        for a single window start, for a multivariate approach:
          - recon_target => shape [window_size, N]
          - pred_target => shape [future_steps, N]
        """
        recon_target = data[st : st + window_size, :]  # [window_size, N]
        if mask is not None:
            recon_m = mask[st : st + window_size, :]
        else:
            recon_m = tf.ones_like(recon_target)

        # forecast
        if prediction_mode == "one_step_ahead":
            next_idx = st + window_size
            if next_idx < T:
                val_data = data[next_idx, :]  # shape [N]
                val_mask = (
                    mask[next_idx, :] if mask is not None else tf.ones_like(val_data)
                )
            else:
                val_data = tf.zeros([N], dtype=tf.float32)
                val_mask = tf.zeros([N], dtype=tf.float32)
            pred_target = tf.reshape(val_data, [1, N])  # => shape [1, N]
            pred_m = tf.reshape(val_mask, [1, N])
        else:
            start_pred = st + window_size
            end_pred = start_pred + future_steps
            if start_pred >= T:
                seg_data = tf.zeros([future_steps, N], dtype=tf.float32)
                seg_mask = tf.zeros([future_steps, N], dtype=tf.float32)
            else:
                seg_data = data[start_pred : min(end_pred, T), :]  # shape [?, N]
                if mask is not None:
                    seg_mask = mask[start_pred : min(end_pred, T), :]
                else:
                    seg_mask = tf.ones_like(seg_data)

                valid_len = tf.shape(seg_data)[0]
                needed = future_steps - valid_len
                if needed > 0:
                    seg_data = tf.concat([seg_data, tf.zeros([needed, N])], axis=0)
                    seg_mask = tf.concat([seg_mask, tf.zeros([needed, N])], axis=0)

            pred_target = seg_data  # shape [future_steps, N]
            pred_m = seg_mask

        return recon_target, recon_m, pred_target, pred_m

    # build the window
    for st in big_meta:
        w_data = data[st : st + window_size, :]  # => hape [window_size, N]
        if mask is not None:
            w_mask = mask[st : st + window_size, :]
        else:
            w_mask = tf.ones_like(w_data)

        if dyn_dim > 0:
            w_dyn = dyn_features[st : st + window_size, :]  # [window_size, dyn_dim]
        else:
            w_dyn = tf.zeros((window_size, 0), dtype=tf.float32)

        recon_t, recon_m, pred_t, pred_m = build_targets(st)

        x_data_list.append(w_data)  # => shape [window_size, N]
        x_mask_list.append(w_mask)  # => shape [window_size, N]
        x_dyn_list.append(w_dyn)  # => shape [window_size, dyn_dim]

        recon_list.append(recon_t)  # shape [window_size, N]
        recon_mask_list.append(recon_m)  # shape [window_size, N]
        pred_list.append(pred_t)  # shape [future_steps, N] or [1, N]
        pred_mask_list.append(pred_m)

    x_data = tf.stack(x_data_list, axis=0)  # => [num_windows, window_size, N]
    x_mask = tf.stack(x_mask_list, axis=0)  # => [num_windows, window_size, N]
    x_dyn = tf.stack(x_dyn_list, axis=0)  # => [num_windows, window_size, dyn_dim]

    recon = tf.stack(recon_list, axis=0)  # => [num_windows, window_size, N]
    recon_m = tf.stack(recon_mask_list, axis=0)  # => [num_windows, window_size, N]
    pred = tf.stack(
        pred_list, axis=0
    )  # => [num_windows, future_steps, N] or [num_windows, 1, N]
    pred_m = tf.stack(pred_mask_list, axis=0)

    big_meta = tf.convert_to_tensor(big_meta, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            (x_data, x_mask, x_dyn, pred_m),
            (recon, recon_m, pred, pred_m),
            big_meta,
        )
    )

    if coverage_mode != "systematic":
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer_size,
            reshuffle_each_iteration=reshuffle_each_epoch,
        )

    block_size = batch_size * windows_batch_size
    dataset = dataset.batch(block_size).prefetch(tf.data.AUTOTUNE)

    if store_dataset:
        import os

        print(f"[build_tf_dataset_multivariate] Saving dataset to {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)
        dataset.save(cache_dir)

    return dataset


def build_forecast_dataset(
    holdout_df: pd.DataFrame,
    unique_ids: List[str],
    lookback_window: int,
    horizon: int,
    dyn_feature_cols: List[str],
) -> Tuple[tf.data.Dataset, List]:
    """
    Build a tf.data.Dataset with exactly 1 sample per series,
    where each sample is (x_data, x_mask, x_dyn, pred_mask).

    x_data = last 'window_size' points from that series
    pred_mask = shape [horizon, 1], all ones.

    holdout_df has columns:
      ['unique_id', 'ds', 'y'] + dynamic feature columns.
    """
    x_data_list = []
    x_mask_list = []
    x_dyn_list = []
    pred_mask_list = []

    meta_list = []

    for uid in unique_ids:
        df_ser = holdout_df[holdout_df["unique_id"] == uid].sort_values("ds")

        # extract the lookback_window from the "history"
        hist_part = (
            df_ser.iloc[-(lookback_window + horizon) : -horizon]
            if horizon > 0
            else df_ser.iloc[-lookback_window:]
        )

        fut_part = df_ser.iloc[-horizon:] if horizon > 0 else pd.DataFrame()

        x_vals = hist_part["y"].values  # [window_size,]
        x_mask = np.ones_like(x_vals, dtype=np.float32)
        if len(x_vals) < lookback_window:
            # zero-pad or skip
            print(f"Series {uid} doesn't have enough history. Skipping.")
            continue

        if len(dyn_feature_cols) > 0:
            hist_dyn = hist_part[dyn_feature_cols].values  # [window_size, dyn_dim]
        else:
            hist_dyn = np.zeros((lookback_window, 0), dtype=np.float32)

        pmask = (
            np.ones((horizon, 1), dtype=np.float32) if horizon > 0 else np.zeros((1, 1))
        )

        x_vals = x_vals.reshape(lookback_window, 1)
        x_mask = x_mask.reshape(lookback_window, 1)

        x_data_list.append(x_vals)
        x_mask_list.append(x_mask)
        x_dyn_list.append(hist_dyn)
        pred_mask_list.append(pmask)

        # keep track of which series + final timestamps for alignment
        meta_list.append(uid)

    x_data = tf.convert_to_tensor(np.stack(x_data_list, axis=0), dtype=tf.float32)
    x_mask = tf.convert_to_tensor(np.stack(x_mask_list, axis=0), dtype=tf.float32)
    x_dyn = tf.convert_to_tensor(np.stack(x_dyn_list, axis=0), dtype=tf.float32)
    pred_mask = tf.convert_to_tensor(np.stack(pred_mask_list, axis=0), dtype=tf.float32)

    # pass placeholders since we only want the forecast_out
    recon_stub = tf.zeros_like(x_data)
    recon_mask_stub = tf.ones_like(x_data)
    pred_stub = tf.zeros_like(pred_mask)
    big_meta = np.arange(len(meta_list))[:, None]

    # build final dataset of shape [num_series, ...]
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            (x_data, x_mask, x_dyn, pred_mask),
            (recon_stub, recon_mask_stub, pred_stub, pred_mask),
            big_meta,
        )
    )

    dataset = dataset.batch(len(meta_list))
    return dataset, meta_list


class Sampling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Sampling, self).__init__(**kwargs)

    def call(self, inputs):
        """
        Performs the reparameterization trick.
        """
        z_mean, z_log_var = inputs

        epsilon = tf.random.normal(tf.shape(z_mean))

        # reparameterization trick
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def masked_mse(y_true, y_pred, mask):
    """
    Compute Mean Squared Error only on the unmasked (non-padded) entries.
    """
    y_true_masked = y_true * mask
    y_pred_masked = y_pred * mask

    mse = tf.reduce_mean(tf.square(y_true_masked - y_pred_masked))
    return mse


class KLScheduleCallback(tf.keras.callbacks.Callback):
    """
    Linearly increases the CVAE's KL weight from `kl_start` to `kl_end`
    over `warmup_epochs` epochs.
    """

    def __init__(self, cvae_model, kl_start=0.0, kl_end=1.0, warmup_epochs=10):
        super().__init__()
        self.cvae_model = cvae_model
        self.kl_start = kl_start
        self.kl_end = kl_end
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        # ramp up linearly until warmup_epochs
        if epoch < self.warmup_epochs:
            ratio = float(epoch) / float(self.warmup_epochs)
            new_kl = self.kl_start + ratio * (self.kl_end - self.kl_start)
        else:
            new_kl = self.kl_end

        self.cvae_model.kl_weight = new_kl

        print(f"[Epoch {epoch}] kl_weight set to {new_kl:.4f}")


class CVAE(keras.Model):
    def __init__(
        self,
        encoder: keras.Model,
        decoder: keras.Model,
        kl_weight_initial: float = 0.0,
        forecasting: bool = True,
        **kwargs,
    ) -> None:
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        self.kl_weight = kl_weight_initial

        self.forecasting = forecasting

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.prediction_loss_tracker = keras.metrics.Mean(name="prediction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs, training=False):
        batch_data, batch_mask, batch_dyn_features, pred_mask = inputs

        z_mean, z_log_var, z = self.encoder(
            [batch_data, batch_mask, batch_dyn_features]
        )

        pred_reconst, pred_forecast = self.decoder(
            [z, batch_mask, batch_dyn_features, pred_mask]
        )

        return pred_reconst, z_mean, z_log_var, pred_forecast

    @property
    def metrics(self):
        """Ensures that Keras tracks the metrics properly during training."""
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.prediction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def compute_loss(
        self,
        recon_true,
        pred_true,
        recon_pred,
        pred_pred,
        z_mean,
        z_log_var,
        recon_mask,
        pred_mask,
    ):
        """
        recon_true:  [B, window_size, 1]
        pred_true:   either [B, 1, 1] for one-step, or [B, future_steps, 1] for multi-step
        recon_pred:  [B, window_size, 1]
        pred_pred:   [B, future_steps, 1] or [B, 1, 1]
        z_mean:      [B, window_size, latent_dim]
        z_log_var:   same shape as z_mean
        """
        reconstruction_loss = masked_mse(
            y_true=recon_true, y_pred=recon_pred, mask=recon_mask
        )

        kl_loss = -0.5 * K.mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

        if self.forecasting:
            prediction_loss = masked_mse(
                y_true=pred_true, y_pred=pred_pred, mask=pred_mask
            )
            total_loss = (
                reconstruction_loss + prediction_loss + self.kl_weight * kl_loss
            )
        else:
            prediction_loss = 0.0
            total_loss = reconstruction_loss + self.kl_weight * kl_loss

        return total_loss, reconstruction_loss, prediction_loss, kl_loss

    @tf.function
    def train_step(self, data):
        (
            (batch_data, batch_mask, batch_dyn, pred_mask),
            (recon_target, recon_mask, pred_target, pred_mask),
            big_meta,
        ) = data

        with tf.GradientTape() as tape:
            pred_reconst, z_mean, z_log_var, pred_forecast = self(
                (batch_data, batch_mask, batch_dyn, pred_mask),
                training=True,
            )

            total_loss, reconstruction_loss, prediction_loss, kl_loss = (
                self.compute_loss(
                    recon_true=recon_target,
                    pred_true=pred_target,
                    recon_pred=pred_reconst,
                    pred_pred=pred_forecast,
                    z_mean=z_mean,
                    z_log_var=z_log_var,
                    recon_mask=recon_mask,
                    pred_mask=pred_mask,
                )
            )

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.prediction_loss_tracker.update_state(prediction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "prediction_loss": self.prediction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        (
            (batch_data, batch_mask, batch_dyn, pred_mask),
            (
                recon_target,
                recon_mask,
                pred_target,
                pred_mask,
            ),
            big_meta,
        ) = data

        pred_reconst, z_mean, z_log_var, pred_forecast = self(
            (batch_data, batch_mask, batch_dyn, pred_mask),
            training=False,
        )
        total_loss, reconstruction_loss, prediction_loss, kl_loss = self.compute_loss(
            recon_true=recon_target,
            recon_mask=recon_mask,
            pred_true=pred_target,
            pred_mask=pred_mask,
            recon_pred=pred_reconst,
            pred_pred=pred_forecast,
            z_mean=z_mean,
            z_log_var=z_log_var,
        )

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.prediction_loss_tracker.update_state(prediction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "prediction_loss": self.prediction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "encoder": self.encoder.get_config(),
                "decoder": self.decoder.get_config(),
                "kl_weight_initial": self.kl_weight,
                "forecasting": self.forecasting,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        encoder_cfg = config.pop("encoder")
        decoder_cfg = config.pop("decoder")
        encoder = keras.Model.from_config(encoder_cfg)
        decoder = keras.Model.from_config(decoder_cfg)
        return cls(encoder=encoder, decoder=decoder, **config)


def get_CVAE(
    window_size: int,
    input_dim: int,
    latent_dim: int,
    pred_dim: int,
    time_dist_units: int = 16,
    n_blocks: int = 3,
    kernel_size: Tuple[int] = (2, 2, 1),
    n_hidden: int = 256,
    forecasting: bool = True,
):
    """
    Constructs the encoder and decoder with input shapes:
      encoder input:  (window_size, input_dim)
      dyn_features:   (window_size, d_something)
    """
    input_shape_main = (window_size, input_dim)
    # 6 dynamic features
    input_shape_dyn_features = (window_size, 6)
    output_shape_pred = (pred_dim, input_dim)

    enc = encoder(
        input_shape=input_shape_main,
        input_shape_dyn_features=input_shape_dyn_features,
        latent_dim=latent_dim,
        time_dist_units=time_dist_units,
        n_blocks=n_blocks,
        kernel_size=kernel_size,
        n_hidden=n_hidden,
    )

    dec = decoder(
        output_shape=input_shape_main,
        output_shape_dyn_features=input_shape_dyn_features,
        latent_dim=latent_dim,
        time_dist_units=time_dist_units,
        n_blocks=n_blocks,
        pred_shape=output_shape_pred,
        kernel_size=kernel_size,
        forecasting=forecasting,
        n_hidden=n_hidden,
    )

    return enc, dec


class UpsampleTimeLayer(tf.keras.layers.Layer):
    def __init__(self, target_len: int, method="bilinear"):
        super().__init__()
        self.target_len = target_len
        self.method = method

    def call(self, x):
        # x shape: [B, current_len, 1]
        x = tf.expand_dims(x, axis=1)  # => [B, 1, current_len, 1]
        x_upsampled = tf.image.resize(
            images=x, size=(1, self.target_len), method=self.method
        )  # => [B, 1, target_len, 1]
        x_upsampled = tf.squeeze(x_upsampled, axis=1)  # => [B, target_len, 1]
        return x_upsampled


class MRHIBlock_backcast(tf.keras.layers.Layer):
    """
    This block produces ONLY a backcast of shape [B, seq_len, hidden_dim],
    so that we can do `residual - backcast`.
    """

    def __init__(
        self,
        backcast_size: Tuple[int, int],  # (seq_len, hidden_dim)
        n_knots: int = 4,
        n_hidden: int = 256,
        n_layers: int = 2,
        kernel_size: Tuple[int] = (2, 2, 1),
        activation="relu",
        dropout_rate: float = 0.2,
        l2_lambda: float = 1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seq_len, self.hidden_dim = backcast_size
        self.n_knots = n_knots
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        # produce knots dimension, then upsample back to seq_len
        self.n_theta_backcast = self.n_knots * self.hidden_dim

        self.upsample_backcast = UpsampleTimeLayer(
            target_len=self.seq_len, method="bilinear"
        )

        self.pooling_layers = []
        for k in kernel_size:
            self.pooling_layers.append(
                layers.MaxPooling1D(pool_size=k, strides=k, padding="valid")
            )

        mlp_layers = [layers.Flatten()]
        for _ in range(n_layers):
            mlp_layers.append(
                layers.Dense(
                    n_hidden,
                    kernel_regularizer=regularizers.l2(l2_lambda),
                    use_bias=False,
                )
            )
            mlp_layers.append(layers.BatchNormalization())
            mlp_layers.append(layers.Activation(activation))
            mlp_layers.append(layers.Dropout(dropout_rate))

        # (n_knots * hidden_dim)
        mlp_layers.append(
            layers.Dense(
                self.n_theta_backcast, kernel_regularizer=regularizers.l2(l2_lambda)
            )
        )

        self.mlp_stack = tf.keras.Sequential(mlp_layers)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        x shape: [B, seq_len, hidden_dim]
        1) pool => shape [B, n_knots, hidden_dim]
        2) flatten => MLP => [B, n_knots*hidden_dim]
        3) reshape => backcast_knots => [B, n_knots, hidden_dim]
        4) upsample => [B, seq_len, hidden_dim]
        """
        for pool_layer in self.pooling_layers:
            x = pool_layer(x)

        theta = self.mlp_stack(x)
        backcast_knots = tf.reshape(theta, [-1, self.n_knots, self.hidden_dim])
        backcast = self.upsample_backcast(backcast_knots)
        return backcast


class MRHIBlock_backcast_forecast(tf.keras.layers.Layer):
    """
    Produces:
      - backcast: [B, seq_len, hidden_dim] so we can do `residual - backcast`.
      - forecast: [B, pred_len, hidden_dim] for the future horizon.
    """

    def __init__(
        self,
        backcast_size: Tuple[int, int],  # (seq_len, hidden_dim)
        forecast_size: Tuple[int, int],  # (pred_len, hidden_dim)
        n_knots: int = 4,
        n_hidden: int = 256,
        n_layers: int = 2,
        kernel_size: Tuple[int] = (2, 2, 1),
        activation="relu",
        dropout_rate: float = 0.2,
        l2_lambda: float = 1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seq_len, self.hidden_dim = backcast_size
        self.pred_len, _ = forecast_size

        self.n_knots = n_knots
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        # produce n_knots * hidden_dim for backcast, plus n_knots*hidden_dim for forecast
        self.n_theta_backcast = self.n_knots * self.hidden_dim
        self.n_theta_forecast = self.n_knots * self.hidden_dim
        self.n_theta = self.n_theta_backcast + self.n_theta_forecast

        self.upsample_backcast = UpsampleTimeLayer(
            target_len=self.seq_len, method="bilinear"
        )
        self.upsample_forecast = UpsampleTimeLayer(
            target_len=self.pred_len, method="bilinear"
        )

        self.pooling_layers = []
        for k in kernel_size:
            self.pooling_layers.append(
                layers.MaxPooling1D(pool_size=k, strides=k, padding="valid")
            )

        mlp_layers = [layers.Flatten()]
        for _ in range(n_layers):
            mlp_layers.append(
                layers.Dense(
                    n_hidden,
                    kernel_regularizer=regularizers.l2(l2_lambda),
                    use_bias=False,
                )
            )
            mlp_layers.append(layers.BatchNormalization())
            mlp_layers.append(layers.Activation(activation))
            mlp_layers.append(layers.Dropout(dropout_rate))

        mlp_layers.append(
            layers.Dense(self.n_theta, kernel_regularizer=regularizers.l2(l2_lambda))
        )
        self.mlp_stack = tf.keras.Sequential(mlp_layers)

    def call(self, x: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """
        x shape: [B, seq_len, hidden_dim].
        1) pool => shape [B, n_knots, hidden_dim]
        2) flatten => MLP => [B, n_theta_backcast + n_theta_forecast]
        3) slice => backcast part vs. forecast part
        4) reshape each => [B, n_knots, hidden_dim]
        5) upsample backcast => [B, seq_len, hidden_dim], forecast => [B, pred_len, hidden_dim]
        """
        for pool_layer in self.pooling_layers:
            x = pool_layer(x)

        theta = self.mlp_stack(x)

        backcast_flat = theta[:, : self.n_theta_backcast]
        forecast_flat = theta[:, self.n_theta_backcast :]

        backcast_knots = tf.reshape(backcast_flat, [-1, self.n_knots, self.hidden_dim])
        forecast_knots = tf.reshape(forecast_flat, [-1, self.n_knots, self.hidden_dim])

        backcast = self.upsample_backcast(backcast_knots)
        forecast = self.upsample_forecast(forecast_knots)

        return backcast, forecast


def encoder(
    input_shape: Tuple[int, int],
    input_shape_dyn_features: Tuple[int, int],
    latent_dim: int,
    time_dist_units: int = 16,
    n_blocks: int = 3,
    n_hidden: int = 256,
    n_layers: int = 2,
    kernel_size: Tuple[int] = (2, 2, 1),
):
    main_input = layers.Input(shape=input_shape, name="main_input")
    mask_input = layers.Input(shape=input_shape, name="mask_input")
    dyn_features_input = layers.Input(
        shape=input_shape_dyn_features, name="dyn_features_input"
    )

    masked_input = layers.Multiply(name="masked_input")([main_input, mask_input])
    masked_input = layers.Concatenate()([dyn_features_input, masked_input])

    masked_input = layers.TimeDistributed(
        layers.Dense(time_dist_units, activation="relu")
    )(masked_input)

    backcast_total = masked_input
    final_output = 0

    for i in range(n_blocks):
        mrhi_block = MRHIBlock_backcast(
            backcast_size=input_shape,
            n_knots=4,
            n_hidden=n_hidden,
            n_layers=n_layers,
            kernel_size=kernel_size[int(3 - n_layers) :],
        )

        backcast = mrhi_block(backcast_total)

        backcast_total = backcast_total - backcast
        final_output += backcast

    final_output = layers.TimeDistributed(
        layers.Dense(latent_dim * 2, activation="relu")
    )(final_output)

    z_mean = layers.Lambda(lambda x: tf.clip_by_value(x[:, :, latent_dim:], -5, 5))(
        final_output
    )
    z_log_var = layers.Lambda(lambda x: tf.clip_by_value(x[:, :, :latent_dim], -5, 5))(
        final_output
    )

    z = Sampling(name="sampling")([z_mean, z_log_var])

    return tf.keras.Model(
        inputs=[main_input, mask_input, dyn_features_input],
        outputs=[z_mean, z_log_var, z],
        name="encoder",
    )


def decoder(
    output_shape: Tuple[int, int],
    output_shape_dyn_features: Tuple[int, int],
    latent_dim: int,
    pred_shape: Tuple[int, int],
    time_dist_units: int = 16,
    n_blocks: int = 3,
    n_hidden: int = 256,
    n_layers: int = 2,
    kernel_size: Tuple[int] = (2, 2, 1),
    forecasting: bool = True,
):
    window_size, n_features = output_shape
    future_steps, n_features_forecast = pred_shape

    latent_input = layers.Input(shape=(window_size, latent_dim), name="latent_input")
    dyn_features_input = layers.Input(
        shape=output_shape_dyn_features, name="dyn_features_input"
    )
    mask_input = layers.Input(shape=output_shape, name="mask_input")
    pred_mask = layers.Input(shape=pred_shape, name="pred_mask")

    x = layers.Concatenate()([dyn_features_input, latent_input])

    x = layers.TimeDistributed(layers.Dense(time_dist_units, activation="relu"))(x)

    residual = x

    final_forecast = layers.Lambda(
        lambda t: tf.zeros((tf.shape(t)[0], future_steps, n_hidden), dtype=t.dtype)
    )(x)
    final_backcast = 0

    if forecasting:
        for i in range(n_blocks):
            mrhi_block = MRHIBlock_backcast_forecast(
                backcast_size=output_shape,
                forecast_size=pred_shape,
                n_knots=4,
                n_hidden=n_hidden,
                n_layers=n_layers,
                kernel_size=kernel_size[int(3 - n_layers) :],
            )

            backcast, forecast = mrhi_block(residual)

            residual = residual - backcast
            final_forecast = final_forecast + forecast
            final_backcast = final_backcast + backcast
    else:
        for i in range(n_blocks):
            mrhi_block = MRHIBlock_backcast(
                backcast_size=output_shape,
                n_hidden=n_hidden,
                n_knots=4,
                n_layers=n_layers,
                kernel_size=kernel_size[int(3 - n_layers) :],
            )
            backcast = mrhi_block(residual)
            residual = residual - backcast
            final_backcast = final_backcast + backcast
        final_forecast = final_backcast

    backcast_out = layers.TimeDistributed(layers.Dense(n_features, activation=None))(
        final_backcast
    )  # => [B, window_size, n_features]

    forecast_out = layers.TimeDistributed(
        layers.Dense(n_features_forecast, activation=None)
    )(final_forecast)

    backcast_out = layers.Multiply(name="masked_backcast_output")(
        [backcast_out, mask_input]
    )

    forecast_out = layers.Multiply(name="masked_forecast_output")(
        [forecast_out, pred_mask]
    )

    return tf.keras.Model(
        inputs=[latent_input, mask_input, dyn_features_input, pred_mask],
        outputs=[backcast_out, forecast_out],
        name="decoder",
    )
