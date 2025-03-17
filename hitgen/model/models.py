import logging
import os
from typing import List, Tuple, Union
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from keras import layers
import tensorflow as tf
from tensorflow.keras import utils
from keras import regularizers
from tensorflow import keras

K = keras.backend


class TemporalizeGenerator(utils.Sequence):
    """
    A generator for univariate time-series data of shape [T, N], optionally with dynamic
    features [T, dyn_dim] and a mask [T, N]. It enumerates all valid windows of length
    `window_size` across the N series, then samples and batches them as follows:

    - coverage_mode='systematic': Uses all windows each epoch (shuffled).
    - coverage_mode='partial': Randomly samples a subset (e.g., coverage_ratio or windows_per_epoch).

    Windows are chunked into batches of size (batch_size * windows_batch_size). For each
    window, two targets are built:
      1. Reconstruction target (the same window).
      2. Forecast target (one-step-ahead or multi-step-ahead, zero-padded if out of range).

    In each __getitem__ call, the generator returns:
      (batch_data, batch_mask, batch_dyn_features),
      (recon_target, recon_mask, pred_target, pred_mask),
    """

    def __init__(
        self,
        data,  # shape [T, N], univariate
        dyn_features=None,  # shape [T, dyn_dim] or None
        mask=None,  # shape [T, N] or None
        window_size=16,
        batch_size=8,  # series per batch
        windows_batch_size=4,  # windows per series
        coverage_mode="systematic",  # or "partial"
        coverage_fraction=0.1,  # fraction of total windows used per epoch if partial
        stride=None,
        prediction_mode="one_step_ahead",
        future_steps=1,
    ):
        if not stride:
            stride = window_size
        self.data = tf.convert_to_tensor(data, dtype=tf.float32)
        if dyn_features is not None:
            self.dyn_features = tf.convert_to_tensor(dyn_features, dtype=tf.float32)
            self.dyn_dim = self.dyn_features.shape[-1]
        else:
            self.dyn_features = None
            self.dyn_dim = 0

        if mask is not None:
            self.mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        else:
            self.mask = None

        self.window_size = window_size
        self.batch_size = batch_size
        self.windows_batch_size = windows_batch_size
        self.coverage_fraction = coverage_fraction
        self.stride = stride
        self.coverage_mode = coverage_mode

        # T = number of time points, N = number of series
        self.T = self.data.shape[0]
        self.N = self.data.shape[1]

        self.series_indices = tf.range(self.N)

        meta_list = []
        for s_idx in range(self.N):
            n_possible = self.T - window_size + 1
            if n_possible < 1:
                # series too short => skip
                continue
            for st in range(0, n_possible, self.stride):
                meta_list.append((s_idx, st))
        self.big_metadata = np.array(meta_list, dtype=object)
        self.num_windows_total = len(self.big_metadata)

        self.block_size = self.batch_size * self.windows_batch_size

        # coverage_mode logic
        if coverage_mode == "systematic":
            self.windows_per_epoch_actual = self.num_windows_total
        elif coverage_mode == "partial":
            # interpret coverage_ratio as fraction of total windows
            if coverage_fraction <= 0 or coverage_fraction > 1:
                raise ValueError("coverage_ratio must be in (0, 1].")
            self.windows_per_epoch_actual = int(
                coverage_fraction * self.num_windows_total
            )
            # at minimum, use 1 block if coverage_ratio is very small:
            self.windows_per_epoch_actual = max(
                self.block_size, self.windows_per_epoch_actual
            )
        else:
            raise ValueError("coverage_mode must be 'systematic' or 'partial'.")

        # shuffle entire big_metadata initially
        np.random.shuffle(self.big_metadata)

        assert prediction_mode in (
            "one_step_ahead",
            "multi_step_ahead",
        ), "prediction_mode must be 'one_step_ahead' or 'multi_step_ahead'."
        self.prediction_mode = prediction_mode
        self.future_steps = future_steps

        self.epoch_indices = None
        self.on_epoch_end()

        self.steps_per_epoch = (
            self.windows_per_epoch_actual + self.block_size - 1
        ) // self.block_size

        # For debugging
        # print(
        #     f"num_windows_total={self.num_windows_total}, windows_per_epoch_actual={self.windows_per_epoch_actual}, "
        #     f"steps_per_epoch={self.steps_per_epoch}"
        # )

    def __len__(self):
        return self.steps_per_epoch

    def on_epoch_end(self):
        """Pick the indices for this epoch. Then define steps_per_epoch accordingly."""
        if self.coverage_mode == "systematic":
            np.random.shuffle(self.big_metadata)
            # define epoch_indices as 0..num_windows_total-1
            self.epoch_indices = np.arange(self.num_windows_total)
        else:
            sampled = np.random.choice(
                self.num_windows_total,
                size=self.windows_per_epoch_actual,
                replace=False,
            )
            self.epoch_indices = sampled

    def __getitem__(self, idx):
        """
        Returns a batch of windows of shape [B, window_size, 1],
        plus dynamic features => [B, window_size, dyn_dim],
        plus the reconstruction and forecast targets.
        """
        start_i = idx * self.block_size
        end_i = min(start_i + self.block_size, len(self.epoch_indices))
        current_inds = self.epoch_indices[start_i:end_i]
        current_meta = self.big_metadata[current_inds]

        batch_data_list = []
        batch_mask_list = []
        batch_dyn_list = []

        batch_recon_list = []
        batch_recon_mask_list = []
        batch_pred_list = []
        batch_pred_mask_list = []

        for s_idx, st in current_meta:
            # gather the window from [st : st+window_size], series s_idx
            w_data = self.data[st : st + self.window_size, s_idx]  # shape [window_size]
            w_mask = self.mask[st : st + self.window_size, s_idx]  # shape [window_size]

            if self.dyn_features is not None:
                w_dyn = self.dyn_features[
                    st : st + self.window_size, :
                ]  # shape [window_size, dyn_dim]
            else:
                w_dyn = tf.zeros((self.window_size, 0), dtype=tf.float32)

            w_data = tf.reshape(w_data, [1, self.window_size, 1])
            w_mask = tf.reshape(w_mask, [1, self.window_size, 1])
            w_dyn = tf.reshape(w_dyn, [1, self.window_size, self.dyn_dim])

            recon_t, recon_m, pred_t, pred_m = self._build_targets(
                s_idx, st, w_data, w_mask
            )

            batch_data_list.append(w_data)
            batch_mask_list.append(w_mask)
            batch_dyn_list.append(w_dyn)

            batch_recon_list.append(recon_t)
            batch_recon_mask_list.append(recon_m)
            batch_pred_list.append(pred_t)
            batch_pred_mask_list.append(pred_m)

        batch_data = tf.concat(batch_data_list, axis=0)  # [B, window_size, 1]
        batch_mask = tf.concat(batch_mask_list, axis=0)  # [B, window_size, 1]
        batch_dyn = tf.concat(batch_dyn_list, axis=0)  # [B, window_size, dyn_dim]

        batch_recon_target = tf.concat(batch_recon_list, axis=0)  # [B, window_size, 1]
        batch_recon_mask = tf.concat(
            batch_recon_mask_list, axis=0
        )  # [B, window_size, 1]
        batch_pred_target = tf.concat(batch_pred_list, axis=0)  # shape [B, * , 1]
        batch_pred_mask = tf.concat(batch_pred_mask_list, axis=0)  # shape [B, * , 1]

        # store metadata for this batch => used in manual inference loop
        self.last_batch_metadata = current_meta  # list of (series_idx, start_idx)

        return (
            (batch_data, batch_mask, batch_dyn),
            (batch_recon_target, batch_recon_mask, batch_pred_target, batch_pred_mask),
        )

    def _build_targets(self, s_idx, st, window_data, window_mask):
        """
        Build:
          recon_target, recon_mask => same shape as input window
          pred_target, pred_mask => shape depends on 'prediction_mode' & 'future_steps'
        """
        # reconstruction
        recon_target = window_data
        recon_mask = window_mask

        # forecast
        if self.prediction_mode == "one_step_ahead":
            next_idx = st + self.window_size
            if next_idx < self.T:
                val_data = self.data[next_idx, s_idx]
                val_mask = self.mask[next_idx, s_idx]
            else:
                val_data = 0.0
                val_mask = 0.0

            pred_target = tf.reshape(val_data, [1, 1, 1])  # => [1, 1, 1]
            pred_mask = tf.reshape(val_mask, [1, 1, 1])  # => [1, 1, 1]

        else:  # "multi_step_ahead"
            start_pred = st + self.window_size
            end_pred = start_pred + self.future_steps
            if start_pred >= self.T:
                # out of range
                seg_data = tf.zeros([self.future_steps], dtype=tf.float32)
                seg_mask = tf.zeros([self.future_steps], dtype=tf.float32)
            else:
                seg_data = self.data[start_pred : min(end_pred, self.T), s_idx]
                seg_mask = self.mask[start_pred : min(end_pred, self.T), s_idx]
                valid_len = tf.shape(seg_data)[0]
                needed = self.future_steps - valid_len
                if needed > 0:
                    seg_data = tf.concat([seg_data, tf.zeros([needed])], axis=0)
                    seg_mask = tf.concat([seg_mask, tf.zeros([needed])], axis=0)

            pred_target = tf.reshape(
                seg_data, [1, self.future_steps, 1]
            )  # => [1, future_steps, 1]
            pred_mask = tf.reshape(seg_mask, [1, self.future_steps, 1])

        return recon_target, recon_mask, pred_target, pred_mask


class _EncoderOnlyGenerator(keras.utils.Sequence):
    """
    This small generator is used for calling `cvae.encoder.predict(...)`.
    We take the same data_mask_temporalized, but yield only the "input" portion
    (batch_data, batch_mask, batch_dyn), ignoring the targets.
    """

    def __init__(self, data_gen: TemporalizeGenerator):
        self.data_gen = data_gen

    def __len__(self):
        return len(self.data_gen)

    def __getitem__(self, index):
        # the original generator returns ((data, mask, dyn), (recon_target, pred_target))
        inputs, _ = self.data_gen[index]
        # inputs => (batch_data, batch_mask, batch_dyn)
        return inputs

    def on_epoch_end(self):
        self.data_gen.on_epoch_end()


class _DecoderOnlyGenerator(keras.utils.Sequence):
    """
    This generator is used to feed the decoder after we have z_augmented.
    We take the mask, dyn features from the original generator, and we supply z_augmented
    as the first input in place of batch_data.
    """

    def __init__(self, data_gen: TemporalizeGenerator, z_augmented: np.ndarray):
        self.data_gen = data_gen
        self.z_augmented = z_augmented

        # z_augmented should have shape [B_total, window_size, latent_dim]
        # We gather the indices where each batch start/ends:
        self.batch_size_list = []
        self.cum_sizes = [0]
        total_B = len(z_augmented)
        for i in range(len(data_gen)):
            # each __getitem__ returns a batch with shape [b_i, window_size, latent_dim]
            (b_data, b_mask, b_dyn), _ = data_gen[i]
            b_len = b_data.shape[0]  # number of windows
            self.batch_size_list.append(b_len)
            self.cum_sizes.append(self.cum_sizes[-1] + b_len)

        # ensure final cum_sizes[-1] == total_B
        if self.cum_sizes[-1] != total_B:
            raise ValueError(
                f"z_augmented shape mismatch: sum of batch sizes {self.cum_sizes[-1]} != {total_B}"
            )

    def __len__(self):
        return len(self.data_gen)

    def __getitem__(self, index):
        # get the slice of z_augmented for this batch
        start_idx = self.cum_sizes[index]
        end_idx = self.cum_sizes[index + 1]

        (b_data, b_mask, b_dyn), _ = self.data_gen[index]
        z_batch = self.z_augmented[
            start_idx:end_idx
        ]  # [b_len, window_size, latent_dim]

        # the decoder expects inputs => [latent_input, mask_input, dyn_input]
        return [z_batch, b_mask, b_dyn]

    def on_epoch_end(self):
        self.data_gen.on_epoch_end()


class Sampling(tf.keras.layers.Layer):
    def __init__(self, noise_scale_init=0.01, **kwargs):
        super(Sampling, self).__init__(**kwargs)
        # define noise_scale as a mutable variable
        self.noise_scale = tf.Variable(
            noise_scale_init, trainable=False, dtype=tf.float32, name="noise_scale"
        )

    def call(self, inputs):
        """
        Performs the reparameterization trick.
        """
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        seq_len = tf.shape(z_mean)[1]
        latent_dim = tf.shape(z_mean)[2]

        epsilon = tf.keras.backend.random_normal(shape=(batch, seq_len, latent_dim))

        # reparameterization trick with dynamic noise scaling
        return z_mean + tf.exp(0.5 * z_log_var) * self.noise_scale * epsilon


def masked_mse(y_true, y_pred, mask):
    """
    Compute Mean Squared Error only on the unmasked (non-padded) entries.
    """
    y_true_masked = y_true * mask
    y_pred_masked = y_pred * mask

    mse = tf.reduce_mean(tf.square(y_true_masked - y_pred_masked))
    return mse


class CVAE(keras.Model):
    def __init__(
        self,
        encoder: keras.Model,
        decoder: keras.Model,
        kl_weight_initial: float = None,
        forecasting: bool = True,
        **kwargs,
    ) -> None:
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        self.kl_weight = tf.Variable(
            kl_weight_initial, trainable=False, name="kl_weight"
        )

        self.forecasting = forecasting

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.prediction_loss_tracker = keras.metrics.Mean(name="prediction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs, training=False):
        batch_data, batch_mask, batch_dyn_features = inputs

        z_mean, z_log_var, z = self.encoder(
            [batch_data, batch_mask, batch_dyn_features]
        )

        pred_reconst, pred_forecast = self.decoder([z, batch_mask, batch_dyn_features])

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
        """
        data => ((batch_data, batch_mask, batch_dyn), (batch_recon_target, batch_pred_target))
        """
        (
            (batch_data, batch_mask, batch_dyn),
            (recon_target, recon_mask, pred_target, pred_mask),
        ) = data

        with tf.GradientTape() as tape:
            pred_reconst, z_mean, z_log_var, pred_forecast = self(
                (batch_data, batch_mask, batch_dyn), training=True
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
        (batch_data, batch_mask, batch_dyn), (
            recon_target,
            recon_mask,
            pred_target,
            pred_mask,
        ) = data

        pred_reconst, z_mean, z_log_var, pred_forecast = self(
            (batch_data, batch_mask, batch_dyn), training=False
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
                "kl_weight_initial": float(self.kl_weight.numpy()),
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
    noise_scale_init: float = 0.01,
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
    # 6 dynamic features => (window_size, 6)
    input_shape_dyn_features = (window_size, 6)

    enc = encoder(
        input_shape=input_shape_main,
        input_shape_dyn_features=input_shape_dyn_features,
        latent_dim=latent_dim,
        noise_scale_init=noise_scale_init,
        kernel_size=kernel_size,
        n_hidden=n_hidden,
    )

    dec = decoder(
        output_shape=input_shape_main,
        output_shape_dyn_features=input_shape_dyn_features,
        latent_dim=latent_dim,
        kernel_size=kernel_size,
        forecasting=forecasting,
        n_hidden=n_hidden,
    )

    return enc, dec


class MRHIBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        backcast_size,
        n_hidden,
        n_layers,
        kernel_size: Tuple[int] = (2, 2, 1),
        **kwargs,
    ):
        """
        Multi-Rate Hierarchical Interpolation Block.
        """
        super(MRHIBlock, self).__init__(**kwargs)
        self.backcast_size = backcast_size

        kernel_size = kernel_size[-1]

        self.pooling_layer = layers.MaxPooling1D(
            pool_size=kernel_size, strides=1, padding="same"
        )

        self.mlp_stack = tf.keras.Sequential(
            [layers.Dense(n_hidden, activation="relu") for _ in range(n_layers)]
        )
        self.backcast_layer = layers.TimeDistributed(
            layers.Dense(backcast_size[1], activation="linear")
        )

    def call(self, inputs):
        x = self.pooling_layer(inputs)
        x = self.mlp_stack(x)

        backcast = self.backcast_layer(x)

        return backcast


class UpsampleTimeLayer(tf.keras.layers.Layer):
    def __init__(self, target_len: int, method="bilinear"):
        super().__init__()
        self.target_len = target_len
        self.method = method

    def call(self, x):
        # x shape: [B, current_len, num_features]
        x = tf.expand_dims(x, axis=1)  # => [B, 1, current_len, num_features]
        x_upsampled = tf.image.resize(
            images=x, size=(1, self.target_len), method=self.method
        )  # => [B, 1, target_len, num_features]
        x_upsampled = tf.squeeze(
            x_upsampled, axis=1
        )  # => [B, target_len, num_features]
        return x_upsampled


class MRHIBlock_backcast_forecast(tf.keras.layers.Layer):
    """
    Block following with MLP projection and upsampling, with added regularization.
    """

    def __init__(
        self,
        backcast_size,
        n_knots: int,
        n_hidden: int = 256,
        n_layers: int = 2,
        kernel_size: Tuple[int] = (2, 2, 1),
        activation="relu",
        dropout_rate: float = 0.2,
        l2_lambda: float = 1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seq_len, self.num_features = backcast_size
        self.n_knots = n_knots
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.n_theta_backcast = self.seq_len * self.num_features
        self.n_theta_forecast = self.n_knots * self.num_features
        self.n_theta = self.n_theta_backcast + self.n_theta_forecast

        self.upsample_layer = UpsampleTimeLayer(
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

        mlp_layers.append(
            layers.Dense(self.n_theta, kernel_regularizer=regularizers.l2(l2_lambda))
        )
        self.mlp_stack = tf.keras.Sequential(mlp_layers)

    def call(self, x: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """
        Forward pass with pooling and MLP.
        """
        for pool_layer in self.pooling_layers:
            x = pool_layer(x)

        theta = self.mlp_stack(x)

        backcast_flat = theta[:, : self.n_theta_backcast]
        knots_flat = theta[:, self.n_theta_backcast :]

        backcast = tf.reshape(backcast_flat, [-1, self.seq_len, self.num_features])
        knots = tf.reshape(knots_flat, [-1, self.n_knots, self.num_features])
        forecast = self.upsample_layer(knots)

        return backcast, forecast


def encoder(
    input_shape: Tuple[int, int],
    input_shape_dyn_features: Tuple[int, int],
    latent_dim: int,
    n_blocks: int = 3,
    n_hidden: int = 256,
    n_layers: int = 2,
    kernel_size: Tuple[int] = (2, 2, 1),
    noise_scale_init: float = 0.01,
):
    main_input = layers.Input(shape=input_shape, name="main_input")
    mask_input = layers.Input(shape=input_shape, name="mask_input")
    dyn_features_input = layers.Input(
        shape=input_shape_dyn_features, name="dyn_features_input"
    )

    masked_input = layers.Multiply(name="masked_input")([main_input, mask_input])
    masked_input = layers.Concatenate()([dyn_features_input, masked_input])

    masked_input = layers.TimeDistributed(
        layers.Dense(input_shape[1], activation="relu")
    )(masked_input)

    backcast_total = masked_input
    final_output = 0

    for i in range(n_blocks):
        mrhi_block = MRHIBlock_backcast_forecast(
            backcast_size=input_shape,
            n_knots=4,
            n_hidden=n_hidden,
            n_layers=n_layers,
            kernel_size=kernel_size,
        )

        backcast, _ = mrhi_block(backcast_total)

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

    z = Sampling(name="sampling", noise_scale_init=noise_scale_init)(
        [z_mean, z_log_var]
    )

    return tf.keras.Model(
        inputs=[main_input, mask_input, dyn_features_input],
        outputs=[z_mean, z_log_var, z],
        name="encoder",
    )


def decoder(
    output_shape: Tuple[int, int],
    output_shape_dyn_features: Tuple[int, int],
    latent_dim: int,
    n_blocks: int = 3,
    n_hidden: int = 256,
    n_layers: int = 2,
    kernel_size: Tuple[int] = (2, 2, 1),
    forecasting: bool = True,
):
    time_steps = output_shape[0]
    num_features = output_shape[1]

    latent_input = layers.Input(shape=(time_steps, latent_dim), name="latent_input")
    dyn_features_input = layers.Input(
        shape=output_shape_dyn_features, name="dyn_features_input"
    )
    mask_input = layers.Input(shape=output_shape, name="mask_input")

    x = layers.TimeDistributed(layers.Dense(num_features, activation="relu"))(
        latent_input
    )

    x = layers.Concatenate()([dyn_features_input, x])

    x = layers.TimeDistributed(layers.Dense(num_features, activation="relu"))(x)

    residual = x
    final_forecast = layers.Lambda(lambda t: tf.zeros_like(t))(x)
    final_backcast = 0

    if forecasting:
        for i in range(n_blocks):
            mrhi_block = MRHIBlock_backcast_forecast(
                backcast_size=output_shape,
                n_knots=4,
                n_hidden=n_hidden,
                n_layers=n_layers,
                kernel_size=kernel_size,
            )

            backcast, forecast = mrhi_block(residual)

            residual = residual - backcast
            final_forecast = final_forecast + forecast
            final_backcast = final_backcast + backcast
    else:
        for i in range(n_blocks):
            mrhi_block = MRHIBlock(
                backcast_size=output_shape,
                n_hidden=n_hidden,
                n_layers=n_layers,
                kernel_size=kernel_size,
            )
            backcast = mrhi_block(residual)
            residual = residual - backcast
            final_backcast = final_backcast + backcast
        final_forecast = final_backcast

    final_backcast = layers.Multiply(name="masked_backcast_output")(
        [final_backcast, mask_input]
    )

    final_forecast = layers.Multiply(name="masked_forecast_output")(
        [final_forecast, mask_input]
    )
    return tf.keras.Model(
        inputs=[latent_input, mask_input, dyn_features_input],
        outputs=[final_backcast, final_forecast],
        name="decoder",
    )
