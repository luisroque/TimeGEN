from keras import layers
from keras import backend as K
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import get_custom_objects


def custom_relu_linear_saturation(x):
    """
    Custom activation:
    - 0 for x < 0 (ReLU behavior)
    - Linear (x) for 0 <= x <= 1
    - Saturates at 1 for x > 1
    """
    relu_part = tf.nn.relu(x)

    # linear between 0 and 1 and saturation at 1
    linear_part = tf.minimum(relu_part, 1.0)

    return linear_part


class TemporalizeGenerator(Sequence):
    def __init__(self, data, mask, window_size, stride=1, batch_size=8, shuffle=True):
        """
        A generator that reshuffles and re-temporalizes the dataset before each epoch.

        Args:
            data (pd.Dataframe): The original time-series data (2D: timesteps x features).
            mask (np.array or Tensor): Mask data (2D: timesteps x features) with 1 for valid and 0 for padded timesteps.
            window_size (int): The size of each temporal window.
            stride (int): The step size for creating temporal windows.
            batch_size (int): The batch size for training.
            shuffle (bool): Whether to shuffle the data and mask at the start of each epoch.
        """
        self.data = tf.convert_to_tensor(data, dtype=tf.float32)
        self.mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.temporalized_data = self.temporalize(self.data)
        self.temporalized_mask = self.temporalize(self.mask)
        self.indices = tf.range(len(self.temporalized_data))
        self.epoch = 0

    def __len__(self):
        """Number of batches per epoch, including the last incomplete batch."""
        total_batches = len(self.indices) // self.batch_size
        if len(self.indices) % self.batch_size != 0:
            total_batches += 1
        return total_batches

    def __getitem__(self, index):
        batch_indices = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        batch_data = tf.gather(self.temporalized_data, batch_indices)
        batch_mask = tf.gather(self.temporalized_mask, batch_indices)
        stacked_tensor = tf.stack([batch_data, batch_mask], axis=-1)

        return stacked_tensor

    def on_epoch_end(self):
        """Shuffle data and re-temporalize."""
        if self.shuffle:
            shuffled_indices = tf.random.shuffle(
                tf.range(tf.shape(self.temporalized_data)[0])
            )
            self.temporalized_data = tf.gather(self.temporalized_data, shuffled_indices)
            self.temporalized_mask = tf.gather(self.temporalized_mask, shuffled_indices)

            tf.print(f"SHUFFLING AND RE-TEMPORALIZING: Epoch {self.epoch}")
        self.epoch += 1

    def temporalize(self, data):
        """
        Create temporal windows from the input data.
        """
        num_windows = (tf.shape(data)[0] - self.window_size) // self.stride + 1
        indices = tf.range(num_windows) * self.stride
        windows = tf.map_fn(
            lambda i: data[i : i + self.window_size],
            indices,
            fn_output_signature=tf.TensorSpec(
                (self.window_size, data.shape[1]), tf.float32
            ),
        )
        return windows


class Sampling(tf.keras.layers.Layer):
    def __init__(self, noise_scale_init=0.01, **kwargs):
        """
        Initializes the Sampling layer.

        Args:
            noise_scale (float): Initial scaling factor for noise in the sampling process.
            **kwargs: Additional keyword arguments (e.g., `name`).
        """
        super(Sampling, self).__init__(**kwargs)
        # Define noise_scale as a mutable variable
        self.noise_scale = tf.Variable(
            noise_scale_init, trainable=False, dtype=tf.float32, name="noise_scale"
        )

    def call(self, inputs):
        """
        Performs the reparameterization trick.

        Args:
            inputs (tuple): A tuple containing (z_mean, z_log_var).

        Returns:
            Tensor: The sampled latent variable.
        """
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        seq_len = tf.shape(z_mean)[1]
        latent_dim = tf.shape(z_mean)[2]

        # Generate epsilon with shape (batch, seq_len, latent_dim)
        epsilon = tf.keras.backend.random_normal(shape=(batch, seq_len, latent_dim))

        # Reparameterization trick with dynamic noise scaling
        return z_mean + tf.exp(0.5 * z_log_var) * self.noise_scale * epsilon


class KLAnnealingAndNoiseScalingCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        model,
        kl_weight_initial,
        kl_weight_final,
        noise_scale_initial,
        noise_scale_final,
        annealing_epochs,
        annealing,
    ):
        super().__init__()
        self.model = model
        self.kl_weight_initial = kl_weight_initial
        self.kl_weight_final = kl_weight_final
        self.noise_scale_initial = noise_scale_initial
        self.noise_scale_final = noise_scale_final
        self.annealing_epochs = annealing_epochs
        self.annealing = annealing

    def compute_progress(self, epoch):
        return min((epoch + 1) / self.annealing_epochs, 1.0)

    def on_epoch_begin(self, epoch, logs=None):
        if self.annealing:
            # Calculate progress and update variables
            progress = self.compute_progress(epoch)
            new_kl_weight = self.kl_weight_initial + progress * (
                self.kl_weight_final - self.kl_weight_initial
            )
            new_noise_scale = self.noise_scale_initial + progress * (
                self.noise_scale_final - self.noise_scale_initial
            )

            # Update the model's tf.Variable instances
            self.model.kl_weight.assign(new_kl_weight)
            sampling_layer = self.model.encoder.get_layer(name="sampling")
            if sampling_layer:
                sampling_layer.noise_scale.assign(new_noise_scale)

            tf.print(
                f"UPDATING OPT VARIABLES: Epoch {epoch + 1}: KL weight = {new_kl_weight:.4f}, Noise scale = {new_noise_scale:.4f}"
            )

    def on_epoch_end(self, epoch, logs=None):
        """Print KL weight and noise scale at the end of the epoch."""
        sampling_layer = self.model.encoder.get_layer(name="sampling")
        current_noise_scale = (
            sampling_layer.noise_scale.numpy() if sampling_layer else "N/A"
        )
        current_kl_weight = self.model.kl_weight.numpy()
        print(
            f"OPT VARIABLES at Epoch {epoch + 1} end: KL weight = {current_kl_weight:.4f}, Noise scale = {current_noise_scale:.4f}"
        )


class CVAE(keras.Model):
    def __init__(
        self,
        encoder: keras.Model,
        decoder: keras.Model,
        kl_weight_initial: int = None,
        **kwargs,
    ) -> None:
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        self.kl_weight = tf.Variable(
            kl_weight_initial, trainable=False, name="kl_weight"
        )

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs, training=None, mask=None):
        batch_data = inputs[..., 0]
        batch_mask = inputs[..., 1]

        z_mean, z_log_var, z = self.encoder([batch_data, batch_mask])

        pred = self.decoder([z, batch_mask])

        return pred, z_mean, z_log_var

    def compute_loss(self, inp_data, pred, z_mean, z_log_var):
        """
        Computes the total loss, including reconstruction, KL divergence, TCN, and SEM loss.

        Args:
            inp_data (tensor): Original input data.
            pred (tensor): Reconstructed data.
            z_mean (tensor): Mean of the latent variable.
            z_log_var (tensor): Log variance of the latent variable.

        Returns:
            tuple: total_loss, reconstruction_loss, kl_loss, tcn_loss, sem_loss.
        """
        reconstruction_loss = tf.keras.losses.MeanAbsoluteError()(inp_data, pred)
        kl_loss = -0.5 * K.mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        total_loss = reconstruction_loss + self.kl_weight * kl_loss

        return total_loss, reconstruction_loss, kl_loss

    def train_step(self, data):
        batch_data = data[..., 0]
        batch_mask = data[..., 1]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([batch_data, batch_mask])
            pred = self.decoder([z, batch_mask])

            total_loss, reconstruction_loss, kl_loss = self.compute_loss(
                batch_data, pred, z_mean, z_log_var
            )

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        batch_data = data[..., 0]
        batch_mask = data[..., 1]

        z_mean, z_log_var, z = self.encoder([batch_data, batch_mask])
        pred = self.decoder([z, batch_mask])

        total_loss, reconstruction_loss, kl_loss = self.compute_loss(
            batch_data, pred, z_mean, z_log_var
        )

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def get_config(self):
        config = super(CVAE, self).get_config()
        config.update(
            {
                "encoder": self.encoder.get_config(),
                "decoder": self.decoder.get_config(),
                "kl_weight_initial": self.kl_weight.numpy(),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        encoder_config = config.pop("encoder")
        decoder_config = config.pop("decoder")

        encoder = keras.Model.from_config(encoder_config)
        decoder = keras.Model.from_config(decoder_config)

        return cls(encoder=encoder, decoder=decoder, **config)


def get_CVAE(
    window_size: int,
    n_series: int,
    latent_dim: int,
    bi_rnn: bool = True,
    noise_scale_init: float = 0.01,
    n_blocks: int = 3,
    n_hidden: int = 64,
    n_layers: int = 2,
    kernel_size: int = 2,
    strides: int = 1,
    pooling_mode: str = "max",
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """
    Constructs and returns the encoder and decoder models for the CVAE.

    Args:
        window_size (int): The size of the input time series window.
        n_series (int): Number of series in the input.
        latent_dim (int): Dimensionality of the latent space.
        bi_rnn (bool): Whether to use bidirectional RNNs.
        noise_scale_init (float): Initial noise scale for the encoder.
        n_blocks (int): Number of blocks in the encoder and decoder.
        n_hidden (int): Number of hidden units in the blocks.
        n_layers (int): Number of layers in each block.
        kernel_size (int): Kernel size for convolutional layers.
        strides (int): Stride for convolutional layers.
        pooling_mode (str): Pooling mode ("max" or "average").

    Returns:
        tuple[tf.keras.Model, tf.keras.Model]: The encoder and decoder models.
    """
    input_shape = (window_size, n_series)

    enc = encoder(
        input_shape=input_shape,
        latent_dim=latent_dim,
        bi_rnn=bi_rnn,
        noise_scale_init=noise_scale_init,
        n_blocks=n_blocks,
        n_hidden=n_hidden,
        n_layers=n_layers,
        kernel_size=kernel_size,
        pooling_mode=pooling_mode,
    )

    dec = decoder(
        output_shape=input_shape,
        latent_dim=latent_dim,
        bi_rnn=bi_rnn,
        n_blocks=n_blocks,
        n_hidden=n_hidden,
        n_layers=n_layers,
        kernel_size=kernel_size,
        pooling_mode=pooling_mode,
    )

    return enc, dec


class NHITSBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        backcast_size,
        n_hidden,
        n_layers,
        pooling_mode="max",
        kernel_size=3,
        **kwargs,
    ):
        """
        N-HiTS Block for time-series decomposition.

        Args:
            backcast_size (tuple): Size of the backcast.
            forecast_size (int): Size of the forecast.
            n_hidden (int): Number of hidden units in the MLP layers.
            n_layers (int): Number of MLP layers.
            pooling_mode (str): Pooling mode, either 'max' or 'average'.
            kernel_size (int): Pooling kernel size.
        """
        super(NHITSBlock, self).__init__(**kwargs)
        self.backcast_size = backcast_size

        if pooling_mode == "max":
            self.pooling_layer = layers.MaxPooling1D(
                pool_size=kernel_size, strides=1, padding="same"
            )
        else:
            self.pooling_layer = layers.AveragePooling1D(
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


def encoder(
    input_shape,
    latent_dim,
    n_blocks=3,
    n_hidden=64,
    n_layers=2,
    kernel_size=2,
    pooling_mode="max",
    bi_rnn=True,
    noise_scale_init=0.01,
):
    main_input = layers.Input(shape=input_shape, name="main_input")
    mask_input = layers.Input(shape=input_shape, name="mask_input")

    # apply mask to ignore padded values
    masked_input = layers.Multiply(name="masked_input")([main_input, mask_input])

    backcast_total = masked_input
    final_output = 0

    for i in range(n_blocks):
        nhits_block = NHITSBlock(
            backcast_size=input_shape,
            n_hidden=n_hidden,
            n_layers=n_layers,
            pooling_mode=pooling_mode,
            kernel_size=kernel_size,
        )

        backcast = nhits_block(backcast_total)

        backcast_total = backcast_total - backcast

        final_output += backcast

    if bi_rnn:
        backcast = layers.Bidirectional(
            layers.LSTM(input_shape[1], return_sequences=True)
        )(backcast_total)

        backcast = layers.TimeDistributed(layers.Dense(input_shape[1]))(backcast)

        final_output += backcast

    final_output = layers.TimeDistributed(layers.Dense(latent_dim * 2))(final_output)

    z_mean = final_output[:, :, latent_dim:]
    z_log_var = final_output[:, :, :latent_dim]

    z = Sampling(name="sampling", noise_scale_init=noise_scale_init)(
        [z_mean, z_log_var]
    )

    return tf.keras.Model(
        inputs=[main_input, mask_input], outputs=[z_mean, z_log_var, z], name="encoder"
    )


def decoder(
    output_shape,
    latent_dim,
    n_blocks=3,
    n_hidden=64,
    n_layers=2,
    kernel_size=2,
    pooling_mode="max",
    bi_rnn=True,
):

    latent_input = layers.Input(
        shape=(output_shape[0], latent_dim),
        name="latent_input",
    )

    mask_input = layers.Input(shape=output_shape, name="mask_input")

    x = layers.TimeDistributed(layers.Dense(output_shape[1]))(latent_input)

    backcast_total = x
    final_output = 0

    for i in range(n_blocks):
        nhits_block = NHITSBlock(
            backcast_size=output_shape,
            n_hidden=n_hidden,
            n_layers=n_layers,
            pooling_mode=pooling_mode,
            kernel_size=kernel_size,
        )

        backcast = nhits_block(backcast_total)

        backcast_total = backcast_total - backcast

        final_output += backcast

    if bi_rnn:
        backcast = layers.Bidirectional(
            layers.LSTM(output_shape[1], return_sequences=True)
        )(backcast_total)

        backcast = layers.TimeDistributed(layers.Dense(output_shape[1]))(backcast)

        final_output += backcast

    final_output = layers.TimeDistributed(
        layers.Dense(output_shape[1], activation="linear")
    )(final_output)

    # apply mask to the output
    final_output = layers.Multiply(name="masked_output")([final_output, mask_input])

    return tf.keras.Model(
        inputs=[latent_input, mask_input], outputs=[final_output], name="decoder"
    )
