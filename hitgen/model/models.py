from keras import layers
from keras import backend as K
from .helper import Sampling
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
    def __init__(self, data, window_size, stride=1, batch_size=8, shuffle=True):
        """
        A generator that reshuffles and re-temporalizes the dataset before each epoch.

        Args:
            data (tf.Tensor): The original time-series data (2D tensor: timesteps x features).
            window_size (int): The size of each temporal window.
            stride (int): The step size for creating temporal windows.
            batch_size (int): The batch size for training.
            shuffle (bool): Whether to shuffle the data at the start of each epoch.
        """
        self.data = tf.convert_to_tensor(data, dtype=tf.float32)
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.temporalized_data = None
        self.indices = None
        self.epoch = 0
        self.on_epoch_end()

    def __len__(self):
        """Number of batches per epoch, including the last incomplete batch."""
        total_batches = len(self.indices) // self.batch_size
        if len(self.indices) % self.batch_size != 0:
            total_batches += 1
        return total_batches

    def __getitem__(self, index):
        """Generate one batch of data."""
        batch_indices = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        batch = tf.gather(self.temporalized_data, batch_indices)
        return batch

    def on_epoch_end(self):
        """Shuffle data and re-temporalize."""
        if self.shuffle:
            shuffled_data = tf.random.shuffle(self.data)
        else:
            shuffled_data = self.data

        self.temporalized_data = self.temporalize(shuffled_data)

        # update indices for new temporalized data
        self.indices = tf.range(len(self.temporalized_data))

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
    def __init__(self, noise_scale=0.01, **kwargs):
        """
        Initializes the Sampling layer.

        Args:
            noise_scale (float): Initial scaling factor for noise in the sampling process.
            **kwargs: Additional keyword arguments (e.g., `name`).
        """
        super(Sampling, self).__init__(**kwargs)
        # Define noise_scale as a mutable variable
        self.noise_scale = tf.Variable(
            noise_scale, trainable=False, dtype=tf.float32, name="noise_scale"
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
    ):
        super().__init__()
        self.model = model
        self.kl_weight_initial = kl_weight_initial
        self.kl_weight_final = kl_weight_final
        self.noise_scale_initial = noise_scale_initial
        self.noise_scale_final = noise_scale_final
        self.annealing_epochs = annealing_epochs

    def compute_progress(self, epoch):
        return min((epoch + 1) / self.annealing_epochs, 1.0)

    def on_epoch_begin(self, epoch, logs=None):
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
        kl_weight_initial: float = 0.1,
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
        inp_data = inputs
        z_mean, z_log_var, z = self.encoder(inp_data)

        pred = self.decoder(z)

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
        inp_data = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(inp_data)
            pred = self.decoder(z)

            total_loss, reconstruction_loss, kl_loss = self.compute_loss(
                inp_data, pred, z_mean, z_log_var
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
        inp_data = data

        z_mean, z_log_var, z = self.encoder(inp_data)
        pred = self.decoder(z)

        total_loss, reconstruction_loss, kl_loss = self.compute_loss(
            inp_data, pred, z_mean, z_log_var
        )

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


def get_CVAE(
    window_size: int, n_series: int, latent_dim: int, last_activation: str, bi_rnn: bool
) -> tuple[tf.keras.Model, tf.keras.Model]:

    input_shape = (window_size, n_series)
    output_shape = (window_size, n_series)

    enc = encoder(
        input_shape=input_shape,
        latent_dim=latent_dim,
        bi_rnn=bi_rnn,
    )
    dec = decoder(
        output_shape=output_shape,
        latent_dim=latent_dim,
        bi_rnn=bi_rnn,
        last_activation=last_activation,
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
        strides=2,
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
                pool_size=kernel_size, strides=strides, padding="same"
            )
        else:
            self.pooling_layer = layers.AveragePooling1D(
                pool_size=kernel_size, strides=strides, padding="same"
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
    latent_dim_expansion=2,
    n_blocks=3,
    n_hidden=64,
    n_layers=2,
    kernel_size=2,
    pooling_mode="max",
    strides=1,
    bi_rnn=True,
):
    main_input = layers.Input(shape=input_shape, name="main_input")

    backcast_total = main_input
    final_output = 0

    for i in range(n_blocks):
        nhits_block = NHITSBlock(
            backcast_size=input_shape,
            n_hidden=n_hidden,
            n_layers=n_layers,
            pooling_mode=pooling_mode,
            kernel_size=kernel_size,
            strides=strides,
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

    final_output = layers.TimeDistributed(
        layers.Dense(latent_dim * latent_dim_expansion)
    )(final_output)

    z_mean = final_output[:, :, latent_dim * latent_dim_expansion // 2 :]
    z_log_var = final_output[:, :, : latent_dim * latent_dim_expansion // 2]

    z = Sampling(name="sampling")([z_mean, z_log_var])

    return tf.keras.Model(
        inputs=[main_input], outputs=[z_mean, z_log_var, z], name="encoder"
    )


def decoder(
    output_shape,
    latent_dim,
    latent_dim_expansion=2,
    n_blocks=3,
    n_hidden=64,
    n_layers=2,
    kernel_size=2,
    strides=1,
    pooling_mode="max",
    bi_rnn=True,
    last_activation="relu",
):
    """
    Decoder with N-HiTS stacking for reconstruction.

    Args:
        output_shape (tuple): Shape of the output (time steps, series size).
        latent_dim (int): Latent dimension size.
        latent_dim_expansion (int): Expansion factor for latent dimension.
        n_blocks (int): Number of N-HiTS blocks in the stack.
        n_hidden (int): Number of hidden units in MLP layers.
        n_layers (int): Number of MLP layers in each block.
        kernel_size (int): Pooling kernel size.
        pooling_mode (str): Pooling mode ('max' or 'average').

    Returns:
        tf.keras.Model: The decoder model.
    """
    get_custom_objects().update(
        {"custom_relu_linear_saturation": custom_relu_linear_saturation}
    )

    if not callable(last_activation):
        try:
            last_activation = tf.keras.activations.get(last_activation)
        except ValueError:
            raise ValueError(f"Unknown activation function: {last_activation}")

    latent_input = layers.Input(
        shape=(output_shape[0], latent_dim_expansion * latent_dim // 2),
        name="latent_input",
    )

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
            strides=strides,
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
        layers.Dense(output_shape[1], activation=last_activation)
    )(final_output)

    return tf.keras.Model(inputs=[latent_input], outputs=[final_output], name="decoder")
