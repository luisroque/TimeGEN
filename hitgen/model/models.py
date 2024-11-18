from keras import layers
from keras.layers import (
    Bidirectional,
    LSTM,
    MultiHeadAttention,
    Dense,
    GlobalAveragePooling1D,
    BatchNormalization,
    Activation,
)
import tensorflow as tf
from keras import backend as K
from .helper import Sampling
from tensorflow import keras


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


class RNNEmbeddingModel(tf.keras.Model):
    def __init__(self, lstm_units=64, reduced_feature_dim=32, **kwargs):
        super(RNNEmbeddingModel, self).__init__(**kwargs)

        self.feature_projection = Dense(reduced_feature_dim, activation="relu")
        self.batch_norm = BatchNormalization()
        self.activation = Activation("relu")

        self.rnn = Bidirectional(
            LSTM(lstm_units, return_sequences=True, name="lstm_rnn_emb"),
            name="bi_rnn_emb",
        )

        self.global_pooling = GlobalAveragePooling1D()

    def call(self, inputs):

        x = self.feature_projection(inputs)
        x = self.batch_norm(x)
        x = self.activation(x)

        x = self.rnn(x)
        x = self.global_pooling(x)

        return x


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
    window_size: int,
    n_series: int,
    latent_dim: int,
    filters=32,
    num_blocks=3,
    kernel_size=3,
    strides=1,
    dynamic_units=16,
) -> tuple[tf.keras.Model, tf.keras.Model]:

    input_shape = (window_size, n_series)
    output_shape = (window_size, n_series)

    enc = encoder(
        input_shape=input_shape,
        latent_dim=latent_dim,
        # num_blocks=num_blocks,
        # filters=filters,
        # kernel_size=kernel_size,
        # strides=strides,
        # dynamic_units=dynamic_units,
    )
    dec = decoder(
        output_shape=output_shape,
        latent_dim=latent_dim,
        # # num_blocks=num_blocks,
        # filters=filters,
        # kernel_size=kernel_size,
    )
    return enc, dec


class IntraWindowAttention(layers.Layer):
    """
    Intra-week attention layer to learn dependencies across time points within a window.
    """

    def __init__(self, reduced_features, latent_dim):
        super(IntraWindowAttention, self).__init__()
        self.query_dense = layers.Dense(latent_dim)
        self.key_dense = layers.Dense(latent_dim)
        self.value_dense = layers.Dense(latent_dim)

    def call(self, x):
        # x shape: (batch_size, window_size, reduced_features)

        # Create query, key, and value tensors for attention
        query = self.query_dense(x)  # (batch_size, window_size, latent_dim)
        key = self.key_dense(x)  # (batch_size, window_size, latent_dim)
        value = self.value_dense(x)  # (batch_size, window_size, latent_dim)

        # Compute attention scores across the time dimension (within the window)
        scores = tf.matmul(
            query, key, transpose_b=True
        )  # (batch_size, window_size, window_size)
        attention_weights = tf.nn.softmax(scores, axis=-1)

        # Apply attention weights to the values
        attended_output = tf.matmul(
            attention_weights, value
        )  # (batch_size, window_size, latent_dim)

        return attended_output  # (batch_size, window_size, latent_dim)


class CorruptionLayer(layers.Layer):
    def __init__(self, noise_stddev=0.1):
        super(CorruptionLayer, self).__init__()
        self.noise_stddev = noise_stddev

    def call(self, x):
        noise = tf.random.normal(shape=tf.shape(x), stddev=self.noise_stddev)
        return x + noise


class TemporalCrossAttention(layers.Layer):
    def __init__(self, num_heads, key_dim):
        super(TemporalCrossAttention, self).__init__()
        self.cross_attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def call(self, query, key, value):
        attention_output, _ = self.cross_attention(
            query=query, key=key, value=value, return_attention_scores=True
        )
        return attention_output


def encoder(input_shape, latent_dim, latent_dim_expansion=16):
    main_input = layers.Input(shape=input_shape, name="main_input")
    x = main_input

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    x = layers.TimeDistributed(layers.Dense(latent_dim * latent_dim_expansion))(x)

    z_mean = x[:, :, latent_dim * latent_dim_expansion // 2 :]
    z_log_var = x[:, :, : latent_dim * latent_dim_expansion // 2]

    z = Sampling(name="sampling")([z_mean, z_log_var])

    return tf.keras.Model(
        inputs=[main_input], outputs=[z_mean, z_log_var, z], name="encoder"
    )


def decoder(output_shape, latent_dim, latent_dim_expansion=16):
    latent_input = layers.Input(
        shape=(output_shape[0], latent_dim_expansion * latent_dim // 2),
        name="latent_input",
    )

    x = layers.Reshape((output_shape[0], -1))(latent_input)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    x = layers.TimeDistributed(layers.Dense(output_shape[1], activation="linear"))(x)

    return tf.keras.Model(inputs=[latent_input], outputs=[x], name="decoder")
