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
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        seq_len = tf.shape(z_mean)[1]
        latent_dim = tf.shape(z_mean)[2]

        # generate epsilon with shape (batch, seq_len, latent_dim)
        epsilon = tf.keras.backend.random_normal(shape=(batch, seq_len, latent_dim))

        # reparameterization trick

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


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
        input_shape,
        kl_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight

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


def encoder(input_shape, latent_dim, window_reduction=12):
    lstm_units = window_reduction * 4
    main_input = layers.Input(shape=input_shape, name="main_input")
    x = main_input
    x_residual = main_input

    x = layers.Bidirectional(layers.LSTM(lstm_units * 2, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)

    x = layers.TimeDistributed(layers.Dense(latent_dim))(x)

    x = layers.Add()([x, x_residual])

    x = layers.Reshape((window_reduction, -1))(x)
    x = layers.Dense(latent_dim * 2)(x)

    z_mean = x[:, :, latent_dim:]
    z_log_var = x[:, :, :latent_dim]

    z = z_mean

    return tf.keras.Model(
        inputs=[main_input], outputs=[z_mean, z_log_var, z], name="encoder"
    )


def decoder(output_shape, latent_dim, window_reduction=12):
    lstm_units = window_reduction * 4
    latent_input = layers.Input(
        shape=(window_reduction, latent_dim), name="latent_input"
    )

    x = layers.TimeDistributed(layers.Dense(output_shape[0], activation="relu"))(
        latent_input
    )

    x = layers.Reshape((output_shape[0], -1))(x)
    rnn_cell = layers.LSTMCell(lstm_units)
    x = layers.RNN(rnn_cell, return_sequences=True)(x)

    x = layers.TimeDistributed(layers.Dense(output_shape[1], activation="linear"))(x)

    return tf.keras.Model(inputs=[latent_input], outputs=[x], name="decoder")
