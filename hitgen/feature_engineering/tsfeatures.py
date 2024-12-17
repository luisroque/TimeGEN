import tensorflow as tf


def statistics_tf(x: tf.Tensor) -> tf.Tensor:
    """
    Computes basic differentiable statistics of x.
    """
    x = tf.cast(x, tf.float32)

    total_sum = tf.reduce_sum(x)
    mean = tf.reduce_mean(x)
    # population variance (differentiable)
    variance = tf.reduce_mean((x - mean) ** 2)
    _max = tf.reduce_max(x)
    _min = tf.reduce_min(x)

    return tf.stack([total_sum, mean, variance, _max, _min], axis=0)


def cosine_similarity(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """
    Computes cosine similarity between two tensors a and b.
    """
    dot_product = tf.reduce_sum(a * b)
    norm_a = tf.norm(a)
    norm_b = tf.norm(b)
    return dot_product / (norm_a * norm_b + 1e-9)


def compute_feature_loss(inp_data: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the feature loss between input data and predictions
    using cosine similarity.
    """
    features_original = statistics_tf(inp_data)
    features_synth = statistics_tf(pred)
    cosine_sim = cosine_similarity(features_original, features_synth)
    return 1.0 - cosine_sim
