import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score


def train_test_divide(ori_data, generated_data, test_ratio=0.2):
    """Divide the data into train and test sets based on a sequential split."""
    n_samples = len(ori_data)
    train_size = int(n_samples * (1 - test_ratio))

    # Use the first part for training and the last part for testing
    train_idx = slice(0, train_size)
    test_idx = slice(train_size, n_samples)

    return (
        ori_data[train_idx],
        generated_data[train_idx],
        ori_data[test_idx],
        generated_data[test_idx],
    )


def batch_generator(data, time, batch_size):
    """Generate a random batch of data."""
    idx = np.random.permutation(len(data))[:batch_size]
    return data[idx], time[idx]


def extract_time(data):
    """Extract the time length of the data."""
    time = np.array([len(seq) for seq in data])
    max_seq_len = max(time)
    return time, max_seq_len


def temporalize(data, window_size):
    """Convert 2D data into 3D format using a sliding window."""
    n_samples = data.shape[0]
    sequences = []

    for i in range(n_samples - window_size + 1):
        # Create overlapping windows
        window = data[i : i + window_size, :]
        sequences.append(window)

    return np.array(sequences)


def compute_discriminative_score(
    original_data, synthetic_data, window_size=24, iterations=500, batch_size=64
):
    """Calculate the discriminative score using a post-hoc RNN classifier."""
    train_x, train_x_hat, test_x, test_x_hat = train_test_divide(
        original_data,
        synthetic_data,
    )

    train_x_temporal = temporalize(train_x, window_size)
    train_x_hat_temporal = temporalize(train_x_hat, window_size)
    test_x_temporal = temporalize(test_x, window_size)
    test_x_hat_temporal = temporalize(test_x_hat, window_size)

    max_seq_len = train_x_temporal.shape[1]

    hidden_dim = int(train_x_temporal.shape[-1] / 2)

    def build_discriminator(hidden_dim, num_layers=2):
        """Build the discriminator model using GRU layers."""
        inputs = tf.keras.Input(shape=(max_seq_len, train_x_temporal.shape[-1]))
        x = inputs
        for _ in range(num_layers):
            x = tf.keras.layers.GRU(
                hidden_dim, activation="relu", return_sequences=True
            )(x)
        x = tf.keras.layers.GRU(hidden_dim, activation="relu", return_sequences=False)(
            x
        )
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        return tf.keras.Model(inputs, outputs)

    discriminator = build_discriminator(hidden_dim)
    discriminator.compile(optimizer="adam", loss="binary_crossentropy")

    real_labels = np.ones((train_x_temporal.shape[0], 1))
    fake_labels = np.zeros((train_x_hat_temporal.shape[0], 1))

    for i in range(iterations):
        idx_real = np.random.randint(0, train_x_temporal.shape[0], batch_size)
        idx_fake = np.random.randint(0, train_x_hat_temporal.shape[0], batch_size)
        real_batch = train_x_temporal[idx_real]
        fake_batch = train_x_hat_temporal[idx_fake]

        discriminator.train_on_batch(real_batch, real_labels[idx_real])
        discriminator.train_on_batch(fake_batch, fake_labels[idx_fake])

    y_pred_real = discriminator.predict(test_x_temporal)
    y_pred_fake = discriminator.predict(test_x_hat_temporal)

    y_pred = np.concatenate([y_pred_real, y_pred_fake], axis=0)
    y_true = np.concatenate(
        [np.ones_like(y_pred_real), np.zeros_like(y_pred_fake)], axis=0
    )

    accuracy = accuracy_score(y_true, (y_pred > 0.5).astype(int))
    discriminative_score = np.abs(accuracy - 0.5)

    return discriminative_score


def evaluate_discriminative_scores(
    X_orig_scaled,
    main_model_data_scaled,
    benchmark_data_dict,
    compute_discriminative_score,
    num_runs=10,
    num_samples=5,
    plot_first_run=True,
):
    """
    Evaluate discriminative scores for different synthetic data baselines.

    Parameters:
    - X_orig_scaled: np.ndarray, scaled original time series data.
    - main_model_data_scaled: np.ndarray, scaled generated data from the main model.
    - benchmark_data_dict: dict, dictionary where keys are benchmark names, and values are scaled benchmark data.
    - compute_discriminative_score: function, computes discriminative score.
    - num_runs: int, number of evaluation runs (default: 10).
    - num_samples: int, number of samples to select in each run (default: 5).
    - plot_first_run: bool, whether to plot the first run (default: True).

    Returns:
    - dict: average and std deviation of discriminative scores for main model and each benchmark.
    """

    discriminative_scores = {"Model": []}
    for name in benchmark_data_dict.keys():
        discriminative_scores[name] = []

    for run in range(num_runs):
        print(f"Run discriminative score {run+1}/{num_runs}")

        # Randomly select time series samples
        idx = np.random.choice(X_orig_scaled.shape[-1], num_samples, replace=False)
        X_orig_selected = X_orig_scaled[:, idx]
        main_model_selected = main_model_data_scaled[:, idx]

        # Compute discriminative score for the main model
        score_main_model = compute_discriminative_score(
            X_orig_selected, main_model_selected
        )
        discriminative_scores["Model"].append(score_main_model)

        # Compute discriminative scores for each benchmark
        for name, benchmark_data in benchmark_data_dict.items():
            benchmark_selected = benchmark_data[:, idx]
            score_benchmark = compute_discriminative_score(
                X_orig_selected, benchmark_selected
            )
            discriminative_scores[name].append(score_benchmark)

        # Plot the first run if specified
        if run == 0 and plot_first_run:
            num_benchmarks = len(benchmark_data_dict)
            fig, axes = plt.subplots(num_samples, num_benchmarks + 1, figsize=(20, 15))

            for i in range(num_samples):
                # Plot the main model
                axes[i, 0].plot(
                    X_orig_selected[:, i], label="Original (Scaled)", color="blue"
                )
                axes[i, 0].plot(
                    main_model_selected[:, i],
                    label="Main Model",
                    color="red",
                    linestyle="dashed",
                )
                axes[i, 0].set_title(f"Main Model - Time Series {i+1}")
                axes[i, 0].legend()

                # Plot each benchmark
                for j, (name, benchmark_data) in enumerate(
                    benchmark_data_dict.items(), start=1
                ):
                    benchmark_selected = benchmark_data[:, idx]
                    axes[i, j].plot(
                        X_orig_selected[:, i], label="Original (Scaled)", color="blue"
                    )
                    axes[i, j].plot(
                        benchmark_selected[:, i],
                        label=f"{name} Benchmark",
                        linestyle="dashed",
                    )
                    axes[i, j].set_title(f"{name} - Time Series {i+1}")
                    axes[i, j].legend()

            plt.tight_layout()
            plt.show()

    results = {}
    for name, scores in discriminative_scores.items():
        results[name] = {"Average": np.mean(scores), "StdDev": np.std(scores)}

    for name, result in results.items():
        print(
            f"Average Discriminative Score ({name}): {result['Average']}, Std Dev: {result['StdDev']}"
        )

    return results
