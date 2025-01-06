import pandas as pd
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from statsmodels.tsa.seasonal import seasonal_decompose
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from hitgen.model.models import (
    TemporalizeGenerator,
)
from hitgen.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE
from hitgen.visualization.model_visualization import (
    plot_loss,
    plot_generated_vs_original,
)
from hitgen.metrics.discriminative_score import (
    compute_discriminative_score,
)
from hitgen.feature_engineering.feature_transformations import detemporalize
from hitgen.benchmarks.timegan import (
    train_timegan_model,
    generate_synthetic_samples,
    train_timegan_with_best_params,
    hyper_tune_timegan,
)
from hitgen.feature_engineering.tsfeatures import compute_feature_loss


DATASET_CONFIGS = {
    "Tourism": {
        "Monthly": {
            "latent_dim": 56,
            "patience": 95,
            "kl_weight": 0.7187173280781497,
            "n_blocks": 3,
            "n_hidden": 16,
            "n_layers": 3,
            "kernel_size": 2,
            "pooling_mode": "average",
            "batch_size": 8,
            "epochs": 801,
            "learning_rate": 6.292263002881687e-05,
            "noise_scale_init": 0.09374107362343356,
        }
    },
    # add other datasets
}

if __name__ == "__main__":
    DATASET = "Tourism"
    DATASET_GROUP = "Monthly"

    FREQ = "M"
    TOP = None
    WINDOW_SIZE = 24
    VAL_STEPS = 0
    LATENT_DIM = DATASET_CONFIGS[DATASET][DATASET_GROUP]["latent_dim"]
    EPOCHS = DATASET_CONFIGS[DATASET][DATASET_GROUP]["epochs"]
    BATCH_SIZE = DATASET_CONFIGS[DATASET][DATASET_GROUP]["batch_size"]
    STRIDE_TEMPORALIZE = 1
    SHUFFLE = True
    BI_RNN = False
    ANNEALING = False
    KL_WEIGHT_INIT = DATASET_CONFIGS[DATASET][DATASET_GROUP]["kl_weight"]
    NOISE_SAMPLE_INIT = DATASET_CONFIGS[DATASET][DATASET_GROUP]["noise_scale_init"]
    N_BLOCKS = DATASET_CONFIGS[DATASET][DATASET_GROUP]["n_blocks"]
    N_HIDDEN = DATASET_CONFIGS[DATASET][DATASET_GROUP]["n_hidden"]
    N_LAYERS = DATASET_CONFIGS[DATASET][DATASET_GROUP]["n_layers"]
    KERNEL_SIZE = DATASET_CONFIGS[DATASET][DATASET_GROUP]["kernel_size"]
    POOLING_MODE = DATASET_CONFIGS[DATASET][DATASET_GROUP]["pooling_mode"]
    LEARNING_RATE = DATASET_CONFIGS[DATASET][DATASET_GROUP]["learning_rate"]
    PATIENCE = DATASET_CONFIGS[DATASET][DATASET_GROUP]["patience"]

    SYNTHETIC_FILE_PATH = (
        f"assets/model_weights/{DATASET}_{DATASET_GROUP}_synthetic_timegan_long.pkl"
    )

    create_dataset_vae = CreateTransformedVersionsCVAE(
        dataset_name=DATASET,
        dataset_group=DATASET_GROUP,
        freq=FREQ,
        top=TOP,
        window_size=WINDOW_SIZE,
        stride_temporalize=STRIDE_TEMPORALIZE,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        bi_rnn=BI_RNN,
        annealing=ANNEALING,
        noise_scale_init=NOISE_SAMPLE_INIT,
        kl_weight_init=KL_WEIGHT_INIT,
    )

    # hypertuning
    create_dataset_vae.hyper_tune_and_train()

    # fit
    model, history, _ = create_dataset_vae.fit(
        latent_dim=LATENT_DIM,
        epochs=EPOCHS,
        patience=PATIENCE,
        learning_rate=LEARNING_RATE,
    )
    plot_loss(history)

    (
        _,
        _,
        original_data,
        train_data_long,
        test_data_long,
        original_data_long,
        _,
        _,
        original_mask,
    ) = create_dataset_vae._feature_engineering()

    data_mask_temporalized = TemporalizeGenerator(
        original_data,
        original_mask,
        window_size=WINDOW_SIZE,
        stride=create_dataset_vae.stride_temporalize,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
    )

    _, synth_hitgen_test_long, _ = create_dataset_vae.predict(
        model,
        samples=data_mask_temporalized.indices.shape[0],
        window_size=WINDOW_SIZE,
        latent_dim=LATENT_DIM,
    )

    # generate more samples into the future to check on overfitting
    # new_latent_samples = np.random.normal(
    #     size=(
    #         data_mask_temporalized.indices.shape[0] + WINDOW_SIZE,
    #         WINDOW_SIZE,
    #         LATENT_DIM,
    #     )
    # )
    # n_series = data.shape[1]
    # future_mask = tf.ones((WINDOW_SIZE, n_series), dtype=tf.float32)
    #
    # mask = tf.concat([create_dataset_vae.mask, future_mask], axis=0)
    # mask_temporalized = create_dataset_vae.temporalize(mask, WINDOW_SIZE)
    # generated_data = model.decoder.predict([new_latent_samples, mask_temporalized])
    #
    # synth_hitgen = detemporalize(generated_data)

    # plot_generated_vs_original(
    #     dec_pred_hat=generated_data,
    #     X_train_raw=X_orig,
    #     dataset_name=DATASET,
    #     dataset_group=DATASET_GROUP,
    #     n_series=8,
    # )

    import matplotlib.pyplot as plt

    unique_ids = synth_hitgen_test_long["unique_id"].unique()[:4]

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    for idx, unique_id in enumerate(unique_ids):
        ax = axes[idx]
        original_series = original_data_long[
            original_data_long["unique_id"] == unique_id
        ]
        synthetic_series = synth_hitgen_test_long[
            synth_hitgen_test_long["unique_id"] == unique_id
        ]

        ax.plot(
            original_series["ds"],
            original_series["y"],
            label="Original",
            linestyle="-",
        )
        ax.plot(
            synthetic_series["ds"],
            synthetic_series["y"],
            label="Synthetic",
            linestyle="--",
        )

        ax.set_title(f"Time Series for ID: {unique_id}")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid()

    plt.xlabel("Time Steps")
    plt.tight_layout()
    plt.show()

    # TimeGAN synthetic data generation

    def run_timegan(target_df, dataset, dataset_group, unique_id):

        hyperparameter_sets = [
            {
                "gan_args": ModelParameters(
                    batch_size=16,
                    lr=2e-4,
                    noise_dim=16,
                    layers_dim=32,
                    latent_dim=32,
                ),
            },
            {
                "gan_args": ModelParameters(
                    batch_size=32,
                    lr=1e-4,
                    noise_dim=8,
                    layers_dim=64,
                    latent_dim=64,
                ),
            },
            {
                "gan_args": ModelParameters(
                    batch_size=8,
                    lr=5e-4,
                    noise_dim=32,
                    layers_dim=16,
                    latent_dim=16,
                ),
            },
        ]

        for idx, params in enumerate(hyperparameter_sets):
            try:
                print(f"Trying hyperparameter set {idx + 1}")
                timegan = train_timegan_model(
                    target_df,
                    gan_args=params["gan_args"],
                    train_args=TrainParameters(
                        epochs=1000, sequence_length=WINDOW_SIZE, number_sequences=4
                    ),
                    model_path=f"assets/model_weights/timegan/timegan_{dataset}_{dataset_group}_{unique_id}.pkl",
                )
                print(f"Training successful with hyperparameter set {idx + 1}")
                break
            except Exception as e:
                print(f"Failed with hyperparameter set {idx + 1}: {e}")
        return timegan

    def train_and_generate_synthetic(
        unique_id, data, dataset, dataset_group, window_size
    ):
        print(f"Training TimeGAN for time series: {unique_id}")

        ts_data = data[data["unique_id"] == unique_id]

        # ts_data = data[data["unique_id"].isin(["m1", "m100", "m101", "m104"])]
        # ts_data["unique_id"] = "m1"
        # ts_data["ds"] = np.arange(ts_data.shape[0])

        target_df = ts_data.pivot(index="ds", columns="unique_id", values="y")
        target_df.columns.name = None
        target_df = target_df.reset_index(drop=True)

        # decompose to make it multivariate
        result = seasonal_decompose(target_df, model="additive", period=12)

        target_df["trend"] = result.trend
        target_df["seasonal"] = result.seasonal
        target_df["residual"] = result.resid

        scaler = MinMaxScaler()
        scaled_target_df = pd.DataFrame(
            scaler.fit_transform(target_df), columns=target_df.columns
        )

        scaled_target_df.fillna(0, inplace=True)

        os.makedirs("assets/model_weights/timegan/", exist_ok=True)

        timegan = run_timegan(scaled_target_df, dataset, dataset_group, unique_id)

        synth_scaled_data = generate_synthetic_samples(
            timegan, ts_data.shape[0] - WINDOW_SIZE + 1, detemporalize
        )

        synth_timegan_data = pd.DataFrame(
            scaler.inverse_transform(synth_scaled_data), columns=target_df.columns
        )

        synthetic_df = pd.DataFrame(synth_timegan_data, columns=[unique_id])

        plot_dir = "assets/plots/timegan/"
        os.makedirs(plot_dir, exist_ok=True)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(
            target_df[unique_id],
            label="Original",
            linestyle="-",
        )
        plt.plot(
            synthetic_df[unique_id],
            label="Synthetic",
            linestyle="--",
        )
        plt.title(f"Original vs Synthetic Time Series for ID: {unique_id}")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()
        plot_path = f"{plot_dir}timegan_{dataset}_{dataset_group}_{unique_id}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()

        return synth_timegan_data[unique_id]

    # parallel timegan training and synthetic data generation
    # synth_timegan_data_all = Parallel(n_jobs=6)(
    #     delayed(train_and_generate_synthetic)(
    #         ts, original_data_long, DATASET, DATASET_GROUP, WINDOW_SIZE
    #     )
    #     for ts in original_data_long["unique_id"].unique()
    # )

    # best_params = hyper_tune_timegan(
    #     train_data_long, DATASET, DATASET_GROUP, window_size=24, n_trials=50
    # )
    # final_model = train_timegan_with_best_params(
    #     test_data_long, best_params, DATASET, DATASET_GROUP, window_size=24
    # )

    test_unique_ids = test_data_long["unique_id"].unique()

    if os.path.exists(SYNTHETIC_FILE_PATH):
        print("Synthetic TimeGAN data already exists. Loading the file...")
        synthetic_timegan_long = pd.read_pickle(SYNTHETIC_FILE_PATH)
    else:
        print("Synthetic TimeGAN data not found. Generating new data...")
        synth_timegan_data_all = []
        count = 0
        for ts in test_unique_ids:
            synth_timegan_data_all.append(
                train_and_generate_synthetic(
                    ts, test_data_long, DATASET, DATASET_GROUP, WINDOW_SIZE
                )
            )
            count += 1
            print(
                f"Generating synth data using TimeGAN for {ts}, which is {count}/{test_data_long['unique_id'].nunique()}"
            )

        synth_timegan_data_all_df = pd.concat(
            synth_timegan_data_all, ignore_index=True, axis=1
        )

        print("Transforming synthetic TimeGAN data into long form...")
        synthetic_timegan_long = create_dataset_vae.create_dataset_long_form(
            data=synth_timegan_data_all_df,
            unique_ids=test_unique_ids,
        )

        synthetic_timegan_long.to_pickle(SYNTHETIC_FILE_PATH)
        print(f"Synthetic TimeGAN data saved to {SYNTHETIC_FILE_PATH}")

    print("\nComputing discriminative score for HiTGen synthetic data...")
    score_hitgen = compute_discriminative_score(
        unique_ids=test_unique_ids,
        original_data=test_data_long,
        synthetic_data=synth_hitgen_test_long,
        freq="M",
        dataset_name=DATASET,
        dataset_group=DATASET_GROUP,
        loss=0.0,
        samples=5,
    )

    print("\nComputing discriminative score for TimeGAN synthetic data...")
    score_timegan = compute_discriminative_score(
        unique_ids=test_unique_ids,
        original_data=test_data_long,
        synthetic_data=synthetic_timegan_long,
        freq="M",
        dataset_name=DATASET,
        dataset_group=DATASET_GROUP,
        loss=0.0,
        samples=5,
    )

    print(f"Discriminative score for HiTGen synthetic data: {score_hitgen:.4f}")
    print(f"Discriminative score for TimeGAN synthetic data: {score_timegan:.4f}")
