import pandas as pd
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
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


if __name__ == "__main__":
    DATASET = "Tourism"
    DATASET_GROUP = "Monthly"
    FREQ = "M"
    TOP = None
    WINDOW_SIZE = 24
    VAL_STEPS = 0
    LATENT_DIM = 50
    EPOCHS = 750
    BATCH_SIZE = 8
    STRIDE_TEMPORALIZE = 1
    # LAST_ACTIVATION = "custom_relu_linear_saturation"
    # BI_RNN = True
    # SHUFFLE = True
    SHUFFLE = True
    BI_RNN = True
    ANNEALING = False
    KL_WEIGHT_INIT = 0.1
    NOISE_SAMPLE_INIT = 0.05

    ###### M5
    # dataset = "m5"
    # freq = "W"
    # top = 500
    # window_size = 26
    # val_steps = 26
    # latent_dim = 500
    # epochs = 5
    # batch_size = 8

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
    # create_dataset_vae.hyper_tune_and_train()

    # fit
    model, history, _ = create_dataset_vae.fit(latent_dim=LATENT_DIM, epochs=EPOCHS)
    # plot_loss(history)

    data = create_dataset_vae._feature_engineering()

    data_mask_temporalized = TemporalizeGenerator(
        data,
        create_dataset_vae.mask,
        window_size=WINDOW_SIZE,
        stride=create_dataset_vae.stride_temporalize,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
    )

    synth_hitgen = create_dataset_vae.predict(
        model,
        samples=data_mask_temporalized.indices.shape[0],
        window_size=WINDOW_SIZE,
        latent_dim=LATENT_DIM,
    )

    ######## GENERATE more samples into the future

    synthetic_hitgen_long = create_dataset_vae.create_dataset_long_form(synth_hitgen)

    # plot_generated_vs_original(
    #     dec_pred_hat=generated_data,
    #     X_train_raw=X_orig,
    #     dataset_name=DATASET,
    #     dataset_group=DATASET_GROUP,
    #     n_series=8,
    # )

    original_data_long = create_dataset_vae.original_data_long

    import matplotlib.pyplot as plt

    unique_ids = synthetic_hitgen_long["unique_id"].unique()[:4]

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    for idx, unique_id in enumerate(unique_ids):
        ax = axes[idx]
        original_series = original_data_long[
            original_data_long["unique_id"] == unique_id
        ]
        synthetic_series = synthetic_hitgen_long[
            synthetic_hitgen_long["unique_id"] == unique_id
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

        scaler = MinMaxScaler()
        scaled_target_df = pd.DataFrame(
            scaler.fit_transform(target_df), columns=target_df.columns
        )

        os.makedirs("assets/model_weights/timegan/", exist_ok=True)

        timegan = train_timegan_model(
            scaled_target_df,
            gan_args=ModelParameters(
                batch_size=8,
                lr=2e-4,
                noise_dim=32,
                layers_dim=128,
                latent_dim=128,
                gamma=1,
            ),
            train_args=TrainParameters(
                epochs=5000, sequence_length=window_size, number_sequences=1
            ),
            model_path=f"assets/model_weights/timegan/timegan_{dataset}_{dataset_group}_{unique_id}.pkl",
        )

        synth_scaled_data = generate_synthetic_samples(
            timegan, ts_data.shape[0] - WINDOW_SIZE + 1, detemporalize
        )

        synth_timegan_data = pd.DataFrame(
            scaler.inverse_transform(synth_scaled_data), columns=target_df.columns
        )

        # Convert synthetic data to DataFrame
        synthetic_df = pd.DataFrame(synth_timegan_data, columns=[unique_id])

        plot_dir = "assets/plots/timegan/"
        os.makedirs(plot_dir, exist_ok=True)

        import matplotlib.pyplot as plt

        # Plot the original vs. synthetic series
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

        return synth_timegan_data

    # parallel timegan training and synthetic data generation
    # synth_timegan_data_all = Parallel(n_jobs=6)(
    #     delayed(train_and_generate_synthetic)(
    #         ts, original_data_long, DATASET, DATASET_GROUP, WINDOW_SIZE
    #     )
    #     for ts in original_data_long["unique_id"].unique()
    # )
    synth_timegan_data_all = []
    for ts in original_data_long["unique_id"].unique():
        synth_timegan_data_all.append(
            train_and_generate_synthetic(
                ts, original_data_long, DATASET, DATASET_GROUP, WINDOW_SIZE
            )
        )

    best_params = hyper_tune_timegan(
        data, DATASET, DATASET_GROUP, window_size=24, n_trials=50
    )
    final_model = train_timegan_with_best_params(
        data, best_params, DATASET, DATASET_GROUP, window_size=24
    )

    # Combine synthetic datasets into long form
    print("Transforming synthetic TimeGAN data into long form...")
    synthetic_timegan_long = create_dataset_vae.create_dataset_long_form(
        synth_timegan_data_all
    )

    # Compute discriminative scores
    print("\nComputing discriminative score for HiTGen synthetic data...")
    score_hitgen = compute_discriminative_score(
        original_data_long, synthetic_hitgen_long, "M", DATASET, DATASET_GROUP, 0.0
    )
    print(f"Discriminative score for HiTGen synthetic data: {score_hitgen:.4f}")

    print("\nComputing discriminative score for TimeGAN synthetic data...")
    score_timegan = compute_discriminative_score(
        original_data_long, synthetic_timegan_long, "M", DATASET, DATASET_GROUP, 0.0
    )
    print(f"Discriminative score for TimeGAN synthetic data: {score_timegan:.4f}")

    # results = evaluate_discriminative_scores(
    #     X_orig_scaled=X_orig_scaled,
    #     main_model_data_scaled=generated_data_scaled,
    #     benchmark_data_dict=benchmark_data_dict,
    #     compute_discriminative_score=compute_discriminative_score,
    #     num_runs=20,
    #     num_samples=2,
    #     plot_first_run=True,

    # )
