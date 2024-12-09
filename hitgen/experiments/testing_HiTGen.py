import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

from hitgen.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE
from hitgen.visualization.model_visualization import (
    plot_loss,
    plot_generated_vs_original,
)
from hitgen.feature_engineering.feature_transformations import detemporalize
from hitgen.benchmarks.timegan import train_timegan_model, generate_synthetic_samples


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
    create_dataset_vae.hyper_tune_and_train()

    # fit
    model, history, _ = create_dataset_vae.fit(latent_dim=LATENT_DIM, epochs=EPOCHS)
    plot_loss(history)

    # prepare data for predictions
    data = create_dataset_vae._feature_engineering(create_dataset_vae.n, val_steps=0)

    X_orig = create_dataset_vae.X_train_raw

    zeros = np.zeros((95, 304))
    X_orig = np.concatenate((X_orig, zeros), axis=0)

    # Generate synthetic data using HiTGen
    # z_mean, z_log_vars, z = model.encoder.predict(gen_data)

    # epsilon = np.random.normal(size=z_mean.shape)
    # z_sampled = z_mean + np.exp(0.5 * z_log_vars) * epsilon
    new_latent_samples = np.random.normal(size=(300, 24, 304))

    generated_data = model.decoder.predict(new_latent_samples)

    generated_data = detemporalize(
        generated_data,
        STRIDE_TEMPORALIZE,
    )

    # plot_generated_vs_original(
    #     dec_pred_hat=generated_data,
    #     X_train_raw=X_orig,
    #     dataset_name=DATASET,
    #     dataset_group=DATASET_GROUP,
    #     n_series=8,
    # )

    # TimeGAN synthetic data generation
    time_gan_data = pd.DataFrame(
        create_dataset_vae.dataset["train"]["data"],
        columns=[
            f"series_{j}"
            for j in range(create_dataset_vae.dataset["train"]["data"].shape[1])
        ],
    )
    timegan = train_timegan_model(
        time_gan_data,
        gan_args=ModelParameters(
            batch_size=128,
            lr=5e-4,
            noise_dim=32,
            layers_dim=128,
            latent_dim=500,
            gamma=1,
        ),
        train_args=TrainParameters(
            epochs=5000,
            sequence_length=WINDOW_SIZE,
            number_sequences=create_dataset_vae.s,
        ),
        model_path=f"assets/model_weights/timegan_{DATASET}.pkl",
    )
    num_samples = len(time_gan_data) + 13
    synth_timegan_data = generate_synthetic_samples(timegan, num_samples, detemporalize)

    # Benchmark evaluation
    benchmark_data_dict = {"TimeGAN": synth_timegan_data}

    original_data = pd.DataFrame(
        create_dataset_vae.dataset["predict"]["data"],
        columns=[
            f"series_{j}"
            for j in range(create_dataset_vae.dataset["train"]["data"].shape[1])
        ],
    )
    original_data["ds"] = create_dataset_vae.dataset["dates"]
    melted_original_data = original_data.melt(
        id_vars=["ds"], var_name="unique_id", value_name="y"
    )

    synthetic_data = pd.DataFrame(
        generated_data,
        columns=[
            f"series_{j}"
            for j in range(create_dataset_vae.dataset["train"]["data"].shape[1])
        ],
    )
    synthetic_data["ds"] = create_dataset_vae.dataset["dates"]
    melted_synthetic_data = synthetic_data.melt(
        id_vars=["ds"], var_name="unique_id", value_name="y"
    )

    melted_synthetic_data.to_csv(
        "assets/results/hi2gen_synthetic_data.csv", index=False
    )

    synthetic_timegan_data = pd.DataFrame(
        synth_timegan_data,
        columns=[
            f"series_{j}"
            for j in range(create_dataset_vae.dataset["train"]["data"].shape[1])
        ],
    )
    synthetic_timegan_data["ds"] = create_dataset_vae.dataset["dates"]
    melted_synthetic_timegan_data = synthetic_timegan_data.melt(
        id_vars=["ds"], var_name="unique_id", value_name="y"
    )

    melted_synthetic_timegan_data.to_csv(
        "assets/results/timegan_synthetic_data.csv", index=False
    )

    # results = evaluate_discriminative_scores(
    #     X_orig_scaled=X_orig_scaled,
    #     main_model_data_scaled=generated_data_scaled,
    #     benchmark_data_dict=benchmark_data_dict,
    #     compute_discriminative_score=compute_discriminative_score,
    #     num_runs=20,
    #     num_samples=2,
    #     plot_first_run=True,

    # )
