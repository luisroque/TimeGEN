import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
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
    # create_dataset_vae.hyper_tune_and_train()

    # fit
    model, history, _ = create_dataset_vae.fit(latent_dim=LATENT_DIM, epochs=EPOCHS)
    plot_loss(history)

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

    synthetic_hitgen_long = create_dataset_vae.create_dataset_long_form(synth_hitgen)

    # plot_generated_vs_original(
    #     dec_pred_hat=generated_data,
    #     X_train_raw=X_orig,
    #     dataset_name=DATASET,
    #     dataset_group=DATASET_GROUP,
    #     n_series=8,
    # )

    # TimeGAN synthetic data generation
    timegan = train_timegan_model(
        data,
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
            number_sequences=data.shape[1],
        ),
        model_path=f"assets/model_weights/timegan_{DATASET}_{DATASET_GROUP}.pkl",
    )
    synth_timegan_data = generate_synthetic_samples(
        timegan, data.shape[0], detemporalize
    )

    # Benchmark evaluation
    original_data_long = create_dataset_vae.original_data_long

    synthetic_timegan_long = create_dataset_vae.create_dataset_long_form(
        synth_timegan_data
    )

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
