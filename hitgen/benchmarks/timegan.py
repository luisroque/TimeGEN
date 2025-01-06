import os
import json
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from hitgen.metrics.discriminative_score import (
    compute_discriminative_score,
)
from hitgen.feature_engineering.feature_transformations import detemporalize


def train_timegan_model(train_data, gan_args, train_args, model_path):
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        synth = TimeSeriesSynthesizer.load(model_path)
    else:
        synth = TimeSeriesSynthesizer(modelname="timegan", model_parameters=gan_args)
        synth.fit(train_data, train_args, num_cols=train_data.columns.tolist())
        synth.save(model_path)
    return synth


def generate_synthetic_samples(synth, num_samples, detemporalize_func):
    synth_data = synth.sample(n_samples=num_samples)
    return detemporalize_func(synth_data)


def train_and_generate_synthetic(unique_id, data, dataset, dataset_group, window_size):
    print(f"Training TimeGAN for time series: {unique_id}")

    ts_data = data[data["unique_id"] == unique_id]

    target_df = ts_data.pivot(index="ds", columns="unique_id", values="y")
    target_df.columns.name = None
    target_df = target_df.reset_index(drop=True)

    os.makedirs("assets/model_weights/timegan/", exist_ok=True)

    timegan = train_timegan_model(
        target_df,
        gan_args=ModelParameters(
            batch_size=128,
            lr=5e-4,
            noise_dim=32,
            layers_dim=128,
            latent_dim=100,
            gamma=1,
        ),
        train_args=TrainParameters(
            epochs=5000, sequence_length=window_size, number_sequences=1
        ),
        model_path=f"assets/model_weights/timegan/timegan_{dataset}_{dataset_group}_{unique_id}.pkl",
    )

    synth_timegan_data = generate_synthetic_samples(
        timegan, ts_data.shape[0], detemporalize
    )

    synthetic_df = pd.DataFrame(synth_timegan_data, columns=[unique_id])

    plot_dir = "assets/plots/timegan/"
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(target_df[unique_id], label="Original", linestyle="-", marker="o")
    plt.plot(synthetic_df[unique_id], label="Synthetic", linestyle="--", marker="x")
    plt.title(f"Original vs Synthetic Time Series for ID: {unique_id}")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plot_path = f"{plot_dir}timegan_{dataset}_{dataset_group}_{unique_id}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return timegan, synth_timegan_data


def objective(trial, data_subset, dataset_name, dataset_group, window_size):
    """
    Objective function for Optuna to tune TimeGAN hyperparameters.
    """
    latent_dim = trial.suggest_int("latent_dim", 8, 512, step=8)
    noise_dim = trial.suggest_int("noise_dim", 8, 64, step=8)
    layers_dim = trial.suggest_int("layers_dim", 64, 256, step=64)
    gamma = trial.suggest_float("gamma", 0.1, 10)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    batch_size = trial.suggest_int("batch_size", 64, 256, step=64)
    epochs = trial.suggest_int("epochs", 100, 2000, step=100)

    gan_args = ModelParameters(
        batch_size=batch_size,
        lr=learning_rate,
        noise_dim=noise_dim,
        layers_dim=layers_dim,
        latent_dim=latent_dim,
        gamma=gamma,
    )

    train_args = TrainParameters(
        epochs=epochs,
        sequence_length=window_size,
        number_sequences=data_subset.shape[1],
    )

    synth_timegan_data = []
    unique_ids = data_subset["unique_id"].unique()
    for ts in unique_ids:
        timegan, synth_timegan_data = train_and_generate_synthetic(
            ts, data_subset, dataset_name, dataset_group, window_size
        )

        synth_timegan_data.append(
            generate_synthetic_samples(timegan, ts.shape[0], detemporalize)
        )

    synthetic_df = pd.DataFrame(synth_timegan_data, columns=[unique_id])
    score = compute_discriminative_score(
        unique_ids,
        original_data_long,
        synthetic_data_long,
        "M",
        dataset_name,
        dataset_group,
        0.0,
    )

    return score


def hyper_tune_timegan(data, dataset_name, dataset_group, window_size, n_trials=50):
    """
    Hyperparameter tuning for TimeGAN using Optuna.
    """
    # randomly select 10% of the time series
    unique_ids = data["unique_id"].unique()
    subset_ids = np.random.choice(
        unique_ids, size=int(len(unique_ids) * 0.1), replace=False
    )
    data_subset = data[data["unique_id"].isin(subset_ids)]

    study = optuna.create_study(direction="minimize", study_name="timegan_optuna")
    study.optimize(
        lambda trial: objective(
            trial, data_subset, dataset_name, dataset_group, window_size
        ),
        n_trials=n_trials,
    )

    best_params_path = f"assets/model_weights/timegan/timegan_{dataset_name}_{dataset_group}_best_params.json"
    os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f)

    print(f"Best Hyperparameters: {study.best_params}")
    return study.best_params


def train_timegan_with_best_params(
    data, best_params, dataset_name, dataset_group, window_size
):
    """
    Train TimeGAN on the full dataset using the best parameters.
    """
    gan_args = ModelParameters(
        batch_size=best_params["batch_size"],
        lr=best_params["learning_rate"],
        noise_dim=best_params["noise_dim"],
        layers_dim=best_params["layers_dim"],
        latent_dim=best_params["latent_dim"],
        gamma=best_params["gamma"],
    )

    train_args = TrainParameters(
        epochs=best_params["epochs"],
        sequence_length=window_size,
        number_sequences=data.shape[1],
    )

    model_path = (
        f"assets/model_weights/timegan/timegan_{dataset_name}_{dataset_group}_final.pkl"
    )
    timegan = train_timegan_model(data, gan_args, train_args, model_path)
    print(f"Final model saved at {model_path}")
    return timegan
