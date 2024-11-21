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
from hitgen.metrics.discriminative_metrics import (
    compute_discriminative_score,
    evaluate_discriminative_scores,
)
from hitgen.benchmarks.timegan import train_timegan_model, generate_synthetic_samples
from hitgen.utils.helper import inverse_transform
from hitgen.model.models import TemporalizeGenerator

###### Tourism
DATASET = "tourism"
FREQ = "M"
TOP = None
WINDOW_SIZE = 24
VAL_STEPS = 0
LATENT_DIM = 25
EPOCHS = 750
BATCH_SIZE = 8
STRIDE_TEMPORALIZE = 1
SHUFFLE = True
NUM_SERIES = 100

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
    freq=FREQ,
    top=TOP,
    window_size=WINDOW_SIZE,
    val_steps=VAL_STEPS,
    num_series=NUM_SERIES,
    stride_temporalize=STRIDE_TEMPORALIZE,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
)

# Fit the CVAE model
model, history, _ = create_dataset_vae.fit(latent_dim=LATENT_DIM, epochs=EPOCHS)
plot_loss(history)

# Prepare data for predictions
data = create_dataset_vae._feature_engineering(create_dataset_vae.n, val_steps=0)

gen_data = TemporalizeGenerator(
    data=data,
    window_size=WINDOW_SIZE,
    stride=STRIDE_TEMPORALIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

X_orig = create_dataset_vae.X_train_raw

# Generate synthetic data using HiTGen
z_mean, z_log_vars, z = model.encoder.predict(gen_data)

epsilon = np.random.normal(size=z_mean.shape)
z_sampled = z_mean + np.exp(0.5 * z_log_vars) * epsilon
generated_data = model.decoder.predict(z_sampled)

generated_data = detemporalize(
    inverse_transform(generated_data, create_dataset_vae.scaler_target),
    STRIDE_TEMPORALIZE,
)

scaler = MinMaxScaler()
X_orig_scaled = scaler.fit_transform(X_orig)
generated_data_scaled = scaler.transform(generated_data)

plot_generated_vs_original(
    dec_pred_hat=generated_data,
    X_train_raw=X_orig,
    dataset_name=DATASET,
    transf_param=1.0,
    model_version="1.0",
    transformation="scaling",
    n_series=8,
    directory=".",
)

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
        batch_size=128, lr=5e-4, noise_dim=32, layers_dim=128, latent_dim=500, gamma=1
    ),
    train_args=TrainParameters(
        epochs=5000, sequence_length=WINDOW_SIZE, number_sequences=create_dataset_vae.s
    ),
    model_path=f"assets/model_weights/timegan_{DATASET}.pkl",
)
num_samples = len(time_gan_data) + 13
synth_timegan_data = generate_synthetic_samples(timegan, num_samples, detemporalize)

# Benchmark evaluation
benchmark_data_dict = {"TimeGAN": synth_timegan_data}
results = evaluate_discriminative_scores(
    X_orig_scaled=X_orig_scaled,
    main_model_data_scaled=generated_data_scaled,
    benchmark_data_dict=benchmark_data_dict,
    compute_discriminative_score=compute_discriminative_score,
    num_runs=20,
    num_samples=2,
    plot_first_run=True,
)
