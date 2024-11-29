import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
import json
from sklearn.preprocessing import MinMaxScaler
import optuna
from tensorflow import keras
import tensorflow as tf
from tensorflow import data as tfdata
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow import (
    float32,
)
from hitgen.model.models import (
    CVAE,
    KLAnnealingAndNoiseScalingCallback,
    TemporalizeGenerator,
)
from hitgen.feature_engineering.feature_transformations import (
    detemporalize,
)
from hitgen.preprocessing.pre_processing_datasets import (
    PreprocessDatasets as ppc,
)
from hitgen.model.models import get_CVAE
from hitgen.metrics.discriminative_score import (
    compute_discriminative_score,
)


class InvalidFrequencyError(Exception):
    pass


class CreateTransformedVersionsCVAE:
    """
    Class for creating transformed versions of the dataset using a Conditional Variational Autoencoder (CVAE).

    This class contains several methods to preprocess data, fit a CVAE, generate new time series, and
    save transformed versions of the dataset. It's designed to be used with time-series data.

    The class follows the Singleton design pattern ensuring that only one instance can exist.

    Args:
        dataset_name: Name of the dataset.
        freq: Frequency of the time series data.
        input_dir: Directory where the input data is located. Defaults to "./".
        transf_data: Type of transformation applied to the data. Defaults to "whole".
        top: Number of top series to select. Defaults to None.
        window_size: Window size for the sliding window. Defaults to 10.
        weekly_m5: If True, use the M5 competition's weekly grouping. Defaults to True.
        test_size: Size of the test set. If None, the size is determined automatically. Defaults to None.

        Below are parameters for the synthetic data creation:
            num_base_series_time_points: Number of base time points in the series. Defaults to 100.
            num_latent_dim: Dimension of the latent space. Defaults to 3.
            num_variants: Number of variants for the transformation. Defaults to 20.
            noise_scale: Scale of the Gaussian noise. Defaults to 0.1.
            amplitude: Amplitude of the time series data. Defaults to 1.0.
    """

    _instance = None

    def __new__(cls, *args, **kwargs) -> "CreateTransformedVersionsCVAE":
        """
        Override the __new__ method to implement the Singleton design pattern.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        dataset_name: str,
        freq: str,
        batch_size: int = 8,
        shuffle: bool = True,
        input_dir: str = "./assets/",
        transf_data: str = "whole",
        top: int = None,
        window_size: int = 12,
        weekly_m5: bool = True,
        test_size: int = None,
        num_base_series_time_points: int = 100,
        num_latent_dim: int = 3,
        num_variants: int = 20,
        noise_scale: float = 0.1,
        amplitude: float = 1.0,
        val_steps: int = 0,
        num_series: int = None,
        stride_temporalize: int = 2,
        last_activation: str = "relu",
        bi_rnn: bool = True,
        annealing: bool = True,
        kl_weight_init: float = None,
        noise_scale_init: float = None,
    ):
        self.dataset_name = dataset_name
        self.input_dir = input_dir
        self.transf_data = transf_data
        self.freq = freq
        self.top = top
        self.test_size = test_size
        self.weekly_m5 = weekly_m5
        self.num_base_series_time_points = num_base_series_time_points
        self.num_latent_dim = num_latent_dim
        self.num_variants = num_variants
        self.noise_scale = noise_scale
        self.amplitude = amplitude
        self.val_steps = val_steps
        self.num_series = num_series
        self.stride_temporalize = stride_temporalize
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.last_activation = last_activation
        self.bi_rnn = bi_rnn
        self.annealing = annealing
        self.kl_weight_init = kl_weight_init
        self.noise_scale_init = noise_scale_init
        self.dataset = self._get_dataset()
        if window_size:
            self.window_size = window_size
        data = self.dataset["predict"]["data"]
        self.y = data
        self.n = data.shape[0]
        self.s = data.shape[1]
        self.n_train = self.n - self.window_size + 1
        self.groups = list(self.dataset["train"]["groups_names"].keys())
        self.df = pd.DataFrame(data)
        self.df = pd.concat(
            [self.df, pd.DataFrame(self.dataset["dates"], columns=["Date"])], axis=1
        )[: self.n]
        self.df = self.df.set_index("Date")
        self.df.asfreq(self.freq)
        self.preprocess_freq()

        self.features_input = (None, None, None)
        self._create_directories()
        self._save_original_file()
        self.original_data_long = self.create_dataset_long_form(data)

    def create_dataset_long_form(self, data):
        data_wider = pd.DataFrame(
            data,
            columns=[f"series_{j}" for j in range(self.s)],
        )
        data_wider["ds"] = self.dataset["dates"]
        data_long = data_wider.melt(
            id_vars=["ds"], var_name="unique_id", value_name="y"
        )
        return data_long

    def preprocess_freq(self):
        end_date = None

        # Create dataset with window_size more dates in the future to be used
        if self.freq in ["Q", "QS"]:
            if self.freq == "Q":
                self.freq += "S"
            end_date = self.df.index[-1] + pd.DateOffset(months=self.window_size * 3)
        elif self.freq in ["M", "MS"]:
            if self.freq == "M":
                self.freq += "S"
            end_date = self.df.index[-1] + pd.DateOffset(months=self.window_size)
        elif self.freq == "W":
            end_date = self.df.index[-1] + pd.DateOffset(weeks=self.window_size)
        elif self.freq == "D":
            end_date = self.df.index[-1] + pd.DateOffset(days=self.window_size)
        else:
            raise InvalidFrequencyError(
                f"Invalid frequency - {self.freq}. Please use one of the defined frequencies: Q, QS, M, MS, W, or D."
            )

        ix = pd.date_range(
            start=self.df.index[0],
            end=end_date,
            freq=self.freq,
        )
        self.df_generate = self.df.copy()
        self.df_generate = self.df_generate.reindex(ix)

    def _get_dataset(self):
        """
        Get dataset and apply preprocessing
        """
        ppc_args = {
            "dataset": self.dataset_name,
            "freq": self.freq,
            "input_dir": self.input_dir,
            "top": self.top,
            "test_size": self.test_size,
            "weekly_m5": self.weekly_m5,
            "num_base_series_time_points": self.num_base_series_time_points,
            "num_latent_dim": self.num_latent_dim,
            "num_variants": self.num_variants,
            "noise_scale": self.noise_scale,
            "amplitude": self.amplitude,
        }

        dataset = ppc(**ppc_args).apply_preprocess()

        return dataset

    def _create_directories(self):
        """
        Create dynamically the directories to store the data if they don't exist
        """
        # Create directory to store transformed datasets if does not exist
        Path(f"{self.input_dir}data").mkdir(parents=True, exist_ok=True)
        Path(f"{self.input_dir}data/transformed_datasets").mkdir(
            parents=True, exist_ok=True
        )

    def _save_original_file(self):
        """
        Store original dataset
        """
        with open(
            f"{self.input_dir}data/transformed_datasets/{self.dataset_name}_original.npy",
            "wb",
        ) as f:
            np.save(f, self.y)

    def _save_version_file(
        self,
        y_new: np.ndarray,
        version: int,
        sample: int,
        transformation: str,
        method: str = "single_transf",
    ) -> None:
        """
        Store the transformed dataset

        :param y_new: transformed data
        :param version: version of the transformation
        :param sample: sample of the transformation
        :param transformation: name of the transformation applied
        """
        with open(
            f"{self.input_dir}data/transformed_datasets/{self.dataset_name}_version_{version}_{sample}samples_{method}_{transformation}_{self.transf_data}.npy",
            "wb",
        ) as f:
            np.save(f, y_new)

    def _feature_engineering(
        self, n: int, val_steps: float = 24
    ) -> tuple[tuple, tuple]:
        """Create static and dynamic features as well as apply preprocessing to raw time series,
           including train-validation split.

        Args:
            n: Total number of samples
            val_steps: Number of data points to use for validation

        Returns:
            Tuple containing processed training and validation feature inputs.
        """
        self.X_train_raw = self.df.astype(np.float32).to_numpy()
        self.X_train_raw = (
            self.X_train_raw[:, : self.num_series]
            if self.num_series
            else self.X_train_raw
        )
        data = self.df.astype(np.float32).to_numpy()
        data = data[:, : self.num_series] if self.num_series else data

        num_train_samples = len(data) - val_steps
        train_data, val_data = data[:num_train_samples], data[num_train_samples:]

        self.scaler_target = MinMaxScaler().fit(train_data)
        train_data_scaled = self.scaler_target.transform(train_data)

        return train_data_scaled

    @staticmethod
    def _generate_noise(self, n_batches, window_size):
        while True:
            yield np.random.uniform(low=0, high=1, size=(n_batches, window_size))

    def get_batch_noise(
        self,
        batch_size,
        size=None,
    ):
        return iter(
            tfdata.Dataset.from_generator(self._generate_noise, output_types=float32)
            .batch(batch_size if size is None else size)
            .repeat()
        )

    def fit(
        self,
        epochs: int = 750,
        patience: int = 100,
        latent_dim: int = 32,
        learning_rate: float = 0.001,
        kl_weight_final: float = 1.0,
        noise_scale_initial: float = 0.01,
        noise_scale_final: float = 1.0,
        annealing_epochs: int = 150,
        hyper_tuning: bool = False,
        load_weights: bool = True,
    ) -> tuple[CVAE, dict, EarlyStopping]:
        """
        Training our CVAE on the dataset supplied with KL annealing and noise scaling.

        :param epochs: Total number of training epochs.
        :param batch_size: Batch size for training.
        :param patience: Early stopping patience.
        :param latent_dim: Latent space dimensionality.
        :param learning_rate: Learning rate for the optimizer.
        :param kl_weight_initial: Initial KL divergence weight.
        :param kl_weight_final: Final KL divergence weight.
        :param noise_scale_initial: Initial noise scale for sampling.
        :param noise_scale_final: Final noise scale for sampling.
        :param annealing_epochs: Number of epochs to fully anneal KL weight and noise scale.
        :return: Trained model, training history, and early stopping object.
        """
        # Prepare features
        data = self._feature_engineering(self.n_train, self.val_steps)

        data_temporalized = TemporalizeGenerator(
            data,
            window_size=self.window_size,
            stride=self.stride_temporalize,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

        # Initialize the CVAE model
        encoder, decoder = get_CVAE(
            window_size=self.window_size,
            n_series=data.shape[-1],
            latent_dim=latent_dim,
            last_activation=self.last_activation,
            bi_rnn=self.bi_rnn,
            noise_scale_init=self.noise_scale_init,
        )

        cvae = CVAE(
            encoder,
            decoder,
            kl_weight_initial=self.kl_weight_init,
        )
        cvae.compile(
            optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate),
            metrics=[cvae.reconstruction_loss_tracker, cvae.kl_loss_tracker],
        )

        # Early stopping and learning rate callbacks
        es = EarlyStopping(
            patience=patience,
            verbose=1,
            monitor="loss",
            mode="auto",
            restore_best_weights=True,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        )

        kl_and_noise_callback = KLAnnealingAndNoiseScalingCallback(
            model=cvae,
            kl_weight_initial=self.kl_weight_init,
            kl_weight_final=kl_weight_final,
            noise_scale_initial=noise_scale_initial,
            noise_scale_final=noise_scale_final,
            annealing_epochs=annealing_epochs,
            annealing=self.annealing,
        )

        weights_folder = "assets/model_weights"
        os.makedirs(weights_folder, exist_ok=True)

        weights_file = os.path.join(
            weights_folder, f"{self.dataset_name}_vae.weights.h5"
        )
        history_file = os.path.join(
            weights_folder, f"{self.dataset_name}_training_history.json"
        )
        history = None

        if os.path.exists(weights_file) and not hyper_tuning and load_weights:
            print("Loading existing weights...")
            cvae.load_weights(weights_file)

            if os.path.exists(history_file):
                print("Loading training history...")
                with open(history_file, "r") as f:
                    history = json.load(f)
            else:
                print("No history file found. Skipping history loading.")
        else:

            mc = ModelCheckpoint(
                weights_file,
                save_best_only=True,
                save_weights_only=True,
                monitor="loss",
                mode="auto",
                verbose=1,
            )

            # Train the model with callbacks
            history = cvae.fit(
                x=data_temporalized,
                epochs=epochs,
                batch_size=self.batch_size,
                shuffle=False,
                callbacks=[es, mc, reduce_lr, kl_and_noise_callback],
            )

            if history is not None:
                history = history.history
                history_dict = {
                    key: [float(val) for val in values]
                    for key, values in history.items()
                }
                with open(history_file, "w") as f:
                    json.dump(history_dict, f)

        return cvae, history, es

    def objective(self, trial):
        """
        Objective function for Optuna to tune the CVAE hyperparameters.
        """

        # Define hyperparameter trials
        latent_dim = trial.suggest_int("latent_dim", 8, 64, step=8)
        window_size = trial.suggest_int("window_size", 6, 24)
        patience = trial.suggest_int("patience", 90, 100, step=5)
        kl_weight = trial.suggest_float("kl_weight", 0.05, 1)
        n_blocks = trial.suggest_int("n_blocks", 1, 5)
        n_hidden = trial.suggest_int("n_hidden", 16, 128, step=16)
        n_layers = trial.suggest_int("n_layers", 1, 5)
        kernel_size = trial.suggest_int("kernel_size", 2, 5)
        pooling_mode = trial.suggest_categorical("pooling_mode", ["max", "average"])
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        epochs = trial.suggest_int("epochs", 1, 2000, step=100)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
        last_activation = trial.suggest_categorical(
            "last_activation", ["relu", "custom_relu_linear_saturation"]
        )
        bi_rnn = trial.suggest_categorical("bi_rnn", [True, False])
        shuffle = trial.suggest_categorical("shuffle", [True, False])
        noise_scale_init = trial.suggest_float("noise_scale_init", 0.01, 0.5)

        data = self._feature_engineering(self.n_train)

        data_temporalized = TemporalizeGenerator(
            data,
            window_size=window_size,
            stride=self.stride_temporalize,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        # Pass hyperparameters to the CVAE model
        encoder, decoder = get_CVAE(
            window_size=window_size,
            n_series=self.s,
            latent_dim=latent_dim,
            last_activation=last_activation,
            bi_rnn=bi_rnn,
            noise_scale_init=noise_scale_init,
            n_blocks=n_blocks,
            n_hidden=n_hidden,
            n_layers=n_layers,
            kernel_size=kernel_size,
            pooling_mode=pooling_mode,
        )

        cvae = CVAE(encoder, decoder, kl_weight_initial=kl_weight)
        cvae.compile(
            optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate),
            metrics=[cvae.reconstruction_loss_tracker, cvae.kl_loss_tracker],
        )

        es = EarlyStopping(
            patience=patience,
            verbose=1,
            monitor="loss",
            mode="auto",
            restore_best_weights=True,
        )

        history = cvae.fit(
            x=data_temporalized,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
        )

        synthetic_data = self.predict(
            cvae,
            samples=data_temporalized.indices.shape[0],
            window_size=window_size,
            latent_dim=latent_dim,
        )
        synthetic_data_long = self.create_dataset_long_form(synthetic_data)

        score = compute_discriminative_score(
            self.original_data_long, synthetic_data_long, "M"
        )

        # Save best parameters and model
        if not hasattr(self, "best_score") or score < self.best_score:
            self.best_score = score
            self.best_params = {
                "latent_dim": latent_dim,
                "window_size": window_size,
                "patience": patience,
                "kl_weight": kl_weight,
                "n_blocks": n_blocks,
                "n_hidden": n_hidden,
                "n_layers": n_layers,
                "kernel_size": kernel_size,
                "pooling_mode": pooling_mode,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "last_activation": last_activation,
                "bi_rnn": bi_rnn,
                "shuffle": shuffle,
                "noise_scale_init": noise_scale_init,
                "score": score,
            }
            with open("assets/model_weights/best_hyperparameters.json", "w") as f:
                json.dump(self.best_params, f)
            print(f"Best hyperparameters saved")

        return score

    def hyper_tune_and_train(self, n_trials=200):
        """
        Run Optuna hyperparameter tuning for the CVAE and train the best model.
        """
        data = self._feature_engineering(self.n_train)

        # optuna study with persistence
        study = optuna.create_study(
            study_name="opt_vae", direction="minimize", load_if_exists=True
        )
        study.optimize(self.objective, n_trials=n_trials)

        # retrieve the best trial
        best_trial = study.best_trial
        self.best_params = best_trial.params

        with open("assets/model_weights/best_params.json", "w") as f:
            json.dump(self.best_params, f)

        print(f"Best Hyperparameters: {self.best_params}")

        data_temporalized = TemporalizeGenerator(
            data,
            window_size=self.best_params["latent_dim"],
            stride=self.stride_temporalize,
            batch_size=self.best_params["batch_size"],
            shuffle=self.best_params["shuffle"],
        )

        encoder, decoder = get_CVAE(
            window_size=self.best_params["latent_dim"],
            n_series=self.s,
            latent_dim=self.best_params["latent_dim"],
            last_activation=self.best_params["last_activation"],
            bi_rnn=self.best_params["bi_rnn"],
            noise_scale_init=self.best_params["noise_scale_init"],
            n_blocks=self.best_params["n_blocks"],
            n_hidden=self.best_params["n_hidden"],
            n_layers=self.best_params["n_layers"],
            kernel_size=self.best_params["kernel_size"],
            strides=self.best_params["strides"],
            pooling_mode=self.best_params["pooling_mode"],
        )

        cvae = CVAE(encoder, decoder, kl_weight=self.best_params["kl_weight"])
        cvae.compile(
            optimizer=keras.optimizers.legacy.Adam(
                learning_rate=self.best_params["learning_rate"]
            ),
            metrics=[cvae.reconstruction_loss_tracker, cvae.kl_loss_tracker],
        )

        # final training with best parameters
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=self.best_params["patience"],
            restore_best_weights=True,
        )
        history = cvae.fit(
            x=data_temporalized,
            epochs=self.best_params["epochs"],
            batch_size=self.best_params["batch_size"],
            callbacks=[early_stopping],
        )

        # Save training history
        self.best_model = cvae
        self.best_history = history.history
        with open("assets/model_weights/training_history.json", "w") as f:
            json.dump(self.best_history, f)

        print("Training completed with the best hyperparameters.")

    def predict(
        self, cvae: CVAE, samples, window_size, latent_dim
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Predict original time series using VAE"""
        new_latent_samples = np.random.normal(size=(samples, window_size, latent_dim))
        generated_data = cvae.decoder.predict(new_latent_samples)

        X_hat = detemporalize(
            self.inverse_transform(generated_data, self.scaler_target)
        )

        X_hat_complete = np.concatenate(
            (self.X_train_raw[: self.window_size], X_hat), axis=0
        )

        return X_hat_complete

    @staticmethod
    def inverse_transform(data, scaler):
        if not scaler:
            return data
        # Reshape from (samples, timesteps, features) to (samples*timesteps, features)
        original_shape = data.shape
        data_reshaped = data.reshape(-1, original_shape[-1])
        data_inverse = scaler.inverse_transform(data_reshaped)
        return data_inverse.reshape(original_shape)

    def generate_new_datasets(
        self,
        cvae: CVAE,
        z_mean: np.ndarray,
        z_log_var: np.ndarray,
        transformation: Optional[str] = None,
        transf_param: List[float] = None,
        n_versions: int = 6,
        n_samples: int = 10,
        save: bool = True,
    ) -> np.ndarray:
        """
        Generate new datasets using the CVAE trained model and different samples from its latent space.

        Args:
            cvae: A trained Conditional Variational Autoencoder (CVAE) model.
            z_mean: Mean parameters of the latent space distribution (Gaussian). Shape: [num_samples, window_size].
            z_log_var: Log variance parameters of the latent space distribution (Gaussian). Shape: [num_samples, window_size].
            transformation: Transformation to apply to the data, if any.
            transf_param: Parameter for the transformation.
            n_versions: Number of versions of the dataset to create.
            n_samples: Number of samples of the dataset to create.
            save: If True, the generated datasets are stored locally.

        Returns:
            An array containing the new generated datasets.
        """
        if transf_param is None:
            transf_param = [0.5, 2, 4, 10, 20, 50]
        y_new = np.zeros((n_versions, n_samples, self.n, self.s))
        s = 0
        for v in range(1, n_versions + 1):
            for s in range(1, n_samples + 1):
                y_new[v - 1, s - 1] = self.generate_transformed_time_series(
                    cvae=cvae,
                    z_mean=z_mean,
                    z_log_var=z_log_var,
                    transformation=transformation,
                    transf_param=transf_param[v - 1],
                )
            if save:
                self._save_version_file(y_new[v - 1], v, s, "vae")
        return y_new
