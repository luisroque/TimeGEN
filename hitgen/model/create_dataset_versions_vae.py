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
from hitgen.model.models import CVAE, get_CVAE, KLAnnealingAndNoiseScalingCallback
from hitgen.feature_engineering.deprecated_static_features import (
    create_static_features,
)
from hitgen.feature_engineering.dynamic_features import (
    create_dynamic_features,
)
from hitgen.feature_engineering.feature_transformations import (
    temporalize,
    combine_inputs_to_model,
    detemporalize,
)
from hitgen.visualization.model_visualization import (
    plot_generated_vs_original,
)

from hitgen.preprocessing.pre_processing_datasets import (
    PreprocessDatasets as ppc,
)
from hitgen.model.models import get_CVAE
from hitgen.metrics.discriminative_metrics import compute_discriminative_score

from hitgen import __version__


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

    def _generate_static_features(self, n: int) -> None:
        """Helper method to create the static feature and scale them

        Args:
            n: number of samples
        """
        self.static_features = create_static_features(self.groups, self.dataset)

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

        self._generate_static_features(n)

        self.dynamic_features_train = create_dynamic_features(
            self.df[:num_train_samples], self.freq
        )

        X_train = temporalize(
            train_data_scaled, self.window_size, self.stride_temporalize
        )

        self.n_features_concat = X_train.shape[1] + self.dynamic_features_train.shape[1]

        inp_train = combine_inputs_to_model(
            X_train,
            self.dynamic_features_train,
            self.static_features,
            self.window_size,
            self.stride_temporalize,
        )

        if val_steps > 0:
            val_data_scaled = self.scaler_target.transform(val_data)
            self.dynamic_features_val = create_dynamic_features(
                self.df[num_train_samples:],
                self.freq,
            )
            X_val = temporalize(val_data_scaled, self.window_size)

            inp_val = combine_inputs_to_model(
                X_val,
                self.dynamic_features_val,
                self.static_features,
                self.window_size,
                self.stride_temporalize,
            )
        else:
            inp_val = None

        return inp_train, inp_val

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
        batch_size: int = 8,
        patience: int = 100,
        latent_dim: int = 32,
        learning_rate: float = 0.001,
        kl_weight_initial: float = 0.01,
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
        self.features_input_train, self.features_input_val = self._feature_engineering(
            self.n_train, self.val_steps
        )
        dynamic_features_dim = len(self.features_input_train[0])

        inp = (
            self.features_input_train[1][0][:, :, : self.num_series]
            if self.num_series
            else self.features_input_train[1][0]
        )

        # Initialize the CVAE model
        encoder, decoder = get_CVAE(
            window_size=self.window_size,
            n_series=inp.shape[-1],
            latent_dim=latent_dim,
        )
        cvae = CVAE(
            encoder,
            decoder,
            kl_weight_initial=kl_weight_initial,
        )
        cvae.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=[cvae.reconstruction_loss_tracker, cvae.kl_loss_tracker],
        )

        # Build the model by calling it on a batch of data
        _ = cvae(inp)
        cvae.summary()

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
            kl_weight_initial=kl_weight_initial,
            kl_weight_final=kl_weight_final,
            noise_scale_initial=noise_scale_initial,
            noise_scale_final=noise_scale_final,
            annealing_epochs=annealing_epochs,
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
            _ = cvae(inp)
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
                x=inp,
                epochs=epochs,
                batch_size=batch_size,
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

        latent_dim = trial.suggest_int("latent_dim", 8, 64, step=8)
        patience = trial.suggest_int("patience", 8, 64, step=8)
        kl_weight = trial.suggest_float("kl_weight", 0.1, 10.0)
        num_blocks = trial.suggest_int("num_blocks", 1, 5)
        filters = trial.suggest_int("filters", 16, 128, step=16)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        epochs = trial.suggest_int("epochs", 200, 2000, step=100)

        dynamic_features_dim = len(self.features_input_train[0])

        encoder, decoder = get_CVAE(
            window_size=self.window_size,
            n_series=self.s,
            latent_dim=latent_dim,
            dynamic_features_dim=dynamic_features_dim,
            num_blocks=num_blocks,
            filters=filters,
        )

        cvae = CVAE(encoder, decoder, kl_weight=kl_weight)
        cvae.compile(optimizer="adam")

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
        history = cvae.fit(
            x=self.features_input_train,
            validation_data=self.features_input_val,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
        )

        synthetic_data = self.predict(cvae)
        ori_data = detemporalize(self.features_input[1][0])

        score = compute_discriminative_score(
            ori_data,
            synthetic_data,
            window_size=self.window_size,
            iterations=500,
            batch_size=64,
        )

        if not hasattr(self, "best_score") or score < self.best_score:
            self.best_score = score

            # Save the best hyperparameters
            self.best_params = {
                "score": score,
                "latent_dim": latent_dim,
                "patience": patience,
                "kl_weight": kl_weight,
                "num_blocks": num_blocks,
                "filters": filters,
                "batch_size": batch_size,
                "epochs": epochs,
            }
            params_path = os.path.join(
                "assets/model_weights", "best_hyperparameters.json"
            )
            with open(params_path, "w") as f:
                json.dump(self.best_params, f)
            print(f"Best hyperparameters saved at {params_path}")

        return score

    def hyper_tune_and_train(self, n_trials=200):
        """
        Run Optuna hyperparameter tuning for the CVAE and train the best model.
        """
        # Prepare data for tuning
        self.features_input_train, self.features_input_val = self._feature_engineering(
            self.n_train
        )

        # Initialize Optuna study
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)

        # Get the best hyperparameters
        best_trial = study.best_trial
        self.best_params = best_trial.params
        print(f"Best Hyperparameters: {self.best_params}")

        dynamic_features_dim = len(self.features_input_train[0])

        # Train the model with the best hyperparameters
        encoder, decoder = get_CVAE(
            window_size=self.window_size,
            latent_dim=self.best_params["latent_dim"],
            num_blocks=self.best_params["num_blocks"],
            filters=self.best_params["filters"],
        )

        cvae = CVAE(encoder, decoder, kl_weight=self.best_params["kl_weight"])
        cvae.compile(optimizer="adam")

        # Final training with best parameters
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.best_params["patience"],
            restore_best_weights=True,
        )
        history = cvae.fit(
            x=self.features_input_train,
            validation_data=self.features_input_val,
            epochs=self.best_params["epochs"],  # Using the optimized epochs
            batch_size=self.best_params["batch_size"],
            callbacks=[early_stopping],
        )

        self.best_model = cvae
        self.best_history = history.history
        with open("best_params.json", "w") as f:
            json.dump(self.best_params, f)
        with open("training_history.json", "w") as f:
            json.dump(self.best_history, f)

        print("Training completed with the best hyperparameters.")

    def evaluate_best_model(self):
        """
        Evaluate the best model using the discriminative score on the test set.
        """
        synthetic_data = self.predict(self.best_model)[0]
        score = compute_discriminative_score(
            self.features_input_val[1],
            synthetic_data,
            window_size=self.window_size,
            iterations=500,
            batch_size=64,
        )
        print(f"Discriminative score of the best model: {score}")
        return score

    def predict(
        self, cvae: CVAE
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Predict original time series using CVAE and capture attention scores.

        Args:
            cvae: The CVAE model with attention layers.

        Returns:
            X_hat_complete: The reconstructed series.
            intra_attention_scores: Intra-window attention scores from the encoder.
            inter_attention_scores: Inter-window attention scores from the encoder.
        """
        self.features_input, _ = self._feature_engineering(self.n_train, val_steps=0)
        dynamic_feat, X_inp, static_feat = self.features_input
        stacked_dynamic_feat = tf.stack(dynamic_feat, axis=-1)

        z_mean, z_log_var, z = cvae.encoder.predict([X_inp, stacked_dynamic_feat])
        generated_data = cvae.decoder.predict([z, stacked_dynamic_feat])

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
