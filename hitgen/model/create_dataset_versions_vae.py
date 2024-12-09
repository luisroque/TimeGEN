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
from hitgen.load_data.config import DATASETS, DATASETS_FREQ
from hitgen.visualization.model_visualization import (
    plot_generated_vs_original,
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
        dataset_group: str,
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
        stride_temporalize: int = 2,
        last_activation: str = "relu",
        bi_rnn: bool = True,
        annealing: bool = True,
        kl_weight_init: float = None,
        noise_scale_init: float = None,
    ):
        self.dataset_name = dataset_name
        self.dataset_group = dataset_group
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
        self.stride_temporalize = stride_temporalize
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.last_activation = last_activation
        self.bi_rnn = bi_rnn
        self.annealing = annealing
        self.kl_weight_init = kl_weight_init
        self.noise_scale_init = noise_scale_init
        (self.data, self.s, self.freq) = self.load_data(
            self.dataset_name, self.dataset_group
        )
        if window_size:
            self.window_size = window_size
        self.y = self.data
        self.n = self.data.shape[0]
        self.df = pd.DataFrame(self.data)
        # self.df = self.df.set_index("Date")
        self.df.asfreq(self.freq)
        # self.preprocess_freq()

        self.features_input = (None, None, None)
        self._create_directories()
        # self._save_original_file()
        self.original_data_long = None
        self.long_properties = {}

    @staticmethod
    def load_data(dataset_name, group):
        data_cls = DATASETS[dataset_name]
        print(dataset_name, group)

        try:
            ds = data_cls.load_data(group)
        except FileNotFoundError as e:
            print(f"Error loading data for {dataset_name} - {group}: {e}")

        h = data_cls.horizons_map[group]
        n_lags = data_cls.context_length[group]
        if dataset_name == "M4":
            freq = data_cls.frequency_map.get(group)
        else:
            freq = data_cls.frequency_pd[group]
        season_len = data_cls.frequency_map[group]
        n_series = ds.nunique()["unique_id"]
        return ds, n_series, freq

    def create_dataset_long_form(self, data):
        df = pd.DataFrame(data)

        df.columns = self.long_properties["unique_id"]
        df["ds"] = pd.date_range(
            self.long_properties["ds"][0],
            periods=data.shape[0],
            freq=self.freq,
        )

        data_long = df.melt(id_vars=["ds"], var_name="unique_id", value_name="y")
        return data_long

    # def preprocess_freq(self):
    #     end_date = None
    #
    #     # Create dataset with window_size more dates in the future to be used
    #     if self.freq in ["Q", "QS"]:
    #         if self.freq == "Q":
    #             self.freq += "S"
    #         end_date = self.df.index[-1] + pd.DateOffset(months=self.window_size * 3)
    #     elif self.freq in ["M", "MS"]:
    #         if self.freq == "M":
    #             self.freq += "S"
    #         end_date = self.df.index[-1] + pd.DateOffset(months=self.window_size)
    #     elif self.freq == "W":
    #         end_date = self.df.index[-1] + pd.DateOffset(weeks=self.window_size)
    #     elif self.freq == "D":
    #         end_date = self.df.index[-1] + pd.DateOffset(days=self.window_size)
    #     elif self.freq == "H":
    #         end_date = self.df.index[-1] + pd.DateOffset(days=self.window_size)
    #     else:
    #         raise InvalidFrequencyError(
    #             f"Invalid frequency - {self.freq}. Please use one of the defined frequencies: Q, QS, M, MS, W, or D."
    #         )
    #
    #     ix = pd.date_range(
    #         start=self.df.index[0],
    #         end=end_date,
    #         freq=self.freq,
    #     )
    #     self.df_generate = self.df.copy()
    #     self.df_generate = self.df_generate.reindex(ix)

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

    def _feature_engineering(
        self,
    ) -> pd.DataFrame:
        """Apply preprocessing to raw time series."""
        # sort the data by 'unique_id' and 'ds' to ensure chronological order for each time series
        self.df = self.df.sort_values(by=["unique_id", "ds"])

        x_train_wide = self.df.pivot(index="ds", columns="unique_id", values="y")

        self.long_properties["ds"] = x_train_wide.reset_index()["ds"].values
        self.long_properties["unique_id"] = x_train_wide.columns.values

        # create mask before padding
        mask_np = (~x_train_wide.isna()).astype(int).values
        self.mask = tf.convert_to_tensor(mask_np, dtype=tf.float32)

        # perform log returns directly, keeping nan values
        # adding small coeff to avoid inf when applying the log
        x_train_wide_log_returns = np.log(x_train_wide + 1)
        x_train_wide_log_returns = x_train_wide_log_returns.diff()

        # pad the data
        x_train_wide_log_returns = x_train_wide_log_returns.fillna(0.0)

        self.original_data_long = x_train_wide_log_returns.reset_index().melt(
            id_vars=["ds"], var_name="unique_id", value_name="y"
        )

        x_train_wide_log_returns = x_train_wide_log_returns.reset_index(drop=True)

        self.X_train_raw = x_train_wide_log_returns

        return x_train_wide_log_returns

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
        """Training our CVAE"""
        # Prepare features
        data = self._feature_engineering()

        data_mask_temporalized = TemporalizeGenerator(
            data,
            self.mask,
            window_size=self.window_size,
            stride=self.stride_temporalize,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

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
            weights_folder, f"{self.dataset_name}_{self.dataset_group}__vae.weights.h5"
        )
        history_file = os.path.join(
            weights_folder,
            f"{self.dataset_name}_{self.dataset_group}_training_history.json",
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

            history = cvae.fit(
                x=data_mask_temporalized,
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

    def update_best_scores(
        self,
        original_data,
        synthetic_data,
        score,
        latent_dim,
        window_size,
        patience,
        kl_weight,
        n_blocks,
        n_hidden,
        n_layers,
        kernel_size,
        pooling_mode,
        batch_size,
        epochs,
        learning_rate,
        last_activation,
        bi_rnn,
        shuffle,
        noise_scale_init,
        loss,
    ):
        scores_path = f"assets/model_weights/{self.dataset_name}_{self.dataset_group}_best_hyperparameters.jsonl"

        if os.path.exists(scores_path):
            with open(scores_path, "r") as f:
                scores_data = [json.loads(line) for line in f.readlines()]
        else:
            scores_data = []

        new_score = {
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
            "loss": loss,
            "score": score,
        }

        # if the list of scores is empty or the current score is better than the worst score
        if not scores_data or score > min([entry["score"] for entry in scores_data]):

            plot_generated_vs_original(
                dec_pred_hat=synthetic_data,
                X_train_raw=original_data,
                score=score,
                loss=loss,
                dataset_name=self.dataset_name,
                dataset_group=self.dataset_group,
                n_series=8,
            )

            scores_data.append(new_score)
            scores_data.sort(key=lambda x: x["score"])

            # keep only the top 20 scores
            scores_data = scores_data[:20]

            os.makedirs(os.path.dirname(scores_path), exist_ok=True)
            with open(scores_path, "w") as f:
                for score_entry in scores_data:
                    f.write(json.dumps(score_entry) + "\n")

            print(f"Best scores updated and saved to {scores_path}")

    def objective(self, trial):
        """
        Objective function for Optuna to tune the CVAE hyperparameters.
        """

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
        epochs = trial.suggest_int("epochs", 1, 50, step=10)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
        last_activation = trial.suggest_categorical(
            "last_activation", ["relu", "custom_relu_linear_saturation"]
        )
        bi_rnn = trial.suggest_categorical("bi_rnn", [True, False])
        shuffle = trial.suggest_categorical("shuffle", [True, False])
        noise_scale_init = trial.suggest_float("noise_scale_init", 0.01, 0.5)

        data = self._feature_engineering()

        data_mask_temporalized = TemporalizeGenerator(
            data,
            self.mask,
            window_size=window_size,
            stride=self.stride_temporalize,
            batch_size=batch_size,
            shuffle=shuffle,
        )

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
            x=data_mask_temporalized,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
        )

        loss = min(history.history["loss"])

        synthetic_data = self.predict(
            cvae,
            samples=data_mask_temporalized.indices.shape[0],
            window_size=window_size,
            latent_dim=latent_dim,
        )
        synthetic_data_long = self.create_dataset_long_form(synthetic_data)

        score = compute_discriminative_score(
            self.original_data_long, synthetic_data_long, "M"
        )

        self.update_best_scores(
            self.original_data_long,
            synthetic_data_long,
            score,
            latent_dim,
            window_size,
            patience,
            kl_weight,
            n_blocks,
            n_hidden,
            n_layers,
            kernel_size,
            pooling_mode,
            batch_size,
            epochs,
            learning_rate,
            last_activation,
            bi_rnn,
            shuffle,
            noise_scale_init,
            loss,
        )

        return score

    def hyper_tune_and_train(self, n_trials=200):
        """
        Run Optuna hyperparameter tuning for the CVAE and train the best model.
        """
        data = self._feature_engineering()

        study = optuna.create_study(
            study_name="opt_vae", direction="minimize", load_if_exists=True
        )
        study.optimize(self.objective, n_trials=n_trials)

        # retrieve the best trial
        best_trial = study.best_trial
        self.best_params = best_trial.params

        with open(
            f"assets/model_weights/{self.dataset_name}_{self.dataset_group}_best_params.json",
            "w",
        ) as f:
            json.dump(self.best_params, f)

        print(f"Best Hyperparameters: {self.best_params}")

        data_mask_temporalized = TemporalizeGenerator(
            data,
            self.mask,
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
            x=data_mask_temporalized,
            epochs=self.best_params["epochs"],
            batch_size=self.best_params["batch_size"],
            callbacks=[early_stopping],
        )

        # Save training history
        self.best_model = cvae
        self.best_history = history.history
        with open(
            f"assets/model_weights/{self.dataset_name}_{self.dataset_group}_training_history.json",
            "w",
        ) as f:
            json.dump(self.best_history, f)

        print("Training completed with the best hyperparameters.")

    def predict(
        self,
        cvae: CVAE,
        samples,
        window_size,
        latent_dim,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Predict original time series using VAE"""
        new_latent_samples = np.random.normal(size=(samples, window_size, latent_dim))
        mask_temporalized = self.temporalize(self.mask, window_size)
        generated_data = cvae.decoder.predict([new_latent_samples, mask_temporalized])

        X_hat = detemporalize(generated_data)

        return X_hat

    @staticmethod
    def temporalize(tensor_2d, window_size):
        shape = tf.shape(tensor_2d)
        output = []

        for idx in range(shape[0] - window_size + 1):
            window = tensor_2d[idx : idx + window_size, :]
            output.append(window)

        output = tf.stack(output)

        return output

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
