import os
import gc
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import optuna
from tensorflow import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from hitgen.model.models import (
    CVAE,
    TemporalizeGenerator,
)
from hitgen.feature_engineering.feature_transformations import (
    detemporalize,
)
from hitgen.model.models import get_CVAE
from hitgen.metrics.discriminative_score import (
    compute_discriminative_score,
    compute_downstream_forecast,
)
from hitgen.load_data.config import DATASETS
from hitgen.visualization.model_visualization import (
    plot_generated_vs_original,
)


class InvalidFrequencyError(Exception):
    pass


class CreateTransformedVersionsCVAE:
    """
    Class for creating transformed versions of the dataset using a Conditional Variational Autoencoder (CVAE).
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
        noise_scale: float = 0.1,
        amplitude: float = 1.0,
        forecasting: bool = True,
        kl_weight_init: float = 0.1,
        noise_scale_init: float = 0.1,
        kernel_size: Tuple[int] = (2, 2, 1),
        patience: int = 30,
        horizon: int = 24,
        opt_score: str = "discriminative_score",
    ):
        self.dataset_name = dataset_name
        self.dataset_group = dataset_group
        self.freq = freq
        self.noise_scale = noise_scale
        self.amplitude = amplitude
        self.batch_size = batch_size
        self.forecasting = forecasting
        self.kl_weight_init = kl_weight_init
        self.noise_scale_init = noise_scale_init
        self.patience = patience
        self.kernel_size = kernel_size
        (self.data, self.s, self.freq) = self.load_data(
            self.dataset_name, self.dataset_group
        )
        self.s_train = None
        self.y = self.data
        self.n = self.data.shape[0]
        self.df = pd.DataFrame(self.data)
        self.df.asfreq(self.freq)
        self.h = horizon
        self.opt_score = opt_score

        num_series = self.df["unique_id"].nunique()
        avg_time_points = self.df.groupby("unique_id").size().mean()

        print(f"Dataset Summary for {dataset_name} ({dataset_group}):")
        print(f"   - Total number of time series: {num_series}")
        print(f"   - Average number of time points per series: {avg_time_points:.2f}")

        self.features_input = (None, None, None)
        self.long_properties = {}
        self.split_path = f"assets/model_weights/data_split/{dataset_name}_{dataset_group}_data_split.json"
        self.unique_ids = self.df["unique_id"].unique()

        # map each frequency string to (index_function, period)
        self.freq_map = {
            "D": (self._daily_index, 365.25),
            "W": (self._weekly_index, 52.18),
            "W-SUN": (self._weekly_index, 52.18),
            "W-MON": (self._weekly_index, 52.18),
            "M": (self._monthly_index, 12),
            "MS": (self._monthly_index, 12),
            "Q": (self._quarterly_index, 4),
            "QS": (self._quarterly_index, 4),
            "Y": (self._yearly_index, 1),
            "YS": (self._yearly_index, 1),
        }

        feature_dict = self._feature_engineering()

        # original Data
        self.original_wide = feature_dict["original_wide"]
        self.original_long = feature_dict["original_long"]
        self.original_long_transf = feature_dict["original_long_transf"]
        self.original_wide_transf = feature_dict["original_wide_transf"]

        # training Data
        self.original_train_wide = feature_dict["train_wide"]
        self.original_train_long = feature_dict["train_long"]
        self.original_train_long_transf = feature_dict["original_train_long_transf"]
        self.original_train_wide_transf = feature_dict["original_train_wide_transf"]

        # validation data
        self.original_val_wide = feature_dict["val_wide"]
        self.original_val_long = feature_dict["val_long"]
        self.original_val_long_transf = feature_dict["original_val_long_transf"]
        self.original_val_wide_transf = feature_dict["original_val_wide_transf"]

        # test data
        self.original_test_wide = feature_dict["test_wide"]
        self.original_test_long = feature_dict["test_long"]
        self.original_test_long_transf = feature_dict["original_test_long_transf"]
        self.original_test_wide_transf = feature_dict["original_test_wide_transf"]

        # mask data
        self.mask_wide = feature_dict["mask_wide"]
        self.mask_train_wide = feature_dict["mask_train_wide"]
        self.mask_val_wide = feature_dict["mask_val_wide"]
        self.mask_test_wide = feature_dict["mask_test_wide"]

        # fourier features
        self.original_dyn_features = feature_dict["fourier_features_original"]
        self.train_dyn_features = feature_dict["fourier_features_train"]
        self.val_dyn_features = feature_dict["fourier_features_val"]
        self.test_dyn_features = feature_dict["fourier_features_test"]

    @staticmethod
    def load_data(dataset_name: str, group: str) -> Tuple[pd.DataFrame, int, str]:
        data_cls = DATASETS[dataset_name]
        print(dataset_name, group)

        try:
            ds = data_cls.load_data(group)

            # for testing purposes only
            # ds = ds[ds["unique_id"].isin(ds["unique_id"].unique()[:20])]
        except FileNotFoundError as e:
            print(f"Error loading data for {dataset_name} - {group}: {e}")

        freq = data_cls.frequency_pd[group]
        n_series = int(ds.nunique()["unique_id"])
        return ds, n_series, freq

    def create_dataset_long_form(self, data, original, unique_ids=None) -> pd.DataFrame:
        df = pd.DataFrame(data)
        df.sort_index(ascending=True, inplace=True)

        if unique_ids is None:
            df.columns = self.long_properties["unique_id"]
            df["ds"] = self.long_properties["ds"]
        else:
            df.columns = unique_ids
            if df.index.name == "ds" or isinstance(df.index, pd.DatetimeIndex):
                df.reset_index(inplace=True)
            else:
                ds_unique_sorted = np.sort(original["ds"].unique())
                if len(df) != len(ds_unique_sorted):
                    raise ValueError(
                        f"Length mismatch: DataFrame has {len(df)} rows, "
                        f"while sorted unique 'ds' has {len(ds_unique_sorted)} entries."
                    )
                df["ds"] = ds_unique_sorted

        data_long = df.melt(id_vars=["ds"], var_name="unique_id", value_name="y")
        data_long = data_long.sort_values(by=["unique_id", "ds"])

        # keep only values that exist in the original
        data_long = data_long.merge(
            original[["unique_id", "ds"]], on=["unique_id", "ds"], how="inner"
        )

        return data_long

    @staticmethod
    def _create_dataset_wide_form(
        data_long: pd.DataFrame,
        ids: List[str],
        full_dates: pd.DatetimeIndex,
        fill_nans: bool = True,
    ) -> pd.DataFrame:
        """
        Transforms a long-form dataset back to wide form ensuring the
        right order of the columns
        """
        if not {"ds", "unique_id", "y"}.issubset(data_long.columns):
            raise ValueError(
                "Input DataFrame must contain 'ds', 'unique_id', and 'y' columns."
            )

        data_long = data_long.sort_values(by=["unique_id", "ds"])
        data_wide = data_long.pivot(index="ds", columns="unique_id", values="y")
        data_wide = data_wide.sort_index(ascending=True)
        data_wide = data_wide.reindex(index=full_dates, columns=ids)
        data_wide.index.name = "ds"

        if fill_nans:
            data_wide = data_wide.fillna(0)

        return data_wide

    def check_series_in_train_test(
        self,
        date,
        train_ids,
        test_ids,
    ):
        series_to_check = self.df.loc[self.df["ds"] == date, "unique_id"].unique()

        in_train = np.intersect1d(train_ids, series_to_check)

        if len(in_train) == 0:
            # none of the series is in train, move one in from test if possible
            in_test = np.intersect1d(test_ids, series_to_check)
            if len(in_test) > 0:
                chosen = in_test[0]
                # remove from test, add to train
                test_ids = np.setdiff1d(test_ids, [chosen])
                train_ids = np.append(train_ids, chosen)

        return train_ids, test_ids

    def _load_or_create_split(
        self,
        train_test_split: float,
        val_split: float = 0.15,
    ) -> (np.ndarray, np.ndarray):
        """Load split from file if it exists, otherwise create and save a new split."""
        if os.path.exists(self.split_path):
            with open(self.split_path, "r") as f:
                split_data = json.load(f)
                return (
                    np.array(split_data["train_ids"]),
                    np.array(split_data["val_ids"]),
                    np.array(split_data["test_ids"]),
                )

        val_size = int(len(self.unique_ids) * val_split)
        np.random.shuffle(self.unique_ids)
        train_size = int(len(self.unique_ids) * (1 - val_split) * train_test_split)

        train_ids = self.unique_ids[:train_size]
        val_ids = self.unique_ids[train_size : train_size + val_size]
        test_ids = self.unique_ids[train_size + val_size :]

        os.makedirs(os.path.dirname(self.split_path), exist_ok=True)
        with open(self.split_path, "w") as f:
            json.dump(
                {
                    "train_ids": train_ids.tolist(),
                    "val_ids": val_ids.tolist(),
                    "test_ids": test_ids.tolist(),
                },
                f,
            )

        # ensuring that we have in the training set at least one series
        # that starts with the min date
        ds_min = self.long_properties["ds"].min()
        ds_max = self.long_properties["ds"].max()

        train_ids, test_ids = self.check_series_in_train_test(
            ds_min, train_ids, test_ids
        )
        train_ids, test_ids = self.check_series_in_train_test(
            ds_max, train_ids, test_ids
        )

        return train_ids, val_ids, test_ids

    @staticmethod
    def _transform_log_returns(x):
        if isinstance(x, np.ndarray):
            x = pd.Series(x)
        x_log = np.log(x + 1)
        x_diff = x_log.diff()
        return x_diff

    @staticmethod
    def _transform_diff(x):
        if isinstance(x, np.ndarray):
            x = pd.Series(x)
        x_diff = x.diff()
        return x_diff

    @staticmethod
    def _transform_diff_minmax(x):
        if isinstance(x, np.ndarray):
            x = pd.Series(x)
        x_diff = x.diff()
        return x_diff

    @staticmethod
    def _backtransform_log_returns(x_diff: pd.DataFrame, initial_value: pd.DataFrame):
        """
        Back-transform log returns.
        """
        x_diff["ds"] = pd.to_datetime(x_diff["ds"])
        initial_value["ds"] = pd.to_datetime(initial_value["ds"])

        # filter x_diff to exclude any value before the first true value
        x_diff = x_diff.merge(
            initial_value[["unique_id", "ds"]],
            on="unique_id",
            suffixes=("", "_initial"),
        )
        x_diff = x_diff[x_diff["ds"] > x_diff["ds_initial"]]
        x_diff = x_diff.drop(columns=["ds_initial"])

        # compute log-transformed initial values
        initial_value = initial_value.set_index("unique_id")
        initial_value["y"] = np.log(initial_value["y"] + 1)

        x_diff = pd.concat(
            [x_diff.reset_index(drop=True), initial_value.reset_index()]
        ).sort_values(by=["unique_id", "ds"])

        # set the index for x_diff for alignment and compute the cumulative sum
        x_diff["y"] = x_diff.groupby("unique_id")["y"].cumsum()

        x_diff["y"] = np.exp(x_diff["y"]) - 1

        return x_diff

    @staticmethod
    def _backtransform_diff(x_diff: pd.DataFrame, initial_value: pd.DataFrame):
        """
        Back-transform log returns.
        """
        x_diff["ds"] = pd.to_datetime(x_diff["ds"])
        initial_value["ds"] = pd.to_datetime(initial_value["ds"])

        # filter x_diff to exclude any value before the first true value
        x_diff = x_diff.merge(
            initial_value[["unique_id", "ds"]],
            on="unique_id",
            suffixes=("", "_initial"),
        )
        x_diff = x_diff[x_diff["ds"] > x_diff["ds_initial"]]
        x_diff = x_diff.drop(columns=["ds_initial"])

        # compute log-transformed initial values
        initial_value = initial_value.set_index("unique_id")

        x_diff = pd.concat(
            [x_diff.reset_index(drop=True), initial_value.reset_index()]
        ).sort_values(by=["unique_id", "ds"])

        # set the index for x_diff for alignment and compute the cumulative sum
        x_diff["y"] = x_diff.groupby("unique_id")["y"].cumsum()

        return x_diff

    def _preprocess_data(
        self, df: pd.DataFrame, ids: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Sort and preprocess the data for feature engineering."""
        self.full_dates = pd.DatetimeIndex(sorted(df.ds.unique()))

        x_wide = self._create_dataset_wide_form(
            data_long=df,
            ids=ids,
            fill_nans=False,
            full_dates=self.full_dates,
        )

        self.long_properties["ds"] = x_wide.reset_index()["ds"].values
        self.long_properties["unique_id"] = x_wide.columns.values

        # create mask before padding
        mask = (~x_wide.isna()).astype(int)

        # padding
        x_wide_filled = x_wide.fillna(0.0)

        return x_wide_filled, mask, x_wide_filled

    @staticmethod
    def _monthly_index(dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Convert each date to an integer that represents
        the number of months since the first date
        """
        d0 = dates[0]
        return (dates.year - d0.year) * 12 + (dates.month - d0.month)

    @staticmethod
    def _quarterly_index(dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Convert each date to an integer that represents
        the number of quarters since the first date
        """
        d0 = dates[0]
        quarter = (dates.month - 1) // 3
        quarter0 = (d0.month - 1) // 3
        return (dates.year - d0.year) * 4 + (quarter - quarter0)

    @staticmethod
    def _yearly_index(dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Convert each date to an integer that represents
        the number of years since the first date
        """
        d0 = dates[0]
        return dates.year - d0.year

    @staticmethod
    def _daily_index(dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Convert each date to an integer that represents
        the number of days since the first date
        """
        d0 = dates[0]
        return (dates - d0).days

    @staticmethod
    def _weekly_index(dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Convert each date to a float that represents
        the number of weeks since the first date
        """
        d0 = dates[0]
        # dividing days by 7 => fractional weeks
        return (dates - d0).days / 7.0

    def compute_fourier_features(self, dates, order=3) -> pd.DataFrame:
        """
        Compute sinusoidal (Fourier) terms for a given frequency.
        """
        dates = pd.to_datetime(dates)

        try:
            index_func, period = self.freq_map[self.freq]
        except KeyError:
            # for weekly, we might have something like "W-THU", so let's handle that generically:
            if self.freq.startswith("W-"):
                index_func, period = self._weekly_index, 52.18
            else:
                raise ValueError(f"Unsupported freq: {self.freq}")

        t = index_func(dates).astype(float)

        features = {}
        for k in range(1, order + 1):
            arg = 2.0 * np.pi * k * t / period
            features[f"sin_{self.freq}_{k}"] = np.sin(arg)
            features[f"cos_{self.freq}_{k}"] = np.cos(arg)

        return pd.DataFrame(features, index=dates)

    def _feature_engineering(
        self, train_test_split=0.7, val_split=0.15
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply preprocessing to raw time series, split into training, validation, and testing,
        compute periodic Fourier features, and return all relevant DataFrames.
        """
        self.ids = self.df["unique_id"].unique().sort()
        original_wide_filled, mask_original_wide, original_wide = self._preprocess_data(
            self.df, self.ids
        )

        # scalers
        self.scaler = StandardScaler()
        self.scaler_train = StandardScaler()
        self.scaler_val = StandardScaler()
        self.scaler_test = StandardScaler()

        original_wide_transf = pd.DataFrame(
            self.scaler.fit_transform(original_wide_filled),
            index=original_wide_filled.index,
            columns=original_wide_filled.columns,
        )

        original_long = self.create_dataset_long_form(original_wide, self.df)
        original_long_transf = self.create_dataset_long_form(
            original_wide_transf, self.df
        )

        mask_original_long = self.create_dataset_long_form(mask_original_wide, self.df)

        # splitting into train, validation, and test sets
        self.train_ids, self.val_ids, self.test_ids = self._load_or_create_split(
            train_test_split=train_test_split, val_split=val_split
        )
        self.train_ids.sort()
        self.val_ids.sort()
        self.test_ids.sort()
        self.s_train = len(self.train_ids)
        self.s_val = len(self.val_ids)

        ### TRAINING DATA ###
        train_long = original_long[original_long["unique_id"].isin(self.train_ids)]
        mask_train_long = mask_original_long[
            mask_original_long["unique_id"].isin(self.train_ids)
        ]

        # convert training long -> wide
        train_wide = self._create_dataset_wide_form(
            data_long=train_long, ids=self.train_ids, full_dates=self.full_dates
        )
        train_wide_transf = pd.DataFrame(
            self.scaler_train.fit_transform(train_wide),
            index=train_wide.index,
            columns=train_wide.columns,
        )
        mask_train_wide = self._create_dataset_wide_form(
            data_long=mask_train_long, ids=self.train_ids, full_dates=self.full_dates
        )
        train_long_transf = self.create_dataset_long_form(
            train_wide_transf, self.df, unique_ids=self.train_ids
        )

        ### VALIDATION DATA ###
        val_long = original_long[original_long["unique_id"].isin(self.val_ids)]
        mask_val_long = mask_original_long[
            mask_original_long["unique_id"].isin(self.val_ids)
        ]

        # convert validation long -> wide
        val_wide = self._create_dataset_wide_form(
            data_long=val_long, ids=self.val_ids, full_dates=self.full_dates
        )
        val_wide_transf = pd.DataFrame(
            self.scaler_val.fit_transform(val_wide),
            index=val_wide.index,
            columns=val_wide.columns,
        )
        mask_val_wide = self._create_dataset_wide_form(
            data_long=mask_val_long, ids=self.val_ids, full_dates=self.full_dates
        )
        val_long_transf = self.create_dataset_long_form(
            val_wide_transf, self.df, unique_ids=self.val_ids
        )

        ### TESTING DATA ###
        test_long = original_long[original_long["unique_id"].isin(self.test_ids)]
        mask_test_long = mask_original_long[
            mask_original_long["unique_id"].isin(self.test_ids)
        ]

        # convert testing long -> wide
        test_wide = self._create_dataset_wide_form(
            data_long=test_long, ids=self.test_ids, full_dates=self.full_dates
        )
        test_wide_transf = pd.DataFrame(
            self.scaler_test.fit_transform(test_wide),
            index=test_wide.index,
            columns=test_wide.columns,
        )
        mask_test_wide = self._create_dataset_wide_form(
            data_long=mask_test_long, ids=self.test_ids, full_dates=self.full_dates
        )
        test_long_transf = self.create_dataset_long_form(
            test_wide_transf, self.df, unique_ids=self.test_ids
        )

        # convert masks to tensors
        self.mask_train_tf = tf.convert_to_tensor(
            mask_train_wide.values, dtype=tf.float32
        )
        self.mask_val_tf = tf.convert_to_tensor(mask_val_wide.values, dtype=tf.float32)
        self.mask_test_tf = tf.convert_to_tensor(
            mask_test_wide.values, dtype=tf.float32
        )
        self.mask_original_tf = tf.convert_to_tensor(
            mask_original_wide.values, dtype=tf.float32
        )

        # store raw datasets
        self.X_train_raw = train_wide.reset_index(drop=True)
        self.X_val_raw = val_wide.reset_index(drop=True)
        self.X_test_raw = test_wide.reset_index(drop=True)
        self.X_orig_raw = original_wide.reset_index(drop=True)

        # compute Fourier features
        fourier_features_train = self.compute_fourier_features(train_wide.index)
        fourier_features_val = self.compute_fourier_features(val_wide.index)
        fourier_features_test = self.compute_fourier_features(test_wide.index)
        fourier_features_original = self.compute_fourier_features(original_wide.index)

        return {
            # wide Data
            "original_wide": original_wide,
            "train_wide": train_wide,
            "val_wide": val_wide,
            "test_wide": test_wide,
            # long Data
            "original_long": original_long,
            "train_long": train_long,
            "val_long": val_long,
            "test_long": test_long,
            # mask Wide
            "mask_train_wide": mask_train_wide,
            "mask_val_wide": mask_val_wide,
            "mask_test_wide": mask_test_wide,
            "mask_wide": mask_original_wide,
            # transformed Long Data
            "original_long_transf": original_long_transf,
            "original_train_long_transf": train_long_transf,
            "original_val_long_transf": val_long_transf,
            "original_test_long_transf": test_long_transf,
            # wide Transformed Data
            "original_wide_transf": original_wide_transf,
            "original_train_wide_transf": train_wide_transf,
            "original_val_wide_transf": val_wide_transf,
            "original_test_wide_transf": test_wide_transf,
            # fourier Features
            "fourier_features_train": fourier_features_train,
            "fourier_features_val": fourier_features_val,
            "fourier_features_test": fourier_features_test,
            "fourier_features_original": fourier_features_original,
        }

    def update_best_scores(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        score: float,
        latent_dim: int,
        window_size: int,
        patience: int,
        kl_weight: float,
        kernel_size: Tuple[int],
        n_hidden: int,
        time_dist_units: int,
        n_blocks: int,
        batch_size: int,
        windows_batch_size: int,
        stride: int,
        coverage_fraction: float,
        prediction_mode: str,
        future_steps: int,
        epochs: int,
        learning_rate: float,
        forecasting: bool,
        noise_scale_init: float,
        loss: float,
        trial: int,
    ) -> None:
        scores_path = f"assets/model_weights/{self.dataset_name}_{self.dataset_group}_best_hyperparameters.jsonl"
        opt_path = f"assets/model_weights/{self.dataset_name}_{self.dataset_group}_best_hyperparameters_opt.json"

        if os.path.exists(scores_path):
            with open(scores_path, "r") as f:
                scores_data = [json.loads(line) for line in f.readlines()]
        else:
            scores_data = []

        new_score = {
            "trial": trial,
            "latent_dim": latent_dim,
            "window_size": window_size,
            "patience": patience,
            "kl_weight": kl_weight,
            "kernel_size": kernel_size,
            "n_hidden": n_hidden,
            "time_dist_units": time_dist_units,
            "n_blocks": n_blocks,
            "batch_size": batch_size,
            "windows_batch_size": windows_batch_size,
            "stride": stride,
            "coverage_fraction": coverage_fraction,
            "prediction_mode": prediction_mode,
            "future_steps": future_steps,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "forecasting": forecasting,
            "noise_scale_init": noise_scale_init,
            "loss": loss,
            self.opt_score: score,
        }

        added_score = False

        # always add the score if there are fewer than 20 entries
        if len(scores_data) < 20:
            scores_data.append(new_score)
            added_score = True

        # if the list is full, add only if the score is better than the worst
        elif score < max([entry[self.opt_score] for entry in scores_data]):
            scores_data.append(new_score)
            added_score = True

        if added_score:
            plot_generated_vs_original(
                synth_data=synthetic_data,
                original_test_data=original_data,
                score=score,
                loss=loss,
                dataset_name=self.dataset_name,
                dataset_group=self.dataset_group,
                n_series=8,
            )

        scores_data.sort(key=lambda x: x[self.opt_score])
        scores_data = scores_data[:20]

        os.makedirs(os.path.dirname(scores_path), exist_ok=True)
        with open(scores_path, "w") as f:
            for score_entry in scores_data:
                f.write(json.dumps(score_entry) + "\n")

        best_score = scores_data[:1]
        opt_meta_info = {"current_trial": trial, "best_score": best_score}

        os.makedirs(os.path.dirname(opt_path), exist_ok=True)
        with open(opt_path, "w") as f:
            json.dumps(opt_meta_info)

        print(f"Best scores updated and saved to {scores_path}")

    @staticmethod
    def compute_mean_discriminative_score(
        unique_ids,
        original_data,
        synthetic_data,
        method,
        freq,
        dataset_name,
        dataset_group,
        loss,
        store_score,
        store_features_synth,
        split,
        num_iterations=1,
        generate_feature_plot=False,
    ):
        scores = []
        for i in range(num_iterations):
            score = compute_discriminative_score(
                unique_ids=unique_ids,
                original_data=original_data,
                synthetic_data=synthetic_data,
                method=method,
                freq=freq,
                dataset_name=dataset_name,
                dataset_group=dataset_group,
                loss=loss,
                samples=3,
                generate_feature_plot=generate_feature_plot,
                store_score=store_score,
                store_features_synth=store_features_synth,
                split=split,
            )
            scores.append(score)

        mean_score = np.mean(scores)
        return mean_score

    def objective(self, trial):
        """
        Objective function for Optuna to tune the CVAE hyperparameters, evaluated using the validation set.
        """
        # try:
        latent_dim = trial.suggest_int("latent_dim", 8, 128, step=8)
        if self.freq in ["M", "MS"]:
            window_size = trial.suggest_int("window_size", 6, 24, step=3)
        elif self.freq in ["Q", "QS"]:
            window_size = trial.suggest_int("window_size", 8, 16, step=2)
        elif self.freq in ["Y", "YS"]:
            window_size = trial.suggest_int("window_size", 4, 8, step=1)
        else:
            window_size = trial.suggest_int("window_size", 4, 24, step=1)

        stride = trial.suggest_int("stride", 1, window_size // 2, step=1)
        patience = trial.suggest_int("patience", 4, 8, step=1)
        kl_weight = trial.suggest_float("kl_weight", 0.05, 0.5)
        n_hidden = trial.suggest_int("n_hidden", 128, 256, step=32)
        time_dist_units = trial.suggest_int("time_dist_units", 16, 32, step=8)
        n_blocks = trial.suggest_int("n_layers", 1, 3)

        predefined_kernel_sizes = [(2, 2, 1), (1, 1, 1), (2, 1, 1), (4, 2, 1)]
        valid_kernel_sizes = [
            ks for ks in predefined_kernel_sizes if all(window_size >= k for k in ks)
        ]
        if not valid_kernel_sizes:
            valid_kernel_sizes.append((1, 1, 1))

        kernel_size = tuple(
            trial.suggest_categorical("kernel_size", valid_kernel_sizes)
        )

        batch_size = trial.suggest_int("batch_size", 8, 24, step=8)
        windows_batch_size = trial.suggest_int("windows_batch_size", 8, 24, step=8)
        coverage_fraction = trial.suggest_float("coverage_fraction", 0.3, 0.6)

        prediction_mode = trial.suggest_categorical(
            "prediction_mode", ["one_step_ahead", "multi_step_ahead"]
        )
        if prediction_mode == "multi_step_ahead":
            future_steps = window_size
        else:
            future_steps = 1
        epochs = trial.suggest_int("epochs", 100, 500, step=25)
        learning_rate = trial.suggest_loguniform("learning_rate", 3e-5, 3e-4)
        noise_scale_init = trial.suggest_float("noise_scale_init", 0.01, 0.5)

        forecasting = True

        data_mask_temporalized_train = TemporalizeGenerator(
            data=self.original_train_wide_transf,
            mask=self.mask_train_wide,
            dyn_features=self.train_dyn_features,
            window_size=window_size,
            stride=stride,
            batch_size=batch_size,
            windows_batch_size=windows_batch_size,
            coverage_mode="partial",
            coverage_fraction=coverage_fraction,
            prediction_mode=prediction_mode,
            future_steps=future_steps,
        )

        data_mask_temporalized_val = TemporalizeGenerator(
            data=self.original_val_wide_transf,
            mask=self.mask_val_wide,
            dyn_features=self.val_dyn_features,
            window_size=window_size,
            stride=1,
            batch_size=batch_size,
            windows_batch_size=windows_batch_size,
            coverage_mode="systematic",
            prediction_mode=prediction_mode,
            future_steps=future_steps,
        )

        encoder, decoder = get_CVAE(
            window_size=window_size,
            input_dim=1,  # univariate series
            latent_dim=latent_dim,
            pred_dim=future_steps,
            time_dist_units=time_dist_units,
            n_blocks=n_blocks,
            noise_scale_init=noise_scale_init,
            kernel_size=kernel_size,
            forecasting=forecasting,
            n_hidden=n_hidden,
        )

        cvae = CVAE(
            encoder, decoder, kl_weight_initial=kl_weight, forecasting=forecasting
        )
        cvae.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=[
                cvae.total_loss_tracker,
                cvae.reconstruction_loss_tracker,
                cvae.kl_loss_tracker,
            ],
        )

        es = EarlyStopping(
            patience=patience,
            verbose=1,
            monitor="val_loss",
            mode="auto",
            restore_best_weights=True,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=10,
            min_lr=1e-6,
            cooldown=3,
            verbose=1,
        )

        history = cvae.fit(
            x=data_mask_temporalized_train,
            validation_data=data_mask_temporalized_val,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            callbacks=[es, reduce_lr],
            validation_freq=3,
        )

        val_loss = min(history.history["loss"])

        synthetic_long = self.predict(
            cvae,
            gen_data=data_mask_temporalized_val,
            scaler=self.scaler_val,
            original_data_wide=self.original_val_wide,
            original_data_long=self.original_val_long,
            unique_ids=self.val_ids,
        )

        if self.opt_score == "discriminative_score":
            score = self.compute_mean_discriminative_score(
                unique_ids=synthetic_long["unique_id"].unique(),
                original_data=self.original_long,
                synthetic_data=synthetic_long,
                method="hitgen",
                freq=self.freq,
                dataset_name=self.dataset_name,
                dataset_group=self.dataset_group,
                loss=val_loss,
                generate_feature_plot=False,
                store_score=False,
                store_features_synth=False,
                split="hypertuning",
            )
        elif self.opt_score == "downstream_score":
            score = compute_downstream_forecast(
                unique_ids=synthetic_long["unique_id"].unique(),
                original_data=self.original_long,
                synthetic_data=synthetic_long,
                method="hitgen",
                freq=self.freq,
                horizon=self.h,
                dataset_name=self.dataset_name,
                dataset_group=self.dataset_group,
                samples=10,
                split="hypertuning",
            )
        else:
            score = val_loss

        if score is None:
            print("No valid scores computed. Pruning this trial.")
            raise optuna.exceptions.TrialPruned()

        self.update_best_scores(
            original_data=self.original_val_long,
            synthetic_data=synthetic_long,
            score=score,
            latent_dim=latent_dim,
            window_size=window_size,
            patience=patience,
            kl_weight=kl_weight,
            kernel_size=kernel_size,
            n_hidden=n_hidden,
            time_dist_units=time_dist_units,
            n_blocks=n_blocks,
            batch_size=batch_size,
            windows_batch_size=windows_batch_size,
            stride=stride,
            coverage_fraction=coverage_fraction,
            epochs=epochs,
            learning_rate=learning_rate,
            forecasting=forecasting,
            noise_scale_init=noise_scale_init,
            prediction_mode=prediction_mode,
            future_steps=future_steps,
            loss=val_loss,
            trial=trial.number,
        )

        # clean GPU memory between trials
        del cvae, encoder, decoder
        tf.keras.backend.clear_session()
        gc.collect()

        return score

        # except Exception as e:
        #     print(f"Error in trial: {e}")
        #     raise optuna.exceptions.TrialPruned()

    def hyper_tune_and_train(self, n_trials=100):
        """
        Run Optuna hyperparameter tuning for the CVAE and train the best model.
        """
        study_name = f"{self.dataset_name}_{self.dataset_group}_opt_vae"

        storage = "sqlite:///optuna_study.db"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            load_if_exists=True,
        )

        weights_folder = "assets/model_weights"
        os.makedirs(weights_folder, exist_ok=True)

        weights_file = os.path.join(
            weights_folder,
            f"{self.dataset_name}_{self.dataset_group}__vae.weights.h5",
        )

        best_params_meta_opt_file = os.path.join(
            weights_folder,
            f"{self.dataset_name}_{self.dataset_group}_best_hyperparameters_opt.json",
        )

        try:
            with open(best_params_meta_opt_file, "r") as f:
                best_params = json.load(f)
                self.best_params = best_params["best_score"][0]
                print("Best params file found. Loading...")
        except (FileNotFoundError, json.JSONDecodeError):
            print(
                f"Error loading best parameters file {best_params_meta_opt_file}. Proceeding with optimization..."
            )
            study.optimize(self.objective, n_trials=n_trials)
            with open(
                best_params_meta_opt_file,
                "r",
            ) as f:
                json.load(f)

            self.best_params = best_params["best_score"][0]

        print(f"Best Hyperparameters: {self.best_params}")

        data_mask_temporalized = TemporalizeGenerator(
            data=self.original_wide_transf,
            mask=self.mask_wide,
            dyn_features=self.original_dyn_features,
            window_size=self.best_params["window_size"],
            batch_size=self.best_params["batch_size"],
            windows_batch_size=self.best_params["windows_batch_size"],
            coverage_mode="systematic",
            stride=1,
            prediction_mode=self.best_params["prediction_mode"],
            future_steps=self.best_params["future_steps"],
        )

        encoder, decoder = get_CVAE(
            window_size=self.best_params["window_size"],
            input_dim=1,  # univariate series
            latent_dim=self.best_params["latent_dim"],
            pred_dim=self.best_params["future_steps"],
            time_dist_units=self.best_params["time_dist_units"],
            noise_scale_init=self.best_params["noise_scale_init"],
            kernel_size=self.best_params["kernel_size"],
            forecasting=self.best_params["forecasting"],
            n_hidden=self.best_params["n_hidden"],
        )

        cvae = CVAE(
            encoder,
            decoder,
            kl_weight_initial=self.best_params["kl_weight"],
            forecasting=self.best_params["forecasting"],
        )
        cvae.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.best_params["learning_rate"]
            ),
            metrics=[
                cvae.total_loss_tracker,
                cvae.reconstruction_loss_tracker,
                cvae.kl_loss_tracker,
            ],
        )

        # final training with best parameters
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=self.best_params["patience"],
            mode="auto",
            restore_best_weights=True,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="loss", factor=0.2, patience=10, min_lr=1e-6, cooldown=3, verbose=1
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

        dummy_input = (
            tf.random.normal((1, self.best_params["window_size"], 1)),  # batch_data
            tf.ones((1, self.best_params["window_size"], 1)),  # batch_mask
            tf.random.normal(
                (1, self.best_params["window_size"], 6)
            ),  # batch_dyn_features
            tf.random.normal((1, self.best_params["future_steps"], 1)),  # pred_mask
        )
        _ = cvae(dummy_input)

        if os.path.exists(weights_file):
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
                epochs=self.best_params["epochs"],
                batch_size=self.best_params["batch_size"],
                shuffle=False,
                callbacks=[early_stopping, mc, reduce_lr],
            )

            if history is not None:
                history = history.history
                history_dict = {
                    key: [float(val) for val in values]
                    for key, values in history.items()
                }
                with open(history_file, "w") as f:
                    json.dump(history_dict, f)

        print("Training completed with the best hyperparameters.")
        return cvae

    def _predict_loop(
        self,
        cvae,
        generator: TemporalizeGenerator,
        T: int,
        N: int,
        use_reconstruction: bool = True,
    ) -> np.ndarray:
        """
        1) Loops over generator.
        2) For each batch, does cvae.predict_on_batch.
        3) Gathers them in a list, plus the metadata.
        4) detemporalize => [T, N].
        5) returns the reconstructed wide array.
        """
        all_preds = []
        all_meta = []

        for i in range(len(generator)):
            (batch_x, targets) = generator[
                i
            ]  # => ( (batch_data, batch_mask, batch_dyn, pred_mask), _ )
            batch_pred = cvae.predict_on_batch(batch_x)  # (recon, forecast)

            if isinstance(batch_pred, (tuple, list)):
                recon_out, z_mean_out, z_log_var_out, forecast_out = batch_pred
                if use_reconstruction:
                    batch_pred = recon_out
                else:
                    batch_pred = forecast_out

            # shape of batch_pred => [B, window_size, 1]
            all_preds.append(batch_pred)
            # gather metadata
            all_meta.extend(generator.last_batch_metadata)

        predictions = np.concatenate(all_preds, axis=0)  # [B_total, window_size, 1]
        reconst_wide = detemporalize(predictions, all_meta, T, N)  # shape [T, N]
        return reconst_wide

    def predict(
        self,
        cvae,
        gen_data,
        scaler,
        original_data_wide,
        original_data_long,
        unique_ids,
        filter_series=None,
        unique_ids_filter=None,
    ):
        T, N = original_data_wide.shape

        reconst_wide = self._predict_loop(cvae, gen_data, T, N, use_reconstruction=True)
        reconst_wide = scaler.inverse_transform(reconst_wide)
        X_hat_long = self.create_dataset_long_form(
            reconst_wide, original_data_long, unique_ids=unique_ids
        )
        if filter_series:
            X_hat_long = X_hat_long.loc[X_hat_long["unique_id"].isin(unique_ids_filter)]

        return X_hat_long
