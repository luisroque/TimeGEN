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
from hitgen.feature_engineering.feature_transformations import (
    detemporalize,
)
from hitgen.model.models import (
    get_CVAE,
    KLScheduleCallback,
    CVAE,
    build_tf_dataset,
    build_forecast_dataset,
    build_tf_dataset_multivariate,
)
from hitgen.metrics.evaluation_metrics import (
    compute_discriminative_score,
    compute_downstream_forecast,
)
from hitgen.load_data.config import DATASETS
from hitgen.visualization.model_visualization import (
    plot_generated_vs_original,
)


class InvalidFrequencyError(Exception):
    pass


class HiTGenPipeline:
    """
    Class for creating transformed versions of the dataset using a Conditional Variational Autoencoder (CVAE).
    """

    _instance = None

    def __new__(cls, *args, **kwargs) -> "HiTGenPipeline":
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
        self.unique_ids = np.sort(self.df["unique_id"].unique())

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

        self.best_params = {}
        self.best_params_forecasting = {}
        self.best_params_multivar = {}

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

        # trainval data
        self.original_trainval_wide = feature_dict["trainval_wide"]
        self.original_trainval_long = feature_dict["trainval_long"]
        self.original_trainval_long_transf = feature_dict[
            "original_trainval_long_transf"
        ]
        self.original_trainval_wide_transf = feature_dict[
            "original_trainval_wide_transf"
        ]

        # test data
        self.original_test_wide = feature_dict["test_wide"]
        self.original_test_long = feature_dict["test_long"]
        self.original_test_long_transf = feature_dict["original_test_long_transf"]
        self.original_test_wide_transf = feature_dict["original_test_wide_transf"]

        # mask data
        self.mask_wide = feature_dict["mask_wide"]
        self.mask_train_wide = feature_dict["mask_train_wide"]
        self.mask_val_wide = feature_dict["mask_val_wide"]
        self.mask_trainval_wide = feature_dict["mask_trainval_wide"]
        self.mask_test_wide = feature_dict["mask_test_wide"]

        # fourier features
        self.original_dyn_features = feature_dict["fourier_features_original"]
        self.train_dyn_features = feature_dict["fourier_features_train"]
        self.val_dyn_features = feature_dict["fourier_features_val"]
        self.trainval_dyn_features = feature_dict["fourier_features_trainval"]
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

    def create_dataset_long_form(
        self,
        data: pd.DataFrame,
        original: pd.DataFrame,
        unique_ids: List = None,
        ds: pd.DatetimeIndex = None,
    ) -> pd.DataFrame:
        df = pd.DataFrame(data)
        df.sort_index(ascending=True, inplace=True)

        if unique_ids is None:
            df.columns = self.long_properties["unique_id"]
            df["ds"] = self.long_properties["ds"]
        else:
            df.columns = unique_ids
            df["ds"] = ds

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
    ) -> (List, List, List):
        """Load split from file if it exists, otherwise create and save a new split."""
        if os.path.exists(self.split_path):
            with open(self.split_path, "r") as f:
                split_data = json.load(f)
                return (
                    np.array(split_data["train_ids"]).tolist(),
                    np.array(split_data["val_ids"]).tolist(),
                    np.array(split_data["test_ids"]).tolist(),
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

        return train_ids.tolist(), val_ids.tolist(), test_ids.tolist()

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
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(dates)

        try:
            index_func, period = self.freq_map[self.freq]
        except KeyError:
            # for weekly, we might have something like "W-THU", so let's handle that generically
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
        self.scaler_trainval = StandardScaler()
        self.scaler_test = StandardScaler()

        original_wide_transf = pd.DataFrame(
            self.scaler.fit_transform(original_wide_filled),
            index=original_wide_filled.index,
            columns=original_wide_filled.columns,
        )

        original_long = self.create_dataset_long_form(original_wide, self.df)
        self.ds_original = pd.DatetimeIndex(original_long["ds"].unique()).sort_values()
        self.unique_ids_original = sorted(original_long["unique_id"].unique())
        original_long_transf = self.create_dataset_long_form(
            original_wide_transf,
            self.df,
            unique_ids=self.unique_ids_original,
            ds=self.ds_original,
        )

        mask_original_long = self.create_dataset_long_form(
            mask_original_wide,
            self.df,
            unique_ids=self.unique_ids_original,
            ds=self.ds_original,
        )

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
        self.ds_train = pd.DatetimeIndex(train_long["ds"].unique()).sort_values()
        self.unique_ids_train = sorted(train_long["unique_id"].unique())

        mask_train_long = mask_original_long[
            mask_original_long["unique_id"].isin(self.train_ids)
        ]

        # convert training long -> wide
        train_wide = self._create_dataset_wide_form(
            data_long=train_long, ids=self.train_ids, full_dates=self.ds_train
        )
        train_wide_transf = pd.DataFrame(
            self.scaler_train.fit_transform(train_wide),
            index=train_wide.index,
            columns=train_wide.columns,
        )
        mask_train_wide = self._create_dataset_wide_form(
            data_long=mask_train_long, ids=self.train_ids, full_dates=self.ds_train
        )
        train_long_transf = self.create_dataset_long_form(
            train_wide_transf, self.df, unique_ids=self.train_ids, ds=self.ds_train
        )

        ### VALIDATION DATA ###
        val_long = original_long[original_long["unique_id"].isin(self.val_ids)]
        self.ds_val = pd.DatetimeIndex(val_long["ds"].unique()).sort_values()
        self.unique_ids_val = sorted(val_long["unique_id"].unique())

        mask_val_long = mask_original_long[
            mask_original_long["unique_id"].isin(self.val_ids)
        ]

        # convert validation long -> wide
        val_wide = self._create_dataset_wide_form(
            data_long=val_long, ids=self.val_ids, full_dates=self.ds_val
        )
        val_wide_transf = pd.DataFrame(
            self.scaler_val.fit_transform(val_wide),
            index=val_wide.index,
            columns=val_wide.columns,
        )
        mask_val_wide = self._create_dataset_wide_form(
            data_long=mask_val_long, ids=self.val_ids, full_dates=self.ds_val
        )
        val_long_transf = self.create_dataset_long_form(
            val_wide_transf, self.df, unique_ids=self.val_ids, ds=self.ds_val
        )

        ### TRAIN + VALIDATION DATA ###
        trainval_ids = self.train_ids + self.val_ids

        trainval_long = original_long[original_long["unique_id"].isin(trainval_ids)]
        self.ds_trainval = pd.DatetimeIndex(trainval_long["ds"].unique()).sort_values()
        self.unique_ids_trainval = sorted(trainval_long["unique_id"].unique())
        mask_trainval_long = mask_original_long[
            mask_original_long["unique_id"].isin(trainval_ids)
        ]

        # convert validation long -> wide
        trainval_wide = self._create_dataset_wide_form(
            data_long=trainval_long,
            ids=trainval_ids,
            full_dates=self.ds_trainval,
        )
        trainval_wide_transf = pd.DataFrame(
            self.scaler_trainval.fit_transform(trainval_wide),
            index=trainval_wide.index,
            columns=trainval_wide.columns,
        )
        mask_trainval_wide = self._create_dataset_wide_form(
            data_long=mask_trainval_long,
            ids=trainval_ids,
            full_dates=self.ds_trainval,
        )
        trainval_long_transf = self.create_dataset_long_form(
            trainval_wide_transf, self.df, unique_ids=trainval_ids, ds=self.ds_trainval
        )

        ### TESTING DATA ###
        test_long = original_long[original_long["unique_id"].isin(self.test_ids)]
        self.ds_test = pd.DatetimeIndex(test_long["ds"].unique()).sort_values()
        self.unique_ids_test = sorted(test_long["unique_id"].unique())

        mask_test_long = mask_original_long[
            mask_original_long["unique_id"].isin(self.test_ids)
        ]

        # convert testing long -> wide
        test_wide = self._create_dataset_wide_form(
            data_long=test_long, ids=self.test_ids, full_dates=self.ds_test
        )
        test_wide_transf = pd.DataFrame(
            self.scaler_test.fit_transform(test_wide),
            index=test_wide.index,
            columns=test_wide.columns,
        )
        mask_test_wide = self._create_dataset_wide_form(
            data_long=mask_test_long, ids=self.test_ids, full_dates=self.ds_test
        )
        test_long_transf = self.create_dataset_long_form(
            test_wide_transf, self.df, unique_ids=self.test_ids, ds=self.ds_test
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
        self.X_trainval_raw = trainval_wide.reset_index(drop=True)
        self.X_test_raw = test_wide.reset_index(drop=True)
        self.X_orig_raw = original_wide.reset_index(drop=True)

        # compute Fourier features
        fourier_features_train = self.compute_fourier_features(train_wide.index)
        fourier_features_val = self.compute_fourier_features(val_wide.index)
        fourier_features_trainval = self.compute_fourier_features(trainval_wide.index)
        fourier_features_test = self.compute_fourier_features(test_wide.index)
        fourier_features_original = self.compute_fourier_features(original_wide.index)

        return {
            # wide Data
            "original_wide": original_wide,
            "train_wide": train_wide,
            "val_wide": val_wide,
            "trainval_wide": trainval_wide,
            "test_wide": test_wide,
            # long Data
            "original_long": original_long,
            "train_long": train_long,
            "val_long": val_long,
            "trainval_long": trainval_long,
            "test_long": test_long,
            # mask Wide
            "mask_train_wide": mask_train_wide,
            "mask_val_wide": mask_val_wide,
            "mask_trainval_wide": mask_trainval_wide,
            "mask_test_wide": mask_test_wide,
            "mask_wide": mask_original_wide,
            # transformed Long Data
            "original_long_transf": original_long_transf,
            "original_train_long_transf": train_long_transf,
            "original_val_long_transf": val_long_transf,
            "original_trainval_long_transf": trainval_long_transf,
            "original_test_long_transf": test_long_transf,
            # wide Transformed Data
            "original_wide_transf": original_wide_transf,
            "original_train_wide_transf": train_wide_transf,
            "original_val_wide_transf": val_wide_transf,
            "original_trainval_wide_transf": trainval_wide_transf,
            "original_test_wide_transf": test_wide_transf,
            # fourier Features
            "fourier_features_train": fourier_features_train,
            "fourier_features_val": fourier_features_val,
            "fourier_features_trainval": fourier_features_trainval,
            "fourier_features_test": fourier_features_test,
            "fourier_features_original": fourier_features_original,
        }

    def build_test_holdout(
        self,
        test_long: pd.DataFrame,
        horizon: int,
        window_size: int,
    ) -> Tuple[pd.DataFrame, List]:
        """
        Create a 'test_holdout_long' DataFrame that:
          - For each test series, includes the last `window_size` historical rows.
          - Adds `horizon` new rows with future timestamps (and Fourier features),
            where `y` is set to NaN.
        """

        holdout_list = []

        for uid in self.test_ids:
            df_ser = test_long[test_long["unique_id"] == uid].copy()
            df_ser.sort_values("ds", inplace=True)

            if len(df_ser) < window_size:
                print(
                    f"[build_test_holdout] Series {uid} has < {window_size} rows. Skipping."
                )
                continue

            hist_tail = df_ser.iloc[-window_size:].copy()

            last_date = hist_tail["ds"].max()
            future_dates = pd.date_range(
                start=last_date + pd.tseries.frequencies.to_offset(self.freq),
                periods=horizon,
                freq=self.freq,
            )

            future_df = pd.DataFrame(
                {
                    "unique_id": [uid] * horizon,
                    "ds": future_dates,
                    "y": np.nan,  # predictions
                }
            )

            ffuture = self.compute_fourier_features(future_df["ds"])
            ffuture.reset_index(drop=True, inplace=True)

            future_df = pd.concat([future_df.reset_index(drop=True), ffuture], axis=1)

            # attach the fourier columns to hist_tail for the last window_size steps
            hist_ff = self.compute_fourier_features(hist_tail["ds"])
            hist_ff.reset_index(drop=True, inplace=True)
            hist_tail = pd.concat([hist_tail.reset_index(drop=True), hist_ff], axis=1)

            holdout_ser = pd.concat([hist_tail, future_df], ignore_index=True)
            holdout_list.append(holdout_ser)

        if not holdout_list:
            return pd.DataFrame(), []

        test_holdout_long = pd.concat(holdout_list, ignore_index=True)
        test_holdout_long.sort_values(["unique_id", "ds"], inplace=True)

        dyn_future_cols = ffuture.columns

        return test_holdout_long, dyn_future_cols

    def fit(self):
        weights_folder = "assets/model_weights"
        os.makedirs(weights_folder, exist_ok=True)

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

        print(f"Best Hyperparameters: {self.best_params}")

        data_mask_temporalized = build_tf_dataset(
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
            cache_dataset_name=self.dataset_name,
            cache_dataset_group=self.dataset_group,
            cache_split="original",
        )

        encoder, decoder = get_CVAE(
            window_size=self.best_params["window_size"],
            input_dim=1,  # univariate series
            latent_dim=self.best_params["latent_dim"],
            pred_dim=self.best_params["future_steps"],
            time_dist_units=self.best_params["time_dist_units"],
            kernel_size=self.best_params["kernel_size"],
            forecasting=self.best_params["forecasting"],
            n_hidden=self.best_params["n_hidden"],
        )

        cvae = CVAE(
            encoder,
            decoder,
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

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=self.best_params["patience"],
            mode="auto",
            restore_best_weights=True,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="loss", factor=0.2, patience=10, min_lr=1e-6, cooldown=3, verbose=1
        )

        weights_file = os.path.join(
            weights_folder,
            f"{self.dataset_name}_{self.dataset_group}__vae_[TNP].weights.h5",
        )
        history_file = os.path.join(
            weights_folder,
            f"{self.dataset_name}_{self.dataset_group}_training_history_[TNP].json",
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

        print("Training for [TNP] completed with the best hyperparameters.")
        return cvae, data_mask_temporalized

    def update_best_scores(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        score: float,
        latent_dim: int,
        window_size: int,
        patience: int,
        kernel_size: Tuple[int],
        n_hidden: int,
        time_dist_units: int,
        n_blocks: int,
        batch_size: int,
        n_knots: int,
        windows_batch_size: int,
        stride: int,
        coverage_fraction: float,
        prediction_mode: str,
        future_steps: int,
        epochs: int,
        learning_rate: float,
        forecasting: bool,
        loss: float,
        trial: int,
        base_dir: str,
    ) -> None:
        os.makedirs(base_dir, exist_ok=True)
        scores_path = f"{base_dir}/{self.dataset_name}_{self.dataset_group}_best_hyperparameters.jsonl"
        opt_path = f"{base_dir}/{self.dataset_name}_{self.dataset_group}_best_hyperparameters_opt.json"

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
            "kernel_size": kernel_size,
            "n_hidden": n_hidden,
            "time_dist_units": time_dist_units,
            "n_blocks": n_blocks,
            "n_knots": n_knots,
            "batch_size": batch_size,
            "windows_batch_size": windows_batch_size,
            "stride": stride,
            "coverage_fraction": coverage_fraction,
            "prediction_mode": prediction_mode,
            "future_steps": future_steps,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "forecasting": forecasting,
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
                original_data=original_data,
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
            f.write(json.dumps(opt_meta_info))

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
                store_features_synth=store_features_synth,
                split=split,
            )
            scores.append(score)

        mean_score = np.mean(scores)
        return mean_score

    def objective(self, trial):
        """
        Objective function for Optuna to tune the CVAE hyperparameters,
        evaluated using the validation set.
        """
        try:
            latent_dim = trial.suggest_int("latent_dim", 8, 96, step=8)

            if self.freq in ["M", "MS"]:
                window_size = trial.suggest_int("window_size", 6, 24, step=3)
            elif self.freq in ["Q", "QS"]:
                window_size = trial.suggest_int("window_size", 8, 16, step=2)
            elif self.freq in ["Y", "YS"]:
                window_size = trial.suggest_int("window_size", 4, 8, step=1)
            else:
                window_size = trial.suggest_int("window_size", 4, 24, step=1)

            stride = trial.suggest_int("stride", 1, window_size // 2, step=1)
            n_knots = trial.suggest_int(
                "knots", int(window_size // 2), window_size, step=1
            )
            patience = trial.suggest_int("patience", 4, 6, step=1)

            n_hidden = trial.suggest_int("n_hidden", 32, 256, step=32)
            time_dist_units = trial.suggest_int("time_dist_units", 16, 32, step=8)
            n_blocks = trial.suggest_int("n_layers", 1, 3)

            predefined_kernel_sizes = [(2, 2, 1), (1, 1, 1), (2, 1, 1), (4, 2, 1)]
            valid_kernel_sizes = [
                ks
                for ks in predefined_kernel_sizes
                if all(window_size >= k for k in ks)
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

            epochs = trial.suggest_int("epochs", 100, 750, step=25)
            learning_rate = trial.suggest_loguniform("learning_rate", 3e-5, 3e-4)

            forecasting = True

            data_mask_temporalized_train = build_tf_dataset(
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
                cache_dataset_name=self.dataset_name,
                cache_dataset_group=self.dataset_group,
                cache_split="train",
                store_dataset=False,
            )

            data_mask_temporalized_val = build_tf_dataset(
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
                cache_dataset_name=self.dataset_name,
                cache_dataset_group=self.dataset_group,
                cache_split="val",
                store_dataset=False,
            )

            encoder, decoder = get_CVAE(
                window_size=window_size,
                input_dim=1,  # univariate series
                latent_dim=latent_dim,
                pred_dim=future_steps,
                time_dist_units=time_dist_units,
                n_knots=n_knots,
                n_blocks=n_blocks,
                kernel_size=kernel_size,
                forecasting=forecasting,
                n_hidden=n_hidden,
            )

            cvae = CVAE(
                encoder=encoder,
                decoder=decoder,
                forecasting=forecasting,
            )
            cvae.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=[
                    cvae.total_loss_tracker,
                    cvae.reconstruction_loss_tracker,
                    cvae.kl_loss_tracker,
                ],
            )

            kl_callback = KLScheduleCallback(
                cvae_model=cvae,
                kl_start=0.0,
                kl_end=1.0,
                warmup_epochs=int(epochs * 0.3),
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
                patience=3,
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
                callbacks=[es, reduce_lr, kl_callback],
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
                ds=self.ds_val,
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
                kernel_size=kernel_size,
                n_hidden=n_hidden,
                time_dist_units=time_dist_units,
                n_blocks=n_blocks,
                batch_size=batch_size,
                windows_batch_size=windows_batch_size,
                stride=stride,
                n_knots=n_knots,
                coverage_fraction=coverage_fraction,
                epochs=epochs,
                learning_rate=learning_rate,
                forecasting=forecasting,
                prediction_mode=prediction_mode,
                future_steps=future_steps,
                loss=val_loss,
                trial=trial.number,
                base_dir="assets/model_weights",
            )

            # clean GPU memory between trials
            del cvae, encoder, decoder
            tf.keras.backend.clear_session()
            gc.collect()

            return score

        except Exception as e:
            print(f"Error in trial: {e}")
            raise optuna.exceptions.TrialPruned()

    def objective_multivariate(self, trial):
        """
        Objective function for Optuna to tune the CVAE hyperparameters,
        evaluated using the validation set.
        """
        try:
            latent_dim = trial.suggest_int("latent_dim", 8, 96, step=8)

            if self.freq in ["M", "MS"]:
                window_size = trial.suggest_int("window_size", 6, 24, step=3)
            elif self.freq in ["Q", "QS"]:
                window_size = trial.suggest_int("window_size", 8, 16, step=2)
            elif self.freq in ["Y", "YS"]:
                window_size = trial.suggest_int("window_size", 4, 8, step=1)
            else:
                window_size = trial.suggest_int("window_size", 4, 24, step=1)

            stride = trial.suggest_int("stride", 1, window_size // 2, step=1)
            patience = trial.suggest_int("patience", 4, 6, step=1)
            n_knots = trial.suggest_int(
                "knots", int(window_size // 2), window_size, step=1
            )

            n_hidden = trial.suggest_int("n_hidden", 32, 256, step=32)
            time_dist_units = trial.suggest_int("time_dist_units", 16, 32, step=8)
            n_blocks = trial.suggest_int("n_layers", 1, 3)

            predefined_kernel_sizes = [(2, 2, 1), (1, 1, 1), (2, 1, 1), (4, 2, 1)]
            valid_kernel_sizes = [
                ks
                for ks in predefined_kernel_sizes
                if all(window_size >= k for k in ks)
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

            epochs = trial.suggest_int("epochs", 100, 750, step=25)
            learning_rate = trial.suggest_loguniform("learning_rate", 3e-5, 3e-4)

            forecasting = True

            data_mask_temporalized_train = build_tf_dataset(
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
                cache_dataset_name=self.dataset_name,
                cache_dataset_group=self.dataset_group,
                cache_split="train",
                store_dataset=False,
            )

            data_mask_temporalized_val = build_tf_dataset(
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
                cache_dataset_name=self.dataset_name,
                cache_dataset_group=self.dataset_group,
                cache_split="val",
                store_dataset=False,
            )

            encoder, decoder = get_CVAE(
                window_size=window_size,
                input_dim=1,  # univariate series
                latent_dim=latent_dim,
                pred_dim=future_steps,
                time_dist_units=time_dist_units,
                n_blocks=n_blocks,
                n_knots=n_knots,
                kernel_size=kernel_size,
                forecasting=forecasting,
                n_hidden=n_hidden,
            )

            cvae = CVAE(
                encoder=encoder,
                decoder=decoder,
                forecasting=forecasting,
            )
            cvae.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=[
                    cvae.total_loss_tracker,
                    cvae.reconstruction_loss_tracker,
                    cvae.kl_loss_tracker,
                ],
            )

            kl_callback = KLScheduleCallback(
                cvae_model=cvae,
                kl_start=0.0,
                kl_end=1.0,
                warmup_epochs=int(epochs * 0.3),
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
                patience=3,
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
                callbacks=[es, reduce_lr, kl_callback],
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
                ds=self.ds_val,
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
                kernel_size=kernel_size,
                n_hidden=n_hidden,
                time_dist_units=time_dist_units,
                n_blocks=n_blocks,
                batch_size=batch_size,
                windows_batch_size=windows_batch_size,
                stride=stride,
                n_knots=n_knots,
                coverage_fraction=coverage_fraction,
                epochs=epochs,
                learning_rate=learning_rate,
                forecasting=forecasting,
                prediction_mode=prediction_mode,
                future_steps=future_steps,
                loss=val_loss,
                trial=trial.number,
                base_dir="assets/model_weights_multivar",
            )

            # clean GPU memory between trials
            del cvae, encoder, decoder
            tf.keras.backend.clear_session()
            gc.collect()

            return score

        except Exception as e:
            print(f"Error in trial: {e}")
            raise optuna.exceptions.TrialPruned()

    def objective_forecast(self, trial):
        """
        Objective function for Optuna to tune CVAE hyperparameters,
        specifically for multi-step forecasting, with future_steps=self.h.
        """

        try:
            latent_dim = trial.suggest_int("latent_dim", 8, 96, step=8)

            if self.freq in ["M", "MS"]:
                window_size = trial.suggest_int("window_size", 6, 24, step=3)
            elif self.freq in ["Q", "QS"]:
                window_size = trial.suggest_int("window_size", 8, 16, step=2)
            elif self.freq in ["Y", "YS"]:
                window_size = trial.suggest_int("window_size", 4, 8, step=1)
            else:
                window_size = trial.suggest_int("window_size", 4, 24, step=1)

            stride = trial.suggest_int("stride", 1, window_size // 2, step=1)
            patience = trial.suggest_int("patience", 4, 6, step=1)
            n_knots = trial.suggest_int(
                "knots", int(window_size // 2), window_size, step=1
            )

            n_hidden = trial.suggest_int("n_hidden", 32, 256, step=32)
            time_dist_units = trial.suggest_int("time_dist_units", 16, 32, step=8)
            n_blocks = trial.suggest_int("n_layers", 1, 3)

            predefined_kernel_sizes = [(2, 2, 1), (1, 1, 1), (2, 1, 1), (4, 2, 1)]
            valid_kernel_sizes = [
                ks
                for ks in predefined_kernel_sizes
                if all(window_size >= k for k in ks)
            ]
            if not valid_kernel_sizes:
                valid_kernel_sizes.append((1, 1, 1))
            kernel_size = tuple(
                trial.suggest_categorical("kernel_size", valid_kernel_sizes)
            )

            batch_size = trial.suggest_int("batch_size", 8, 24, step=8)
            windows_batch_size = trial.suggest_int("windows_batch_size", 8, 24, step=8)
            coverage_fraction = trial.suggest_float("coverage_fraction", 0.3, 0.6)

            # force multi-step
            prediction_mode = "multi_step_ahead"
            future_steps = self.h

            epochs = trial.suggest_int("epochs", 100, 750, step=25)
            learning_rate = trial.suggest_loguniform("learning_rate", 3e-5, 3e-4)

            forecasting = True

            data_mask_temporalized_train = build_tf_dataset(
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
                cache_dataset_name=self.dataset_name,
                cache_dataset_group=self.dataset_group + "_forecasting",
                cache_split="train",
                store_dataset=False,
            )

            data_mask_temporalized_val = build_tf_dataset(
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
                cache_dataset_name=self.dataset_name,
                cache_dataset_group=self.dataset_group + "_forecasting",
                cache_split="val",
                store_dataset=False,
            )

            encoder, decoder = get_CVAE(
                window_size=window_size,
                input_dim=1,  # univariate
                latent_dim=latent_dim,
                pred_dim=future_steps,
                time_dist_units=time_dist_units,
                n_blocks=n_blocks,
                n_knots=n_knots,
                kernel_size=kernel_size,
                forecasting=forecasting,
                n_hidden=n_hidden,
                n_knots=n_knots,
            )

            cvae = CVAE(encoder, decoder, forecasting=forecasting)
            cvae.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=[
                    cvae.total_loss_tracker,
                    cvae.reconstruction_loss_tracker,
                    cvae.kl_loss_tracker,
                ],
            )

            kl_callback = KLScheduleCallback(
                cvae_model=cvae,
                kl_start=0.0,
                kl_end=1.0,
                warmup_epochs=int(epochs * 0.3),
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
                patience=3,
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
                callbacks=[es, reduce_lr, kl_callback],
                validation_freq=3,
            )

            val_loss = min(history.history["loss"])

            score = val_loss

            synthetic_long = self.predict_future(cvae, window_size=window_size)

            self.update_best_scores(
                original_data=self.original_val_long,
                synthetic_data=synthetic_long,
                score=score,
                latent_dim=latent_dim,
                window_size=window_size,
                patience=patience,
                kernel_size=kernel_size,
                n_hidden=n_hidden,
                time_dist_units=time_dist_units,
                n_blocks=n_blocks,
                batch_size=batch_size,
                windows_batch_size=windows_batch_size,
                stride=stride,
                n_knots=n_knots,
                coverage_fraction=coverage_fraction,
                epochs=epochs,
                learning_rate=learning_rate,
                forecasting=forecasting,
                prediction_mode=prediction_mode,
                future_steps=future_steps,
                loss=val_loss,
                trial=trial.number,
                base_dir="assets/model_weights_forecasting",
            )

            # clean GPU memory
            del cvae, encoder, decoder
            tf.keras.backend.clear_session()
            gc.collect()

            return score

        except Exception as e:
            print(f"Error in objective_forecast trial: {e}")
            raise optuna.exceptions.TrialPruned()

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
                best_params = json.load(f)

            self.best_params = best_params["best_score"][0]

        print(f"Best Hyperparameters: {self.best_params}")

        data_mask_temporalized = build_tf_dataset(
            data=self.original_trainval_wide_transf,
            mask=self.mask_trainval_wide,
            dyn_features=self.trainval_dyn_features,
            window_size=self.best_params["window_size"],
            batch_size=self.best_params["batch_size"],
            windows_batch_size=self.best_params["windows_batch_size"],
            coverage_mode="systematic",
            stride=1,
            prediction_mode=self.best_params["prediction_mode"],
            future_steps=self.best_params["future_steps"],
            cache_dataset_name=self.dataset_name,
            cache_dataset_group=self.dataset_group,
            cache_split="trainval",
        )

        encoder, decoder = get_CVAE(
            window_size=self.best_params["window_size"],
            input_dim=1,  # univariate series
            latent_dim=self.best_params["latent_dim"],
            pred_dim=self.best_params["future_steps"],
            n_knots=self.best_params["n_knots"],
            time_dist_units=self.best_params["time_dist_units"],
            kernel_size=self.best_params["kernel_size"],
            forecasting=self.best_params["forecasting"],
            n_hidden=self.best_params["n_hidden"],
        )

        cvae = CVAE(
            encoder,
            decoder,
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

    def hyper_tune_and_train_multivariate(self, n_trials=100):
        """
        Run Optuna hyperparameter tuning for the CVAE and train the best model.
        """
        study_name = f"{self.dataset_name}_{self.dataset_group}_opt_vae_multivar"

        storage = "sqlite:///optuna_study.db"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            load_if_exists=True,
        )

        weights_folder = "assets/model_weights_multivar"
        os.makedirs(weights_folder, exist_ok=True)

        best_params_meta_opt_file = os.path.join(
            weights_folder,
            f"{self.dataset_name}_{self.dataset_group}_best_hyperparameters_opt.json",
        )

        try:
            with open(best_params_meta_opt_file, "r") as f:
                best_params = json.load(f)
                self.best_params_multivar = best_params["best_score"][0]
                print("Best Multivar params file found. Loading...")
        except (FileNotFoundError, json.JSONDecodeError):
            print(
                f"Error loading best Multivar parameters file {best_params_meta_opt_file}. Proceeding with optimization..."
            )
            study.optimize(self.objective_multivariate, n_trials=n_trials)
            with open(
                best_params_meta_opt_file,
                "r",
            ) as f:
                best_params = json.load(f)

            self.best_params_multivar = best_params["best_score"][0]

        print(f"Best Multivar Hyperparameters: {self.best_params_multivar}")

        data_mask_temporalized = build_tf_dataset_multivariate(
            data=self.original_trainval_wide_transf,
            mask=self.mask_trainval_wide,
            dyn_features=self.trainval_dyn_features,
            window_size=self.best_params_multivar["window_size"],
            batch_size=self.best_params_multivar["batch_size"],
            windows_batch_size=self.best_params_multivar["windows_batch_size"],
            coverage_mode="systematic",
            stride=1,
            prediction_mode=self.best_params_multivar["prediction_mode"],
            future_steps=self.best_params_multivar["future_steps"],
            cache_dataset_name=self.dataset_name,
            cache_dataset_group=self.dataset_group + "_multivar",
            cache_split="trainval",
        )

        encoder, decoder = get_CVAE(
            window_size=self.best_params_multivar["window_size"],
            input_dim=1,  # univariate series
            latent_dim=self.best_params_multivar["latent_dim"],
            pred_dim=self.best_params_multivar["future_steps"],
            n_knots=self.best_params_multivar["n_knots"],
            time_dist_units=self.best_params_multivar["time_dist_units"],
            kernel_size=self.best_params_multivar["kernel_size"],
            forecasting=self.best_params_multivar["forecasting"],
            n_hidden=self.best_params_multivar["n_hidden"],
        )

        cvae = CVAE(
            encoder,
            decoder,
            forecasting=self.best_params_multivar["forecasting"],
        )
        cvae.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.best_params_multivar["learning_rate"]
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
            patience=self.best_params_multivar["patience"],
            mode="auto",
            restore_best_weights=True,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="loss", factor=0.2, patience=10, min_lr=1e-6, cooldown=3, verbose=1
        )

        weights_file = os.path.join(
            weights_folder, f"{self.dataset_name}_{self.dataset_group}__vae.weights.h5"
        )
        history_file = os.path.join(
            weights_folder,
            f"{self.dataset_name}_{self.dataset_group}_training_history.json",
        )
        history = None

        dummy_input = (
            tf.random.normal(
                (1, self.best_params_multivar["window_size"], 1)
            ),  # batch_data
            tf.ones((1, self.best_params_multivar["window_size"], 1)),  # batch_mask
            tf.random.normal(
                (1, self.best_params_multivar["window_size"], 6)
            ),  # batch_dyn_features
            tf.random.normal(
                (1, self.best_params_multivar["future_steps"], 1)
            ),  # pred_mask
        )
        _ = cvae(dummy_input)

        if os.path.exists(weights_file):
            print("Loading existing Multivar weights...")
            cvae.load_weights(weights_file)

            if os.path.exists(history_file):
                print("Loading training Multivar history...")
                with open(history_file, "r") as f:
                    history = json.load(f)
            else:
                print("No history Multivar file found. Skipping history loading.")
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
                epochs=self.best_params_multivar["epochs"],
                batch_size=self.best_params_multivar["batch_size"],
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

        print("Training Multivar completed with the best hyperparameters.")
        return cvae

    def hyper_tune_and_train_forecasting(self, n_trials=100):
        """
        Run Optuna hyperparameter tuning for the CVAE purely for multi-step forecasting,
        then train the best model.
        """
        import optuna

        study_name = f"{self.dataset_name}_{self.dataset_group}_opt_vae_forecasting"
        storage = "sqlite:///optuna_study.db"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            load_if_exists=True,
        )

        weights_folder = "assets/model_weights_forecasting"
        os.makedirs(weights_folder, exist_ok=True)

        best_params_meta_opt_file = os.path.join(
            weights_folder,
            f"{self.dataset_name}_{self.dataset_group}_best_hyperparameters_opt.json",
        )

        try:
            with open(best_params_meta_opt_file, "r") as f:
                best_params = json.load(f)
                self.best_params_forecasting = best_params["best_score"][0]
                print("Best forecast params file found. Loading...")
        except (FileNotFoundError, json.JSONDecodeError):
            print(
                f"No best forecast params found in {best_params_meta_opt_file}. Proceeding with optimization..."
            )
            study.optimize(self.objective_forecast, n_trials=n_trials)

            with open(best_params_meta_opt_file, "r") as f:
                best_params = json.load(f)
            self.best_params_forecasting = best_params["best_score"][0]

        print(f"Best Forecasting Hyperparameters: {self.best_params_forecasting}")

        data_mask_temporalized = build_tf_dataset(
            data=self.original_trainval_wide_transf,
            mask=self.mask_trainval_wide,
            dyn_features=self.trainval_dyn_features,
            window_size=self.best_params_forecasting["window_size"],
            batch_size=self.best_params_forecasting["batch_size"],
            windows_batch_size=self.best_params_forecasting["windows_batch_size"],
            coverage_mode="systematic",
            stride=1,
            prediction_mode="multi_step_ahead",
            future_steps=self.h,
            cache_dataset_name=self.dataset_name,
            cache_dataset_group=self.dataset_group + "_forecasting",
            cache_split="trainval",
        )

        encoder, decoder = get_CVAE(
            window_size=self.best_params_forecasting["window_size"],
            input_dim=1,
            latent_dim=self.best_params_forecasting["latent_dim"],
            pred_dim=self.best_params_forecasting["future_steps"],
            n_knots=self.best_params_forecasting["n_knots"],
            time_dist_units=self.best_params_forecasting["time_dist_units"],
            kernel_size=self.best_params_forecasting["kernel_size"],
            forecasting=True,
            n_hidden=self.best_params_forecasting["n_hidden"],
        )

        cvae = CVAE(encoder, decoder, forecasting=True)
        cvae.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.best_params_forecasting["learning_rate"]
            ),
            metrics=[
                cvae.total_loss_tracker,
                cvae.reconstruction_loss_tracker,
                cvae.kl_loss_tracker,
            ],
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=self.best_params_forecasting["patience"],
            mode="auto",
            restore_best_weights=True,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="loss", factor=0.2, patience=10, min_lr=1e-6, cooldown=3, verbose=1
        )

        weights_file = os.path.join(
            weights_folder,
            f"{self.dataset_name}_{self.dataset_group}__vae.weights.h5",
        )
        history_file = os.path.join(
            weights_folder,
            f"{self.dataset_name}_{self.dataset_group}_training_history.json",
        )

        dummy_input = (
            tf.random.normal((1, self.best_params_forecasting["window_size"], 1)),
            tf.ones((1, self.best_params_forecasting["window_size"], 1)),
            tf.random.normal((1, self.best_params_forecasting["window_size"], 6)),
            tf.random.normal((1, self.best_params_forecasting["future_steps"], 1)),
        )
        _ = cvae(dummy_input)

        if os.path.exists(weights_file):
            print("Loading existing forecasting weights...")
            cvae.load_weights(weights_file)

            if os.path.exists(history_file):
                print("Loading training history for forecasting...")
                with open(history_file, "r") as f:
                    history = json.load(f)
            else:
                print("No forecasting history file found. Skipping load.")
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
                epochs=self.best_params_forecasting["epochs"],
                batch_size=self.best_params_forecasting["batch_size"],
                shuffle=False,
                callbacks=[early_stopping, mc, reduce_lr],
            )

            if history is not None:
                history_dict = {
                    key: [float(val) for val in values]
                    for key, values in history.history.items()
                }
                with open(history_file, "w") as f:
                    json.dump(history_dict, f)

        print(
            "Training for multi-step forecasting completed with best hyperparameters."
        )
        return cvae

    @staticmethod
    def _predict_loop(
        cvae: keras.Model,
        dataset: tf.data.Dataset,
        T: int,
        N: int,
        use_reconstruction: bool = True,
    ) -> np.ndarray:
        pred_tuple = cvae.predict(dataset, verbose=0)

        recon_out, z_mean_out, z_log_var_out, forecast_out = pred_tuple
        predictions = recon_out if use_reconstruction else forecast_out

        # gather big_meta in same order => second pass
        # make sure that there was no shuffle (systematic does not shuffle!)
        meta_list = []
        for _, _, batch_meta in dataset:
            meta_list.append(batch_meta.numpy())
        big_meta = np.concatenate(meta_list, axis=0)

        reconst_wide = detemporalize(predictions, big_meta, T, N)
        return reconst_wide

    def predict(
        self,
        cvae: keras.Model,
        gen_data: tf.data.Dataset,
        scaler: StandardScaler,
        original_data_wide: pd.DataFrame,
        original_data_long: pd.DataFrame,
        unique_ids: List,
        ds: pd.DatetimeIndex,
        filter_series: bool = None,
        unique_ids_filter: List = None,
    ):
        T, N = original_data_wide.shape

        reconst_wide = self._predict_loop(cvae, gen_data, T, N, use_reconstruction=True)
        reconst_wide = scaler.inverse_transform(reconst_wide)
        X_hat_long = self.create_dataset_long_form(
            reconst_wide, original_data_long, unique_ids=unique_ids, ds=ds
        )
        if filter_series:
            X_hat_long = X_hat_long.loc[X_hat_long["unique_id"].isin(unique_ids_filter)]

        return X_hat_long

    def _predict_random_loop(
        self,
        cvae: keras.Model,
        dataset: tf.data.Dataset,
        T: int,
        N: int,
        use_reconstruction: bool = True,
    ) -> np.ndarray:
        """
        Generate predictions by sampling z ~ N(0,I) from the prior, ignoring the encoder.
        This yields more 'global' diversity, but no guidance from the input data.
        """

        all_predictions = []
        all_metas = []

        for (
            (batch_data, batch_mask, batch_dyn, batch_pred_mask),
            _,
            batch_meta,
        ) in dataset:

            latent_dim = self.best_params["latent_dim"]
            window_size = self.best_params["window_size"]

            z_random = tf.random.normal(
                shape=(tf.shape(batch_data)[0], window_size, latent_dim)
            )

            recon_out, forecast_out = cvae.decoder(
                [z_random, batch_mask, batch_dyn, batch_pred_mask], training=False
            )

            chosen_out = recon_out if use_reconstruction else forecast_out

            all_predictions.append(chosen_out)
            all_metas.append(batch_meta)

        all_predictions = tf.concat(all_predictions, axis=0).numpy()
        all_metas = tf.concat(all_metas, axis=0).numpy()

        reconst_wide = detemporalize(all_predictions, all_metas, T, N)
        return reconst_wide

    def predict_random_latent(
        self,
        cvae: keras.Model,
        gen_data: tf.data.Dataset,
        scaler: StandardScaler,
        original_data_wide: pd.DataFrame,
        original_data_long: pd.DataFrame,
        unique_ids: List,
        ds: pd.DatetimeIndex,
        filter_series: bool = None,
        unique_ids_filter: List = None,
    ) -> pd.DataFrame:
        """
        Public method that calls _predict_random_loop, then transforms outputs to your usual wide/long format.
        """
        T, N = original_data_wide.shape

        reconst_wide = self._predict_random_loop(
            cvae, gen_data, T, N, use_reconstruction=True
        )

        reconst_wide = scaler.inverse_transform(reconst_wide)

        X_hat_long = self.create_dataset_long_form(
            reconst_wide, original_data_long, unique_ids=unique_ids, ds=ds
        )
        if filter_series:
            X_hat_long = X_hat_long.loc[X_hat_long["unique_id"].isin(unique_ids_filter)]

        return X_hat_long

    @staticmethod
    def _predict_guided_extra_noise_loop(
        cvae: keras.Model,
        dataset: tf.data.Dataset,
        T: int,
        N: int,
        noise_scale: float = 1.0,
        use_reconstruction: bool = True,
    ) -> np.ndarray:
        """
        Like _predict_loop, but we manually do:
          z_mean, z_log_var = encoder(...)
          z = z_mean + noise_scale * exp(0.5 * z_log_var) * epsilon
        so that we can inject more diversity
        """

        all_predictions = []
        all_metas = []

        for (
            (batch_data, batch_mask, batch_dyn, batch_pred_mask),
            _,
            batch_meta,
        ) in dataset:
            z_mean, z_log_var, _ = cvae.encoder(
                [batch_data, batch_mask, batch_dyn], training=False
            )

            eps = tf.random.normal(tf.shape(z_mean))
            z_sample = z_mean + noise_scale * tf.exp(0.5 * z_log_var) * eps

            recon_out, forecast_out = cvae.decoder(
                [z_sample, batch_mask, batch_dyn, batch_pred_mask], training=False
            )

            chosen_out = recon_out if use_reconstruction else forecast_out

            all_predictions.append(chosen_out)
            all_metas.append(batch_meta)

        all_predictions = tf.concat(all_predictions, axis=0).numpy()
        all_metas = tf.concat(all_metas, axis=0).numpy()

        reconst_wide = detemporalize(all_predictions, all_metas, T, N)
        return reconst_wide

    def predict_guided_with_extra_noise(
        self,
        cvae: keras.Model,
        gen_data: tf.data.Dataset,
        scaler: StandardScaler,
        original_data_wide: pd.DataFrame,
        original_data_long: pd.DataFrame,
        unique_ids: List,
        ds: pd.DatetimeIndex,
        noise_scale: float = 2.0,
        filter_series: bool = None,
        unique_ids_filter: List = None,
    ) -> pd.DataFrame:
        """
        Generate data that is guided by the new input (through the encoder)
        but adds extra noise for diversity.
        """
        T, N = original_data_wide.shape

        reconst_wide = self._predict_guided_extra_noise_loop(
            cvae, gen_data, T, N, noise_scale=noise_scale, use_reconstruction=True
        )
        reconst_wide = scaler.inverse_transform(reconst_wide)

        X_hat_long = self.create_dataset_long_form(
            reconst_wide, original_data_long, unique_ids=unique_ids, ds=ds
        )
        if filter_series:
            X_hat_long = X_hat_long.loc[X_hat_long["unique_id"].isin(unique_ids_filter)]

        return X_hat_long

    def predict_future(
        self,
        cvae: keras.Model,
        window_size: int = None,
    ) -> pd.DataFrame:
        """
          - For each test series, we take the last 'window_size' points as input,
            plus the next 'horizon' dynamic features.
          - We pass them into the CVAE => get forecast_out for each series.
          - We build a DataFrame that merges y_true with y_pred.

        Returns:
          DataFrame [unique_id, ds, y, y_pred]
          for the final horizon steps of each series.
        """
        if window_size is None:
            window_size = self.best_params_forecasting["window_size"]

        holdout_df, dyn_feature_cols = self.build_test_holdout(
            test_long=self.original_test_long,
            horizon=self.h,
            window_size=window_size,
        )

        forecast_dataset, meta_list = build_forecast_dataset(
            holdout_df=holdout_df,
            unique_ids=self.test_ids,
            lookback_window=window_size,
            horizon=self.h,
            dyn_feature_cols=dyn_feature_cols,
        )
        if forecast_dataset is None:
            print("No valid data to forecast.")
            return pd.DataFrame()

        pred_tuple = cvae.predict(forecast_dataset, verbose=0)
        recon_out, z_mean_out, z_log_var_out, forecast_out = pred_tuple

        # forecast_out => [num_series, horizon, 1]
        forecast_out = self.scaler_test.inverse_transform(forecast_out.squeeze(-1).T).T[
            :, :, None
        ]

        # align the forecast to the real data
        # create a new DataFrame with [unique_id, ds, y_pred] and then we merge on the real y
        # each row in 'meta_list' corresponds to forecast_out[i, ...]

        results = []
        for i, uid in enumerate(meta_list):
            y_pred = forecast_out[i, :, 0]  # [horizon]

            df_ser = holdout_df.loc[holdout_df["unique_id"] == uid].sort_values("ds")
            # the last horizon points we are predicting
            fut_part = df_ser.iloc[-self.h :] if self.h > 0 else pd.DataFrame()

            if not fut_part.empty:
                subres = fut_part.copy()
                subres["y_pred"] = y_pred
                subres = subres[["unique_id", "ds", "y", "y_pred"]]
                subres = subres.rename(columns={"y": "y_true"})
                subres = subres.rename(columns={"y_pred": "y"})
                results.append(subres)

        if not results:
            return pd.DataFrame()

        df_forecast = pd.concat(results, ignore_index=True)
        return df_forecast
