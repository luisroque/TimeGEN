import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import json
from sklearn.preprocessing import StandardScaler
from hitgen.load_data.config import DATASETS


class InvalidFrequencyError(Exception):
    pass


class DataPipeline:
    """
    Class for creating transformed versions of the dataset using a Conditional Variational Autoencoder (CVAE).
    """

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
        window_size: int = 24,
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
        self.window_size = window_size

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

        self._feature_engineering_basic_forecast()
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

    @staticmethod
    def _skip_short_series(df_long: pd.DataFrame, min_length: int) -> pd.DataFrame:
        """
        keep only the series that have >= min_length points.
        """
        lengths = df_long.groupby("unique_id")["ds"].count()

        valid_ids = lengths[lengths >= min_length].index

        df_filtered = df_long[df_long["unique_id"].isin(valid_ids)].copy()
        return df_filtered

    def _feature_engineering(
        self, train_test_split=0.7, val_split=0.15
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply preprocessing to raw time series, split into training, validation, and testing,
        compute periodic Fourier features, and return all relevant DataFrames.
        """
        min_length = self.window_size + 2 * self.h

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
        train_long = self._skip_short_series(train_long, min_length)

        self.ds_train = pd.DatetimeIndex(train_long["ds"].unique()).sort_values()
        self.unique_ids_train = sorted(train_long["unique_id"].unique())

        # convert training long -> wide
        train_wide = self._create_dataset_wide_form(
            data_long=train_long, ids=self.train_ids, full_dates=self.ds_train
        )
        train_wide_transf = pd.DataFrame(
            self.scaler_train.fit_transform(train_wide),
            index=train_wide.index,
            columns=train_wide.columns,
        )
        train_long_transf = self.create_dataset_long_form(
            train_wide_transf, self.df, unique_ids=self.train_ids, ds=self.ds_train
        )

        ### VALIDATION DATA ###
        val_long = original_long[original_long["unique_id"].isin(self.val_ids)]
        val_long = self._skip_short_series(val_long, min_length)

        self.ds_val = pd.DatetimeIndex(val_long["ds"].unique()).sort_values()
        self.unique_ids_val = sorted(val_long["unique_id"].unique())

        # convert validation long -> wide
        val_wide = self._create_dataset_wide_form(
            data_long=val_long, ids=self.val_ids, full_dates=self.ds_val
        )
        val_wide_transf = pd.DataFrame(
            self.scaler_val.fit_transform(val_wide),
            index=val_wide.index,
            columns=val_wide.columns,
        )
        val_long_transf = self.create_dataset_long_form(
            val_wide_transf, self.df, unique_ids=self.val_ids, ds=self.ds_val
        )

        ### TRAIN + VALIDATION DATA ###
        trainval_ids = self.train_ids + self.val_ids

        trainval_long = original_long[original_long["unique_id"].isin(trainval_ids)]
        trainval_long = self._skip_short_series(trainval_long, min_length)

        self.ds_trainval = pd.DatetimeIndex(trainval_long["ds"].unique()).sort_values()
        self.unique_ids_trainval = sorted(trainval_long["unique_id"].unique())

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

        trainval_long_transf = self.create_dataset_long_form(
            trainval_wide_transf, self.df, unique_ids=trainval_ids, ds=self.ds_trainval
        )

        ### TESTING DATA ###
        test_long = original_long[original_long["unique_id"].isin(self.test_ids)]
        test_long = self._skip_short_series(test_long, min_length)

        self.ds_test = pd.DatetimeIndex(test_long["ds"].unique()).sort_values()
        self.unique_ids_test = sorted(test_long["unique_id"].unique())

        # convert testing long -> wide
        test_wide = self._create_dataset_wide_form(
            data_long=test_long, ids=self.test_ids, full_dates=self.ds_test
        )
        test_wide_transf = pd.DataFrame(
            self.scaler_test.fit_transform(test_wide),
            index=test_wide.index,
            columns=test_wide.columns,
        )
        self.scaler_test_dict = {}
        for uid in test_wide.columns:
            X_col = test_wide[uid].values.reshape(-1, 1)
            scaler = StandardScaler().fit(X_col)
            self.scaler_test_dict[uid] = scaler

        test_long_transf = self.create_dataset_long_form(
            test_wide_transf, self.df, unique_ids=self.test_ids, ds=self.ds_test
        )

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

    @staticmethod
    def _mark_train_val_test(group, window_size, horizon):
        """
        Given one group of rows for a single unique_id,
        return a Series of labels: 'train', 'val', 'test', or 'skip'.
        """
        group = group.sort_values(by="ds")
        n = len(group)
        # we need at least window size + val_size (h) + h
        if n < window_size + 2 * horizon:
            # mark as skip for all rows in this group
            return pd.Series(["skip"] * n, index=group.index)

        test_start_idx = n - horizon
        val_start_idx = n - 2 * horizon

        labels = np.array(["train"] * n)  # default
        labels[val_start_idx:test_start_idx] = "val"
        labels[test_start_idx:] = "test"
        return pd.Series(labels, index=group.index)

    def _feature_engineering_basic_forecast(self):
        df = self.df.sort_values(by=["unique_id", "ds"]).copy()

        # mark each row with its fold (train, val, test, or skip)
        df["fold"] = df.groupby("unique_id", group_keys=False).apply(
            lambda g: self._mark_train_val_test(g, self.window_size, self.h)
        )

        df = df[df["fold"] != "skip"]

        train_long = df[df["fold"] == "train"]
        val_long = df[df["fold"] == "val"]
        test_long = df[df["fold"] == "test"]
        trainval_long = df[df["fold"] != "test"]

        self.original_train_long_basic_forecast = train_long
        self.original_val_long_basic_forecast = val_long
        self.original_test_long_basic_forecast = test_long
        self.original_trainval_long_basic_forecast = trainval_long
