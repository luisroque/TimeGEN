import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from pathlib import Path
from typing import Dict, Tuple
import joblib
import json
from pandas.api.types import CategoricalDtype
from timegen.load_data.config import DATASETS


def build_mixed_trainval(pipelines, dataset_source, dataset_group):
    """
    Returns a single long-form dataframe with all series, prefixing
    each unique_id with the dataset name to avoid clashes.
    """
    long_frames = []
    for dp in pipelines:
        lf = dp.original_trainval_long.copy()
        lf["unique_id"] = f"{dp.dataset_name}_{dp.dataset_group}::" + lf[
            "unique_id"
        ].astype(str)
        long_frames.append(lf)

    mixed = pd.concat(long_frames, ignore_index=True)

    num_series = mixed["unique_id"].nunique()
    avg_time_points = mixed.groupby("unique_id", observed=True).size().mean()

    print(f"Dataset Summary for {dataset_source} ({dataset_group}):")
    print(f"   - Total number of time series: {num_series}")
    print(f"   - Average number of time points per series: {avg_time_points:.2f}")
    return mixed.sort_values(["unique_id", "ds"])


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
        # self.df.asfreq(self.freq)
        self.h = horizon
        self.window_size = window_size

        num_series = self.df["unique_id"].nunique()
        avg_time_points = self.df.groupby("unique_id", observed=True).size().mean()

        print(f"Dataset Summary for {dataset_name} ({dataset_group}):")
        print(f"   - Total number of time series: {num_series}")
        print(f"   - Average number of time points per series: {avg_time_points:.2f}")

        self.features_input = (None, None, None)
        self.split_path = f"assets/model_weights/data_split/{dataset_name}_{dataset_group}_data_split.json"
        self.unique_ids = np.sort(self.df["unique_id"].unique())

        # map each frequency string to (index_function, period)
        self.freq_map = {
            "D": (self._daily_index, 365),
            "W": (self._weekly_index, 52),
            "W-SUN": (self._weekly_index, 52),
            "W-MON": (self._weekly_index, 52),
            "M": (self._monthly_index, 12),
            "MS": (self._monthly_index, 12),
            "Q": (self._quarterly_index, 4),
            "QS": (self._quarterly_index, 4),
            "Y": (self._yearly_index, 1),
            "YS": (self._yearly_index, 1),
        }
        self.period = self.freq_map[freq][1]

        self._feature_engineering_basic_forecast()
        feature_dict = self._feature_engineering()

        self.original_long = feature_dict["original_long"]
        self.original_train_long = feature_dict["train_long"]
        self.original_val_long = feature_dict["val_long"]
        self.original_trainval_long = feature_dict["trainval_long"]
        self.original_test_long = feature_dict["test_long"]

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
        ds_min = self.df["ds"].min()
        ds_max = self.df["ds"].max()

        train_ids, test_ids = self.check_series_in_train_test(
            ds_min, train_ids, test_ids
        )
        train_ids, test_ids = self.check_series_in_train_test(
            ds_max, train_ids, test_ids
        )

        return train_ids.tolist(), val_ids.tolist(), test_ids.tolist()

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
        """
        cache_dir = Path("assets/processed_datasets")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = (
            cache_dir / f"{self.dataset_name}_{self.dataset_group}_features.pkl"
        )

        if cache_path.exists():
            print("✓ Loaded cached feature‑engineering dictionary.")
            return joblib.load(cache_path)

        print("• Computing heavy feature‑engineering")
        min_length = self.window_size + 2 * self.h

        if isinstance(self.df["unique_id"].dtype, CategoricalDtype):
            self.df["unique_id"] = self.df["unique_id"].astype(str)
        self.ids = np.sort(self.df["unique_id"].unique())

        original_long = self.df.copy()

        self.ds_original = pd.DatetimeIndex(original_long["ds"].unique()).sort_values()
        self.unique_ids_original = sorted(original_long["unique_id"].unique())

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

        ### VALIDATION DATA ###
        val_long = original_long[original_long["unique_id"].isin(self.val_ids)]
        val_long = self._skip_short_series(val_long, min_length)

        self.ds_val = pd.DatetimeIndex(val_long["ds"].unique()).sort_values()
        self.unique_ids_val = sorted(val_long["unique_id"].unique())

        ### TRAIN + VALIDATION DATA ###
        trainval_ids = self.train_ids + self.val_ids

        trainval_long = original_long[original_long["unique_id"].isin(trainval_ids)]
        trainval_long = self._skip_short_series(trainval_long, min_length)

        self.ds_trainval = pd.DatetimeIndex(trainval_long["ds"].unique()).sort_values()
        self.unique_ids_trainval = sorted(trainval_long["unique_id"].unique())

        ### TESTING DATA ###
        test_long = original_long[original_long["unique_id"].isin(self.test_ids)]
        test_long = self._skip_short_series(test_long, min_length)

        self.ds_test = pd.DatetimeIndex(test_long["ds"].unique()).sort_values()
        self.unique_ids_test = sorted(test_long["unique_id"].unique())

        feature_dict = {
            # long Data
            "original_long": original_long,
            "train_long": train_long,
            "val_long": val_long,
            "trainval_long": trainval_long,
            "test_long": test_long,
            # splits/meta
            "train_ids": self.train_ids,
            "val_ids": self.val_ids,
            "test_ids": self.test_ids,
            "ds_train": self.ds_train,
            "ds_val": self.ds_val,
            "ds_trainval": self.ds_trainval,
            "ds_test": self.ds_test,
        }

        joblib.dump(feature_dict, cache_path)
        print(f"  → cached feature‑engineering dictionary at {cache_path.name}")
        return feature_dict

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
        cache_dir = Path("assets/processed_datasets")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = (
            cache_dir / f"{self.dataset_name}_{self.dataset_group}_basic_forecast.pkl"
        )

        if cache_path.exists():
            basic_dict = joblib.load(cache_path)
            print("✓ Loaded cached basic‑forecast splits.")
        else:
            df = self.df.sort_values(by=["unique_id", "ds"]).copy()

            # mark each row with its fold (train, val, test, or skip)
            df["fold"] = df.groupby("unique_id", group_keys=False, observed=True).apply(
                lambda g: self._mark_train_val_test(g, self.window_size, self.h)
            )

            df = df[df["fold"] != "skip"]

            basic_dict = {
                "train_long": df[df["fold"] == "train"],
                "val_long": df[df["fold"] == "val"],
                "test_long": df[df["fold"] == "test"],
                "trainval_long": df[df["fold"] != "test"],
            }
            joblib.dump(basic_dict, cache_path)
            print(f"  → cached basic‑forecast splits at {cache_path.name}")

        self.original_train_long_basic_forecast = basic_dict["train_long"]
        self.original_val_long_basic_forecast = basic_dict["val_long"]
        self.original_test_long_basic_forecast = basic_dict["test_long"]
        self.original_trainval_long_basic_forecast = basic_dict["trainval_long"]
