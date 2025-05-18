import fsspec
import pickle
from typing import List, Union, Optional
import pandas as pd
import numpy as np
import warnings

from timegen.model_pipeline.TimeGEN_S import TimeGEN_S
from timegen.model_pipeline.TimeGEN_M import TimeGEN_M
from timegen.model_pipeline.TimeGEN_D import TimeGEN_D
from timegen.model_pipeline.TimeGEN import TimeGEN


from neuralforecast import NeuralForecast
from neuralforecast import core as nf_core
from neuralforecast.core import (
    validate_freq,
    DataFrame,
    SparkDataFrame,
    pl_DataFrame,
    ufp,
    LocalFilesTimeSeriesDataset,
    _FilesDataset,
    get_prediction_interval_method,
    PredictionIntervals,
    DistributedConfig,
    Sequence,
)

nf_core.MODEL_FILENAME_DICT.update(
    {
        "autotimegen": TimeGEN,
        "autotimegen_s": TimeGEN_S,
        "autotimegen_m": TimeGEN_M,
        "autotimegen_d": TimeGEN_D,
    }
)


class CustomNeuralForecast(NeuralForecast):
    """
    NeuralForecast that overrides the static load() method
    to allow loading custom (user-defined) models
    """

    @staticmethod
    def load(
        path: str,
        verbose: bool = False,
        **kwargs,
    ) -> "CustomNeuralForecast":
        """
        Load a NeuralForecast object from disk, allowing for user-defined models.
        """
        if path.endswith("/"):
            path = path[:-1]

        fs, _, _ = fsspec.get_fs_token_paths(path)
        files = [f.split("/")[-1] for f in fs.ls(path) if fs.isfile(f)]

        models_ckpt = [f for f in files if f.endswith(".ckpt")]
        if len(models_ckpt) == 0:
            raise FileNotFoundError(f"No .ckpt model files found in directory: {path}")

        if verbose:
            print(10 * "-" + " Loading models " + 10 * "-")

        try:
            with fsspec.open(f"{path}/alias_to_model.pkl", "rb") as f:
                alias_to_model = pickle.load(f)
        except FileNotFoundError:
            alias_to_model = {}

        models = []
        for ckpt_file in models_ckpt:
            model_name = "_".join(ckpt_file.split("_")[:-1])
            model_class_name = alias_to_model.get(model_name, model_name)

            if model_class_name.lower() in nf_core.MODEL_FILENAME_DICT:
                cls_ = nf_core.MODEL_FILENAME_DICT[model_class_name.lower()]
            else:
                raise ValueError(
                    f"Could not find model_class_name='{model_class_name}' in Nixtla's "
                    "core models or in custom_model_dict. "
                    f"Ensure you have spelled '{model_class_name}' correctly or "
                    "provided it in custom_model_dict."
                )

            # load the actual checkpoint file
            loaded_model = cls_.load(f"{path}/{ckpt_file}", **kwargs)
            loaded_model.alias = model_name
            models.append(loaded_model)

            if verbose:
                print(f"Model '{model_name}' loaded as '{model_class_name}'.")

        dataset = None
        if verbose:
            print(10 * "-" + " Loading dataset " + 10 * "-")
        try:
            with fsspec.open(f"{path}/dataset.pkl", "rb") as f:
                dataset = pickle.load(f)
            if verbose:
                print("Dataset loaded.")
        except FileNotFoundError:
            if verbose:
                print("No dataset found in directory. (Skipping)")

        if verbose:
            print(10 * "-" + " Loading configuration " + 10 * "-")
        try:
            with fsspec.open(f"{path}/configuration.pkl", "rb") as f:
                config_dict = pickle.load(f)
            if verbose:
                print("Configuration loaded.")
        except FileNotFoundError:
            raise FileNotFoundError(
                "No configuration.pkl file found in directory. Cannot restore metadata."
            )

        freq = config_dict.get("freq", "D")
        local_scaler_type = config_dict.get("local_scaler_type", None)
        my_nf = CustomNeuralForecast(
            models=models,
            freq=freq,
            local_scaler_type=local_scaler_type,
        )

        my_nf.id_col = config_dict.get("id_col", "unique_id")
        my_nf.time_col = config_dict.get("time_col", "ds")
        my_nf.target_col = config_dict.get("target_col", "y")

        if dataset is not None:
            my_nf.dataset = dataset
            for attr in ["uids", "last_dates", "ds"]:
                setattr(my_nf, attr, config_dict.get(attr, None))

        my_nf._fitted = config_dict.get("_fitted", False)
        my_nf.scalers_ = config_dict.get("scalers_", {})

        my_nf.prediction_intervals = config_dict.get("prediction_intervals", None)
        my_nf._cs_df = config_dict.get("_cs_df", None)

        return my_nf

    def fit(
        self,
        df: Optional[Union[DataFrame, SparkDataFrame, Sequence[str]]] = None,
        static_df: Optional[Union[DataFrame, SparkDataFrame]] = None,
        val_size: Optional[int] = 0,
        use_init_models: bool = False,
        verbose: bool = False,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        distributed_config: Optional[DistributedConfig] = None,
        prediction_intervals: Optional[PredictionIntervals] = None,
    ) -> None:
        """Fit the core.NeuralForecast.

        Fit `models` to a large set of time series from DataFrame `df`.
        and store fitted models for later inspection.

        Parameters
        ----------
        df : pandas, polars or spark DataFrame, or a list of parquet files containing the series, optional (default=None)
            DataFrame with columns [`unique_id`, `ds`, `y`] and exogenous variables.
            If None, a previously stored dataset is required.
        static_df : pandas, polars or spark DataFrame, optional (default=None)
            DataFrame with columns [`unique_id`] and static exogenous.
        val_size : int, optional (default=0)
            Size of validation set.
        use_init_models : bool, optional (default=False)
            Use initial model passed when NeuralForecast object was instantiated.
        verbose : bool (default=False)
            Print processing steps.
        id_col : str (default='unique_id')
            Column that identifies each serie.
        time_col : str (default='ds')
            Column that identifies each timestep, its values can be timestamps or integers.
        target_col : str (default='y')
            Column that contains the target.
        distributed_config : neuralforecast.DistributedConfig
            Configuration to use for DDP training. Currently only spark is supported.
        prediction_intervals : PredictionIntervals, optional (default=None)
            Configuration to calibrate prediction intervals (Conformal Prediction).

        Returns
        -------
        self : NeuralForecast
            Returns `NeuralForecast` class with fitted `models`.
        """
        if (df is None) and not (hasattr(self, "dataset")):
            raise Exception("You must pass a DataFrame or have one stored.")

        # Model and datasets interactions protections
        if (
            any(model.early_stop_patience_steps > 0 for model in self.models)
            and val_size == 0
        ):
            raise Exception("Set val_size>0 if early stopping is enabled.")

        self._cs_df: Optional[DataFrame] = None
        self.prediction_intervals: Optional[PredictionIntervals] = None

        # Process and save new dataset (in self)
        if isinstance(df, (pd.DataFrame, pl_DataFrame)):
            # validate_freq(df[time_col], self.freq)
            self.dataset, self.uids, self.last_dates, self.ds = self._prepare_fit(
                df=df,
                static_df=static_df,
                predict_only=False,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
            )
            if prediction_intervals is not None:
                self.prediction_intervals = prediction_intervals
                self._cs_df = self._conformity_scores(
                    df=df,
                    id_col=id_col,
                    time_col=time_col,
                    target_col=target_col,
                    static_df=static_df,
                )

        elif isinstance(df, SparkDataFrame):
            if static_df is not None and not isinstance(static_df, SparkDataFrame):
                raise ValueError(
                    "`static_df` must be a spark dataframe when `df` is a spark dataframe."
                )
            self.dataset = self._prepare_fit_distributed(
                df=df,
                static_df=static_df,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                distributed_config=distributed_config,
            )

            if prediction_intervals is not None:
                raise NotImplementedError(
                    "Prediction intervals are not supported for distributed training."
                )

        elif isinstance(df, Sequence):
            if not all(isinstance(val, str) for val in df):
                raise ValueError(
                    "All entries in the list of files must be of type string"
                )
            self.dataset = self._prepare_fit_for_local_files(
                files_list=df,
                static_df=static_df,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
            )
            self.uids = self.dataset.indices
            self.last_dates = self.dataset.last_times

            if prediction_intervals is not None:
                raise NotImplementedError(
                    "Prediction intervals are not supported for local files."
                )

        elif df is None:
            if verbose:
                print("Using stored dataset.")
        else:
            raise ValueError(
                f"`df` must be a pandas, polars or spark DataFrame, or a list of parquet files containing the series, or `None`, got: {type(df)}"
            )

        if val_size is not None:
            if self.dataset.min_size < val_size:
                warnings.warn(
                    "Validation set size is larger than the shorter time-series."
                )

        # Recover initial model if use_init_models
        if use_init_models:
            self._reset_models()

        for i, model in enumerate(self.models):
            self.models[i] = model.fit(
                self.dataset, val_size=val_size, distributed_config=distributed_config
            )

        self._fitted = True

    def predict(
        self,
        df: Optional[Union[DataFrame, SparkDataFrame]] = None,
        freq: str = "D",
        static_df: Optional[Union[DataFrame, SparkDataFrame]] = None,
        futr_df: Optional[Union[DataFrame, SparkDataFrame]] = None,
        verbose: bool = False,
        engine=None,
        level: Optional[List[Union[int, float]]] = None,
        **data_kwargs,
    ):
        """Predict with core.NeuralForecast.

        Use stored fitted `models` to predict large set of time series from DataFrame `df`.

        Parameters
        ----------
        df : pandas, polars or spark DataFrame, optional (default=None)
            DataFrame with columns [`unique_id`, `ds`, `y`] and exogenous variables.
            If a DataFrame is passed, it is used to generate forecasts.
        static_df : pandas, polars or spark DataFrame, optional (default=None)
            DataFrame with columns [`unique_id`] and static exogenous.
        futr_df : pandas, polars or spark DataFrame, optional (default=None)
            DataFrame with [`unique_id`, `ds`] columns and `df`'s future exogenous.
        verbose : bool (default=False)
            Print processing steps.
        engine : spark session
            Distributed engine for inference. Only used if df is a spark dataframe or if fit was called on a spark dataframe.
        level : list of ints or floats, optional (default=None)
            Confidence levels between 0 and 100.
        data_kwargs : kwargs
            Extra arguments to be passed to the dataset within each model.

        Returns
        -------
        fcsts_df : pandas or polars DataFrame
            DataFrame with insample `models` columns for point predictions and probabilistic
            predictions for all fitted `models`.
        """
        if df is None and not hasattr(self, "dataset"):
            raise Exception("You must pass a DataFrame or have one stored.")

        if not self._fitted:
            raise Exception("You must fit the model before predicting.")

        needed_futr_exog = self._get_needed_futr_exog()
        if needed_futr_exog:
            if futr_df is None:
                raise ValueError(
                    f"Models require the following future exogenous features: {needed_futr_exog}. "
                    "Please provide them through the `futr_df` argument."
                )
            else:
                missing = needed_futr_exog - set(futr_df.columns)
                if missing:
                    raise ValueError(
                        f"The following features are missing from `futr_df`: {missing}"
                    )

        # distributed df or NeuralForecast instance was trained with a distributed input and no df is provided
        # we assume the user wants to perform distributed inference as well
        is_files_dataset = isinstance(getattr(self, "dataset", None), _FilesDataset)
        is_dataset_local_files = isinstance(
            getattr(self, "dataset", None), LocalFilesTimeSeriesDataset
        )
        if isinstance(df, SparkDataFrame) or (df is None and is_files_dataset):
            return self._predict_distributed(
                df=df,
                static_df=static_df,
                futr_df=futr_df,
                engine=engine,
            )

        if is_dataset_local_files and df is None:
            raise ValueError(
                "When the model has been trained on a dataset that is split between multiple files, you must pass in a specific dataframe for prediciton."
            )

        # Process new dataset but does not store it.
        if df is not None:
            validate_freq(df[self.time_col], freq)
            dataset, uids, last_dates, _ = self._prepare_fit(
                df=df,
                static_df=static_df,
                predict_only=True,
                id_col=self.id_col,
                time_col=self.time_col,
                target_col=self.target_col,
            )
        else:
            dataset = self.dataset
            uids = self.uids
            last_dates = self.last_dates
            if verbose:
                print("Using stored dataset.")

        cols = self._get_model_names()

        # Placeholder dataframe for predictions with unique_id and ds
        fcsts_df = ufp.make_future_dataframe(
            uids=uids,
            last_times=last_dates,
            freq=freq,
            h=self.h,
            id_col=self.id_col,
            time_col=self.time_col,
        )

        # Update and define new forecasting dataset
        if futr_df is None:
            futr_df = fcsts_df
        else:
            futr_orig_rows = futr_df.shape[0]
            futr_df = ufp.join(futr_df, fcsts_df, on=[self.id_col, self.time_col])
            if futr_df.shape[0] < fcsts_df.shape[0]:
                if df is None:
                    expected_cmd = "make_future_dataframe()"
                    missing_cmd = "get_missing_future(futr_df)"
                else:
                    expected_cmd = "make_future_dataframe(df)"
                    missing_cmd = "get_missing_future(futr_df, df)"
                raise ValueError(
                    "There are missing combinations of ids and times in `futr_df`.\n"
                    f"You can run the `{expected_cmd}` method to get the expected combinations or "
                    f"the `{missing_cmd}` method to get the missing combinations."
                )
            if futr_orig_rows > futr_df.shape[0]:
                dropped_rows = futr_orig_rows - futr_df.shape[0]
                warnings.warn(f"Dropped {dropped_rows:,} unused rows from `futr_df`.")
            if any(ufp.is_none(futr_df[col]).any() for col in needed_futr_exog):
                raise ValueError("Found null values in `futr_df`")
        futr_dataset = dataset.align(
            futr_df,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
        )
        self._scalers_transform(futr_dataset)
        dataset = dataset.append(futr_dataset)

        col_idx = 0
        fcsts = np.full(
            (self.h * len(uids), len(cols)), fill_value=np.nan, dtype=np.float32
        )
        for model in self.models:
            old_test_size = model.get_test_size()
            model.set_test_size(self.h)  # To predict h steps ahead
            model_fcsts = model.predict(dataset=dataset, **data_kwargs)
            # Append predictions in memory placeholder
            output_length = len(model.loss.output_names)
            fcsts[:, col_idx : col_idx + output_length] = model_fcsts
            col_idx += output_length
            model.set_test_size(old_test_size)  # Set back to original value
        if self.scalers_:
            indptr = np.append(0, np.full(len(uids), self.h).cumsum())
            fcsts = self._scalers_target_inverse_transform(fcsts, indptr)

        # Declare predictions pd.DataFrame
        cols = (
            self._get_model_names()
        )  # Needed for IQLoss as column names may have changed during the call to .predict()
        if isinstance(fcsts_df, pl_DataFrame):
            fcsts = pl_DataFrame(dict(zip(cols, fcsts.T)))
        else:
            fcsts = pd.DataFrame(fcsts, columns=cols)
        fcsts_df = ufp.horizontal_concat([fcsts_df, fcsts])

        # add prediction intervals
        if level is not None:
            if self._cs_df is None or self.prediction_intervals is None:
                raise Exception(
                    "You must fit the model with prediction_intervals to use level."
                )
            else:
                level_ = sorted(level)
                model_names = self._get_model_names(add_level=True)
                prediction_interval_method = get_prediction_interval_method(
                    self.prediction_intervals.method
                )

                fcsts_df = prediction_interval_method(
                    fcsts_df,
                    self._cs_df,
                    model_names=list(model_names),
                    level=level_,
                    cs_n_windows=self.prediction_intervals.n_windows,
                    n_series=len(uids),
                    horizon=self.h,
                )

        return fcsts_df
