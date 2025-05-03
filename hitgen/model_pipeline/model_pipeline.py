import os
import numpy as np
from typing import Tuple, Union
import pandas as pd
from ray import tune
from neuralforecast.auto import (
    AutoNHITS,
    AutoKAN,
    AutoPatchTST,
    AutoiTransformer,
    AutoTSMixer,
    AutoTFT,
)
from hitgen.model_pipeline.auto.AutoModels import (
    AutoHiTGen,
    AutoHiTGenDeep,
    AutoHiTGenMixture,
    AutoHiTGenDeepMixture,
    AutoHiTGenDynamicMixture,
)
from hitgen.visualization.model_visualization import (
    plot_generated_vs_original,
)
from hitgen.model_pipeline.core.core_extension import CustomNeuralForecast


AutoModelType = Union[
    AutoNHITS,
    AutoKAN,
    AutoPatchTST,
    AutoiTransformer,
    AutoTSMixer,
    AutoTFT,
    AutoHiTGen,
    AutoHiTGenDeep,
    AutoHiTGenMixture,
    AutoHiTGenDeepMixture,
    AutoHiTGenDynamicMixture,
]


class _ModelListMixin:
    """
    Mixin that provides a `get_model_list()` method.
    Subclasses can override `MODEL_LIST` or `get_model_list`
    to control exactly which Auto-models are trained.
    """

    MODEL_LIST: list[tuple[str, AutoModelType]] = [
        ("AutoHiTGen", AutoHiTGen),
        ("AutoHiTGenDeep", AutoHiTGenDeep),
        ("AutoHiTGenMixture", AutoHiTGenMixture),
        ("AutoHiTGenDeepMixture", AutoHiTGenDeepMixture),
        ("AutoHiTGenDynamicMixture", AutoHiTGenDynamicMixture),
        ("AutoNHITS", AutoNHITS),
        ("AutoKAN", AutoKAN),
        ("AutoPatchTST", AutoPatchTST),
        ("AutoiTransformer", AutoiTransformer),
        ("AutoTSMixer", AutoTSMixer),
        ("AutoTFT", AutoTFT),
    ]

    def get_model_list(self):
        return self.MODEL_LIST


class ModelPipeline(_ModelListMixin):
    """
    pipeline that:
      - Re-uses an existing DataPipeline instance for data splits/freq/horizon.
      - Hyper-tunes and trains models
      - Predict functions for different strategies
    """

    def __init__(self, data_pipeline):
        """
        data_pipeline : DataPipeline
            A fully initialized instance of existing DataPipeline,
            used to retrieve train/val/test splits, freq, horizon, etc
        """
        self.hp = data_pipeline
        self.freq = self.hp.freq
        self.h = self.hp.h

        # TRAIN+VAL (transfer learning)
        self.trainval_long = (
            self.hp.original_trainval_long[["unique_id", "ds", "y"]]
            .copy()
            .sort_values(["unique_id", "ds"])
        )

        # TRAIN+VAL (basic forecasting)
        self.trainval_long_basic_forecast = (
            self.hp.original_trainval_long_basic_forecast[["unique_id", "ds", "y"]]
            .copy()
            .sort_values(["unique_id", "ds"])
        )

        # TEST data (transfer learning)
        self.test_long = (
            self.hp.original_test_long[["unique_id", "ds", "y"]]
            .copy()
            .sort_values(["unique_id", "ds"])
        )

        # TEST (basic forecasting)
        self.test_long_basic_forecast = (
            self.hp.original_test_long_basic_forecast[["unique_id", "ds", "y"]]
            .copy()
            .sort_values(["unique_id", "ds"])
        )

        # combined basic-forecast dataset (train+test)
        self.original_long_basic_forecast = pd.concat(
            [self.trainval_long_basic_forecast, self.test_long_basic_forecast],
            ignore_index=True,
        )

        self.models = {}

    def hyper_tune_and_train(
        self, dataset_source, dataset_group_source, max_evals=20, mode="in_domain"
    ):
        """
        Trains and hyper-tunes all six models.
        Each data_pipeline does internal time-series cross-validation to select its best hyperparameters.
        """
        if mode in ("in_domain", "out_domain"):
            trainval_long = self.trainval_long
            mode_suffix = ""
        elif mode == "basic_forecasting":
            trainval_long = self.trainval_long_basic_forecast
            mode_suffix = "_basic_forecasting"
        else:
            raise ValueError(
                f"Unsupported mode: '{mode}'. Supported modes are: 'in_domain', 'out_domain', 'basic_forecasting'."
            )

        model_list = self.get_model_list()

        weights_folder = f"assets/model_weights_{mode}"
        os.makedirs(weights_folder, exist_ok=True)

        save_dir = f"assets/model_weights_{mode}/hypertuning{mode_suffix}"
        os.makedirs(save_dir, exist_ok=True)

        for name, ModelClass in model_list:
            print(f"\n=== Handling {name} ===")
            if name in ("AutoTSMixer", "AutoiTransformer"):
                init_kwargs = dict(
                    h=self.h,
                    n_series=1,
                    num_samples=max_evals,
                    verbose=True,
                )
                base_config = ModelClass.get_default_config(
                    h=self.h,
                    backend="ray",
                    n_series=1,
                )
            else:
                init_kwargs = dict(h=self.h, num_samples=max_evals, verbose=True)
                base_config = ModelClass.get_default_config(h=self.h, backend="ray")
                base_config["start_padding_enabled"] = True

            base_config["input_size"] = self.h
            base_config["scaler_type"] = tune.choice([None, "standard"])

            init_kwargs["config"] = base_config

            nf_save_path = os.path.join(
                save_dir,
                f"{dataset_source}_{dataset_group_source}_{name}_neuralforecast",
            )

            if os.path.exists(nf_save_path):
                print(
                    f"Found saved data_pipeline for {name}. Reinitializing wrapper and loading weights..."
                )

                auto_model = ModelClass(**init_kwargs)
                nf = CustomNeuralForecast(models=[auto_model], freq=self.freq)

                model = nf.load(path=nf_save_path)
            else:
                print(f"No saved {name} found. Training & tuning from scratch...")
                auto_model = ModelClass(**init_kwargs)
                model = CustomNeuralForecast(models=[auto_model], freq=self.freq)
                model.fit(df=trainval_long, val_size=self.h)

                model.save(path=nf_save_path, overwrite=True, save_dataset=False)
                print(f"Saved {name} NeuralForecast object to {nf_save_path}")

                results = model.models[0].results.get_dataframe()
                results_file = os.path.join(
                    save_dir,
                    f"{dataset_source}_{dataset_group_source}_{name}_results.csv",
                )
                results.to_csv(results_file, index=False)
                print(f"Saved tuning results to {results_file}")

            self.models[name] = model

        print("\nAll Auto-models have been trained/tuned or loaded from disk.\n")

    @staticmethod
    def _mark_context_rows(
        group: pd.DataFrame, window_size_source: int, horizon: int
    ) -> pd.DataFrame:
        """
        Given rows for a single unique_id (already sorted by ds),
        slice off the last horizon points so we only keep the context portion.
        Return an empty DataFrame if not enough data.
        """
        n = len(group)
        if n < window_size_source + horizon:
            return pd.DataFrame(columns=group.columns)

        cutoff = min(window_size_source, horizon)

        last_window_end = n - cutoff
        return group.iloc[:last_window_end].copy()

    def _preprocess_context(
        self, window_size: int, window_size_source: int = None
    ) -> pd.DataFrame:
        if not window_size_source:
            window_size_source = window_size

        df_test = self.test_long.sort_values(["unique_id", "ds"])

        df_context = df_test.groupby(
            "unique_id", group_keys=True, as_index=False
        ).apply(
            lambda g: self._mark_context_rows(
                group=g, window_size_source=window_size_source, horizon=window_size
            )
        )
        df_context = df_context.reset_index(drop=True)

        if "y_true" in df_context.columns:
            df_context = df_context.rename(columns={"y_true": "y"})

        return df_context[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])

    @staticmethod
    def _mark_prediction_rows(group: pd.DataFrame, horizon: int) -> pd.DataFrame:

        last_window_end = horizon
        return group.iloc[:last_window_end].copy()

    def predict_from_last_window_one_pass(
        self,
        model: CustomNeuralForecast,
        window_size: int,
        window_size_source: int,
        dataset_target: str,
        dataset_group_target: str,
        dataset_source: str,
        dataset_group_source: str,
        freq: str,
        h: int,
        mode: str = "in_domain",
    ) -> pd.DataFrame:
        """
        Predicts exactly the last horizon h points for each test series in a single pass.
        """
        model_name = str(model.models[0])

        dataset_desc = f"{dataset_source}-{dataset_group_source}"
        dataset_name_for_title = dataset_source
        dataset_group_for_title = dataset_group_source

        if mode == "in_domain":
            df_y_preprocess = self._preprocess_context(window_size=window_size)
            df_y_hat = model.predict(df=df_y_preprocess, freq=freq)
            # df_y are only the series on the test set bucket of series
            df_y = self.test_long
        elif mode == "out_domain":
            df_y_preprocess = self._preprocess_context(
                window_size_source=window_size_source,
                window_size=window_size,
            )

            if df_y_preprocess.empty:
                # no series has enough context, so nothing to predict
                print(
                    f"[SKIP] '{dataset_source or dataset_target}' – "
                    f"no series meets the context requirement "
                    f"(window = {window_size_source or window_size})."
                )
                return pd.DataFrame()

            df_y_hat_raw = model.predict(df=df_y_preprocess, freq=freq)

            df_y_hat = df_y_hat_raw.groupby(
                "unique_id", group_keys=True, as_index=False
            ).apply(lambda g: self._mark_prediction_rows(group=g, horizon=window_size))
            df_y_hat = df_y_hat.reset_index(drop=True)

            df_y_hat.sort_values(["unique_id", "ds"], inplace=True)

            # df_y are only the series on the test set bucket of series
            df_y = self.test_long

            dataset_desc = (
                f"{dataset_source}-{dataset_group_source}"
                f"_to_"
                f"{dataset_target}-{dataset_group_target}"
            )
            dataset_name_for_title = f"{dataset_source}→{dataset_target}"
            dataset_group_for_title = f"{dataset_group_source}→{dataset_group_target}"
        elif mode == "basic_forecasting":
            df_y_hat = model.predict(df=self.trainval_long_basic_forecast, freq=freq)
            # df_y is the complete original dataset
            df_y = self.original_long_basic_forecast
        else:
            raise ValueError(
                f"Unsupported mode: '{mode}'. Supported modes are: 'in_domain', 'out_domain', 'basic_forecasting'."
            )

        df_y_hat.rename(columns={model_name: "y"}, inplace=True)
        df_y_hat["y"] = df_y_hat["y"].clip(lower=0)
        df_y_hat = df_y_hat.groupby("unique_id", group_keys=False).tail(h)

        if "y_true" in df_y.columns:
            df_y = df_y.rename(columns={"y_true": "y"})

        suffix_name = f"{model_name}-last-window-one-pass_{mode}_{dataset_desc}"
        title = (
            f"{model_name} • Last-window one-pass ({mode.replace('_', ' ')}) — "
            f"{dataset_name_for_title} [{dataset_group_for_title}]"
        )

        plot_generated_vs_original(
            synth_data=df_y_hat[["unique_id", "ds", "y"]],
            original_data=df_y[["unique_id", "ds", "y"]],
            dataset_name=dataset_name_for_title,
            dataset_group=dataset_group_for_title,
            model_name=model_name,
            n_series=8,
            suffix_name=suffix_name,
            title=title,
        )

        df_y.rename(columns={"y": "y_true"}, inplace=True)

        df_y_y_hat = df_y.merge(df_y_hat, on=["unique_id", "ds"], how="left")

        return df_y_y_hat


class ModelPipelineCoreset(ModelPipeline):
    """
    A lightweight version of ModelPipeline that trains the
    best-performing models on a mixed training set (the coreset)
    """

    MODEL_LIST = [
        ("AutoHiTGenMixture", AutoHiTGenMixture),
        ("AutoHiTGenDynamicMixture", AutoHiTGenDynamicMixture),
        ("AutoPatchTST", AutoPatchTST),
        ("AutoTFT", AutoTFT),
    ]

    def __init__(
        self,
        long_df: pd.DataFrame,
        freq: str,
        h: int,
    ):
        self.freq = freq
        self.h = h

        # coreset itself (train + val)
        self.trainval_long = long_df.sort_values(["unique_id", "ds"])

        # dummy placeholders so inherited code that references them does not fail
        self.trainval_long_basic_forecast = self.trainval_long
        self.original_long_basic_forecast = self.trainval_long

        empty = pd.DataFrame(columns=self.trainval_long.columns)
        self.test_long = empty
        self.test_long_basic_forecast = empty

        self.models = {}
