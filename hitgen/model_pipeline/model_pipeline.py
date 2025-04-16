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
]


class ModelPipeline:
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

    def hyper_tune_and_train(self, max_evals=20, mode="in_domain"):
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

        model_list = [
            ("AutoHiTGen", AutoHiTGen),
            ("AutoHiTGenDeep", AutoHiTGenDeep),
            ("AutoHiTGenMixture", AutoHiTGenMixture),
            ("AutoHiTGenDeepMixture", AutoHiTGenDeepMixture),
            ("AutoNHITS", AutoNHITS),
            ("AutoKAN", AutoKAN),
            ("AutoPatchTST", AutoPatchTST),
            ("AutoiTransformer", AutoiTransformer),
            ("AutoTSMixer", AutoTSMixer),
            ("AutoTFT", AutoTFT),
        ]

        weights_folder = "assets/model_weights"
        os.makedirs(weights_folder, exist_ok=True)

        save_dir = f"assets/model_weights/hypertuning{mode_suffix}"
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
                f"{self.hp.dataset_name}_{self.hp.dataset_group}_{name}_neuralforecast",
            )

            if os.path.exists(nf_save_path):
                print(
                    f"Found saved data_pipeline for {name}. Reinitializing wrapper and loading weights..."
                )

                auto_model = ModelClass(**init_kwargs)
                nf = CustomNeuralForecast(models=[auto_model], freq=self.hp.freq)

                model = nf.load(path=nf_save_path)
            else:
                print(f"No saved {name} found. Training & tuning from scratch...")
                auto_model = ModelClass(**init_kwargs)
                model = CustomNeuralForecast(models=[auto_model], freq=self.hp.freq)
                model.fit(df=trainval_long, val_size=self.hp.h)

                model.save(path=nf_save_path, overwrite=True, save_dataset=False)
                print(f"Saved {name} NeuralForecast object to {nf_save_path}")

                results = model.models[0].results.get_dataframe()
                results_file = os.path.join(
                    save_dir,
                    f"{self.hp.dataset_name}_{self.hp.dataset_group}_{name}_results.csv",
                )
                results.to_csv(results_file, index=False)
                print(f"Saved tuning results to {results_file}")

            self.models[name] = model

        print("\nAll Auto-models have been trained/tuned or loaded from disk.\n")

    @staticmethod
    def _mark_context_rows(
        group: pd.DataFrame, window_size: int, horizon: int
    ) -> pd.DataFrame:
        """
        Given rows for a single unique_id (already sorted by ds),
        slice off the last horizon points so we only keep the context portion.
        Return an empty DataFrame if not enough data.
        """
        n = len(group)
        if n < window_size + horizon:
            return pd.DataFrame(columns=group.columns)

        last_window_end = n - horizon
        return group.iloc[:last_window_end].copy()

    def _preprocess_context(self, window_size: int) -> pd.DataFrame:
        df_test = self.hp.original_test_long.sort_values(["unique_id", "ds"])

        df_context = df_test.groupby(
            "unique_id", group_keys=True, as_index=False
        ).apply(
            lambda g: self._mark_context_rows(
                group=g, window_size=window_size, horizon=self.h
            )
        )
        df_context = df_context.reset_index(drop=True)

        return df_context[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])

    def predict_from_last_window_one_pass(
        self,
        model: CustomNeuralForecast,
        window_size: int,
        mode: str = "in_domain",
    ) -> pd.DataFrame:
        """
        Predicts exactly the last horizon h points for each test series in a single pass.
        """
        model_name = str(model.models[0])

        mode_suffix = f"_{mode}"

        if mode in ("in_domain", "out_domain"):
            df_y_preprocess = self._preprocess_context(window_size)
            df_y_hat = model.predict(df=df_y_preprocess)
            # df_y are only the series on the test set bucket of series
            df_y = self.test_long
        elif mode == "basic_forecasting":
            df_y_hat = model.predict(df=self.trainval_long_basic_forecast)
            # df_y is the complete original dataset
            df_y = self.original_long_basic_forecast
        else:
            raise ValueError(
                f"Unsupported mode: '{mode}'. Supported modes are: 'in_domain', 'out_domain', 'basic_forecasting'."
            )

        df_y_hat.rename(columns={model_name: "y"}, inplace=True)
        df_y_hat = df_y_hat.groupby("unique_id", group_keys=False).tail(self.hp.h)

        if "y_true" in df_y.columns:
            df_y = df_y.rename(columns={"y_true": "y"})

        plot_generated_vs_original(
            synth_data=df_y_hat[["unique_id", "ds", "y"]],
            original_data=df_y[["unique_id", "ds", "y"]],
            dataset_name=self.hp.dataset_name,
            dataset_group=self.hp.dataset_group,
            model_name=model_name,
            n_series=8,
            suffix_name=f"{model_name}-last-window-one-pass_{mode_suffix}",
        )

        df_y.rename(columns={"y": "y_true"}, inplace=True)

        df_y_y_hat = df_y.merge(df_y_hat, on=["unique_id", "ds"], how="left")

        return df_y_y_hat
