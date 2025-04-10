import os
import numpy as np
from typing import Tuple, Union
import pandas as pd
from neuralforecast.auto import (
    AutoNHITS,
    AutoKAN,
    AutoPatchTST,
    AutoiTransformer,
    AutoTSMixer,
    AutoTFT,
)
from hitgen.benchmarks.auto.AutoModels import AutoHiTGen, AutoHiTGenDeep
from hitgen.visualization.model_visualization import (
    plot_generated_vs_original,
)
from hitgen.benchmarks.core.core_extension import CustomNeuralForecast

AutoModelType = Union[
    AutoNHITS,
    AutoKAN,
    AutoPatchTST,
    AutoiTransformer,
    AutoTSMixer,
    AutoTFT,
    AutoHiTGen,
    AutoHiTGenDeep,
]


class ModelPipeline:
    """
    pipeline that:
      - Re-uses an existing HiTGenPipeline instance for data splits/freq/horizon.
      - Hyper-tunes and trains models
      - Predict functions for different strategies
    """

    def __init__(self, hitgen_pipeline):
        """
        hitgen_pipeline : HiTGenPipeline
            A fully initialized instance of existing HiTGenPipeline,
            used to retrieve train/val/test splits, freq, horizon, etc
        """
        self.hp = hitgen_pipeline
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
        Each model does internal time-series cross-validation to select its best hyperparameters.
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
            init_kwargs["config"] = base_config

            nf_save_path = os.path.join(
                save_dir,
                f"{self.hp.dataset_name}_{self.hp.dataset_group}_{name}_neuralforecast",
            )

            if os.path.exists(nf_save_path):
                print(
                    f"Found saved model for {name}. Reinitializing wrapper and loading weights..."
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

    def _preprocess_context(self, window_size: int):
        df_for_inference = []
        skip_count = 0

        for uid in self.hp.test_ids:
            df_ser = self.hp.original_test_long.loc[
                self.hp.original_test_long["unique_id"] == uid
            ].copy()
            df_ser.sort_values("ds", inplace=True)

            T = len(df_ser)
            if T < window_size + self.h:
                print(
                    f"[predict_from_last_window_one_pass] Skipping uid={uid}, "
                    f"series length={T} < window_size + h={window_size + self.h}"
                )
                skip_count += 1
                continue

            last_window_end = T - self.h

            y_true = df_ser["y"].values.astype(np.float32)
            window_data = y_true[:last_window_end]

            ds_slice = df_ser["ds"].values[:last_window_end]

            y_context = list(window_data)

            tmp_df = pd.DataFrame(
                {
                    "unique_id": [str(uid)] * last_window_end,
                    "ds": ds_slice,
                    "y": y_context,
                }
            )

            df_for_inference.append(tmp_df)

        if skip_count > 0:
            print(
                f"Skipped {skip_count} short series that couldn't fit window_size + h."
            )

        if not df_for_inference:
            print("No valid series for inference. Returning empty DataFrames.")
            return pd.DataFrame()

        df_y = pd.concat(df_for_inference, ignore_index=True)
        df_y.sort_values(["unique_id", "ds"], inplace=True)

        return df_y

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
            df_y_hat = model.predict()
            # df_y is the complete original dataset
            df_y = self.original_long_basic_forecast
        else:
            raise ValueError(
                f"Unsupported mode: '{mode}'. Supported modes are: 'in_domain', 'out_domain', 'basic_forecasting'."
            )

        df_y_hat.rename(columns={model_name: "y"}, inplace=True)
        df_y_hat = df_y_hat.groupby("unique_id", group_keys=False).tail(self.hp.h)

        plot_generated_vs_original(
            synth_data=df_y_hat[["unique_id", "ds", "y"]],
            original_data=self.hp.original_test_long,
            dataset_name=self.hp.dataset_name,
            dataset_group=self.hp.dataset_group,
            model_name=model_name,
            n_series=8,
            suffix_name=f"{model_name}-last-window-one-pass_{mode_suffix}",
        )

        df_y.rename(columns={"y": "y_true"}, inplace=True)

        df_y_y_hat = df_y.merge(df_y_hat, on=["unique_id", "ds"], how="left")

        return df_y_y_hat
