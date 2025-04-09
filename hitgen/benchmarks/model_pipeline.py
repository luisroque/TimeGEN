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

        self.train_long = self.hp.original_train_long[["unique_id", "ds", "y"]].copy()
        self.val_long = self.hp.original_val_long[["unique_id", "ds", "y"]].copy()
        self.test_long = self.hp.original_test_long[["unique_id", "ds", "y"]].copy()

        self.trainval_long = pd.concat(
            [self.train_long, self.val_long], ignore_index=True
        )
        self.trainval_long.sort_values(["unique_id", "ds"], inplace=True)

        self.models = {}

    def hyper_tune_and_train(self, max_evals=20):
        """
        Trains and hyper-tunes all six models
        Each model does internal time-series cross-validation to select its best hyperparameters.
        """
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

            save_dir = "assets/model_weights/hypertuning"
            os.makedirs(save_dir, exist_ok=True)

            nf_save_path = os.path.join(
                save_dir,
                f"{self.hp.dataset_name}_{self.hp.dataset_group}_{name}_neuralforecast",
            )

            init_kwargs["config"] = base_config

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
                model.fit(df=self.trainval_long, val_size=self.hp.h)

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

        test_dict = {}
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

            test_dict[uid] = df_ser.copy()

            last_window_start = T - self.h - window_size
            last_window_end = T - self.h

            y_true = df_ser["y"].values.astype(np.float32)
            window_data = y_true[:last_window_end]

            # last window
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
        prediction_mode: str = None,
    ) -> pd.DataFrame:
        """
        Predicts exactly the last horizon h points for each test series in a single pass.

          1. For each test series, take the last `window_size` points ending at T-h
             as context, then produce h NaNs after that window.
          2. Concatenate all these 'context+future' segments into one DataFrame.
          3. Create a single TimeSeriesDataset and pass it once to `model.predict()`.
          4. Inverse-transform each group's predictions using local scaling,
             then assemble the final horizon and full test predictions.
        """
        model_name = str(model.models[0])

        df_y_preprocess = self._preprocess_context(window_size)

        df_y_hat = model.predict(df=df_y_preprocess)

        df_y_hat.rename(columns={model_name: "y"}, inplace=True)
        df_y_hat = df_y_hat.groupby("unique_id", group_keys=False).tail(self.hp.h)

        plot_generated_vs_original(
            synth_data=df_y_hat[["unique_id", "ds", "y"]],
            original_data=self.hp.original_test_long,
            dataset_name=self.hp.dataset_name,
            dataset_group=self.hp.dataset_group,
            model_name=model_name,
            n_series=8,
            suffix_name=f"{model_name}-last-window-one-pass_{prediction_mode}",
        )

        df_y = self.hp.original_test_long.rename(columns={"y": "y_true"})

        df_y_y_hat = df_y.merge(df_y_hat, on=["unique_id", "ds"], how="left")

        return df_y_y_hat
