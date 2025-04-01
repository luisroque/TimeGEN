import os
import numpy as np
import pandas as pd
from neuralforecast.auto import (
    AutoNHITS,
    AutoKAN,
    AutoPatchTST,
    AutoiTransformer,
    AutoTSMixer,
    AutoTFT,
)
from neuralforecast.tsdataset import TimeSeriesDataset


class BenchmarkPipeline:
    """
    benchmark pipeline that:
      - Re-uses an existing HiTGenPipeline instance for data splits/freq/horizon.
      - Hyper-tunes and trains benchmark models
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
            ("AutoNHITS", AutoNHITS),
            ("AutoKAN", AutoKAN),
            ("AutoPatchTST", AutoPatchTST),
            ("AutoiTransformer", AutoiTransformer),
            ("AutoTSMixer", AutoTSMixer),
            ("AutoTFT", AutoTFT),
        ]

        train_dset, *_ = TimeSeriesDataset.from_df(
            df=self.trainval_long,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
        )

        for name, ModelClass in model_list:
            print(f"\n=== Training & tuning {name} ===")
            model = ModelClass(
                h=self.h,
                num_samples=max_evals,
                verbose=True,
            )
            # fit on train+val (internal cross-validation for best hyperparams)
            model.fit(train_dset)
            self.models[name] = model

        print("\nAll Nixtla Auto-models have been trained and tuned.\n")

    def predict_from_first_window(
        self,
        model: str,
        use_direct_forecast: bool = True,
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        Predicts the entire test range by taking exactly the first window_size points
        of each test series as context (observed).

        Then:
          - If use_direct_forecast=True, calls .predict(steps=full_remaining)
            to get the entire forecast in one shot.
          - Otherwise, does an autoregressive loop in increments of self.h,
            repeatedly calling model.predict(...).

        Returns
        first_horizon_df, autoregressive_df
            Both with columns: [unique_id, ds, y_true, y].

            - `first_horizon_df` is just the first chunk of forecast within the test set
              (window_size to window_size + h).
            - `autoregressive_df` is from window_size up to T in the test set
              (one record per step).
            - In the *direct* forecast scenario, both DataFrames end up basically the same
              (the difference is in how we slice out the first horizon chunk).
        """
        if model not in self.models:
            raise ValueError(
                f"Model '{model}' not found in self.models. "
                f"Available keys: {list(self.models.keys())}"
            )

        model = self.models[model]

        test_long = self.test_long.copy()
        test_long = test_long.sort_values(["unique_id", "ds"], ascending=[True, True])

        results_all = []
        results_first_horizon = []

        for uid in test_long["unique_id"].unique():
            df_ser = test_long[test_long["unique_id"] == uid].copy()
            df_ser.sort_values("ds", inplace=True)

            T = len(df_ser)
            if T < self.hp.best_params_forecasting["window_size"]:
                continue

            context_df = df_ser.iloc[
                : self.hp.best_params_forecasting["window_size"]
            ].copy()
            context_df = context_df.reset_index(drop=True)

            future_needed = (
                T - self.hp.best_params_forecasting["window_size"]
            )  # how many test steps remain after the initial window
            df_ser["y_true"] = df_ser["y"]
            df_ser["y"] = np.nan  # fill forecast column with NaN initially

            if use_direct_forecast:
                # direct multi-step forecast for the entire remainder
                df_context = context_df[["unique_id", "ds", "y"]].copy()

                if future_needed > 0:
                    context_dset, *_ = TimeSeriesDataset.from_df(
                        df=df_context,
                        id_col="unique_id",
                        time_col="ds",
                        target_col="y",
                    )
                    fcst = model.predict(dataset=context_dset, steps=future_needed)

                    df_merge = df_ser.merge(fcst, on=["unique_id", "ds"], how="left")
                    df_merge = df_merge.rename(columns={"y_pred": "y"})

                    df_ser = df_merge[["unique_id", "ds", "y_true", "y"]]

                first_horizon_end = min(
                    self.hp.best_params_forecasting["window_size"] + self.h, T
                )
                first_chunk = df_ser.iloc[
                    self.hp.best_params_forecasting["window_size"] : first_horizon_end
                ].copy()

            else:
                # autoregressive => Move forward in increments of self.h
                pointer = 0
                current_history = context_df[["unique_id", "ds", "y"]].copy()

                current_history_dset, *_ = TimeSeriesDataset.from_df(
                    df=current_history,
                    id_col="unique_id",
                    time_col="ds",
                    target_col="y",
                )

                while pointer < future_needed:
                    steps_to_forecast = min(self.h, future_needed - pointer)

                    fcst = model.predict(
                        df=current_history_dset, steps=steps_to_forecast
                    )

                    start_idx = self.hp.best_params_forecasting["window_size"] + pointer
                    end_idx = (
                        self.hp.best_params_forecasting["window_size"]
                        + pointer
                        + steps_to_forecast
                    )
                    slice_df = df_ser.iloc[start_idx:end_idx].copy()

                    slice_df = slice_df.merge(fcst, on=["unique_id", "ds"], how="left")
                    slice_df = slice_df.rename(columns={"y_pred": "y"})

                    df_ser.iloc[start_idx:end_idx, df_ser.columns.get_loc("y")] = (
                        slice_df["y"].values
                    )

                    # treat as newly observed for the next iteration
                    new_history = slice_df[["unique_id", "ds", "y"]].copy()
                    current_history = pd.concat(
                        [current_history, new_history], ignore_index=True
                    )
                    current_history.sort_values("ds", inplace=True)

                    pointer += steps_to_forecast

                first_horizon_end = min(
                    self.hp.best_params_forecasting["window_size"] + self.h, T
                )
                first_chunk = df_ser.iloc[
                    self.hp.best_params_forecasting["window_size"] : first_horizon_end
                ].copy()

            results_all.append(df_ser)
            results_first_horizon.append(first_chunk)

        if not results_all:
            return pd.DataFrame(), pd.DataFrame()

        df_autoregressive = pd.concat(results_all, ignore_index=True)
        df_first_window = pd.concat(results_first_horizon, ignore_index=True)

        df_autoregressive = df_autoregressive[["unique_id", "ds", "y_true", "y"]]
        df_first_window = df_first_window[["unique_id", "ds", "y_true", "y"]]

        return df_first_window, df_autoregressive
