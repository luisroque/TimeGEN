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
from neuralforecast.tsdataset import TimeSeriesDataset
from hitgen.visualization.model_visualization import (
    plot_generated_vs_original,
)

AutoModelType = Union[
    AutoNHITS,
    AutoKAN,
    AutoPatchTST,
    AutoiTransformer,
    AutoTSMixer,
    AutoTFT,
]


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
            # ("AutoKAN", AutoKAN),
            # ("AutoPatchTST", AutoPatchTST),
            # ("AutoiTransformer", AutoiTransformer),
            # ("AutoTSMixer", AutoTSMixer),
            # ("AutoTFT", AutoTFT),
        ]

        train_dset, *_ = TimeSeriesDataset.from_df(
            df=self.trainval_long,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
        )

        weights_folder = "assets/model_weights_benchmarks"
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
            else:
                init_kwargs = dict(h=self.h, num_samples=max_evals, verbose=True)

            model_save_path = os.path.join(
                weights_folder,
                f"{self.hp.dataset_name}_{self.hp.dataset_group}_{name}.ckpt",
            )

            if os.path.exists(model_save_path):
                print(
                    f"Found saved model for {name}. Reinitializing wrapper and loading weights..."
                )

                model = ModelClass(**init_kwargs)

                model.model = model.cls_model.load(path=model_save_path)
            else:
                print(f"No saved {name} found. Training & tuning from scratch...")
                model = ModelClass(**init_kwargs)
                model.fit(train_dset)
                print(f"Saving {name} to {model_save_path} ...")
                model.save(path=model_save_path)

            self.models[name] = model

        print("\nAll Auto-models have been trained/tuned or loaded from disk.\n")

    def predict_from_first_window(
        self,
        model: AutoModelType,
        window_size: int = None,
        prediction_mode: str = "in_domain",
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
        model_name = model.__class__.__name__
        model.set_test_size(self.hp.h)

        if window_size is None:
            window_size = self.hp.best_params_forecasting["window_size"]

        results = []
        first_horizon_list = []
        autoregressive_list = []

        for uid in self.hp.test_ids:
            df_ser = self.hp.original_test_long.loc[
                self.hp.original_test_long["unique_id"] == uid
            ].copy()
            df_ser.sort_values("ds", inplace=True)

            # if series is too short, skip
            if len(df_ser) < window_size:
                print(
                    f"[predict_from_first_window] Series {uid} < window_size={window_size}. Skipping."
                )
                continue

            # build a date range up to T+H for dynamic features
            T = len(df_ser)
            # T actual points and additional H beyond that
            last_date = df_ser["ds"].iloc[-1]
            future_dates = pd.date_range(
                start=last_date + pd.tseries.frequencies.to_offset(self.freq),
                periods=self.h,
                freq=self.freq,
            )
            ds_full = pd.concat(
                [df_ser["ds"], pd.Series(future_dates)], ignore_index=True
            )

            # ground-truth from 0..T-1
            y_all = np.full(shape=(T,), fill_value=np.nan, dtype=np.float32)
            y_all[:T] = df_ser["y"].values
            y_context = np.full(shape=(T,), fill_value=np.nan, dtype=np.float32)
            y_context[:window_size] = df_ser["y"][:window_size].values

            # predicted fill from window_size onward
            y_hat = np.full(shape=(T,), fill_value=np.nan, dtype=np.float32)

            # -- autoregressive loop --
            # move in steps of horizon until we reach T
            step = 0
            while step + window_size < T:
                window_end = step + window_size

                # next chunk [window_end .. window_end + horizon)
                future_end = window_end + self.h
                horizon_length = min(future_end, T)

                # growing window of context
                window_data = y_context[:window_end].copy()

                # models that have fixed input size
                if model_name in ("AutoiTransformer", "AutoTSMixer"):
                    min_input_size = model.model.hparams["input_size"]
                    needed_pad = min_input_size - len(window_data)

                    if needed_pad > 0:
                        first_real_date = ds_full.iloc[0]
                        ds_padding = pd.date_range(
                            end=first_real_date
                            - pd.tseries.frequencies.to_offset(self.freq),
                            periods=needed_pad,
                            freq=self.freq,
                        )

                        pad_array = np.zeros(needed_pad, dtype=window_data.dtype)

                        window_data = np.concatenate([pad_array, window_data])

                        padded_dates = pd.Index(ds_padding).append(
                            pd.Index(ds_full.iloc[:horizon_length].values)
                        )

                        ds_array = padded_dates.values

                        horizon_length += needed_pad
                    else:
                        ds_array = ds_full.iloc[:horizon_length].values
                else:
                    ds_array = ds_full.iloc[:horizon_length].values

                tmp_df = pd.DataFrame(
                    {
                        "unique_id": [str(uid)] * horizon_length,
                        "ds": ds_array,
                        "y": window_data.tolist()
                        + [np.nan] * (horizon_length - window_data.shape[0]),
                    }
                )

                tmp_df.dropna(subset=["unique_id"], inplace=True)
                tmp_df = tmp_df[["unique_id", "ds", "y"]]

                single_ds, *_ = TimeSeriesDataset.from_df(
                    df=tmp_df,
                    id_col="unique_id",
                    time_col="ds",
                    target_col="y",
                )

                if single_ds is None:
                    break

                forecast_out = model.predict(dataset=single_ds)

                y_hat[window_end:horizon_length] = forecast_out[
                    0, : (horizon_length - window_end)
                ]

                y_context[window_end:horizon_length] = forecast_out[
                    0, : (horizon_length - window_end)
                ]

                # move the window forward
                step += self.h

            df_out = pd.DataFrame(
                {
                    "unique_id": [str(uid)] * T,
                    "ds": ds_full[:T].values,
                    "y_true": y_all,  # original values
                    "y": y_hat,  # predictions (NaN up to window_size)
                }
            )

            # first horizon data
            first_horizon_end = min(window_size + self.h, T)
            df_first = df_out.iloc[window_size:first_horizon_end].copy()

            df_auto = df_out.iloc[window_size:T].copy()

            results.append(df_out)
            first_horizon_list.append(df_first)
            autoregressive_list.append(df_auto)

        if not results:
            return pd.DataFrame(), pd.DataFrame()

        df_res = pd.concat(results, ignore_index=True)
        df_first_window = pd.concat(first_horizon_list, ignore_index=True)
        df_autoregressive = pd.concat(autoregressive_list, ignore_index=True)

        plot_generated_vs_original(
            synth_data=df_res[["unique_id", "ds", "y"]],
            original_data=self.hp.original_test_long,
            score=0.0,
            loss=0.0,
            dataset_name=self.hp.dataset_name,
            dataset_group=self.hp.dataset_group,
            n_series=8,
            suffix_name=f"{model}-initial-window-{prediction_mode}",
        )

        return df_first_window, df_autoregressive

    def predict_from_last_window(
        self,
        model: AutoModelType,
        window_size: int = None,
        prediction_mode: str = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predicts a horizon of H points by taking exactly the LAST window_size points
        from each test series (ending at T-H) and forecasting the next H points
        (from T-H .. T-1).
        """
        model_name = model.__class__.__name__
        model.set_test_size(self.hp.h)

        if window_size is None:
            window_size = self.hp.best_params_forecasting["window_size"]

        results_all = []
        results_horizon = []

        for uid in self.hp.test_ids:
            df_ser = self.hp.original_test_long.loc[
                self.hp.original_test_long["unique_id"] == uid
            ].copy()
            df_ser.sort_values("ds", inplace=True)

            T = len(df_ser)

            if T < window_size + self.hp.h:
                print(
                    f"[predict_from_last_window] Series {uid} length={T} "
                    f"< window_size + H={window_size + self.hp.h}. Skipping."
                )
                continue

            ds_full = df_ser["ds"].values  # shape (T,)

            y_true = df_ser["y"].values.astype(np.float32)
            y_pred = np.full(shape=(T,), fill_value=np.nan, dtype=np.float32)

            last_window_start = T - self.hp.h - window_size
            last_window_end = T - self.hp.h  # exclusive index
            window_data = y_true[last_window_start:last_window_end]

            horizon_length = window_size + self.hp.h
            ds_array = ds_full[last_window_start : (last_window_end + self.hp.h)]

            if model_name in ("AutoiTransformer", "AutoTSMixer"):
                min_input_size = model.model.hparams["input_size"]
                needed_pad = min_input_size - len(window_data)

                if needed_pad > 0:
                    first_real_date = ds_array[0]

                    ds_padding = pd.date_range(
                        end=first_real_date
                        - pd.tseries.frequencies.to_offset(self.freq),
                        periods=needed_pad,
                        freq=self.freq,
                    )
                    pad_array = np.zeros(needed_pad, dtype=window_data.dtype)

                    window_data = np.concatenate([pad_array, window_data])
                    padded_dates = pd.Index(ds_padding).append(pd.Index(ds_array))
                    ds_array = padded_dates.values

                    horizon_length += needed_pad

            tmp_df = pd.DataFrame(
                {
                    "unique_id": [str(uid)] * horizon_length,
                    "ds": ds_array,
                    "y": list(window_data) + [np.nan] * self.hp.h,
                }
            )

            tmp_df.dropna(subset=["unique_id"], inplace=True)
            tmp_df = tmp_df[["unique_id", "ds", "y"]]

            single_ds, *_ = TimeSeriesDataset.from_df(
                df=tmp_df,
                id_col="unique_id",
                time_col="ds",
                target_col="y",
            )

            if single_ds is None:
                print(
                    f"[predict_from_last_window] Could not build dataset for {uid}. Skipping."
                )
                continue

            forecast_out = model.predict(dataset=single_ds)

            y_pred[last_window_end : last_window_end + self.hp.h] = forecast_out[
                0, : self.hp.h
            ]

            df_all = pd.DataFrame(
                {
                    "unique_id": [str(uid)] * T,
                    "ds": ds_full,
                    "y_true": y_true,  # entire ground truth
                    "y": y_pred,  # forecast in last H
                }
            )

            df_hor = df_all.iloc[last_window_end : last_window_end + self.hp.h].copy()

            results_all.append(df_all)
            results_horizon.append(df_hor)

        if not results_all:
            return pd.DataFrame(), pd.DataFrame()

        df_all_final = pd.concat(results_all, ignore_index=True)
        df_horizon_final = pd.concat(results_horizon, ignore_index=True)

        plot_generated_vs_original(
            synth_data=df_all_final[["unique_id", "ds", "y"]],
            original_data=self.hp.original_test_long,
            score=0.0,
            loss=0.0,
            dataset_name=self.hp.dataset_name,
            dataset_group=self.hp.dataset_group,
            n_series=8,
            suffix_name=f"{model_name}-last-window_{prediction_mode}",
        )

        return df_horizon_final, df_all_final
