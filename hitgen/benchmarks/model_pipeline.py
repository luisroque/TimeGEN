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
from hitgen.benchmarks.auto.AutoModels import (
    AutoHiTGen,
    AutoHiTGenFlow,
    AutoHiTGenAttn,
    AutoHiTGenHier,
    AutoHiTGenTCN,
)
from neuralforecast.tsdataset import TimeSeriesDataset
from sklearn.preprocessing import StandardScaler

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
    AutoHiTGen,
    AutoHiTGenFlow,
    AutoHiTGenAttn,
    AutoHiTGenHier,
    AutoHiTGenTCN,
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

        self.train_long = self.hp.original_train_long_transf[
            ["unique_id", "ds", "y"]
        ].copy()
        self.val_long = self.hp.original_val_long_transf[
            ["unique_id", "ds", "y"]
        ].copy()
        self.test_long = self.hp.original_test_long_transf[
            ["unique_id", "ds", "y"]
        ].copy()

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
            ("AutoHiTGenFlow", AutoHiTGenFlow),
            ("AutoHiTGenAttn", AutoHiTGenAttn),
            ("AutoHiTGenHier", AutoHiTGenHier),
            ("AutoHiTGenTCN", AutoHiTGenTCN),
            ("AutoNHITS", AutoNHITS),
            ("AutoKAN", AutoKAN),
            ("AutoPatchTST", AutoPatchTST),
            ("AutoiTransformer", AutoiTransformer),
            ("AutoTSMixer", AutoTSMixer),
            ("AutoTFT", AutoTFT),
        ]

        trainval_dset, *_ = TimeSeriesDataset.from_df(
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

            model_save_path = os.path.join(
                weights_folder,
                f"{self.hp.dataset_name}_{self.hp.dataset_group}_{name}.ckpt",
            )

            base_config["scaler_type"] = None
            init_kwargs["config"] = base_config

            if os.path.exists(model_save_path):
                print(
                    f"Found saved model for {name}. Reinitializing wrapper and loading weights..."
                )

                model = ModelClass(**init_kwargs)

                model.model = model.cls_model.load(path=model_save_path)
            else:
                print(f"No saved {name} found. Training & tuning from scratch...")
                model = ModelClass(**init_kwargs)
                model.fit(dataset=trainval_dset)
                print(f"Saving {name} to {model_save_path} ...")
                model.save(path=model_save_path)

            self.models[name] = model

        print("\nAll Auto-models have been trained/tuned or loaded from disk.\n")

    def predict_from_first_window(
        self,
        model: AutoModelType,
        window_size: int,
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

        results = []
        first_horizon_list = []
        autoregressive_list = []

        local_scaler = {}

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

                if step == 0:
                    local_scaler[uid] = StandardScaler()
                    scaled_window = local_scaler[uid].fit_transform(
                        window_data.reshape(-1, 1)
                    )
                else:
                    scaled_window = local_scaler[uid].transform(
                        window_data.reshape(-1, 1)
                    )

                scaled_window = scaled_window.squeeze(-1)

                ds_array = ds_full.iloc[:future_end].values

                tmp_df = pd.DataFrame(
                    {
                        "unique_id": [str(uid)] * future_end,
                        "ds": ds_array,
                        "y": scaled_window.tolist()
                        + [np.nan] * (future_end - scaled_window.shape[0]),
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

                forecast_out = local_scaler[uid].inverse_transform(forecast_out).T

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
        window_size: int,
        prediction_mode: str = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predicts a horizon of H points by taking exactly the LAST window_size points
        from each test series (ending at T-H) and forecasting the next H points
        (from T-H .. T-1).
        """
        model_name = model.__class__.__name__
        model.set_test_size(self.hp.h)

        results_all = []
        results_horizon = []

        local_scaler = {}

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

            local_scaler[uid] = StandardScaler()
            scaled_window = (
                local_scaler[uid].fit_transform(window_data.reshape(-1, 1)).squeeze(-1)
            )

            horizon_length = window_size + self.hp.h
            ds_array = ds_full[last_window_start : (last_window_end + self.hp.h)]

            if model_name in ("AutoiTransformer", "AutoTSMixer"):
                min_input_size = model.model.hparams["input_size"]
                needed_pad = min_input_size - len(scaled_window)

                if needed_pad > 0:
                    first_real_date = ds_array[0]

                    ds_padding = pd.date_range(
                        end=first_real_date
                        - pd.tseries.frequencies.to_offset(self.freq),
                        periods=needed_pad,
                        freq=self.freq,
                    )
                    pad_array = np.zeros(needed_pad, dtype=scaled_window.dtype)

                    scaled_window = np.concatenate([pad_array, scaled_window])
                    padded_dates = pd.Index(ds_padding).append(pd.Index(ds_array))
                    ds_array = padded_dates.values

                    horizon_length += needed_pad

            tmp_df = pd.DataFrame(
                {
                    "unique_id": [str(uid)] * horizon_length,
                    "ds": ds_array,
                    "y": list(scaled_window) + [np.nan] * self.hp.h,
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

            forecast_out = (
                local_scaler[uid]
                .inverse_transform(forecast_out.squeeze(-1).reshape(-1, 1))
                .squeeze(-1)
            )

            y_pred[last_window_end : last_window_end + self.hp.h] = forecast_out

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

    def predict_from_last_window_one_pass(
        self,
        model: AutoModelType,
        window_size: int,
        prediction_mode: str = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predicts exactly the last horizon h points for each test series in a single pass.

          1. For each test series, take the last `window_size` points ending at T-h
             as context, then produce h NaNs after that window.
          2. Concatenate all these 'context+future' segments into one DataFrame.
          3. Create a single TimeSeriesDataset and pass it once to `model.predict()`.
          4. Inverse-transform each group's predictions using local scaling,
             then assemble the final horizon and full test predictions.
        """
        model_name = model.__class__.__name__
        model.set_test_size(self.h)

        # context+NaN windows
        df_for_inference = []
        # keep track of per-series info to help unscale and re-inject predictions
        series_scalers = {}
        series_lengths = {}
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
            window_data = y_true[last_window_start:last_window_end]

            local_scaler = StandardScaler()
            scaled_window = local_scaler.fit_transform(
                window_data.reshape(-1, 1)
            ).squeeze(-1)

            # last window + h future placeholders
            ds_slice = df_ser["ds"].values[
                last_window_start : (last_window_end + self.h)
            ]

            # scaled_window for context, then NaNs for h future steps
            y_context_plus_future = list(scaled_window) + [np.nan] * self.h

            tmp_df = pd.DataFrame(
                {
                    "unique_id": [str(uid)] * (window_size + self.h),
                    "ds": ds_slice,
                    "y": y_context_plus_future,
                }
            )

            df_for_inference.append(tmp_df)
            series_scalers[uid] = local_scaler
            series_lengths[uid] = T

        if skip_count > 0:
            print(
                f"Skipped {skip_count} short series that couldn't fit window_size + h."
            )

        if not df_for_inference:
            print("No valid series for inference. Returning empty DataFrames.")
            return pd.DataFrame(), pd.DataFrame()

        df_in_big = pd.concat(df_for_inference, ignore_index=True)
        df_in_big.sort_values(["unique_id", "ds"], inplace=True)

        dataset_test, *_ = TimeSeriesDataset.from_df(
            df=df_in_big,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
        )

        forecast_big = model.predict(dataset=dataset_test)
        forecast_big = forecast_big.detach().cpu().numpy()  # (N, h)

        grouped = df_in_big.groupby("unique_id", sort=False)
        unique_ids_ordered = list(grouped.groups.keys())

        df_predictions_list = []
        row_idx = 0

        for i, uid in enumerate(unique_ids_ordered):
            group_idx = grouped.groups[uid]
            group_df = df_in_big.loc[group_idx].copy()

            local_scaler = series_scalers[uid]
            inv_preds = local_scaler.inverse_transform(
                forecast_big[i, :].reshape(-1, 1)
            ).squeeze(-1)

            group_df.loc[group_df.tail(self.h).index, "y"] = inv_preds

            df_predictions_list.append(group_df)

        # each group last h steps have predicted y, the rest are context
        df_pred_all = pd.concat(df_predictions_list, ignore_index=True)

        horizon_frames = []
        all_frames = []

        for uid in unique_ids_ordered:
            sub_pred = df_pred_all[df_pred_all["unique_id"] == uid].copy()
            sub_pred.sort_values("ds", inplace=True)

            T = series_lengths[uid]
            df_ser = test_dict[uid]

            y_true = df_ser["y"].values.astype(np.float32)
            ds_full = df_ser["ds"].values

            y_hat = np.full(shape=(T,), fill_value=np.nan, dtype=np.float32)

            # predictions correspond to T-h..T-1
            last_window_end = T - self.h
            predicted_values = sub_pred["y"].tail(self.h).to_numpy(dtype=np.float32)
            y_hat[last_window_end : last_window_end + self.h] = predicted_values

            df_all = pd.DataFrame(
                {
                    "unique_id": [uid] * T,
                    "ds": ds_full,
                    "y_true": y_true,
                    "y": y_hat,
                }
            )

            df_horizon = df_all.iloc[last_window_end : last_window_end + self.h].copy()

            horizon_frames.append(df_horizon)
            all_frames.append(df_all)

        df_horizon_final = pd.concat(horizon_frames, ignore_index=True)
        df_all_final = pd.concat(all_frames, ignore_index=True)

        plot_generated_vs_original(
            synth_data=df_all_final[["unique_id", "ds", "y"]],
            original_data=self.hp.original_test_long,
            score=0.0,
            loss=0.0,
            dataset_name=self.hp.dataset_name,
            dataset_group=self.hp.dataset_group,
            n_series=8,
            suffix_name=f"{model_name}-last-window-one-pass_{prediction_mode}",
        )

        return df_horizon_final, df_all_final
