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

    def _preprocess_context(self, window_size: int):
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

        return df_in_big, series_scalers, series_lengths, test_dict

    def _store_rescale_predictions(self, df, forecast_df, scalers):
        grouped = df.groupby("unique_id", sort=False)
        unique_ids_ordered = list(grouped.groups.keys())
        step = 0
        df_predictions_list = []
        for i, uid in enumerate(unique_ids_ordered):
            group_idx = grouped.groups[uid]
            group_df = df.loc[group_idx].copy()

            local_scaler = scalers[uid]
            inv_preds = local_scaler.inverse_transform(
                forecast_df[step : step + self.hp.h]
            ).squeeze(-1)

            group_df.loc[group_df.tail(self.h).index, "y"] = inv_preds

            df_predictions_list.append(group_df)
            step += self.hp.h

        df_pred_all = pd.concat(df_predictions_list, ignore_index=True)
        return df_pred_all, unique_ids_ordered

    def _final_df_realigned_predictions(
        self, df_pred_all, unique_ids_ordered, series_lengths, test_dict
    ):
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

        df_in_big, series_scalers, series_lengths, test_dict = self._preprocess_context(
            window_size
        )

        dataset_test, *_ = TimeSeriesDataset.from_df(
            df=df_in_big,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
        )

        forecast_big = model.predict(dataset=dataset_test)

        df_pred_all, unique_ids_ordered = self._store_rescale_predictions(
            df=df_in_big, forecast_df=forecast_big, scalers=series_scalers
        )

        df_horizon_final, df_all_final = self._final_df_realigned_predictions(
            df_pred_all=df_pred_all,
            unique_ids_ordered=unique_ids_ordered,
            series_lengths=series_lengths,
            test_dict=test_dict,
        )

        plot_generated_vs_original(
            synth_data=df_all_final[["unique_id", "ds", "y"]],
            original_data=self.hp.original_test_long,
            dataset_name=self.hp.dataset_name,
            dataset_group=self.hp.dataset_group,
            model_name=model_name,
            n_series=8,
            suffix_name=f"{model_name}-last-window-one-pass_{prediction_mode}",
        )

        return df_horizon_final, df_all_final
