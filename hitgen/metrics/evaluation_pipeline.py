import os
import json
import numpy as np

from hitgen.model_pipeline.model_pipeline import ModelPipeline
from hitgen.model_pipeline.core.core_extension import CustomNeuralForecast
from hitgen.metrics.evaluation_metrics import smape, mase, mae, rmse, rmsse


def evaluation_pipeline_hitgen_forecast(
    dataset: str,
    dataset_group: str,
    pipeline: ModelPipeline,
    model: CustomNeuralForecast,
    horizon: int,
    freq: str,
    row_forecast: dict,
    period: int = None,
    window_size: int = None,
    window_size_source: int = None,
    dataset_source: str = None,
    dataset_group_source: str = None,
    mode: str = "in_domain",
) -> None:
    """
    Evaluate forecast for different modes: basic forecasting and transfer learning
    in domain and out of domain
    """
    results_folder = f"assets/results_forecast_{mode}"
    os.makedirs(results_folder, exist_ok=True)

    model_name = str(model.models[0])

    if dataset_source:
        results_file = os.path.join(
            results_folder,
            f"{dataset}_{dataset_group}_{model_name}_{horizon}_trained_on_{dataset_source}_{dataset_group_source}.json",
        )
    else:
        results_file = os.path.join(
            results_folder, f"{dataset}_{dataset_group}_{model_name}_{horizon}.json"
        )

    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            existing_results = json.load(f)
        row_forecast.update(existing_results)
        print(
            f"[SKIP] Results file '{results_file}' already exists. "
            "Loading existing results into row_forecast and skipping fresh compute."
        )
        return

    print(f"\n\n=== {dataset} {dataset_group} Forecast Evaluation ===\n")
    print(f"Forecast horizon = {horizon}, freq = {freq}, mode = {mode}\n")

    row_forecast["Dataset Source"] = dataset_source
    row_forecast["Dataset Group Source"] = dataset_group_source
    row_forecast["Dataset"] = dataset
    row_forecast["Group"] = dataset_group
    row_forecast["Forecast Horizon"] = horizon
    row_forecast["Method"] = model_name

    forecast_df_last_window_horizon = pipeline.predict_from_last_window_one_pass(
        model=model,
        window_size=window_size,
        window_size_source=window_size_source,
        mode=mode,
        dataset_target=dataset,
        dataset_group_target=dataset_group,
        dataset_source=dataset_source,
        dataset_group_source=dataset_group_source,
        freq=freq,
        h=horizon,
    )

    metric_prefix = f"Forecast {{metric}} {{stat}} (last window) Per Series_{mode}"

    if forecast_df_last_window_horizon.empty:
        print(f"[Last Window ({mode})] No forecast results found.")
        row_forecast[f"Forecast SMAPE MEAN (last window) Per Series_{mode}"] = None
    else:
        forecast_horizon = forecast_df_last_window_horizon.dropna(
            subset=["y", "y_true"]
        ).copy()
        if forecast_horizon.empty:
            print(
                f"[Last Window ({mode})] No valid y,y_true pairs. Can't compute sMAPE."
            )
            row_forecast[f"Forecast SMAPE (last window) Per Series_{mode}"] = None
        else:
            smape_series = forecast_horizon.groupby("unique_id").apply(
                lambda df: smape(df["y_true"], df["y"])
            )

            for stat_name, agg_func in {
                "MEDIAN": np.nanmedian,
                "MEAN": np.nanmean,
            }.items():
                value = float(round(agg_func(smape_series), 4))
                key = metric_prefix.format(metric="SMAPE", stat=stat_name)
                row_forecast[key] = value
                print(f"[sMAPE/{stat_name} ({mode})] = {value:.4f}")

            mase_series = forecast_df_last_window_horizon.groupby("unique_id").apply(
                lambda df: mase(
                    df["y_true"],
                    df["y"],
                    m=period,
                    h=int(min(window_size, window_size_source)),
                )
            )

            for stat_name, agg_func in {
                "MEDIAN": np.nanmedian,
                "MEAN": np.nanmean,
            }.items():
                value = float(round(agg_func(mase_series), 4))
                key = metric_prefix.format(metric="MASE", stat=stat_name)
                row_forecast[key] = value
                print(f"[MASE/{stat_name} ({mode})] = {value:.4f}")

            mae_series = forecast_horizon.groupby("unique_id").apply(
                lambda df: mae(df["y_true"], df["y"])
            )
            for stat, agg in {"MEDIAN": np.nanmedian, "MEAN": np.nanmean}.items():
                val = float(round(agg(mae_series), 4))
                row_forecast[metric_prefix.format(metric="MAE", stat=stat)] = val
                print(f"[MAE/{stat} ({mode})]  = {val:.4f}")

            rmse_series = forecast_horizon.groupby("unique_id").apply(
                lambda df: rmse(df["y_true"], df["y"])
            )
            for stat, agg in {"MEDIAN": np.nanmedian, "MEAN": np.nanmean}.items():
                val = float(round(agg(rmse_series), 4))
                row_forecast[metric_prefix.format(metric="RMSE", stat=stat)] = val
                print(f"[RMSE/{stat} ({mode})] = {val:.4f}")

            rmsse_series = forecast_df_last_window_horizon.groupby("unique_id").apply(
                lambda d: rmsse(
                    d["y_true"],
                    d["y"],
                    h=int(min(window_size, window_size_source)),
                    m=period,
                )
            )
            for stat, agg in {"MEDIAN": np.nanmedian, "MEAN": np.nanmean}.items():
                val = float(round(agg(rmsse_series), 4))
                row_forecast[metric_prefix.format(metric="RMSSE", stat=stat)] = val
                print(f"[RMSSE/{stat} ({mode})] = {val:.4f}")

    with open(results_file, "w") as f:
        json.dump(row_forecast, f)
    print(f"Results for forecast saved to '{results_file}'")
