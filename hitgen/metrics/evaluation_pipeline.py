import os
import json

from hitgen.model_pipeline.model_pipeline import ModelPipeline
from hitgen.model_pipeline.core.core_extension import CustomNeuralForecast
from hitgen.metrics.evaluation_metrics import smape


def evaluation_pipeline_hitgen_forecast(
    dataset: str,
    dataset_group: str,
    pipeline: ModelPipeline,
    model: CustomNeuralForecast,
    horizon: int,
    freq: str,
    row_forecast: dict,
    window_size: int = None,
    dataset_source: str = None,
    dataset_group_source: str = None,
    mode: str = "in_domain",
) -> None:
    """
    Evaluate direct forecasting for up to three forecast approaches:
      1) 'first window' forecast horizon
      2) 'autoregressive' forecast of the entire series
      3) 'last window' forecast horizon
    and compute the sMAPE per series.
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
        model=model, window_size=window_size, mode=mode
    )

    if forecast_df_last_window_horizon.empty:
        print(f"[Last Window ({mode})] No forecast results found.")
        row_forecast[f"Forecast SMAPE MEAN (last window) Per Series_{mode}"] = None
    else:
        forecast_df_last_window_horizon = forecast_df_last_window_horizon.dropna(
            subset=["y", "y_true"]
        )
        if forecast_df_last_window_horizon.empty:
            print(
                f"[Last Window ({mode})] No valid y,y_true pairs. Can't compute sMAPE."
            )
            row_forecast[f"Forecast SMAPE (last window) Per Series_{mode}"] = None
        else:
            smape_result_lw_per_series = forecast_df_last_window_horizon.groupby(
                "unique_id"
            ).apply(lambda df: smape(df["y_true"], df["y"]))
            smape_per_series_lw_median = smape_result_lw_per_series.median()
            print(
                f"\n[Last Window Forecast per Series ({mode})] "
                f"sMAPE MEDIAN = {smape_per_series_lw_median:.4f}\n"
            )
            row_forecast[f"Forecast SMAPE MEDIAN (last window) Per Series_{mode}"] = (
                float(round(smape_per_series_lw_median, 4))
            )

            smape_per_series_lw_mean = smape_result_lw_per_series.mean()
            print(
                f"\n[Last Window Forecast per Series ({mode})] "
                f"sMAPE MEAN = {smape_per_series_lw_mean:.4f}\n"
            )
            row_forecast[f"Forecast SMAPE MEAN (last window) Per Series_{mode}"] = (
                float(round(smape_per_series_lw_mean, 4))
            )

    with open(results_file, "w") as f:
        json.dump(row_forecast, f)
    print(f"Results for forecast saved to '{results_file}'")
