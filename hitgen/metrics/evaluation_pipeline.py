import os
import json

from hitgen.benchmarks.model_pipeline import ModelPipeline, AutoModelType
from hitgen.metrics.evaluation_metrics import smape


def evaluation_pipeline_hitgen_forecast(
    dataset: str,
    dataset_group: str,
    pipeline: ModelPipeline,
    model: AutoModelType,
    horizon: int,
    freq: str,
    row_forecast: dict,
    window_size: int = None,
    dataset_source: str = None,
    prediction_mode: str = "in_domain",
) -> None:
    """
    Evaluate direct forecasting for up to three forecast approaches:
      1) 'first window' forecast horizon
      2) 'autoregressive' forecast of the entire series
      3) 'last window' forecast horizon
    and compute the sMAPE per series.
    """
    os.makedirs("assets/results_forecast", exist_ok=True)
    os.makedirs("assets/results_forecast_tl", exist_ok=True)

    if isinstance(model, AutoModelType):
        model_name = model.__class__.__name__
    else:
        raise TypeError(
            f"Unsupported model type: {type(model).__name__}. "
            "Expected a Keras model or an AutoModelType instance."
        )

    if dataset_source:
        results_file = f"assets/results_forecast_tl/{dataset}_{dataset_group}_{model_name}_{horizon}_TL_trained_on_{dataset_source}.json"
    else:
        results_file = f"assets/results_forecast/{dataset}_{dataset_group}_{model_name}_{horizon}.json"

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
    print(f"Forecast horizon = {horizon}, freq = {freq}\n")

    row_forecast["Dataset"] = dataset
    row_forecast["Group"] = dataset_group
    row_forecast["Forecast Horizon"] = horizon
    row_forecast["Method"] = model_name

    forecast_df_last_window_horizon, forecast_df_last_window_all = (
        pipeline.predict_from_last_window_one_pass(
            model=model, window_size=window_size, prediction_mode=prediction_mode
        )
    )

    if forecast_df_last_window_horizon.empty:
        print("[Last Window] No forecast results found.")
        row_forecast["Forecast SMAPE (last window) Per Series"] = None
    else:
        forecast_df_last_window_horizon = forecast_df_last_window_horizon.dropna(
            subset=["y", "y_true"]
        )
        if forecast_df_last_window_horizon.empty:
            print("[Last Window] No valid y,y_true pairs. Can't compute sMAPE.")
            row_forecast["Forecast SMAPE (last window) Per Series"] = None
        else:
            smape_result_lw_per_series = forecast_df_last_window_horizon.groupby(
                "unique_id"
            ).apply(lambda df: smape(df["y_true"], df["y"]))
            smape_per_series_lw_median = smape_result_lw_per_series.median()
            print(
                f"\n[Last Window Forecast per Series] sMAPE MEDIAN = {smape_per_series_lw_median:.4f}\n"
            )
            row_forecast["Forecast SMAPE MEDIAN (last window) Per Series"] = float(
                round(smape_per_series_lw_median, 4)
            )

            smape_per_series_lw_mean = smape_result_lw_per_series.mean()
            print(
                f"\n[Last Window Forecast per Series] sMAPE MEAN = {smape_per_series_lw_mean:.4f}\n"
            )
            row_forecast["Forecast SMAPE MEAN (last window) Per Series"] = float(
                round(smape_per_series_lw_mean, 4)
            )

    with open(results_file, "w") as f:
        json.dump(row_forecast, f)
    print(f"Results for forecast saved to '{results_file}'")
