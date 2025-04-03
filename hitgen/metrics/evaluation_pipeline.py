from typing import List, Dict, Union
import numpy as np
import os
import json
from tensorflow import keras
import tensorflow as tf

from hitgen.benchmarks.benchmark_models import BenchmarkPipeline
from hitgen.metrics.evaluation_metrics import (
    compute_discriminative_score,
    compute_downstream_forecast,
    tstr,
)
from hitgen.visualization import plot_generated_vs_original
from hitgen.model.create_dataset_versions_vae import HiTGenPipeline
from hitgen.metrics.evaluation_metrics import smape


def evaluation_pipeline_hitgen(
    dataset: str,
    dataset_group: str,
    model: keras.Model,
    pipeline: HiTGenPipeline,
    gen_data: tf.data.Dataset,
    sampling_strategy: str,
    freq: str,
    h: int,
    test_unique_ids: List,
    row_hitgen: Dict,
    noise_scale: int = 2.0,
):
    """
    Evaluate HiTGen synthetic data with the selected sampling strategy.

    Args:
        sampling_strategy: e.g.
           - "MR" for the usual .predict()
           - "NP" (No Prior) for .predict_random_latent()
           - "IR" (Increased Randomness) for .predict_guided_with_extra_noise()
    """
    print(f"\n\n{dataset}")
    print(f"{dataset_group}")
    print(f"\n\n----- HiTGen {sampling_strategy} synthetic data -----\n\n")

    if sampling_strategy == "MR":
        synth_hitgen_test_long = pipeline.predict(
            cvae=model,
            gen_data=gen_data,
            scaler=pipeline.scaler_test,
            original_data_wide=pipeline.original_test_wide,
            original_data_long=pipeline.original_test_long,
            unique_ids=pipeline.test_ids,
            ds=pipeline.ds_test,
        )

    elif sampling_strategy == "NP":
        synth_hitgen_test_long = pipeline.predict_random_latent(
            cvae=model,
            gen_data=gen_data,
            scaler=pipeline.scaler_test,
            original_data_wide=pipeline.original_test_wide,
            original_data_long=pipeline.original_test_long,
            unique_ids=pipeline.test_ids,
            ds=pipeline.ds_test,
        )

    elif sampling_strategy == "IR":
        synth_hitgen_test_long = pipeline.predict_guided_with_extra_noise(
            cvae=model,
            gen_data=gen_data,
            scaler=pipeline.scaler_test,
            original_data_wide=pipeline.original_test_wide,
            original_data_long=pipeline.original_test_long,
            unique_ids=pipeline.test_ids,
            ds=pipeline.ds_test,
            noise_scale=noise_scale,
        )

    else:
        raise ValueError(f"Unknown sampling_strategy='{sampling_strategy}'")

    plot_generated_vs_original(
        synth_data=synth_hitgen_test_long,
        original_data=pipeline.original_test_long,
        score=0.0,
        loss=0.0,
        dataset_name=dataset,
        dataset_group=dataset_group,
        n_series=8,
        suffix_name=f"hitgen_{sampling_strategy}",
    )

    print("\nComputing discriminative score for HiTGen synthetic data...")
    hitgen_score_disc = compute_discriminative_score(
        unique_ids=test_unique_ids,
        original_data=pipeline.original_test_long,
        synthetic_data=synth_hitgen_test_long,
        method=f"hitgen[{sampling_strategy}]",
        freq=freq,
        dataset_name=dataset,
        dataset_group=dataset_group,
        loss=0.0,
        samples=5,
        split="test",
    )

    print("\nComputing TSTR score for HiTGen synthetic data...")
    hitgen_score_tstr = tstr(
        unique_ids=test_unique_ids,
        original_data=pipeline.original_test_long,
        synthetic_data=synth_hitgen_test_long,
        method=f"hitgen[{sampling_strategy}]",
        freq=freq,
        horizon=h,
        dataset_name=dataset,
        dataset_group=dataset_group,
        samples=5,
        split="test",
    )

    print("\nComputing downstream task forecasting score for HiTGen synthetic data...")
    hitgen_score_dtf = compute_downstream_forecast(
        unique_ids=test_unique_ids,
        original_data=pipeline.original_test_long,
        synthetic_data=synth_hitgen_test_long,
        method=f"hitgen[{sampling_strategy}]",
        freq=freq,
        horizon=h,
        dataset_name=dataset,
        dataset_group=dataset_group,
        samples=10,
        split="test",
    )

    print(
        f"Discriminative score for HiTGen {sampling_strategy} synthetic data: {hitgen_score_disc:.4f}"
    )

    print(
        f"TSTR score for HiTGen {sampling_strategy} synthetic data: "
        f"TSTR {hitgen_score_tstr['avg_smape_tstr']:.4f} vs TRTR "
        f"score {hitgen_score_tstr['avg_smape_trtr']:.4f}"
    )

    print(
        f"Downstream task forecasting score for HiTGen {sampling_strategy} synthetic data: "
        f"concat avg score {hitgen_score_dtf['avg_smape_concat']:.4f} vs original "
        f"score {hitgen_score_dtf['avg_smape_original']:.4f}\n\n"
        f"concat std score {hitgen_score_dtf['std_smape_concat']:.4f} vs original "
        f"score {hitgen_score_dtf['std_smape_original']:.4f}\n\n"
    )

    row_hitgen[f"Dataset"] = dataset
    row_hitgen[f"Group"] = dataset_group
    row_hitgen[f"Method"] = f"hitgen"

    row_hitgen[f"Discriminative Score [{sampling_strategy}]"] = hitgen_score_disc
    row_hitgen[f"TSTR (avg_smape_tstr) [{sampling_strategy}]"] = hitgen_score_tstr[
        "avg_smape_tstr"
    ]
    row_hitgen[f"TRTR (avg_smape_trtr) [{sampling_strategy}]"] = hitgen_score_tstr[
        "avg_smape_trtr"
    ]
    row_hitgen[f"DTF Concat Avg Score [{sampling_strategy}]"] = hitgen_score_dtf[
        "avg_smape_concat"
    ]
    row_hitgen[f"DTF Original Avg Score [{sampling_strategy}]"] = hitgen_score_dtf[
        "avg_smape_original"
    ]
    row_hitgen[f"DTF Concat Std Score [{sampling_strategy}]"] = hitgen_score_dtf[
        "std_smape_concat"
    ]
    row_hitgen[f"DTF Original Std Score [{sampling_strategy}]"] = hitgen_score_dtf[
        "std_smape_original"
    ]


def evaluation_pipeline_hitgen_forecast(
    dataset: str,
    dataset_group: str,
    pipeline: Union[HiTGenPipeline, BenchmarkPipeline],
    model: Union[keras.Model, str],
    horizon: int,
    freq: str,
    row_forecast: dict,
    window_size: int = None,
) -> None:
    """
    Evaluate direct forecasting for the two different forecast approaches
    forecast_df_first_window, forecast_df_autoregressive and computes SMAPE
    """
    os.makedirs("assets/results_forecast", exist_ok=True)
    if isinstance(model, keras.Model):
        model_name = "hitgen"
    else:
        model_name = model
    results_file = (
        f"assets/results_forecast/{dataset}_{dataset_group}_{model_name}_{horizon}.json"
    )

    if os.path.exists(results_file):
        print(f"[SKIP] Results file '{results_file}' already exists. Skipping compute.")
        return

    print(f"\n\n=== {dataset} {dataset_group} Forecast Evaluation ===\n")
    print(f"Forecast horizon = {horizon}, freq = {freq}\n")

    forecast_df_first_window, forecast_df_autoregressive = (
        pipeline.predict_from_first_window(model=model, window_size=window_size)
    )

    if forecast_df_first_window.empty:
        print("[First Window] No forecast results found.")
        row_forecast["Forecast SMAPE (first window)"] = None
    else:
        forecast_df_first_window = forecast_df_first_window.dropna(
            subset=["y", "y_true"]
        )
        if forecast_df_first_window.empty:
            print("[First Window] No valid y,y_true pairs. Can't compute sMAPE.")
            row_forecast["Forecast SMAPE (first window)"] = None
        else:
            smape_result_fw_per_series = (
                forecast_df_first_window.dropna(subset=["y", "y_true"])
                .groupby("unique_id")
                .apply(lambda df: smape(df["y_true"], df["y"]))
            )
            smape_per_series_fw_median = smape_result_fw_per_series.median()

            print(
                f"\n[First Window Forecast per Series] sMAPE = {smape_per_series_fw_median:.4f}\n"
            )
            row_forecast["Forecast SMAPE (first window) Per Series"] = float(
                round(smape_per_series_fw_median, 4)
            )

    if forecast_df_autoregressive.empty:
        print("[Autoregressive] No forecast results found.")
        row_forecast["Forecast SMAPE (autoregressive)"] = None
    else:
        forecast_df_autoregressive = forecast_df_autoregressive.dropna(
            subset=["y", "y_true"]
        )
        if forecast_df_autoregressive.empty:
            print("[Autoregressive] No valid y,y_true pairs. Can't compute sMAPE.")
            row_forecast["Forecast SMAPE (autoregressive)"] = None
        else:
            smape_result_ar_per_series = (
                forecast_df_autoregressive.dropna(subset=["y", "y_true"])
                .groupby("unique_id")
                .apply(lambda df: smape(df["y_true"], df["y"]))
            )
            smape_per_series_ar_median = smape_result_ar_per_series.median()

            print(
                f"\n[Autoregressive Forecast] sMAPE = {smape_per_series_ar_median:.4f}\n"
            )
            row_forecast["Forecast SMAPE (autoregressive)"] = float(
                round(smape_per_series_ar_median, 4)
            )

    row_forecast["Dataset"] = dataset
    row_forecast["Group"] = dataset_group
    row_forecast["Forecast Horizon"] = horizon

    with open(results_file, "w") as f:
        json.dump(row_forecast, f)
    print(f"Results for forecast saved to '{results_file}'")
