from typing import List, Dict
from tensorflow import keras
import tensorflow as tf
from hitgen.metrics.evaluation_metrics import (
    compute_discriminative_score,
    compute_downstream_forecast,
    tstr,
)
from hitgen.visualization import plot_generated_vs_original
from hitgen.model.create_dataset_versions_vae import HiTGenPipeline


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

    elif sampling_strategy == "TNP":
        synth_hitgen_test_long = pipeline.predict_random_latent(
            cvae=model,
            gen_data=gen_data,
            scaler=pipeline.scaler,
            original_data_wide=pipeline.original_wide,
            original_data_long=pipeline.original_long,
            unique_ids=pipeline.unique_ids_original,
            ds=pipeline.ds_original,
            filter_series=True,
            unique_ids_filter=pipeline.test_ids,
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

    return row_hitgen
