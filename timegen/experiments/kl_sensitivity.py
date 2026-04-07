"""
KL Weight Sensitivity Analysis.

Trains TimeGEN with different kl_weight values on a few representative datasets
in the multi-source (LOO) setting and evaluates on the held-out target.

Usage:
    conda run -n TimeGEN python -m timegen.experiments.kl_sensitivity --use-gpu

Results are saved to assets/results_kl_sensitivity/
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from timegen.data_pipeline.data_pipeline_setup import DataPipeline, build_mixed_trainval
from timegen.model_pipeline.model_pipeline import ModelPipeline
from timegen.model_pipeline.core.core_extension import CustomNeuralForecast
from timegen.model_pipeline.auto.AutoModels import AutoTimeGEN
from timegen.metrics.evaluation_metrics import mase
from timegen.experiments.helper import set_device

from ray.tune.search.basic_variant import BasicVariantGenerator
from neuralforecast.losses.pytorch import MAE
from ray import tune


# Datasets to test (representative: one Monthly, one Quarterly, one Daily)
TARGET_DATASETS = [
    {"dataset_name": "M3", "dataset_group": "Monthly", "freq": "M", "h": 24},
    {"dataset_name": "M4", "dataset_group": "Quarterly", "freq": "Q", "h": 8},
    {"dataset_name": "Traffic", "dataset_group": "Daily", "freq": "D", "h": 30},
]

# All datasets for building multi-source training sets
ALL_DATASETS = [
    {"dataset_name": "M1", "dataset_group": "Monthly", "freq": "M", "h": 24},
    {"dataset_name": "M1", "dataset_group": "Quarterly", "freq": "Q", "h": 8},
    {"dataset_name": "M3", "dataset_group": "Monthly", "freq": "M", "h": 24},
    {"dataset_name": "M3", "dataset_group": "Quarterly", "freq": "Q", "h": 8},
    {"dataset_name": "M3", "dataset_group": "Yearly", "freq": "Y", "h": 4},
    {"dataset_name": "M4", "dataset_group": "Monthly", "freq": "M", "h": 24},
    {"dataset_name": "M4", "dataset_group": "Quarterly", "freq": "Q", "h": 8},
    {"dataset_name": "M5", "dataset_group": "Daily", "freq": "D", "h": 30},
    {"dataset_name": "Tourism", "dataset_group": "Monthly", "freq": "M", "h": 24},
    {"dataset_name": "Traffic", "dataset_group": "Daily", "freq": "D", "h": 30},
]

KL_WEIGHTS = [0.0, 0.1, 0.3, 0.5, 1.0]

RESULTS_DIR = PROJECT_ROOT / "assets" / "results_kl_sensitivity"


def get_source_pipelines(target_config):
    """Build data pipelines for all datasets EXCEPT the target (LOO)."""
    pipelines = []
    for cfg in ALL_DATASETS:
        if cfg["dataset_name"] == target_config["dataset_name"] and \
           cfg["dataset_group"] == target_config["dataset_group"]:
            continue
        # Only include datasets with matching horizon for simplicity
        if cfg["h"] != target_config["h"]:
            continue
        dp = DataPipeline(
            dataset_name=cfg["dataset_name"],
            dataset_group=cfg["dataset_group"],
            freq=cfg["freq"],
            horizon=cfg["h"],
            window_size=cfg["h"],
        )
        pipelines.append(dp)
    return pipelines


def run_kl_sensitivity(use_gpu: bool = False):
    """Main KL sensitivity sweep."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_device(use_gpu)

    all_results = []

    for target_config in TARGET_DATASETS:
        target_key = f"{target_config['dataset_name']}_{target_config['dataset_group']}"
        print(f"\n{'='*60}")
        print(f"Target: {target_key}")
        print(f"{'='*60}")

        # Build source training data (all except target)
        source_pipelines = get_source_pipelines(target_config)
        if not source_pipelines:
            print(f"  No source datasets with matching horizon, skipping")
            continue

        mixed_train = build_mixed_trainval(
            source_pipelines,
            target_config["dataset_name"],
            target_config["dataset_group"],
        )

        # Build target pipeline for evaluation
        target_pipeline = DataPipeline(
            dataset_name=target_config["dataset_name"],
            dataset_group=target_config["dataset_group"],
            freq=target_config["freq"],
            horizon=target_config["h"],
            window_size=target_config["h"],
        )

        for kl_weight in KL_WEIGHTS:
            result_file = RESULTS_DIR / f"{target_key}_kl_{kl_weight}.json"
            if result_file.exists():
                print(f"  [SKIP] kl_weight={kl_weight} already computed")
                with open(result_file) as f:
                    all_results.append(json.load(f))
                continue

            print(f"\n  Training with kl_weight={kl_weight}...")

            # Create AutoTimeGEN with fixed kl_weight
            config = AutoTimeGEN.get_default_config(
                h=target_config["h"], backend="ray"
            )
            # Override kl_weight to fixed value
            config["kl_weight"] = kl_weight

            model = AutoTimeGEN(
                h=target_config["h"],
                loss=MAE(),
                config=config,
                num_samples=20,
                search_alg=BasicVariantGenerator(random_state=1),
            )

            # Train
            nf = CustomNeuralForecast(
                models=[model],
                freq=target_config["freq"],
            )

            nf.fit(df=mixed_train)

            # Evaluate on target
            mp = ModelPipeline(target_pipeline)
            forecast_df = mp.predict_from_last_window_one_pass(
                model=nf,
                window_size=target_config["h"],
                window_size_source=target_config["h"],
                dataset_target=target_config["dataset_name"],
                dataset_group_target=target_config["dataset_group"],
                dataset_source="MIXED",
                dataset_group_source="ALL",
                freq=target_config["freq"],
                h=target_config["h"],
                mode="out_domain",
            )

            if forecast_df.empty:
                print(f"    No predictions for kl_weight={kl_weight}")
                continue

            # Compute MASE
            forecast_df = forecast_df.dropna(subset=["y", "y_true"])
            period = {"M": 12, "Q": 4, "Y": 1, "D": 7}.get(target_config["freq"], 1)
            mase_series = forecast_df.groupby("unique_id").apply(
                lambda df: mase(
                    df["y_true"], df["y"],
                    m=period, h=target_config["h"]
                )
            )
            mase_mean = float(np.nanmean(mase_series))

            result = {
                "target": target_key,
                "kl_weight": kl_weight,
                "mase_mean": round(mase_mean, 4),
            }
            all_results.append(result)

            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"    MASE={mase_mean:.4f}")

    # Summary table
    print(f"\n{'='*60}")
    print("KL Weight Sensitivity Summary")
    print(f"{'='*60}")
    df = pd.DataFrame(all_results)
    if not df.empty:
        pivot = df.pivot(index="kl_weight", columns="target", values="mase_mean")
        pivot["Average"] = pivot.mean(axis=1)
        print(pivot.round(4).to_string())
        pivot.round(4).to_csv(RESULTS_DIR / "kl_sensitivity_summary.csv")
        print(f"\nSaved to {RESULTS_DIR / 'kl_sensitivity_summary.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KL Weight Sensitivity Analysis")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()

    run_kl_sensitivity(use_gpu=args.use_gpu)
