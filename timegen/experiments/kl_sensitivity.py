"""
KL Weight Sensitivity Analysis.

Trains TimeGEN with different kl_weight values on a few representative datasets
in the multi-source (LOO) setting and evaluates on the held-out target.

Follows the same coreset pipeline pattern as run_pipeline.py --coreset.

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

from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator
from neuralforecast.losses.pytorch import MAE

from timegen.data_pipeline.data_pipeline_setup import DataPipeline, build_mixed_trainval
from timegen.model_pipeline.model_pipeline import ModelPipeline
from timegen.model_pipeline.core.core_extension import CustomNeuralForecast
from timegen.model_pipeline.auto.AutoModels import AutoTimeGEN
from timegen.metrics.evaluation_pipeline import evaluation_pipeline_timegen_forecast
from timegen.experiments.helper import set_device


# All datasets for building multi-source training sets
DATASET_GROUP_FREQ = {
    "M1": {"Monthly": {"FREQ": "M", "H": 24}, "Quarterly": {"FREQ": "Q", "H": 8}},
    "M3": {"Monthly": {"FREQ": "M", "H": 24}, "Quarterly": {"FREQ": "Q", "H": 8}, "Yearly": {"FREQ": "Y", "H": 4}},
    "M4": {"Monthly": {"FREQ": "M", "H": 24}, "Quarterly": {"FREQ": "Q", "H": 8}},
    "M5": {"Daily": {"FREQ": "D", "H": 30}},
    "Tourism": {"Monthly": {"FREQ": "M", "H": 24}},
    "Traffic": {"Daily": {"FREQ": "D", "H": 30}},
}

# Targets to sweep (representative: one Monthly, one Quarterly, one Daily)
TARGET_DATASETS = [
    ("M3", "Monthly"),
    ("M4", "Quarterly"),
    ("Traffic", "Daily"),
]

KL_WEIGHTS = [0.0, 0.1, 0.3, 0.5, 1.0]

RESULTS_DIR = PROJECT_ROOT / "assets" / "results_kl_sensitivity"


def run_kl_sensitivity(use_gpu: bool = False, max_evals: int = 20):
    """Main KL sensitivity sweep using the coreset pipeline pattern."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_device(use_gpu)

    # Build all data pipelines
    all_data_pipelines = {}
    for ds, groups in DATASET_GROUP_FREQ.items():
        for grp, meta in groups.items():
            all_data_pipelines[(ds, grp)] = DataPipeline(
                dataset_name=ds,
                dataset_group=grp,
                freq=meta["FREQ"],
                horizon=meta["H"],
                window_size=meta["H"],
            )

    all_results = []

    for target_ds, target_grp in TARGET_DATASETS:
        target_key = f"{target_ds}_{target_grp}"
        target_pipeline = all_data_pipelines[(target_ds, target_grp)]
        target_meta = DATASET_GROUP_FREQ[target_ds][target_grp]

        print(f"\n{'='*60}")
        print(f"Target: {target_key} (h={target_meta['H']}, freq={target_meta['FREQ']})")
        print(f"{'='*60}")

        # Build source pipelines (all except target)
        source_pipelines = [
            dp for (ds, grp), dp in all_data_pipelines.items()
            if (ds, grp) != (target_ds, target_grp)
        ]

        dataset_source = "MIXED"
        dataset_group_source = f"ALL_BUT_{target_key}"
        mixed_trainval = build_mixed_trainval(
            source_pipelines,
            dataset_source=dataset_source,
            dataset_group=dataset_group_source,
        )

        # Evaluation pipeline for held-out target
        heldout_mp = ModelPipeline(target_pipeline)

        for kl_weight in KL_WEIGHTS:
            result_file = RESULTS_DIR / f"{target_key}_kl_{kl_weight}.json"
            if result_file.exists():
                with open(result_file) as f:
                    existing = json.load(f)
                if existing.get("mase_mean") is not None and not np.isnan(existing["mase_mean"]):
                    print(f"  [SKIP] kl_weight={kl_weight} already computed: MASE={existing['mase_mean']}")
                    all_results.append(existing)
                    continue
                else:
                    print(f"  [RETRY] kl_weight={kl_weight} had NaN, re-running...")

            print(f"\n  Training with kl_weight={kl_weight}...")

            # Build config with fixed kl_weight (following hyper_tune_and_train pattern)
            base_config = AutoTimeGEN.get_default_config(h=target_meta["H"], backend="ray")
            base_config["start_padding_enabled"] = True
            base_config["scaler_type"] = tune.choice(["revin"])
            base_config["kl_weight"] = kl_weight  # Fixed, not tuned

            model = AutoTimeGEN(
                h=target_meta["H"],
                loss=MAE(),
                config=base_config,
                num_samples=max_evals,
                search_alg=BasicVariantGenerator(random_state=1),
                verbose=True,
            )

            nf = CustomNeuralForecast(
                models=[model],
                freq="mixed",
            )

            # Save path for this specific kl_weight run
            save_dir = str(RESULTS_DIR / "models")
            os.makedirs(save_dir, exist_ok=True)
            nf_save_path = os.path.join(
                save_dir,
                f"{dataset_source}_{dataset_group_source}_kl{kl_weight}_neuralforecast",
            )

            if os.path.exists(nf_save_path):
                print(f"    Loading existing model from {nf_save_path}")
                nf_trained = CustomNeuralForecast.load(path=nf_save_path)
            else:
                nf.fit(df=mixed_trainval)
                nf.save(path=nf_save_path, overwrite=True)
                nf_trained = nf

            # Evaluate on held-out target
            row = {}
            evaluation_pipeline_timegen_forecast(
                dataset=target_ds,
                dataset_group=target_grp,
                pipeline=heldout_mp,
                model=nf_trained,
                horizon=target_pipeline.h,
                freq=target_pipeline.freq,
                period=target_pipeline.period,
                row_forecast=row,
                dataset_source=dataset_source,
                dataset_group_source=f"{dataset_group_source}_kl{kl_weight}",
                mode="out_domain",
                window_size=target_pipeline.h,
                window_size_source=target_pipeline.h,
            )

            # Extract MASE from the evaluation result
            mase_key = [k for k in row if "MASE" in k and "MEAN" in k]
            mase_mean = float(row[mase_key[0]]) if mase_key and row[mase_key[0]] is not None else float("nan")

            result = {
                "target": target_key,
                "kl_weight": kl_weight,
                "mase_mean": round(mase_mean, 4) if not np.isnan(mase_mean) else None,
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
    parser.add_argument("--max-evals", type=int, default=20, help="HP search samples per run")
    args = parser.parse_args()

    run_kl_sensitivity(use_gpu=args.use_gpu, max_evals=args.max_evals)
