import multiprocessing
import os
import argparse
import pandas as pd
from hitgen.model.create_dataset_versions_vae import (
    CreateTransformedVersionsCVAE,
)
from hitgen.model.models import build_tf_dataset
from hitgen.metrics.discriminative_score import (
    compute_discriminative_score,
    compute_downstream_forecast,
    tstr,
)
from hitgen.visualization import plot_generated_vs_original
from hitgen.benchmarks.metaforecast import workflow_metaforecast_methods


DATASET_GROUP_FREQ = {
    # "Tourism": {
    #     "Monthly": {"FREQ": "M", "H": 24},
    # },
    # "M1": {
    #     "Monthly": {"FREQ": "M", "H": 24},
    #     "Quarterly": {"FREQ": "Q", "H": 8},
    # },
    "M3": {
        "Monthly": {"FREQ": "M", "H": 24},
        # "Quarterly": {"FREQ": "Q", "H": 8},
        # "Yearly": {"FREQ": "Y", "H": 4},
    },
}


METAFORECAST_METHODS = [
    "DBA",
    "Jitter",
    "Scaling",
    "MagWarp",
    "TimeWarp",
    "MBB",
    # "TSMixup",
]


def extract_frequency(dataset_group):
    """Safely extracts frequency from dataset group."""
    freq = dataset_group[1]["FREQ"]
    return freq


def extract_horizon(dataset_group):
    """Safely extracts horizon from dataset group."""
    h = dataset_group[1]["H"]
    return h


def extract_score(dataset_group):
    """Safely extracts frequency from dataset group."""
    score = dataset_group[1]["final_score"]
    return score


def has_final_score_in_tuple(tpl):
    """Check if the second element is a dictionary and contains 'final_score'"""
    return isinstance(tpl[1], dict) and "final_score" in tpl[1]


def set_device(
    use_gpu: bool,
):
    """Configures TensorFlow to use GPU or CPU."""
    if not use_gpu:
        print("Using CPU as specified by the user.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(
        description="Run synthetic data generation using HiTGen."
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU if available (default: False, meaning it runs on CPU).",
    )
    parser.add_argument(
        "--opt-score",
        choices=["discriminative_score", "downstream_score", "val_loss"],
        default="val_loss",
        help="Select the score for the hyperparameter tuning optimization. Choices: "
        "'discriminative_score', 'downstream_score' or 'val_loss' (default: 'val_loss').",
    )
    args = parser.parse_args()

    set_device(args.use_gpu)

    results = []

    for DATASET, SUBGROUPS in DATASET_GROUP_FREQ.items():
        for subgroup in SUBGROUPS.items():
            dataset_group_results = []

            FREQ = extract_frequency(subgroup)
            H = extract_horizon(subgroup)
            DATASET_GROUP = subgroup[0]
            hitgen_score_disc = None
            if has_final_score_in_tuple(subgroup):
                print(
                    f"Dataset: {DATASET}, Dataset-group: {DATASET_GROUP}, Frequency: {FREQ} "
                    f"has a final score already, skipping HiTGen scores computations..."
                )
                hitgen_score_disc = extract_score(subgroup)

            print(
                f"Dataset: {DATASET}, Dataset-group: {DATASET_GROUP}, Frequency: {FREQ}"
            )

            print(
                f"\n\nOptimization score to use for hyptertuning: {args.opt_score}\n\n"
            )

            SYNTHETIC_FILE_PATH_HITGEN = (
                f"assets/model_weights/{DATASET}_{DATASET_GROUP}_synthetic_hitgen.pkl"
            )
            SYNTHETIC_FILE_PATH_TIMEGAN = (
                f"assets/model_weights/{DATASET}_{DATASET_GROUP}_synthetic_timegan.pkl"
            )

            create_dataset_vae = CreateTransformedVersionsCVAE(
                dataset_name=DATASET,
                dataset_group=DATASET_GROUP,
                freq=FREQ,
                horizon=H,
                opt_score=args.opt_score,
            )

            # hypertuning
            model = create_dataset_vae.hyper_tune_and_train()

            data_mask_temporalized_test = build_tf_dataset(
                data=create_dataset_vae.original_test_wide_transf,
                mask=create_dataset_vae.mask_test_wide,
                dyn_features=create_dataset_vae.test_dyn_features,
                window_size=create_dataset_vae.best_params["window_size"],
                batch_size=create_dataset_vae.best_params["batch_size"],
                windows_batch_size=create_dataset_vae.best_params["windows_batch_size"],
                stride=1,
                coverage_mode="systematic",
                prediction_mode=create_dataset_vae.best_params["prediction_mode"],
                future_steps=create_dataset_vae.best_params["future_steps"],
            )

            synth_hitgen_test_long = create_dataset_vae.predict(
                model,
                gen_data=data_mask_temporalized_test,
                scaler=create_dataset_vae.scaler_test,
                original_data_wide=create_dataset_vae.original_test_wide,
                original_data_long=create_dataset_vae.original_test_long,
                unique_ids=create_dataset_vae.test_ids,
                ds=create_dataset_vae.ds_test,
            )

            test_unique_ids = create_dataset_vae.original_test_long[
                "unique_id"
            ].unique()

            # metaforecast methods
            synthetic_metaforecast_long = workflow_metaforecast_methods(
                df=create_dataset_vae.original_test_long,
                freq=FREQ,
                dataset=DATASET,
                dataset_group=DATASET_GROUP,
            )

            plot_generated_vs_original(
                synth_data=synth_hitgen_test_long,
                original_data=create_dataset_vae.original_test_long,
                score=0.0,
                loss=0.0,
                dataset_name=DATASET,
                dataset_group=DATASET_GROUP,
                n_series=8,
                suffix_name="hitgen",
            )

            if not hitgen_score_disc:
                print("\nComputing discriminative score for HiTGen synthetic data...")
                hitgen_score_disc = compute_discriminative_score(
                    unique_ids=test_unique_ids,
                    original_data=create_dataset_vae.original_test_long,
                    synthetic_data=synth_hitgen_test_long,
                    method="hitgen",
                    freq=FREQ,
                    dataset_name=DATASET,
                    dataset_group=DATASET_GROUP,
                    loss=0.0,
                    samples=5,
                    split="test",
                )

            print("\nComputing TSTR score for HiTGen synthetic data...")
            hitgen_score_tstr = tstr(
                unique_ids=test_unique_ids,
                original_data=create_dataset_vae.original_test_long,
                synthetic_data=synth_hitgen_test_long,
                method="hitgen",
                freq=FREQ,
                horizon=H,
                dataset_name=DATASET,
                dataset_group=DATASET_GROUP,
                samples=5,
                split="test",
            )

            print(
                "\nComputing downstream task forecasting score for HiTGen synthetic data..."
            )
            hitgen_score_dtf = compute_downstream_forecast(
                unique_ids=test_unique_ids,
                original_data=create_dataset_vae.original_test_long,
                synthetic_data=synth_hitgen_test_long,
                method="hitgen",
                freq=FREQ,
                horizon=H,
                dataset_name=DATASET,
                dataset_group=DATASET_GROUP,
                samples=10,
                split="test",
            )

            print(f"\n\n{DATASET}")
            print(f"{DATASET_GROUP}")

            print(
                f"Discriminative score for HiTGen synthetic data: {hitgen_score_disc:.4f}"
            )

            print(
                f"TSTR score for HiTGen synthetic data: "
                f"TSTR {hitgen_score_tstr['avg_smape_tstr']:.4f} vs TRTR "
                f"score {hitgen_score_tstr['avg_smape_trtr']:.4f}"
            )

            print(
                f"Downstream task forecasting score for HiTGen synthetic data: "
                f"concat avg score {hitgen_score_dtf['avg_smape_concat']:.4f} vs original "
                f"score {hitgen_score_dtf['avg_smape_original']:.4f}\n\n"
                f"concat std score {hitgen_score_dtf['std_smape_concat']:.4f} vs original "
                f"score {hitgen_score_dtf['std_smape_original']:.4f}\n\n"
            )

            row_hitgen = {
                "Dataset": DATASET,
                "Group": DATASET_GROUP,
                "Method": "HiTGen",
                "Discriminative Score": hitgen_score_disc,
                "TSTR (avg_smape_tstr)": hitgen_score_tstr["avg_smape_tstr"],
                "TRTR (avg_smape_trtr)": hitgen_score_tstr["avg_smape_trtr"],
                "DTF Concat Avg Score": hitgen_score_dtf["avg_smape_concat"],
                "DTF Original Avg Score": hitgen_score_dtf["avg_smape_original"],
                "DTF Concat Std Score": hitgen_score_dtf["std_smape_concat"],
                "DTF Original Std Score": hitgen_score_dtf["std_smape_original"],
            }
            dataset_group_results.append(row_hitgen)
            results.append(row_hitgen)

            for method in METAFORECAST_METHODS:
                synthetic_metaforecast_long_method = synthetic_metaforecast_long.loc[
                    synthetic_metaforecast_long["method"] == method
                ].copy()

                synthetic_metaforecast_long_method = (
                    synthetic_metaforecast_long_method.merge(
                        create_dataset_vae.original_test_long[["unique_id", "ds"]],
                        on=["unique_id", "ds"],
                        how="right",
                    )
                )
                synthetic_metaforecast_long_method.fillna(0, inplace=True)

                plot_generated_vs_original(
                    synth_data=synthetic_metaforecast_long_method,
                    original_data=create_dataset_vae.original_test_long,
                    score=0.0,
                    loss=0.0,
                    dataset_name=DATASET,
                    dataset_group=DATASET_GROUP,
                    n_series=4,
                    suffix_name=f"{method}",
                )

                print(
                    f"\nComputing discriminative score for {method} synthetic data generation..."
                )

                score_disc = compute_discriminative_score(
                    unique_ids=test_unique_ids,
                    original_data=create_dataset_vae.original_test_long,
                    synthetic_data=synthetic_metaforecast_long_method,
                    method=method,
                    freq=FREQ,
                    dataset_name=DATASET,
                    dataset_group=DATASET_GROUP,
                    loss=0.0,
                    samples=5,
                    split="test",
                )

                print(f"\nComputing TSTR score for {method} synthetic data...")
                score_tstr = tstr(
                    unique_ids=test_unique_ids,
                    original_data=create_dataset_vae.original_test_long,
                    synthetic_data=synthetic_metaforecast_long_method,
                    method=method,
                    freq=FREQ,
                    horizon=H,
                    dataset_name=DATASET,
                    dataset_group=DATASET_GROUP,
                    samples=5,
                    split="test",
                )

                print(
                    f"\nComputing downstream task forecasting score for {method} synthetic data..."
                )
                score_dtf = compute_downstream_forecast(
                    unique_ids=test_unique_ids,
                    original_data=create_dataset_vae.original_test_long,
                    synthetic_data=synthetic_metaforecast_long_method,
                    method=method,
                    freq=FREQ,
                    horizon=H,
                    dataset_name=DATASET,
                    dataset_group=DATASET_GROUP,
                    samples=10,
                    split="test",
                )

                print(f"\n\n{DATASET}")
                print(f"{DATASET_GROUP}")

                print(
                    f"Discriminative score for {method} synthetic data: {score_disc:.4f}"
                )

                print(
                    f"TSTR score for {method} synthetic data: "
                    f"TSTR {score_tstr['avg_smape_tstr']:.4f} vs TRTR "
                    f"score {score_tstr['avg_smape_trtr']:.4f}"
                )

                print(
                    f"Downstream task forecasting score for {method} synthetic data: "
                    f"concat score {score_dtf['avg_smape_concat']:.4f} vs original "
                    f"score {score_dtf['avg_smape_original']:.4f}\n\n"
                    f"concat std score {score_dtf['std_smape_concat']:.4f} vs original "
                    f"score {score_dtf['std_smape_original']:.4f}\n\n"
                )

                row_method = {
                    "Dataset": DATASET,
                    "Group": DATASET_GROUP,
                    "Method": method,
                    "Discriminative Score": score_disc,
                    "TSTR (avg_smape_tstr)": score_tstr["avg_smape_tstr"],
                    "TRTR (avg_smape_trtr)": score_tstr["avg_smape_trtr"],
                    "DTF Concat Avg Score": score_dtf["avg_smape_concat"],
                    "DTF Original Avg Score": score_dtf["avg_smape_original"],
                    "DTF Concat Std Score": score_dtf["std_smape_concat"],
                    "DTF Original Std Score": score_dtf["std_smape_original"],
                }
                dataset_group_results.append(row_method)
                results.append(row_method)

            dataset_group_df = pd.DataFrame(dataset_group_results).round(3)
            os.makedirs("assets/results", exist_ok=True)

            dataset_group_df_results_path = (
                f"assets/results/{DATASET}_{DATASET_GROUP}_synthetic_data_results.csv"
            )
            dataset_group_df.to_csv(dataset_group_df_results_path, index=False)
            print(
                f"\n==> Saved results for {DATASET} {DATASET_GROUP} to {dataset_group_df_results_path}\n"
            )

    all_results_df = pd.DataFrame(results).round(3)
    all_results_path = "assets/results/synthetic_data_results.csv"
    all_results_df.to_csv(all_results_path, index=False)

    print(f"==> Saved consolidated results to {all_results_path}")
