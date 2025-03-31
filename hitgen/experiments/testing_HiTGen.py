import multiprocessing
import os
import pandas as pd
from hitgen.model.create_dataset_versions_vae import (
    HiTGenPipeline,
)
from hitgen.model.models import build_tf_dataset, build_tf_dataset_multivariate
from hitgen.metrics.evaluation_metrics import (
    compute_discriminative_score,
    compute_downstream_forecast,
    tstr,
)
from hitgen.visualization import plot_generated_vs_original
from hitgen.benchmarks.metaforecast import workflow_metaforecast_methods
from hitgen.metrics.evaluation_pipeline import (
    evaluation_pipeline_hitgen,
    evaluation_pipeline_hitgen_forecast,
)
from hitgen.experiments.helper import (
    extract_score,
    extract_frequency,
    extract_horizon,
    has_final_score_in_tuple,
    cmd_parser,
)


DATASET_GROUP_FREQ = {
    "Tourism": {
        "Monthly": {"FREQ": "M", "H": 24},
    },
    # "M1": {
    # "Monthly": {"FREQ": "M", "H": 24},
    # "Quarterly": {"FREQ": "Q", "H": 8},
    # },
    # "M3": {
    #     "Monthly": {"FREQ": "M", "H": 24},
    # "Quarterly": {"FREQ": "Q", "H": 8},
    # "Yearly": {"FREQ": "Y", "H": 4},
    # },
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


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    args = cmd_parser()

    results = []

    for DATASET, SUBGROUPS in DATASET_GROUP_FREQ.items():
        for subgroup in SUBGROUPS.items():
            dataset_group_results = []

            SAMPLING_STRATEGIES = ["MR", "IR", "NP"]
            SAMPLING_STRATEGIES_MULTIVAR = ["MR_Multivar", "IR_Multivar", "NP_Multivar"]
            FREQ = extract_frequency(subgroup)
            H = extract_horizon(subgroup)
            DATASET_GROUP = subgroup[0]
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

            hitgen_pipeline = HiTGenPipeline(
                dataset_name=DATASET,
                dataset_group=DATASET_GROUP,
                freq=FREQ,
                horizon=H,
                opt_score=args.opt_score,
            )

            # hypertuning
            # model = hitgen_pipeline.hyper_tune_and_train()
            model_forecasting = hitgen_pipeline.hyper_tune_and_train_forecasting()
            # model_multivariate = hitgen_pipeline.hyper_tune_and_train_multivariate()

            test_unique_ids = hitgen_pipeline.original_test_long["unique_id"].unique()

            # data_mask_temporalized_test = build_tf_dataset(
            #     data=hitgen_pipeline.original_test_wide_transf,
            #     mask=hitgen_pipeline.mask_test_wide,
            #     dyn_features=hitgen_pipeline.test_dyn_features,
            #     window_size=hitgen_pipeline.best_params["window_size"],
            #     batch_size=hitgen_pipeline.best_params["batch_size"],
            #     windows_batch_size=hitgen_pipeline.best_params["windows_batch_size"],
            #     stride=1,
            #     coverage_mode="systematic",
            #     prediction_mode=hitgen_pipeline.best_params["prediction_mode"],
            #     future_steps=hitgen_pipeline.best_params["future_steps"],
            #     cache_dataset_name=DATASET,
            #     cache_dataset_group=DATASET_GROUP,
            #     cache_split="test",
            # )

            # data_mask_temporalized_multivar_test = build_tf_dataset_multivariate(
            #     data=hitgen_pipeline.original_test_wide_transf,
            #     mask=hitgen_pipeline.mask_test_wide,
            #     dyn_features=hitgen_pipeline.test_dyn_features,
            #     window_size=hitgen_pipeline.best_params_multivar["window_size"],
            #     batch_size=hitgen_pipeline.best_params_multivar["batch_size"],
            #     windows_batch_size=hitgen_pipeline.best_params_multivar[
            #         "windows_batch_size"
            #     ],
            #     stride=1,
            #     coverage_mode="systematic",
            #     prediction_mode=hitgen_pipeline.best_params_multivar["prediction_mode"],
            #     future_steps=hitgen_pipeline.best_params_multivar["future_steps"],
            #     cache_dataset_name=DATASET,
            #     cache_dataset_group=DATASET_GROUP + "_multivar",
            #     cache_split="test",
            # )

            # ----------------------------------------------------------------
            # HiTGen
            # ----------------------------------------------------------------
            row_hitgen = {}

            # for sampling_strategy in SAMPLING_STRATEGIES:
            #     evaluation_pipeline_hitgen(
            #         dataset=DATASET,
            #         dataset_group=DATASET_GROUP,
            #         model=model,
            #         pipeline=hitgen_pipeline,
            #         gen_data=data_mask_temporalized_test,
            #         sampling_strategy=sampling_strategy,
            #         freq=FREQ,
            #         h=H,
            #         test_unique_ids=test_unique_ids,
            #         row_hitgen=row_hitgen,
            #         noise_scale=5,
            #     )

            # for sampling_strategy in SAMPLING_STRATEGIES_MULTIVAR:
            #     evaluation_pipeline_hitgen(
            #         dataset=DATASET,
            #         dataset_group=DATASET_GROUP,
            #         model=model_multivariate,
            #         pipeline=hitgen_pipeline,
            #         gen_data=data_mask_temporalized_test,
            #         sampling_strategy=sampling_strategy,
            #         freq=FREQ,
            #         h=H,
            #         test_unique_ids=test_unique_ids,
            #         row_hitgen=row_hitgen,
            #         noise_scale=5,
            #     )

            evaluation_pipeline_hitgen_forecast(
                dataset=DATASET,
                dataset_group=DATASET_GROUP,
                model=model_forecasting,
                pipeline=hitgen_pipeline,
                horizon=H,
                freq=FREQ,
                row_forecast=row_hitgen,
            )

            # append all hitgen results
            dataset_group_results.append(row_hitgen)
            results.append(row_hitgen)

            # ----------------------------------------------------------------
            # Metaforecast Methods
            # ----------------------------------------------------------------
            # synthetic_metaforecast_long = workflow_metaforecast_methods(
            #     df=hitgen_pipeline.original_test_long,
            #     freq=FREQ,
            #     dataset=DATASET,
            #     dataset_group=DATASET_GROUP,
            # )
            #
            # for method in METAFORECAST_METHODS:
            #     synthetic_metaforecast_long_method = synthetic_metaforecast_long.loc[
            #         synthetic_metaforecast_long["method"] == method
            #     ].copy()
            #
            #     synthetic_metaforecast_long_method = (
            #         synthetic_metaforecast_long_method.merge(
            #             hitgen_pipeline.original_test_long[["unique_id", "ds"]],
            #             on=["unique_id", "ds"],
            #             how="right",
            #         )
            #     )
            #     synthetic_metaforecast_long_method.fillna(0, inplace=True)
            #
            #     plot_generated_vs_original(
            #         synth_data=synthetic_metaforecast_long_method,
            #         original_data=hitgen_pipeline.original_test_long,
            #         score=0.0,
            #         loss=0.0,
            #         dataset_name=DATASET,
            #         dataset_group=DATASET_GROUP,
            #         n_series=4,
            #         suffix_name=f"{method}",
            #     )
            #
            #     print(
            #         f"\nComputing discriminative score for {method} synthetic data generation..."
            #     )
            #
            #     score_disc = compute_discriminative_score(
            #         unique_ids=test_unique_ids,
            #         original_data=hitgen_pipeline.original_test_long,
            #         synthetic_data=synthetic_metaforecast_long_method,
            #         method=method,
            #         freq=FREQ,
            #         dataset_name=DATASET,
            #         dataset_group=DATASET_GROUP,
            #         loss=0.0,
            #         samples=5,
            #         split="test",
            #     )
            #
            #     print(f"\nComputing TSTR score for {method} synthetic data...")
            #     score_tstr = tstr(
            #         unique_ids=test_unique_ids,
            #         original_data=hitgen_pipeline.original_test_long,
            #         synthetic_data=synthetic_metaforecast_long_method,
            #         method=method,
            #         freq=FREQ,
            #         horizon=H,
            #         dataset_name=DATASET,
            #         dataset_group=DATASET_GROUP,
            #         samples=5,
            #         split="test",
            #     )
            #
            #     print(
            #         f"\nComputing downstream task forecasting score for {method} synthetic data..."
            #     )
            #     score_dtf = compute_downstream_forecast(
            #         unique_ids=test_unique_ids,
            #         original_data=hitgen_pipeline.original_test_long,
            #         synthetic_data=synthetic_metaforecast_long_method,
            #         method=method,
            #         freq=FREQ,
            #         horizon=H,
            #         dataset_name=DATASET,
            #         dataset_group=DATASET_GROUP,
            #         samples=10,
            #         split="test",
            #     )
            #
            #     print(f"\n\n{DATASET}")
            #     print(f"{DATASET_GROUP}")
            #
            #     print(
            #         f"Discriminative score for {method} synthetic data: {score_disc:.4f}"
            #     )
            #
            #     print(
            #         f"TSTR score for {method} synthetic data: "
            #         f"TSTR {score_tstr['avg_smape_tstr']:.4f} vs TRTR "
            #         f"score {score_tstr['avg_smape_trtr']:.4f}"
            #     )
            #
            #     print(
            #         f"Downstream task forecasting score for {method} synthetic data: "
            #         f"concat score {score_dtf['avg_smape_concat']:.4f} vs original "
            #         f"score {score_dtf['avg_smape_original']:.4f}\n\n"
            #         f"concat std score {score_dtf['std_smape_concat']:.4f} vs original "
            #         f"score {score_dtf['std_smape_original']:.4f}\n\n"
            #     )
            #
            #     row_method = {
            #         "Dataset": DATASET,
            #         "Group": DATASET_GROUP,
            #         "Method": method,
            #         "Discriminative Score": score_disc,
            #         "TSTR (avg_smape_tstr)": score_tstr["avg_smape_tstr"],
            #         "TRTR (avg_smape_trtr)": score_tstr["avg_smape_trtr"],
            #         "DTF Concat Avg Score": score_dtf["avg_smape_concat"],
            #         "DTF Original Avg Score": score_dtf["avg_smape_original"],
            #         "DTF Concat Std Score": score_dtf["std_smape_concat"],
            #         "DTF Original Std Score": score_dtf["std_smape_original"],
            #     }
            #     dataset_group_results.append(row_method)
            #     results.append(row_method)
            #
            # dataset_group_df = pd.DataFrame(dataset_group_results).round(3)
            # os.makedirs("assets/results", exist_ok=True)
            #
            # dataset_group_df_results_path = (
            #     f"assets/results/{DATASET}_{DATASET_GROUP}_synthetic_data_results.csv"
            # )
            # dataset_group_df.to_csv(dataset_group_df_results_path, index=False)
            # print(
            #     f"\n==> Saved results for {DATASET} {DATASET_GROUP} to {dataset_group_df_results_path}\n"
            # )

    all_results_df = pd.DataFrame(results).round(3)
    all_results_path = "assets/results/synthetic_data_results.csv"
    all_results_df.to_csv(all_results_path, index=False)

    print(f"==> Saved consolidated results to {all_results_path}")
