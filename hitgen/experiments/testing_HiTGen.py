import multiprocessing
import pandas as pd
from hitgen.model.create_dataset_versions_vae import (
    HiTGenPipeline,
)
from hitgen.metrics.evaluation_pipeline import (
    evaluation_pipeline_hitgen_forecast,
)
from hitgen.benchmarks.benchmark_models import BenchmarkPipeline
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
    #     "Monthly": {"FREQ": "M", "H": 24},
    #     "Quarterly": {"FREQ": "Q", "H": 8},
    # },
    # "M3": {
    #     "Monthly": {"FREQ": "M", "H": 24},
    #     "Quarterly": {"FREQ": "Q", "H": 8},
    #     "Yearly": {"FREQ": "Y", "H": 4},
    # },
}


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

            model_forecasting = hitgen_pipeline.hyper_tune_and_train_forecasting()

            test_unique_ids = hitgen_pipeline.original_test_long["unique_id"].unique()

            row_hitgen = {}

            evaluation_pipeline_hitgen_forecast(
                dataset=DATASET,
                dataset_group=DATASET_GROUP,
                model=model_forecasting,
                pipeline=hitgen_pipeline,
                horizon=H,
                freq=FREQ,
                row_forecast=row_hitgen,
            )

            dataset_group_results.append(row_hitgen)
            results.append(row_hitgen)

            ######### Benchmark #########

            benchmark_pipeline = BenchmarkPipeline(hitgen_pipeline=hitgen_pipeline)

            benchmark_pipeline.hyper_tune_and_train(max_evals=20)

            for model_name in benchmark_pipeline.models.keys():
                row_forecast_benchmark = {}

                evaluation_pipeline_hitgen_forecast(
                    dataset=DATASET,
                    dataset_group=DATASET_GROUP,
                    model=model_name,
                    pipeline=benchmark_pipeline,
                    horizon=H,
                    freq=FREQ,
                    row_forecast=row_forecast_benchmark,
                )

                dataset_group_results.append(row_forecast_benchmark)
                results.append(row_forecast_benchmark)

    all_results_df = pd.DataFrame(results).round(3)
    all_results_path = "assets/results/synthetic_data_results.csv"
    all_results_df.to_csv(all_results_path, index=False)

    print(f"==> Saved consolidated results to {all_results_path}")
