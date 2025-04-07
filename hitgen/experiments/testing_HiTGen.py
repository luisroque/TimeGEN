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

SOURCE_DATASET_GROUP_FREQ_TRANSFER_LEARNING = {
    "Tourism": {"Monthly": {"FREQ": "M", "H": 24}}
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

            benchmark_pipeline = BenchmarkPipeline(hitgen_pipeline=hitgen_pipeline)

            test_unique_ids = hitgen_pipeline.original_test_long["unique_id"].unique()

            if not args.transfer_learning:
                model_forecasting = hitgen_pipeline.hyper_tune_and_train_forecasting()

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

                benchmark_pipeline.hyper_tune_and_train(max_evals=20)

                for model_name, model in benchmark_pipeline.models.items():
                    row_forecast_benchmark = {}

                    evaluation_pipeline_hitgen_forecast(
                        dataset=DATASET,
                        dataset_group=DATASET_GROUP,
                        model=model,
                        pipeline=benchmark_pipeline,
                        horizon=H,
                        freq=FREQ,
                        row_forecast=row_forecast_benchmark,
                        window_size=H,
                    )

                    dataset_group_results.append(row_forecast_benchmark)
                    results.append(row_forecast_benchmark)

            else:
                for (
                    DATASET_TL,
                    SUBGROUPS_TL,
                ) in SOURCE_DATASET_GROUP_FREQ_TRANSFER_LEARNING.items():
                    for subgroup_tl in SUBGROUPS_TL.items():
                        FREQ_TL = extract_frequency(subgroup_tl)
                        H_TL = extract_horizon(subgroup_tl)
                        DATASET_GROUP_TL = subgroup_tl[0]

                        hitgen_pipeline_transfer_learning = HiTGenPipeline(
                            dataset_name=DATASET_TL,
                            dataset_group=DATASET_GROUP_TL,
                            freq=FREQ_TL,
                            horizon=H_TL,
                            opt_score=args.opt_score,
                        )

                        model_forecasting_tl = (
                            hitgen_pipeline_transfer_learning.hyper_tune_and_train_forecasting()
                        )

                        row_hitgen_tl = {}

                        evaluation_pipeline_hitgen_forecast(
                            dataset=DATASET,
                            dataset_group=DATASET_GROUP,
                            model=model_forecasting_tl,
                            pipeline=hitgen_pipeline,
                            horizon=H,
                            freq=FREQ,
                            row_forecast=row_hitgen_tl,
                            dataset_source=DATASET_TL,
                            window_size=hitgen_pipeline_transfer_learning.best_params_forecasting[
                                "window_size"
                            ],
                            prediction_mode="out_domain",
                        )

                        dataset_group_results.append(row_hitgen_tl)
                        results.append(row_hitgen_tl)

                        ######### Benchmark #########

                        benchmark_pipeline_transfer_learning = BenchmarkPipeline(
                            hitgen_pipeline=hitgen_pipeline_transfer_learning
                        )

                        benchmark_pipeline_transfer_learning.hyper_tune_and_train(
                            max_evals=20
                        )

                        for (
                            model_name,
                            model,
                        ) in benchmark_pipeline_transfer_learning.models.items():
                            row_forecast_benchmark_tl = {}

                            evaluation_pipeline_hitgen_forecast(
                                dataset=DATASET,
                                dataset_group=DATASET_GROUP,
                                model=model,
                                pipeline=benchmark_pipeline,
                                horizon=H,
                                freq=FREQ,
                                row_forecast=row_forecast_benchmark_tl,
                                dataset_source=DATASET_TL,
                                prediction_mode="out_domain",
                                window_size=H,
                            )

                            dataset_group_results.append(row_forecast_benchmark_tl)
                            results.append(row_forecast_benchmark_tl)
