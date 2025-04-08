import multiprocessing
import pandas as pd
import os
from hitgen.model.create_dataset_versions_vae import (
    HiTGenPipeline,
)
from hitgen.metrics.evaluation_pipeline import (
    evaluation_pipeline_hitgen_forecast,
)
from hitgen.benchmarks.model_pipeline import ModelPipeline
from hitgen.experiments.helper import (
    extract_frequency,
    extract_horizon,
    cmd_parser,
)


DATASET_GROUP_FREQ = {
    # "Tourism": {
    #     "Monthly": {"FREQ": "M", "H": 24},
    # },
    # "M1": {
    #     "Monthly": {"FREQ": "M", "H": 24},
    #     "Quarterly": {"FREQ": "Q", "H": 8},
    # },
    # "M3": {
    #     "Monthly": {"FREQ": "M", "H": 24},
    #     "Quarterly": {"FREQ": "Q", "H": 8},
    #     "Yearly": {"FREQ": "Y", "H": 4},
    # },
    # "Labour": {
    #     "Monthly": {"FREQ": "M", "H": 24},
    # },
    # "Traffic": {
    #     "Daily": {"FREQ": "D", "H": 30},
    # },
    # "ETTh1": {
    #     "Daily": {"FREQ": "D", "H": 30},
    # },
    "M4": {
        "Monthly": {"FREQ": "M", "H": 24},
        #     "Quarterly": {"FREQ": "Q", "H": 8},
        #     "Yearly": {"FREQ": "Y", "H": 4},
    },
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

            FREQ = extract_frequency(subgroup)
            H = extract_horizon(subgroup)
            DATASET_GROUP = subgroup[0]

            print(
                f"Dataset: {DATASET}, Dataset-group: {DATASET_GROUP}, Frequency: {FREQ}"
            )

            hitgen_pipeline = HiTGenPipeline(
                dataset_name=DATASET,
                dataset_group=DATASET_GROUP,
                freq=FREQ,
                horizon=H,
                opt_score=args.opt_score,
            )

            benchmark_pipeline = ModelPipeline(hitgen_pipeline=hitgen_pipeline)

            test_unique_ids = hitgen_pipeline.original_test_long["unique_id"].unique()

            if not args.transfer_learning:

                benchmark_pipeline.hyper_tune_and_train(max_evals=20)

                for model_name, model in benchmark_pipeline.models.items():
                    row_forecast = {}

                    evaluation_pipeline_hitgen_forecast(
                        dataset=DATASET,
                        dataset_group=DATASET_GROUP,
                        model=model,
                        pipeline=benchmark_pipeline,
                        horizon=H,
                        freq=FREQ,
                        row_forecast=row_forecast,
                        window_size=H,
                    )

                    dataset_group_results.append(row_forecast)
                    results.append(row_forecast)

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

                        benchmark_pipeline_transfer_learning = ModelPipeline(
                            hitgen_pipeline=hitgen_pipeline_transfer_learning
                        )

                        benchmark_pipeline_transfer_learning.hyper_tune_and_train(
                            max_evals=20
                        )

                        for (
                            model_name,
                            model,
                        ) in benchmark_pipeline_transfer_learning.models.items():
                            row_forecast_tl = {}

                            evaluation_pipeline_hitgen_forecast(
                                dataset=DATASET,
                                dataset_group=DATASET_GROUP,
                                model=model,
                                pipeline=benchmark_pipeline,
                                horizon=H,
                                freq=FREQ,
                                row_forecast=row_forecast_tl,
                                dataset_source=DATASET_TL,
                                prediction_mode="out_domain",
                                window_size=H,
                            )

                            dataset_group_results.append(row_forecast_tl)
                            results.append(row_forecast_tl)

    df_results = pd.DataFrame(results)

    os.makedirs("assets", exist_ok=True)
    df_results.to_csv("assets/results_forecast/final_results.csv", index=False)

    print("Final forecast results saved to assets/results_forecast/final_results.csv")
