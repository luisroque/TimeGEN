import multiprocessing
import pandas as pd
from hitgen.model.models import (
    TemporalizeGenerator,
)
from hitgen.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE
from hitgen.metrics.discriminative_score import (
    compute_discriminative_score,
    compute_downstream_forecast,
    tstr,
)
from hitgen.visualization import plot_generated_vs_original
from hitgen.benchmarks.metaforecast import workflow_metaforecast_methods
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

DATASETS_HYPERPARAMS_CONFIGS = {
    "Tourism": {
        "Monthly": {
            "hitgen": {
                "latent_dim": 150,
                "window_size": 12,
                "patience": 30,
                "kl_weight": 0.25,
                "n_blocks_encoder": 3,
                "n_blocks_decoder": 3,
                "n_hidden": 64,
                "n_layers": 2,
                "kernel_size": 2,
                "pooling_mode": "average",
                "batch_size": 8,
                "epochs": 1000,
                "learning_rate": 0.001,
                "bi_rnn": True,
                "shuffle": True,
                "noise_scale_init": 0.1,
                "machine": "liacc-11gb",
            },
            "timegan": {
                "gan_args": ModelParameters(
                    batch_size=16,
                    lr=2e-4,
                    noise_dim=16,
                    layers_dim=32,
                    latent_dim=32,
                    gamma=1.0,
                ),
                "train_args": TrainParameters(
                    epochs=1000,
                    sequence_length=24,
                    number_sequences=4,
                ),
            },
        }
    },
    "M1": {
        "Monthly": {
            "hitgen": {
                "latent_dim": 150,
                "window_size": 12,
                "patience": 30,
                "kl_weight": 0.25,
                "n_blocks_encoder": 3,
                "n_blocks_decoder": 3,
                "n_hidden": 16,
                "n_layers": 2,
                "kernel_size": 2,
                "pooling_mode": "average",
                "batch_size": 16,
                "epochs": 1000,
                "learning_rate": 0.001,
                "bi_rnn": True,
                "shuffle": True,
                "noise_scale_init": 0.1,
                "machine": "liacc-48gb",
            },
            "timegan": {
                "gan_args": ModelParameters(
                    batch_size=16,
                    lr=2e-4,
                    noise_dim=16,
                    layers_dim=32,
                    latent_dim=32,
                    gamma=1.0,
                ),
                "train_args": TrainParameters(
                    epochs=1000,
                    sequence_length=24,
                    number_sequences=4,
                ),
            },
        },
        "Quarterly": {
            "hitgen": {
                "latent_dim": 150,
                "window_size": 12,
                "patience": 30,
                "kl_weight": 0.25,
                "n_blocks_encoder": 3,
                "n_blocks_decoder": 3,
                "n_hidden": 16,
                "n_layers": 2,
                "kernel_size": 2,
                "pooling_mode": "average",
                "batch_size": 8,
                "epochs": 1000,
                "learning_rate": 0.001,
                "bi_rnn": True,
                "shuffle": True,
                "noise_scale_init": 0.1,
                "machine": "liacc-48gb",
            },
            "timegan": {
                "gan_args": ModelParameters(
                    batch_size=16,
                    lr=2e-4,
                    noise_dim=16,
                    layers_dim=32,
                    latent_dim=32,
                    gamma=1.0,
                ),
                "train_args": TrainParameters(
                    epochs=1000,
                    sequence_length=24,
                    number_sequences=4,
                ),
            },
        },
    },
    "M3": {
        "Monthly": {
            "hitgen": {
                "latent_dim": 150,
                "window_size": 12,
                "patience": 30,
                "kl_weight": 0.25,
                "n_blocks_encoder": 3,
                "n_blocks_decoder": 3,
                "n_hidden": 16,
                "n_layers": 2,
                "kernel_size": 2,
                "pooling_mode": "average",
                "batch_size": 16,
                "epochs": 1000,
                "learning_rate": 0.001,
                "bi_rnn": True,
                "shuffle": True,
                "noise_scale_init": 0.1,
            },
            "timegan": {
                "gan_args": ModelParameters(
                    batch_size=16,
                    lr=2e-4,
                    noise_dim=16,
                    layers_dim=32,
                    latent_dim=32,
                    gamma=1.0,
                ),
                "train_args": TrainParameters(
                    epochs=1000,
                    sequence_length=24,
                    number_sequences=4,
                ),
            },
        },
        "Quarterly": {
            "hitgen": {
                "latent_dim": 150,
                "window_size": 12,
                "patience": 30,
                "kl_weight": 0.25,
                "n_blocks_encoder": 3,
                "n_blocks_decoder": 3,
                "n_hidden": 16,
                "n_layers": 2,
                "kernel_size": 2,
                "pooling_mode": "average",
                "batch_size": 8,
                "epochs": 1000,
                "learning_rate": 0.001,
                "bi_rnn": True,
                "shuffle": True,
                "noise_scale_init": 0.1,
            },
            "timegan": {
                "gan_args": ModelParameters(
                    batch_size=16,
                    lr=2e-4,
                    noise_dim=16,
                    layers_dim=32,
                    latent_dim=32,
                    gamma=1.0,
                ),
                "train_args": TrainParameters(
                    epochs=1000,
                    sequence_length=24,
                    number_sequences=4,
                ),
            },
        },
        "Yearly": {
            "hitgen": {
                "latent_dim": 150,
                "window_size": 12,
                "patience": 30,
                "kl_weight": 0.25,
                "n_blocks_encoder": 3,
                "n_blocks_decoder": 3,
                "n_hidden": 16,
                "n_layers": 2,
                "kernel_size": 2,
                "pooling_mode": "average",
                "batch_size": 8,
                "epochs": 1000,
                "learning_rate": 0.001,
                "bi_rnn": True,
                "shuffle": True,
                "noise_scale_init": 0.1,
            },
            "timegan": {
                "gan_args": ModelParameters(
                    batch_size=16,
                    lr=2e-4,
                    noise_dim=16,
                    layers_dim=32,
                    latent_dim=32,
                    gamma=1.0,
                ),
                "train_args": TrainParameters(
                    epochs=1000,
                    sequence_length=24,
                    number_sequences=4,
                ),
            },
        },
    },
    "M4": {
        "Monthly": {
            "hitgen": {
                "latent_dim": 150,
                "window_size": 12,
                "patience": 30,
                "kl_weight": 0.25,
                "n_blocks_encoder": 3,
                "n_blocks_decoder": 3,
                "n_hidden": 16,
                "n_layers": 2,
                "kernel_size": 2,
                "pooling_mode": "average",
                "batch_size": 16,
                "epochs": 1000,
                "learning_rate": 0.001,
                "bi_rnn": True,
                "shuffle": True,
                "noise_scale_init": 0.1,
            },
            "timegan": {
                "gan_args": ModelParameters(
                    batch_size=16,
                    lr=2e-4,
                    noise_dim=16,
                    layers_dim=32,
                    latent_dim=32,
                    gamma=1.0,
                ),
                "train_args": TrainParameters(
                    epochs=1000,
                    sequence_length=24,
                    number_sequences=4,
                ),
            },
        },
        "Quarterly": {
            "hitgen": {
                "latent_dim": 150,
                "window_size": 12,
                "patience": 30,
                "kl_weight": 0.25,
                "n_blocks_encoder": 3,
                "n_blocks_decoder": 3,
                "n_hidden": 16,
                "n_layers": 2,
                "kernel_size": 2,
                "pooling_mode": "average",
                "batch_size": 8,
                "epochs": 1000,
                "learning_rate": 0.001,
                "bi_rnn": True,
                "shuffle": True,
                "noise_scale_init": 0.1,
            },
            "timegan": {
                "gan_args": ModelParameters(
                    batch_size=16,
                    lr=2e-4,
                    noise_dim=16,
                    layers_dim=32,
                    latent_dim=32,
                    gamma=1.0,
                ),
                "train_args": TrainParameters(
                    epochs=1000,
                    sequence_length=24,
                    number_sequences=4,
                ),
            },
        },
    },
}

DATASET_GROUP_FREQ = {
    # "Tourism": {
    #     "Monthly": {"FREQ": "M"},
    # },
    # "M1": {
    #     "Monthly": {"FREQ": "M"},
    #     "Quarterly": {"FREQ": "Q"},
    # },
    # "M3": {
    #     "Monthly": {"FREQ": "M"},
    #     "Quarterly": {"FREQ": "Q"},
    #     "Yearly": {"FREQ": "Y"},
    # },
    "M4": {
        "Monthly": {"FREQ": "M"},
        "Quarterly": {"FREQ": "Q"},
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


def extract_score(dataset_group):
    """Safely extracts frequency from dataset group."""
    score = dataset_group[1]["final_score"]
    return score


def has_final_score_in_tuple(tpl):
    """Check if the second element is a dictionary and contains 'final_score'"""
    return isinstance(tpl[1], dict) and "final_score" in tpl[1]


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    results = []

    for DATASET, SUBGROUPS in DATASET_GROUP_FREQ.items():
        for subgroup in SUBGROUPS.items():
            FREQ = extract_frequency(subgroup)
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
            if (
                DATASET not in DATASETS_HYPERPARAMS_CONFIGS
                or DATASET_GROUP not in DATASETS_HYPERPARAMS_CONFIGS[DATASET]
            ):
                raise ValueError(
                    f"Configuration for {DATASET} - {DATASET_GROUP} not found."
                )

            dataset_config = DATASETS_HYPERPARAMS_CONFIGS[DATASET][DATASET_GROUP]

            TOP = None
            VAL_STEPS = 0

            # HITGEN Configurations
            hitgen_config = dataset_config["hitgen"]
            WINDOW_SIZE = hitgen_config["window_size"]
            LATENT_DIM_HITGEN = hitgen_config["latent_dim"]
            EPOCHS_HITGEN = hitgen_config["epochs"]
            BATCH_SIZE_HITGEN = hitgen_config["batch_size"]
            KL_WEIGHT_INIT = hitgen_config["kl_weight"]
            NOISE_SAMPLE_INIT = hitgen_config["noise_scale_init"]
            N_BLOCKS_ENCODER_HITGEN = hitgen_config["n_blocks_encoder"]
            N_BLOCKS_DECODER_HITGEN = hitgen_config["n_blocks_decoder"]
            N_HIDDEN_HITGEN = hitgen_config["n_hidden"]
            N_LAYERS_HITGENS = hitgen_config["n_layers"]
            KERNEL_SIZE_HITGEN = hitgen_config["kernel_size"]
            POOLING_MODE_HITGEN = hitgen_config["pooling_mode"]
            LEARNING_RATE_HITGEN = hitgen_config["learning_rate"]
            PATIENCE_HITGEN = hitgen_config["patience"]
            STRIDE_TEMPORALIZE_HITGEN = 1
            SHUFFLE_HITGEN = hitgen_config["shuffle"]
            BI_RNN_HITGEN = hitgen_config["bi_rnn"]
            ANNEALING_HITGEN = False

            # TIMEGAN Configurations
            timegan_config = dataset_config["timegan"]
            gan_args = timegan_config["gan_args"]
            train_args = timegan_config["train_args"]

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
                top=TOP,
                window_size=WINDOW_SIZE,
                stride_temporalize=STRIDE_TEMPORALIZE_HITGEN,
                batch_size=BATCH_SIZE_HITGEN,
                shuffle=SHUFFLE_HITGEN,
                bi_rnn=BI_RNN_HITGEN,
                annealing=ANNEALING_HITGEN,
                noise_scale_init=NOISE_SAMPLE_INIT,
                kl_weight_init=KL_WEIGHT_INIT,
                n_blocks_encoder=N_BLOCKS_ENCODER_HITGEN,
                n_blocks_decoder=N_BLOCKS_DECODER_HITGEN,
                n_hidden=N_HIDDEN_HITGEN,
                n_layers=N_LAYERS_HITGENS,
                patience=PATIENCE_HITGEN,
            )

            # hypertuning
            # create_dataset_vae.hyper_tune_and_train()

            # fit
            model, history, _ = create_dataset_vae.fit(
                latent_dim=LATENT_DIM_HITGEN,
                epochs=EPOCHS_HITGEN,
                patience=PATIENCE_HITGEN,
                learning_rate=LEARNING_RATE_HITGEN,
            )
            # plot_loss(history)

            (
                _,
                _,
                original_data,
                train_data_long,
                test_data_long,
                original_data_long,
                _,
                _,
                original_mask,
                original_data_no_transf_long,
                test_data_no_transf_long,
                _,
                test_dyn_features,
                original_dyn_features,
            ) = create_dataset_vae._feature_engineering()

            data_mask_temporalized = TemporalizeGenerator(
                original_data,
                original_mask,
                original_dyn_features,
                window_size=WINDOW_SIZE,
                stride=create_dataset_vae.stride_temporalize,
                batch_size=BATCH_SIZE_HITGEN,
                shuffle=SHUFFLE_HITGEN,
            )

            _, synth_hitgen_test_long, _, _, synth_hitgen_test_long_no_transf, _ = (
                create_dataset_vae.predict(
                    model,
                    data_mask_temporalized=data_mask_temporalized,
                    samples=data_mask_temporalized.indices.shape[0],
                    window_size=WINDOW_SIZE,
                    latent_dim=LATENT_DIM_HITGEN,
                )
            )

            # generate more samples into the future to check on overfitting
            # new_latent_samples = np.random.normal(
            #     size=(
            #         data_mask_temporalized.indices.shape[0] + WINDOW_SIZE,
            #         WINDOW_SIZE,
            #         LATENT_DIM,
            #     )
            # )
            # n_series = data.shape[1]
            # future_mask = tf.ones((WINDOW_SIZE, n_series), dtype=tf.float32)
            #
            # mask = tf.concat([create_dataset_vae.mask, future_mask], axis=0)
            # mask_temporalized = create_dataset_vae.temporalize(mask, WINDOW_SIZE)
            # generated_data = model.decoder.predict([new_latent_samples, mask_temporalized])
            #
            # synth_hitgen = detemporalize(generated_data)

            plot_generated_vs_original(
                synth_data=synth_hitgen_test_long_no_transf,
                original_test_data=test_data_no_transf_long,
                score=0.0,
                loss=0.0,
                dataset_name=DATASET,
                dataset_group=DATASET_GROUP,
                n_series=8,
                suffix_name="hitgen_no_transf",
            )
            plot_generated_vs_original(
                synth_data=synth_hitgen_test_long,
                original_test_data=test_data_long,
                score=0.0,
                loss=0.0,
                dataset_name=DATASET,
                dataset_group=DATASET_GROUP,
                n_series=8,
                suffix_name="hitgen",
            )

            # import matplotlib.pyplot as plt
            #
            # unique_ids = synth_hitgen_test_long["unique_id"].unique()[:4]
            #
            # fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
            # for idx, unique_id in enumerate(unique_ids):
            #     ax = axes[idx]
            #     original_series = original_data_long[
            #         original_data_long["unique_id"] == unique_id
            #     ]
            #     synthetic_series = synth_hitgen_test_long[
            #         synth_hitgen_test_long["unique_id"] == unique_id
            #     ]
            #
            #     ax.plot(
            #         original_series["ds"],
            #         original_series["y"],
            #         label="Original",
            #         linestyle="-",
            #     )
            #     ax.plot(
            #         synthetic_series["ds"],
            #         synthetic_series["y"],
            #         label="Synthetic",
            #         linestyle="--",
            #     )
            #
            #     ax.set_title(f"Time Series for ID: {unique_id}")
            #     ax.set_ylabel("Value")
            #     ax.legend()
            #     ax.grid()
            #
            # plt.xlabel("Time Steps")
            # plt.tight_layout()
            # plt.show()

            # TimeGAN synthetic data generation

            # parallel timegan training and synthetic data generation
            # synth_timegan_data_all = Parallel(n_jobs=6)(
            #     delayed(train_and_generate_synthetic)(
            #         ts, original_data_long, DATASET, DATASET_GROUP, WINDOW_SIZE
            #     )
            #     for ts in original_data_long["unique_id"].unique()
            # )

            # best_params = hyper_tune_timegan(
            #     train_data_long, DATASET, DATASET_GROUP, window_size=24, n_trials=50
            # )
            # final_model = train_timegan_with_best_params(
            #     test_data_long, best_params, DATASET, DATASET_GROUP, window_size=24
            # )

            test_unique_ids = test_data_no_transf_long["unique_id"].unique()

            # hypertuning timegan
            # hyper_tune_timegan(
            #     data=original_data_long,
            #     dataset_name=DATASET,
            #     dataset_group=DATASET_GROUP,
            #     window_size=WINDOW_SIZE,
            #     long_properties=create_dataset_vae.long_properties,
            #     freq=FREQ,
            # )

            # synthetic_timegan_long = workflow_timegan(
            #     test_unique_ids,
            #     SYNTHETIC_FILE_PATH_TIMEGAN,
            #     test_data_long,
            #     DATASET,
            #     DATASET_GROUP,
            #     WINDOW_SIZE,
            #     create_dataset_vae.long_properties,
            #     FREQ,
            #     timegan_config,
            # )

            # metaforecast methods
            synthetic_metaforecast_long_no_transf = workflow_metaforecast_methods(
                df=original_data_no_transf_long.fillna(value=0),
                freq=FREQ,
            )

            if not hitgen_score_disc:
                print("\nComputing discriminative score for HiTGen synthetic data...")
                hitgen_score_disc = compute_discriminative_score(
                    unique_ids=test_unique_ids,
                    original_data=test_data_no_transf_long.fillna(value=0),
                    synthetic_data=synth_hitgen_test_long_no_transf,
                    method="hitgen",
                    freq="M",
                    dataset_name=DATASET,
                    dataset_group=DATASET_GROUP,
                    loss=0.0,
                    samples=5,
                )

            print("\nComputing TSTR score for HiTGen synthetic data...")
            hitgen_score_tstr = tstr(
                unique_ids=test_unique_ids,
                original_data=test_data_no_transf_long.fillna(value=0),
                synthetic_data=synth_hitgen_test_long_no_transf,
                method="hitgen",
                freq="M",
                horizon=24,
                dataset_name=DATASET,
                dataset_group=DATASET_GROUP,
                samples=5,
            )

            print(
                "\nComputing downstream task forecasting score for HiTGen synthetic data..."
            )
            hitgen_score_dtf = compute_downstream_forecast(
                unique_ids=test_unique_ids,
                original_data=test_data_no_transf_long.fillna(value=0),
                synthetic_data=synth_hitgen_test_long_no_transf,
                method="hitgen",
                freq="M",
                horizon=24,
                dataset_name=DATASET,
                dataset_group=DATASET_GROUP,
                samples=5,
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
                f"concat score {hitgen_score_dtf['avg_smape_concat']:.4f} vs original "
                f"score {hitgen_score_dtf['avg_smape_original']:.4f}\n\n"
            )

            results.append(
                {
                    "Dataset": DATASET,
                    "Group": DATASET_GROUP,
                    "Method": "HiTGen",
                    "Discriminative Score": hitgen_score_disc,
                    "TSTR (avg_smape_tstr)": hitgen_score_tstr["avg_smape_tstr"],
                    "TRTR (avg_smape_trtr)": hitgen_score_tstr["avg_smape_trtr"],
                    "DTF Concat Score": hitgen_score_dtf["avg_smape_concat"],
                    "DTF Original Score": hitgen_score_dtf["avg_smape_original"],
                }
            )

            # print("\nComputing discriminative score for TimeGAN synthetic data...")
            # score_timegan = compute_discriminative_score(
            #     unique_ids=test_unique_ids,
            #     original_data=test_data_long,
            #     synthetic_data=synthetic_timegan_long,
            #     freq="M",
            #     dataset_name=DATASET,
            #     dataset_group=DATASET_GROUP,
            #     loss=0.0,
            #     samples=5,
            # )

            # print(f"Discriminative score for TimeGAN synthetic data: {score_timegan:.4f}")

            for method in METAFORECAST_METHODS:
                print(
                    f"\nComputing discriminative score for {method} synthetic data generation..."
                )

                score_disc = compute_discriminative_score(
                    unique_ids=test_unique_ids,
                    original_data=test_data_no_transf_long.fillna(value=0),
                    synthetic_data=synthetic_metaforecast_long_no_transf.loc[
                        synthetic_metaforecast_long_no_transf["method"] == method
                    ],
                    method=method,
                    freq="M",
                    dataset_name=DATASET,
                    dataset_group=DATASET_GROUP,
                    loss=0.0,
                    samples=5,
                )

                print(f"\nComputing TSTR score for {method} synthetic data...")
                score_tstr = tstr(
                    unique_ids=test_unique_ids,
                    original_data=test_data_no_transf_long.fillna(value=0),
                    synthetic_data=synthetic_metaforecast_long_no_transf.loc[
                        synthetic_metaforecast_long_no_transf["method"] == method
                    ],
                    method=method,
                    freq="M",
                    horizon=24,
                    dataset_name=DATASET,
                    dataset_group=DATASET_GROUP,
                    samples=5,
                )

                print(
                    f"\nComputing downstream task forecasting score for {method} synthetic data..."
                )
                score_dtf = compute_downstream_forecast(
                    unique_ids=test_unique_ids,
                    original_data=test_data_no_transf_long.fillna(value=0),
                    synthetic_data=synthetic_metaforecast_long_no_transf.loc[
                        synthetic_metaforecast_long_no_transf["method"] == method
                    ],
                    method=method,
                    freq="M",
                    horizon=24,
                    dataset_name=DATASET,
                    dataset_group=DATASET_GROUP,
                    samples=5,
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
                )

                results.append(
                    {
                        "Dataset": DATASET,
                        "Group": DATASET_GROUP,
                        "Method": method,
                        "Discriminative Score": score_disc,
                        "TSTR (avg_smape_tstr)": score_tstr["avg_smape_tstr"],
                        "TRTR (avg_smape_trtr)": score_tstr["avg_smape_trtr"],
                        "DTF Concat Score": score_dtf["avg_smape_concat"],
                        "DTF Original Score": score_dtf["avg_smape_original"],
                    }
                )

    results_df = pd.DataFrame(results)
    print(results_df)

    results_path = "assets/results/synthetic_data_results.csv"
    results_df.to_csv(results_path, index=False)
