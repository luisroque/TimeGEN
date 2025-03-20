import os
import argparse


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


def cmd_parser():
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

    return args
