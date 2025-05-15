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


def set_device(use_gpu: bool):
    """
    Configures whether PyTorch can see the GPU.
    """
    if not use_gpu:
        print("Forcing CPU usage (GPU disabled by user).")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    else:
        print("Using GPU if available")


def cmd_parser():
    parser = argparse.ArgumentParser(
        description="Run synthetic data generation using TimeGEN."
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU if available (default: False, meaning it runs on CPU).",
    )
    parser.add_argument(
        "--transfer-learning",
        action="store_true",
        help="Perform transfer learning (TL).",
    )
    parser.add_argument(
        "--coreset",
        action="store_true",
        help="Perform transfer learning with coreset dataset and a leave-one-out strategy.",
    )
    parser.add_argument(
        "--basic-forecasting",
        action="store_true",
        help="Perform basic forecasting.",
    )
    args = parser.parse_args()

    set_device(args.use_gpu)

    return args
