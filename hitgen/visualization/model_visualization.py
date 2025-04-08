import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def plot_loss(history_dict):
    plt.figure(figsize=(12, 8))

    # Total Loss
    plt.subplot(2, 1, 1)
    plt.plot(history_dict["loss"], label="Training Loss")
    plt.title("Total Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Reconstruction Loss
    plt.subplot(2, 1, 2)
    plt.plot(history_dict["reconstruction_loss"], label="Training Reconstruction Loss")
    plt.title("Reconstruction Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # KL Loss
    plt.figure(figsize=(12, 4))
    plt.plot(history_dict["kl_loss"], label="Training KL Loss")
    plt.title("KL Divergence Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()


def plot_generated_vs_original(
    synth_data: pd.DataFrame,
    original_data: pd.DataFrame,
    model_name: str,
    dataset_name: str,
    dataset_group: str,
    n_series: int = 8,
    suffix_name: str = "hitgen",
) -> None:
    """
    Plot generated series and the original series for comparison and store as PDF.
    """

    plt.style.use("bmh")

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")

    if not n_series % 2 == 0:
        n_series -= 1

    fig, axes = plt.subplots(n_series // 2, 2, figsize=(18, 10))
    axes = axes.ravel()

    unique_ids = synth_data["unique_id"].unique()[:n_series]

    for i in range(n_series):
        ts_id = unique_ids[i]

        axes[i].plot(
            synth_data.loc[synth_data["unique_id"] == ts_id]["ds"],
            synth_data.loc[synth_data["unique_id"] == ts_id]["y"],
            label="Generated",
        )

        axes[i].plot(
            original_data.loc[original_data["unique_id"] == ts_id]["ds"],
            original_data.loc[original_data["unique_id"] == ts_id]["y"],
            label="Original",
        )

        axes[i].set_title(f"Time Series ID: {ts_id}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Date", fontsize=9)
        axes[i].set_ylabel("Value", fontsize=9)
        axes[i].grid(True, linestyle="--", alpha=0.6)

    axes[-1].legend(loc="lower right")
    fig.suptitle(
        f"Predicted vs. Original\n"
        f"Dataset: {dataset_name} | Group: {dataset_group} | Model: {model_name}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    os.makedirs("assets/plots", exist_ok=True)
    plt.savefig(
        f"assets/plots/"
        f"{current_datetime}_vae_generated_vs_original_{suffix_name}_"
        f"{dataset_name}_{dataset_group}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
