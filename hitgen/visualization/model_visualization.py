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
    original_test_data: pd.DataFrame,
    score: float,
    loss: float,
    dataset_name: str,
    dataset_group: str,
    n_series: int = 8,
    suffix_name: str = "hitgen",
) -> None:
    """
    Plot generated series and the original series and store as pdf
    """
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")

    if not n_series % 2 == 0:
        n_series -= 1

    fig, ax = plt.subplots(n_series // 2, 2, figsize=(18, 10))
    ax = ax.ravel()

    unique_ids = synth_data["unique_id"].unique()[:n_series]

    for i in range(n_series):
        ts_id = unique_ids[i]

        ax[i].plot(
            synth_data.loc[synth_data["unique_id"] == ts_id]["ds"],
            synth_data.loc[synth_data["unique_id"] == ts_id]["y"],
            label="Generated",
        )
        ax[i].plot(
            original_test_data.loc[original_test_data["unique_id"] == ts_id]["ds"],
            original_test_data.loc[original_test_data["unique_id"] == ts_id]["y"],
            label="Original",
        )

        ax[i].set_title(f"Time Series: {ts_id}", fontsize=12)
        ax[i].set_xlabel("Date")
        ax[i].set_ylabel("Value")

    ax[-1].legend(loc="lower right")

    plt.suptitle(
        f"VAE generated dataset vs original -> {dataset_name}: {dataset_group}",
        fontsize=14,
    )

    plt.tight_layout()
    os.makedirs("assets/plots", exist_ok=True)
    plt.savefig(
        f"assets/plots/{current_datetime}_vae_generated_vs_original_{suffix_name}_"
        f"{dataset_name}_{dataset_group}_{round(score, 2)}_{round(loss, 2)}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
