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
    title: str | None = None,
    save_dir: str = "assets/plots",
    formats: tuple[str, ...] = ("pdf",),
) -> None:

    plt.style.use("bmh")

    available = len(synth_data["unique_id"].unique())
    n_series = min(n_series, available)
    if n_series % 2:  # make it even
        n_series -= 1

    n_rows = n_series // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(18, 10), sharex=False, sharey=False)
    axes = axes.ravel()

    unique_ids = synth_data["unique_id"].unique()[:n_series]

    for i, ts_id in enumerate(unique_ids):
        ax = axes[i]

        ax.plot(
            synth_data.loc[synth_data["unique_id"] == ts_id, "ds"],
            synth_data.loc[synth_data["unique_id"] == ts_id, "y"],
            label="Generated",
        )
        ax.plot(
            original_data.loc[original_data["unique_id"] == ts_id, "ds"],
            original_data.loc[original_data["unique_id"] == ts_id, "y"],
            label="Original",
        )

        ax.set_title(f"Series: {ts_id}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Date", fontsize=9)
        ax.set_ylabel("Value", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.6)

    axes[-1].legend(loc="lower right")

    if title is None:
        title = (
            f"{model_name} — Predicted vs. Original\n"
            f"Dataset: {dataset_name}  |  Group: {dataset_group}"
        )
    fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    base_fname = f"{timestamp}_{suffix_name}"

    for ext in formats:
        fig.savefig(
            os.path.join(save_dir, f"{base_fname}.{ext}"),
            format=ext,
            bbox_inches="tight",
        )

    plt.close(fig)
