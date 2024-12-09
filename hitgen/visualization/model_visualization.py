import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def plot_loss(history_dict):
    plt.figure(figsize=(12, 8))

    # Total Loss
    plt.subplot(2, 1, 1)
    plt.plot(history_dict["loss"], label="Training Loss")
    # plt.plot(history_dict["val_loss"], label="Validation Loss")
    plt.title("Total Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Reconstruction Loss
    plt.subplot(2, 1, 2)
    plt.plot(history_dict["reconstruction_loss"], label="Training Reconstruction Loss")
    # plt.plot(
    #     history_dict["val_reconstruction_loss"], label="Validation Reconstruction Loss"
    # )
    plt.title("Reconstruction Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # KL Loss
    plt.figure(figsize=(12, 4))
    plt.plot(history_dict["kl_loss"], label="Training KL Loss")
    # plt.plot(history_dict["val_kl_loss"], label="Validation KL Loss")
    plt.title("KL Divergence Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()


def plot_generated_vs_original(
    dec_pred_hat: pd.DataFrame,
    X_train_raw: pd.DataFrame,
    score: float,
    loss: float,
    dataset_name: str,
    dataset_group: str,
    n_series: int = 8,
) -> None:
    """
    Plot generated series and the original series and store as pdf
    """
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
    # n_series needs to be even
    if not n_series % 2 == 0:
        n_series -= 1
    _, ax = plt.subplots(int(n_series // 2), 2, figsize=(18, 10))
    ax = ax.ravel()
    unique_ids = dec_pred_hat["unique_id"].unique()[:n_series]
    for i in range(n_series):
        ax[i].plot(
            dec_pred_hat.loc[dec_pred_hat["unique_id"] == unique_ids[i]]["ds"],
            dec_pred_hat.loc[dec_pred_hat["unique_id"] == unique_ids[i]]["y"],
            label="new sample",
        )
        ax[i].plot(
            X_train_raw.loc[X_train_raw["unique_id"] == unique_ids[i]]["ds"],
            X_train_raw.loc[X_train_raw["unique_id"] == unique_ids[i]]["y"],
            label="orig",
        )
    plt.legend()
    plt.suptitle(
        f"VAE generated dataset vs original -> {dataset_name}: {dataset_group}",
        fontsize=14,
    )
    os.makedirs("assets/plots", exist_ok=True)
    plt.savefig(
        f"assets/plots/{current_datetime}_vae_generated_vs_original_{dataset_name}_{dataset_group}_{round(score,2)}_{round(loss,2)}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
