"""
Latent space analysis for TimeGEN: shared temporal structure vs specialization.

Extracts mu vectors from trained TimeGEN (VAE) models, computes quantitative
metrics, and produces t-SNE/UMAP visualizations comparing latent representations
across domains.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import LabelEncoder

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from timegen.model_pipeline.core.core_extension import CustomNeuralForecast
from timegen.data_pipeline.data_pipeline_setup import DataPipeline

# Dataset configurations matching run_pipeline.py
DATASET_CONFIGS = {
    "M1_Monthly": {"dataset_name": "M1", "dataset_group": "Monthly", "freq": "M", "h": 24},
    "M1_Quarterly": {"dataset_name": "M1", "dataset_group": "Quarterly", "freq": "Q", "h": 8},
    "M3_Monthly": {"dataset_name": "M3", "dataset_group": "Monthly", "freq": "M", "h": 24},
    "M3_Quarterly": {"dataset_name": "M3", "dataset_group": "Quarterly", "freq": "Q", "h": 8},
    "M3_Yearly": {"dataset_name": "M3", "dataset_group": "Yearly", "freq": "Y", "h": 4},
    "M4_Monthly": {"dataset_name": "M4", "dataset_group": "Monthly", "freq": "M", "h": 24},
    "M4_Quarterly": {"dataset_name": "M4", "dataset_group": "Quarterly", "freq": "Q", "h": 8},
    "M5_Daily": {"dataset_name": "M5", "dataset_group": "Daily", "freq": "D", "h": 30},
    "Tourism_Monthly": {"dataset_name": "Tourism", "dataset_group": "Monthly", "freq": "M", "h": 24},
    "Traffic_Daily": {"dataset_name": "Traffic", "dataset_group": "Daily", "freq": "D", "h": 30},
}

MODELS_DIR = PROJECT_ROOT / "assets" / "model_weights_out_domain" / "hypertuning"
OUTPUT_DIR = PROJECT_ROOT / "assets" / "plots" / "latent_analysis"


def load_model(dataset_key: str, model_type: str = "AutoTimeGEN") -> CustomNeuralForecast:
    """Load a trained model from disk. Supports both single-source and multi-source (MIXED) naming."""
    # Try single-source naming first
    model_dir = MODELS_DIR / f"{dataset_key}_{model_type}_neuralforecast"
    if model_dir.exists():
        return CustomNeuralForecast.load(path=str(model_dir))

    # Try multi-source (MIXED_ALL_BUT_) naming
    mixed_dir = MODELS_DIR / f"MIXED_ALL_BUT_{dataset_key}_{model_type}_neuralforecast"
    if mixed_dir.exists():
        return CustomNeuralForecast.load(path=str(mixed_dir))

    raise FileNotFoundError(
        f"Model not found. Tried:\n  {model_dir}\n  {mixed_dir}"
    )


def load_data_pipeline(config: dict) -> DataPipeline:
    """Initialize a DataPipeline for a given dataset configuration."""
    return DataPipeline(
        dataset_name=config["dataset_name"],
        dataset_group=config["dataset_group"],
        freq=config["freq"],
        horizon=config["h"],
        window_size=config["h"],
    )


def extract_latents(
    model: CustomNeuralForecast,
    data_pipeline: DataPipeline,
    max_windows: int = 2000,
) -> dict:
    """
    Extract latent mu vectors from a trained TimeGEN model on test data.

    Returns dict with keys: mu, logvar, domain_label
    """
    timegen_model = model.models[0]
    timegen_model.eval()

    # Get test data -- use trainval (what the model would see at prediction time)
    df = data_pipeline.original_trainval_long.copy()
    if df.empty:
        df = data_pipeline.df.copy()

    h = data_pipeline.h
    window_size = timegen_model.input_size

    # Build windows manually from the data
    all_mu = []
    all_logvar = []

    series_ids = df["unique_id"].unique()
    windows_collected = 0

    for uid in series_ids:
        if windows_collected >= max_windows:
            break

        series = df[df["unique_id"] == uid].sort_values("ds")
        values = series["y"].values.astype(np.float32)

        if len(values) < window_size + h:
            continue

        # Take the last window (matching prediction behavior)
        start = len(values) - window_size - h
        window = values[start : start + window_size]

        insample_y = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        insample_mask = torch.ones_like(insample_y)

        # Create empty exogenous tensors matching the encoder's expectations
        enc = timegen_model.encoder
        futr_size = enc.futr_input_size
        hist_size = enc.hist_input_size
        stat_size = enc.stat_input_size

        futr_exog = torch.zeros(1, window_size + h, futr_size)
        hist_exog = torch.zeros(1, window_size, hist_size)
        stat_exog = torch.zeros(1, stat_size) if stat_size > 0 else torch.zeros(1, 0)

        with torch.no_grad():
            mu, logvar = enc(insample_y, futr_exog, hist_exog, stat_exog)

        all_mu.append(mu.cpu().numpy())
        all_logvar.append(logvar.cpu().numpy())
        windows_collected += 1

    if not all_mu:
        return {"mu": np.array([]), "logvar": np.array([])}

    return {
        "mu": np.concatenate(all_mu, axis=0),
        "logvar": np.concatenate(all_logvar, axis=0),
    }


def extract_latents_from_all_datasets(
    model_type: str = "AutoTimeGEN",
    max_windows_per_dataset: int = 500,
    datasets: list = None,
) -> tuple:
    """
    DEPRECATED: Each dataset uses a different model, so comparing latent spaces
    across datasets is not meaningful. Use extract_latents_single_model instead.
    """
    raise NotImplementedError(
        "Use extract_latents_single_model() instead -- comparing latents "
        "from different models is not meaningful."
    )


def extract_latents_single_model(
    source_dataset_key: str,
    target_dataset_keys: list,
    model_type: str = "AutoTimeGEN",
    max_windows_per_dataset: int = 500,
) -> tuple:
    """
    Extract latent representations from a SINGLE trained model applied to
    data from MULTIPLE domains. This shows how one model organizes cross-domain
    inputs in its latent space -- the meaningful comparison for transfer learning.

    Args:
        source_dataset_key: The dataset the model was trained on (e.g. "M3_Monthly")
        target_dataset_keys: List of datasets to extract latents from
        model_type: Model variant to use
        max_windows_per_dataset: Max windows per target dataset

    Returns:
        all_mu: np.ndarray [N, latent_dim]
        domain_labels: list of domain names
        dataset_keys_processed: list of dataset keys
    """
    print(f"  Loading model trained on {source_dataset_key}...")
    model = load_model(source_dataset_key, model_type)
    timegen_model = model.models[0]
    timegen_model.eval()

    all_mu = []
    all_windows = []
    domain_labels = []
    dataset_keys_processed = []

    for dataset_key in target_dataset_keys:
        config = DATASET_CONFIGS[dataset_key]
        print(f"  Extracting latents for {dataset_key} (using {source_dataset_key} model)...")

        try:
            dp = load_data_pipeline(config)

            # Use the source model's encoder on target data
            latents = _extract_latents_with_model(
                timegen_model, dp, max_windows=max_windows_per_dataset
            )

            if latents["mu"].shape[0] == 0:
                print(f"    No windows extracted for {dataset_key}")
                continue

            n = latents["mu"].shape[0]
            all_mu.append(latents["mu"])
            all_windows.append(latents["windows"])
            domain_labels.extend([config["dataset_name"]] * n)
            dataset_keys_processed.append(dataset_key)
            print(f"    Extracted {n} latent vectors (dim={latents['mu'].shape[1]})")
        except Exception as e:
            print(f"    Error processing {dataset_key}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_mu:
        raise RuntimeError("No latent vectors extracted from any dataset")

    mu_out = np.concatenate(all_mu, axis=0)
    windows_out = np.concatenate(all_windows, axis=0)

    return mu_out, domain_labels, dataset_keys_processed, windows_out


def _extract_latents_with_model(
    timegen_model,
    data_pipeline: DataPipeline,
    max_windows: int = 500,
) -> dict:
    """
    Extract latent mu vectors from a specific model on data from a data pipeline.
    The model and data can come from different domains. When the data has more
    time points than the model's input_size, the window is truncated (last
    input_size points) so cross-frequency extraction is possible.
    """
    df = data_pipeline.original_trainval_long.copy()
    if df.empty:
        df = data_pipeline.df.copy()

    h = timegen_model.h
    window_size = timegen_model.input_size
    enc = timegen_model.encoder

    all_mu = []
    all_logvar = []
    all_windows = []
    series_ids = df["unique_id"].unique()
    windows_collected = 0

    # Use model's expected window_size regardless of data's native horizon
    min_len = window_size + h

    for uid in series_ids:
        if windows_collected >= max_windows:
            break

        series = df[df["unique_id"] == uid].sort_values("ds")
        values = series["y"].values.astype(np.float32)

        if len(values) < min_len:
            continue

        # Take the last window_size points as input (truncate if data is longer)
        start = len(values) - window_size - h
        window = values[start : start + window_size]

        insample_y = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

        futr_size = enc.futr_input_size
        hist_size = enc.hist_input_size
        stat_size = enc.stat_input_size

        futr_exog = torch.zeros(1, window_size + h, futr_size)
        hist_exog = torch.zeros(1, window_size, hist_size)
        stat_exog = torch.zeros(1, stat_size) if stat_size > 0 else torch.zeros(1, 0)

        with torch.no_grad():
            mu, logvar = enc(insample_y, futr_exog, hist_exog, stat_exog)

        all_mu.append(mu.cpu().numpy())
        all_logvar.append(logvar.cpu().numpy())
        all_windows.append(window)
        windows_collected += 1

    if not all_mu:
        return {"mu": np.array([]), "logvar": np.array([]), "windows": np.array([])}

    return {
        "mu": np.concatenate(all_mu, axis=0),
        "logvar": np.concatenate(all_logvar, axis=0),
        "windows": np.stack(all_windows, axis=0),
    }


def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float = None) -> float:
    """Compute Maximum Mean Discrepancy between two sample sets using RBF kernel."""
    if gamma is None:
        # Use median heuristic for bandwidth
        from scipy.spatial.distance import cdist
        all_data = np.vstack([X[:100], Y[:100]])  # subsample for efficiency
        dists = cdist(all_data, all_data, metric="sqeuclidean")
        median_dist = np.median(dists[dists > 0])
        gamma = 1.0 / (median_dist + 1e-8)

    def rbf_kernel(A, B, gamma):
        dists = np.sum(A**2, axis=1, keepdims=True) + np.sum(B**2, axis=1, keepdims=True).T - 2 * A @ B.T
        return np.exp(-gamma * np.clip(dists, 0, 500))

    Kxx = rbf_kernel(X, X, gamma)
    Kyy = rbf_kernel(Y, Y, gamma)
    Kxy = rbf_kernel(X, Y, gamma)

    return float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())


def compute_domain_metrics(mu: np.ndarray, labels: list) -> dict:
    """
    Compute quantitative domain-structure metrics on the latent space.

    Returns dict with:
        - domain_classifier_accuracy: linear classifier accuracy (lower = more shared)
        - silhouette_score: clustering quality by domain (lower = more mixing)
        - pairwise_mmd: dict of MMD distances between domain pairs
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = le.classes_

    # Domain classifier accuracy
    clf = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    clf.fit(mu, y)
    train_acc = accuracy_score(y, clf.predict(mu))

    # Silhouette score (only if we have > 1 domain and enough samples)
    sil = None
    if len(classes) > 1 and len(y) > len(classes):
        try:
            sil = silhouette_score(mu, y, sample_size=min(5000, len(y)), random_state=42)
        except Exception:
            pass

    # Pairwise MMD
    pairwise_mmd = {}
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            if j <= i:
                continue
            mask1 = y == i
            mask2 = y == j
            if mask1.sum() > 5 and mask2.sum() > 5:
                mmd_val = compute_mmd(mu[mask1], mu[mask2])
                pairwise_mmd[f"{c1}_vs_{c2}"] = mmd_val

    return {
        "domain_classifier_accuracy": train_acc,
        "silhouette_score": sil,
        "pairwise_mmd": pairwise_mmd,
        "n_domains": len(classes),
        "n_samples": len(y),
        "domains": list(classes),
    }


def compute_temporal_features(windows: np.ndarray) -> pd.DataFrame:
    """
    Compute temporal characteristics for each time series window.

    Args:
        windows: np.ndarray of shape [N, window_size]

    Returns:
        DataFrame with columns: trend_strength, seasonality_strength,
        coefficient_of_variation, mean_level, volatility
    """
    features = []
    for i in range(windows.shape[0]):
        w = windows[i]
        n = len(w)

        # Mean level (log-scale to handle wide range)
        mean_val = np.mean(np.abs(w)) + 1e-8
        log_mean = np.log10(mean_val)

        # Coefficient of variation (relative variability)
        cv = np.std(w) / mean_val if mean_val > 1e-6 else 0.0

        # Trend strength: correlation with linear time index
        t = np.arange(n, dtype=np.float64)
        if np.std(w) > 1e-8:
            trend = np.corrcoef(t, w.astype(np.float64))[0, 1]
        else:
            trend = 0.0

        # Volatility: std of first differences (captures local variation)
        diffs = np.diff(w)
        volatility = np.std(diffs) / mean_val if mean_val > 1e-6 else 0.0

        # Seasonality strength via autocorrelation
        # Look for periodicity at common lags
        w_centered = w - np.mean(w)
        var = np.var(w_centered)
        if var > 1e-8 and n > 12:
            # Check lags 2-12 for seasonal patterns
            acf_vals = []
            for lag in range(2, min(13, n // 2)):
                acf = np.sum(w_centered[lag:] * w_centered[:-lag]) / (var * (n - lag))
                acf_vals.append(acf)
            seasonality = max(acf_vals) if acf_vals else 0.0
        else:
            seasonality = 0.0

        features.append({
            "trend_strength": float(trend),
            "seasonality_strength": float(seasonality),
            "coefficient_of_variation": float(np.clip(cv, 0, 10)),
            "log_mean_level": float(log_mean),
            "volatility": float(np.clip(volatility, 0, 10)),
        })

    return pd.DataFrame(features)


def plot_tsne_temporal(
    mu: np.ndarray,
    windows: np.ndarray,
    domain_labels: list,
    title_prefix: str = "TimeGEN Latent Space",
    save_path: str = None,
    perplexity: int = 30,
):
    """
    Create a multi-panel t-SNE visualization colored by temporal characteristics.
    Same t-SNE coordinates, different colorings to show what organizes the latent space.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(mu)

    # Compute temporal features
    feats = compute_temporal_features(windows)

    feature_configs = [
        ("Domain", domain_labels, "Set2", True),
        ("Trend Strength", feats["trend_strength"].values, "RdBu_r", False),
        ("Seasonality", feats["seasonality_strength"].values, "viridis", False),
        ("CV (Relative Variability)", feats["coefficient_of_variation"].values, "magma", False),
        ("Log Mean Level", feats["log_mean_level"].values, "plasma", False),
        ("Volatility", feats["volatility"].values, "inferno", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for ax, (label, values, cmap, is_categorical) in zip(axes, feature_configs):
        if is_categorical:
            df_plot = pd.DataFrame({
                "t-SNE 1": coords[:, 0],
                "t-SNE 2": coords[:, 1],
                "color": values,
            })
            sns.scatterplot(
                data=df_plot, x="t-SNE 1", y="t-SNE 2",
                hue="color", palette=cmap, alpha=0.6, s=15, ax=ax,
                legend=True,
            )
            ax.legend(fontsize=7, loc="upper right")
        else:
            sc = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=values, cmap=cmap, alpha=0.6, s=15,
            )
            plt.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("t-SNE 1", fontsize=9)
        ax.set_ylabel("t-SNE 2", fontsize=9)

    plt.suptitle(f"{title_prefix} -- Colored by Temporal Characteristics", fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.close(fig)
    return coords, feats


def plot_tsne(
    mu: np.ndarray,
    labels: list,
    title: str = "Latent Space (t-SNE)",
    save_path: str = None,
    perplexity: int = 30,
):
    """Create a t-SNE visualization of the latent space, colored by domain."""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(mu)

    df_plot = pd.DataFrame({
        "t-SNE 1": coords[:, 0],
        "t-SNE 2": coords[:, 1],
        "Domain": labels,
    })

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.scatterplot(
        data=df_plot,
        x="t-SNE 1",
        y="t-SNE 2",
        hue="Domain",
        palette="Set2",
        alpha=0.6,
        s=20,
        ax=ax,
    )
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.close(fig)
    return coords


def plot_tsne_comparison(
    mu_vae: np.ndarray,
    labels_vae: list,
    mu_ae: np.ndarray,
    labels_ae: list,
    save_path: str = None,
):
    """Side-by-side t-SNE comparison of VAE vs AE latent spaces."""
    tsne_vae = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    coords_vae = tsne_vae.fit_transform(mu_vae)

    tsne_ae = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    coords_ae = tsne_ae.fit_transform(mu_ae)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, coords, labels, subtitle in [
        (axes[0], coords_vae, labels_vae, "TimeGEN (VAE)"),
        (axes[1], coords_ae, labels_ae, "TimeGEN-AE (Deterministic)"),
    ]:
        df_plot = pd.DataFrame({
            "t-SNE 1": coords[:, 0],
            "t-SNE 2": coords[:, 1],
            "Domain": labels,
        })
        sns.scatterplot(
            data=df_plot,
            x="t-SNE 1",
            y="t-SNE 2",
            hue="Domain",
            palette="Set2",
            alpha=0.6,
            s=20,
            ax=ax,
        )
        ax.set_title(subtitle)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.suptitle("Latent Space Structure: VAE vs Deterministic AE", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.close(fig)


def run_vae_analysis(
    source_dataset: str = None,
    target_datasets: list = None,
    max_windows_per_dataset: int = 500,
):
    """
    Run the latent analysis using a SINGLE model applied to multiple domains.
    This is the meaningful analysis: how does one model's encoder represent
    data from different domains?
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Default: use M4_Monthly as source (large dataset, monthly frequency)
    # and test on diverse targets
    if source_dataset is None:
        # Pick a Monthly source so window_size is compatible with other Monthly datasets
        source_dataset = "M4_Monthly"
    if target_datasets is None:
        # Same-frequency datasets that the source model can process
        source_config = DATASET_CONFIGS[source_dataset]
        target_datasets = [
            k for k, v in DATASET_CONFIGS.items()
            if v["h"] == source_config["h"]  # same horizon = same input_size compatibility
        ]

    print("=" * 60)
    print("TimeGEN Latent Space Analysis: Shared Structure vs Specialization")
    print(f"Source model: {source_dataset}")
    print(f"Target domains: {target_datasets}")
    print("=" * 60)

    # Extract latents from single model across domains
    print("\n[1/4] Extracting latent representations...")
    mu_vae, labels_vae, keys_vae, windows_vae = extract_latents_single_model(
        source_dataset_key=source_dataset,
        target_dataset_keys=target_datasets,
        model_type="AutoTimeGEN",
        max_windows_per_dataset=max_windows_per_dataset,
    )
    print(f"  Total: {mu_vae.shape[0]} vectors, dim={mu_vae.shape[1]}")

    # Compute metrics
    print("\n[2/4] Computing domain-structure metrics...")
    metrics_vae = compute_domain_metrics(mu_vae, labels_vae)
    print(f"  Domain classifier accuracy: {metrics_vae['domain_classifier_accuracy']:.3f}")
    if metrics_vae['silhouette_score'] is not None:
        print(f"  Silhouette score: {metrics_vae['silhouette_score']:.3f}")
    print(f"  Pairwise MMD distances:")
    for pair, mmd in sorted(metrics_vae["pairwise_mmd"].items()):
        print(f"    {pair}: {mmd:.6f}")

    # t-SNE by domain
    print("\n[3/4] Generating domain t-SNE visualization...")
    plot_tsne(
        mu_vae,
        labels_vae,
        title=f"TimeGEN Latent Space (model trained on {source_dataset})",
        save_path=str(OUTPUT_DIR / f"tsne_vae_{source_dataset}_to_targets.pdf"),
    )

    # t-SNE by temporal characteristics
    print("\n[4/4] Generating temporal characteristics t-SNE...")
    plot_tsne_temporal(
        mu_vae,
        windows_vae,
        labels_vae,
        title_prefix=f"TimeGEN Latent Space ({source_dataset} model)",
        save_path=str(OUTPUT_DIR / f"tsne_temporal_{source_dataset}.pdf"),
    )

    # Save metrics
    import json
    metrics_path = OUTPUT_DIR / f"latent_metrics_vae_{source_dataset}.json"
    serializable = {"source_model": source_dataset, "target_datasets": keys_vae}
    for k, v in metrics_vae.items():
        if k == "pairwise_mmd":
            serializable[k] = {pk: round(float(pv), 6) for pk, pv in v.items()}
        elif isinstance(v, (np.floating, np.integer)):
            serializable[k] = round(float(v), 6)
        elif isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Metrics saved: {metrics_path}")

    return mu_vae, labels_vae, metrics_vae


def run_vae_vs_ae_analysis(
    source_dataset: str = None,
    target_datasets: list = None,
    max_windows_per_dataset: int = 500,
):
    """
    Run comparative analysis between VAE and AE latent spaces.
    Uses a single source model for each variant, applied to multiple target domains.
    Requires both AutoTimeGEN and AutoTimeGEN_AE models to be trained.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if source_dataset is None:
        source_dataset = "M4_Monthly"
    if target_datasets is None:
        source_config = DATASET_CONFIGS[source_dataset]
        target_datasets = [
            k for k, v in DATASET_CONFIGS.items()
            if v["h"] == source_config["h"]
        ]

    print("=" * 60)
    print("Comparative Latent Analysis: VAE vs Deterministic AE")
    print(f"Source model: {source_dataset}")
    print("=" * 60)

    # Extract VAE latents
    print("\n[1/4] Extracting VAE latents...")
    mu_vae, labels_vae, _, _ = extract_latents_single_model(
        source_dataset_key=source_dataset,
        target_dataset_keys=target_datasets,
        model_type="AutoTimeGEN",
        max_windows_per_dataset=max_windows_per_dataset,
    )

    # Extract AE latents
    print("\n[2/4] Extracting AE latents...")
    mu_ae, labels_ae, _, _ = extract_latents_single_model(
        source_dataset_key=source_dataset,
        target_dataset_keys=target_datasets,
        model_type="AutoTimeGEN_AE",
        max_windows_per_dataset=max_windows_per_dataset,
    )

    # Compute metrics for both
    print("\n[3/4] Computing metrics...")
    metrics_vae = compute_domain_metrics(mu_vae, labels_vae)
    metrics_ae = compute_domain_metrics(mu_ae, labels_ae)

    print(f"\n  {'Metric':<35} {'VAE':>10} {'AE':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Domain classifier accuracy':<35} {metrics_vae['domain_classifier_accuracy']:>10.3f} {metrics_ae['domain_classifier_accuracy']:>10.3f}")
    if metrics_vae["silhouette_score"] is not None and metrics_ae["silhouette_score"] is not None:
        print(f"  {'Silhouette score':<35} {metrics_vae['silhouette_score']:>10.3f} {metrics_ae['silhouette_score']:>10.3f}")

    # Side-by-side t-SNE
    print("\n[4/4] Generating comparison visualization...")
    plot_tsne_comparison(
        mu_vae, labels_vae,
        mu_ae, labels_ae,
        save_path=str(OUTPUT_DIR / f"tsne_vae_vs_ae_{source_dataset}.pdf"),
    )

    return metrics_vae, metrics_ae


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TimeGEN Latent Space Analysis")
    parser.add_argument("--mode", choices=["vae", "compare"], default="vae",
                        help="'vae' for single-model analysis, 'compare' for VAE vs AE")
    parser.add_argument("--max-windows", type=int, default=500,
                        help="Max windows to extract per dataset")
    parser.add_argument("--source", type=str, default=None,
                        help="Source dataset key for the model (e.g. M4_Monthly)")
    parser.add_argument("--targets", nargs="+", default=None,
                        help="Target dataset keys (e.g. M1_Monthly M3_Monthly Tourism_Monthly)")
    args = parser.parse_args()

    if args.mode == "vae":
        run_vae_analysis(
            source_dataset=args.source,
            target_datasets=args.targets,
            max_windows_per_dataset=args.max_windows,
        )
    elif args.mode == "compare":
        run_vae_vs_ae_analysis(
            source_dataset=args.source,
            target_datasets=args.targets,
            max_windows_per_dataset=args.max_windows,
        )
