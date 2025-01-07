from typing import Tuple, List
import pandas as pd
import numpy as np
import json
import os
import warnings
from tsfeatures import tsfeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from datetime import datetime


def split_train_test(
    unique_ids,
    key,
    split_dir,
    train_ratio=0.8,
    max_train_series=100,
    max_test_series=20,
):
    """
    Splits data indices into train and test indices, stores them in a JSON file,
    and retrieves them by a numeric key.
    """
    os.makedirs(split_dir, exist_ok=True)
    split_file = os.path.join(split_dir, "splits.json")

    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            splits = json.load(f)
    else:
        splits = {}

    if str(key) in splits:
        print(f"         Key {key} already exists. Skipping split creation.")
        return splits[str(key)]["train_indices"], splits[str(key)]["test_indices"]

    n_series = len(unique_ids)

    n_series_train = min(int(train_ratio * n_series), max_train_series)
    n_series_test = min(n_series - n_series_train, max_test_series)

    train_indices = np.random.choice(unique_ids, size=n_series_train, replace=False)
    test_indices = np.setdiff1d(unique_ids, train_indices)[:n_series_test]

    splits[str(key)] = {
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
    }

    with open(split_file, "w") as f:
        json.dump(splits, f, indent=4)

    return splits[str(key)]["train_indices"], splits[str(key)]["test_indices"]


def filter_data_by_indices(
    data: pd.DataFrame, indices: List[str], label_value: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filters data by indices and assigns labels."""
    filtered_data = data[data["unique_id"].isin(indices)]
    unique_ids_n = filtered_data["unique_id"].nunique()
    labels = pd.DataFrame([label_value] * unique_ids_n)
    return filtered_data, labels


def safe_generate_features(data, freq):
    """
    Safely generates time series features using tsfeatures, with warning filtering.
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            features = tsfeatures(data, freq=freq)
            for warning in w:
                if "divide by zero" in str(warning.message):
                    print("Detected problematic data. Skipping.")
                    return None
            return features
        except Exception as e:
            print(f"Error generating features: {e}")
            return None


def plot_feature_importance(
    feature_names, feature_importances, score, loss, dataset_name, dataset_group
):
    """
    Plots and saves feature importance.
    """
    sorted_idx = feature_importances.argsort()
    sorted_features = feature_names[sorted_idx]
    sorted_importances = feature_importances[sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importances, color="skyblue")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance")
    plt.tight_layout()

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = (
        f"assets/plots/{current_datetime}_feature_importance_vae_generated_vs_original_"
        f"{dataset_name}_{dataset_group}_{round(score, 2)}_{round(loss, 2)}.pdf"
    )
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Feature importance plot saved to {filename}")


def compute_discriminative_score(
    unique_ids,
    original_data,
    synthetic_data,
    freq,
    dataset_name,
    dataset_group,
    loss,
    generate_feature_plot=True,
    samples=1,
):
    scores = []
    FREQS = {"H": 24, "D": 1, "M": 12, "Q": 4, "W": 1, "Y": 1}
    freq = FREQS[freq]
    for sample in range(samples):
        print(f"    Sample {sample} of {samples}")
        train_idx, test_idx = split_train_test(
            unique_ids,
            sample,
            split_dir=f"assets/model_weights/{dataset_name}_{dataset_group}_data_split_discriminator",
        )

        # original data
        original_data_train, original_data_train_y = filter_data_by_indices(
            original_data, train_idx, label_value=0
        )
        original_data_test, original_data_test_y = filter_data_by_indices(
            original_data, test_idx, label_value=0
        )

        original_features_train = safe_generate_features(original_data_train, freq=freq)
        original_features_test = safe_generate_features(original_data_test, freq=freq)

        if original_features_train is None or original_features_test is None:
            print("Feature generation failed for original data. Skipping iteration.")
            continue

        # synthetic data
        synthetic_data_train, synthetic_data_train_y = filter_data_by_indices(
            synthetic_data, train_idx, label_value=1
        )
        synthetic_data_test, synthetic_data_test_y = filter_data_by_indices(
            synthetic_data, test_idx, label_value=1
        )

        synthetic_features_train = safe_generate_features(
            synthetic_data_train, freq=freq
        )
        synthetic_features_test = safe_generate_features(synthetic_data_test, freq=freq)

        if synthetic_features_train is None or synthetic_features_test is None:
            print("Feature generation failed for synthetic data. Skipping iteration.")
            continue

        # Classifier
        X_train = pd.concat(
            (original_features_train, synthetic_features_train), ignore_index=True
        ).drop(columns=["unique_id"], errors="ignore")
        y_train = pd.concat(
            (original_data_train_y, synthetic_data_train_y), ignore_index=True
        )

        X_test = pd.concat(
            (original_features_test, synthetic_features_test), ignore_index=True
        ).drop(columns=["unique_id"], errors="ignore")
        y_test = pd.concat(
            (original_data_test_y, synthetic_data_test_y), ignore_index=True
        )

        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)

        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # feature importance
        feature_importances = classifier.feature_importances_
        score = f1_score(y_test, y_pred)
        print("F1 score:", score)
        if generate_feature_plot:
            plot_feature_importance(
                X_train.columns,
                feature_importances,
                score,
                loss,
                dataset_name,
                dataset_group,
            )
        scores.append(score)

    if scores:
        final_score = np.average(scores)
        print(f"\n\n### -> Final score: {final_score:.4f}")
    else:
        print("No valid iterations completed. Final score is undefined.")
        final_score = None

    return final_score
