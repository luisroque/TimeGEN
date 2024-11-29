import pandas as pd
import numpy as np
from tsfeatures import tsfeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from hitgen.metrics.discriminative_score import compute_discriminative_score


def load_data(file_path):
    """Loads data from the given CSV file."""
    return pd.read_csv(file_path)


def split_train_test(data, train_ratio=0.8):
    """Splits data indices into train and test indices."""
    n_series = data["unique_id"].nunique()
    n_series_train = int(train_ratio * n_series)
    all_indices = np.arange(n_series)
    train_indices = np.random.choice(all_indices, size=n_series_train, replace=False)
    test_indices = np.setdiff1d(all_indices, train_indices)
    return train_indices, test_indices


def filter_data_by_indices(data, indices, label_value):
    """Filters data by indices and assigns labels."""
    series_names = [f"series_{i}" for i in indices]
    filtered_data = data[data["unique_id"].isin(series_names)].reset_index(drop=True)
    labels = pd.DataFrame([label_value] * len(indices))
    return filtered_data, labels


def generate_features(data, freq):
    """Generates time series features using tsfeatures."""
    return tsfeatures(data, freq=freq)


def main():
    # Load data
    original_data = load_data("assets/results/hi2gen_original_data.csv")
    synthetic_data = load_data("assets/results/hi2gen_synthetic_data.csv")
    synthetic_timegan_data = load_data("assets/results/timegan_synthetic_data.csv")

    compute_discriminative_score(original_data, synthetic_data, "M")
    compute_discriminative_score(original_data, synthetic_timegan_data, "M")


if __name__ == "__main__":
    main()
