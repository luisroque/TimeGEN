import pandas as pd
import numpy as np
from tsfeatures import tsfeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


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

    # Split train and test indices
    train_idx, test_idx = split_train_test(original_data)

    # Frequency mapping
    FREQS = {"H": 24, "D": 1, "M": 12, "Q": 4, "W": 1, "Y": 1}
    freq = FREQS["M"]

    # Original data
    original_data_train, original_data_train_y = filter_data_by_indices(
        original_data, train_idx, label_value=0
    )
    original_data_test, original_data_test_y = filter_data_by_indices(
        original_data, test_idx, label_value=0
    )

    original_features_train = generate_features(original_data_train, freq=freq)
    original_features_test = generate_features(original_data_test, freq=freq)

    # Hi2Gen synthetic data
    synthetic_data_train, synthetic_data_train_y = filter_data_by_indices(
        synthetic_data, train_idx, label_value=1
    )
    synthetic_data_test, synthetic_data_test_y = filter_data_by_indices(
        synthetic_data, test_idx, label_value=1
    )

    synthetic_features_train = generate_features(synthetic_data_train, freq=freq)
    synthetic_features_test = generate_features(synthetic_data_test, freq=freq)

    # TimeGAN synthetic data
    synthetic_timegan_train, synthetic_timegan_train_y = filter_data_by_indices(
        synthetic_timegan_data, train_idx, label_value=1
    )
    synthetic_timegan_test, synthetic_timegan_test_y = filter_data_by_indices(
        synthetic_timegan_data, test_idx, label_value=1
    )

    synthetic_timegan_features_train = generate_features(
        synthetic_timegan_train, freq=freq
    )
    synthetic_timegan_features_test = generate_features(
        synthetic_timegan_test, freq=freq
    )

    # Classifier for Hi2Gen
    X_train_hi2gen = pd.concat(
        (original_features_train, synthetic_features_train), ignore_index=True
    ).drop(columns=["unique_id"], errors="ignore")
    y_train_hi2gen = pd.concat(
        (original_data_train_y, synthetic_data_train_y), ignore_index=True
    )

    X_test_hi2gen = pd.concat(
        (original_features_test, synthetic_features_test), ignore_index=True
    ).drop(columns=["unique_id"], errors="ignore")
    y_test_hi2gen = pd.concat(
        (original_data_test_y, synthetic_data_test_y), ignore_index=True
    )

    classifier_hi2gen = DecisionTreeClassifier()
    classifier_hi2gen.fit(X_train_hi2gen, y_train_hi2gen)

    y_pred_hi2gen = classifier_hi2gen.predict(X_test_hi2gen)
    print("Classification Report for Hi2Gen:")
    print(classification_report(y_test_hi2gen, y_pred_hi2gen))

    # Classifier for TimeGAN
    X_train_timegan = pd.concat(
        (original_features_train, synthetic_timegan_features_train), ignore_index=True
    ).drop(columns=["unique_id"], errors="ignore")
    y_train_timegan = pd.concat(
        (original_data_train_y, synthetic_timegan_train_y), ignore_index=True
    )

    X_test_timegan = pd.concat(
        (original_features_test, synthetic_timegan_features_test), ignore_index=True
    ).drop(columns=["unique_id"], errors="ignore")
    y_test_timegan = pd.concat(
        (original_data_test_y, synthetic_timegan_test_y), ignore_index=True
    )

    classifier_timegan = DecisionTreeClassifier()
    classifier_timegan.fit(X_train_timegan, y_train_timegan)

    y_pred_timegan = classifier_timegan.predict(X_test_timegan)
    print("Classification Report for TimeGAN:")
    print(classification_report(y_test_timegan, y_pred_timegan))


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
