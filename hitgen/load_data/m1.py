import pandas as pd
from gluonts.dataset.repository.datasets import get_dataset, dataset_names
from hitgen.load_data.base import LoadDataset


class M1Dataset(LoadDataset):
    DATASET_NAME = "M1"

    horizons_map = {
        "Quarterly": 2,
        "Monthly": 8,
    }

    frequency_map = {
        "Quarterly": 4,
        "Monthly": 12,
    }

    context_length = {
        "Quarterly": 4,
        "Monthly": 12,
    }

    min_samples = {
        "Quarterly": 22,
        "Monthly": 52,
    }

    frequency_pd = {
        "Quarterly": "Q",
        "Monthly": "M",
    }

    data_group = [*horizons_map]
    horizons = [*horizons_map.values()]
    frequency = [*frequency_map.values()]

    @classmethod
    def load_data(cls, group, min_n_instances=None):

        dataset = get_dataset(f"m1_{group.lower()}", regenerate=False)
        train_list = dataset.train

        df_list = []
        for i, series in enumerate(train_list):
            s = pd.Series(
                series["target"],
                index=pd.date_range(
                    start=series["start"].to_timestamp(),
                    freq=series["start"].freq,
                    periods=len(series["target"]),
                ),
            )

            if group == "australian_electricity_demand":
                s = s.resample("W").sum()

            s_df = s.reset_index()
            s_df.columns = ["ds", "y"]
            s_df["unique_id"] = f"ID{i}"

            df_list.append(s_df)

        df = pd.concat(df_list).reset_index(drop=True)
        df = df[["unique_id", "ds", "y"]]

        if min_n_instances is not None:
            df = cls.prune_df_by_size(df, min_n_instances)

        return df
