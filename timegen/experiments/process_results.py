import os
import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd


BASE_OUT_DOMAIN = Path("assets/results_forecast_out_domain")
BASE_IN_DOMAIN = Path("assets/results_forecast_in_domain")
BASE_BASIC_FORECASTING = Path("assets/results_forecast_basic_forecasting")
SUMMARY_DIR = Path("assets/results_forecast_out_domain_summary")

FM_PATHS = {
    "moirai": SUMMARY_DIR / "moirai_results.csv",
    "timemoe": SUMMARY_DIR / "timemoe_results.csv",
}


def load_json_files(base_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load all JSON files from a directory into a DataFrame.
    """
    base_path = Path(base_path)
    data = []
    for file in base_path.glob("*.json"):
        with file.open("r") as f:
            data.append(json.load(f))
    return pd.DataFrame(data)


def rename_columns(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """
    Rename columns for out-domain or in-domain data to unified format.
    """
    metric_suffix = f"Per Series_{domain}"
    rename_map = {
        f"Forecast SMAPE MEAN (last window) {metric_suffix}": "SMAPE Mean",
        f"Forecast MASE MEAN (last window) {metric_suffix}": "MASE Mean",
        f"Forecast MAE MEAN (last window) {metric_suffix}": "MAE Mean",
        f"Forecast RMSE MEAN (last window) {metric_suffix}": "RMSE Mean",
        f"Forecast RMSSE MEAN (last window) {metric_suffix}": "RMSSE Mean",
        "Dataset": "Dataset Target",
        "Group": "Dataset Group Target",
    }
    return df.rename(columns=rename_map)


def align_columns(df: pd.DataFrame, reference_cols: List[str]) -> pd.DataFrame:
    """
    Align the DataFrame columns to a reference column list.
    Missing columns will be added with NA values.
    """
    for col in reference_cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[reference_cols]


def add_source_target_pair_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column for identifying source-target dataset pairs.
    """
    return df.assign(
        **{
            "Source-Target Pair": df["Dataset Source"]
            + " ("
            + df["Dataset Group Source"]
            + ") → "
            + df["Dataset Target"]
            + " ("
            + df["Dataset Group Target"]
            + ")"
        }
    )


def apply_fm_results(base_df: pd.DataFrame, path_fm: Union[str, Path]) -> pd.DataFrame:
    """
    Merge forecast model (FM) results with the filtered coreset.
    """
    fm_df = pd.read_csv(path_fm)
    required_columns = base_df.columns

    for col in required_columns:
        if col not in fm_df.columns:
            fm_df[col] = pd.NA

    fm_df["Dataset Source"] = "MIXED"
    fm_df["Dataset Group Source"] = fm_df.apply(
        lambda x: f"ALL_BUT_{x['Dataset Target']}_{x['Dataset Group Target']}", axis=1
    )
    fm_df = fm_df[required_columns]

    return pd.concat([base_df, fm_df], ignore_index=True)


def summarize_metric(
    df: pd.DataFrame,
    metric: str,
    mode: str,
    aggregate_by: List[str],
    rank_within: Optional[List[str]] = None,
    filter_same_seasonality: bool = False,
    src_seas_col: str = "Dataset Group Source",
    tgt_seas_col: str = "Dataset Group Target",
    out_path: Optional[Path] = None,
    fname: Optional[str] = None,
    rank_method: str = "min",
    agg_func=np.nanmean,
) -> pd.DataFrame:
    """
    Summarize a metric either by rank or mean.
    """
    work = df.copy()

    if filter_same_seasonality:
        work = work[work[src_seas_col] == work[tgt_seas_col]]

    if mode == "rank":
        if not rank_within:
            raise ValueError("`rank_within` must be given when mode='rank'")
        work["Rank"] = work.groupby(rank_within)[metric].rank(method=rank_method)
        summary = work.groupby(aggregate_by)["Rank"].apply(agg_func).reset_index()
        summary.rename(columns={"Rank": "Rank"}, inplace=True)
        sort_by = ["Rank"] if aggregate_by == ["Method"] else aggregate_by + ["Rank"]
    elif mode == "mean":
        summary = work.groupby(aggregate_by)[metric].apply(agg_func).reset_index()
        summary.rename(columns={metric: metric}, inplace=True)
        sort_by = [metric] if aggregate_by == ["Method"] else aggregate_by + [metric]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    summary.sort_values(by=sort_by, inplace=True)

    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)
        if fname is None:
            stem = "_".join(aggregate_by)
            fname = f"{mode}_{metric.replace(' ','_').lower()}_{stem}.csv"
        summary.to_csv(out_path / fname, index=False)

    return summary


def main():
    out_df = load_json_files(BASE_OUT_DOMAIN)
    in_df = load_json_files(BASE_IN_DOMAIN)
    basic_for_df = load_json_files(BASE_BASIC_FORECASTING)

    out_df = out_df[out_df["Dataset Source"] != out_df["Dataset"]]
    out_df = rename_columns(out_df, "out_domain")
    in_df = rename_columns(in_df, "in_domain")
    basic_for_df = rename_columns(basic_for_df, "basic_forecasting")

    common_cols = [
        "Dataset Source",
        "Dataset Group Source",
        "Dataset Target",
        "Dataset Group Target",
        "Method",
        "SMAPE Mean",
        "MASE Mean",
        "MAE Mean",
        "RMSE Mean",
        "RMSSE Mean",
    ]

    in_df["Dataset Source"] = "None"
    in_df["Dataset Group Source"] = "None"
    in_df = align_columns(in_df, common_cols)
    basic_for_df["Dataset Source"] = "None"
    basic_for_df["Dataset Group Source"] = "None"
    basic_for_df = align_columns(basic_for_df, common_cols)
    out_df = align_columns(out_df, common_cols)

    # separate coreset and add FM results and add pair columns
    coreset_df = out_df[out_df["Dataset Source"] == "MIXED"].copy()
    coreset_df = apply_fm_results(coreset_df, FM_PATHS["moirai"])
    coreset_df = apply_fm_results(coreset_df, FM_PATHS["timemoe"])

    out_df = out_df[out_df["Dataset Source"] != "MIXED"]

    out_df = add_source_target_pair_column(out_df)
    in_df = add_source_target_pair_column(in_df)
    basic_for_df = add_source_target_pair_column(basic_for_df)
    coreset_df = add_source_target_pair_column(coreset_df)

    # summarize results
    metric = "MASE Mean"
    metric_store = metric.replace(" ", "_").lower()

    for df, suffix in zip(
        [basic_for_df, in_df, out_df, coreset_df],
        ["_basic_forecasting", "_in_domain", "_out_domain", "_out_domain_coreset"],
    ):
        summarize_metric(
            df,
            metric,
            "rank",
            aggregate_by=["Dataset Source", "Dataset Group Source", "Method"],
            rank_within=["Source-Target Pair"],
            out_path=SUMMARY_DIR,
            fname=f"results_ranks_all_seasonalities{suffix}.csv",
        )
        summarize_metric(
            df,
            metric,
            "mean",
            aggregate_by=["Dataset Source", "Dataset Group Source", "Method"],
            out_path=SUMMARY_DIR,
            fname=f"results_all_seasonalities_{metric_store}{suffix}.csv",
        )
        summarize_metric(
            df,
            metric,
            "mean",
            aggregate_by=[
                "Dataset Source",
                "Dataset Group Source",
                "Dataset Target",
                "Dataset Group Target",
                "Method",
            ],
            out_path=SUMMARY_DIR,
            fname=f"results_all_seasonalities_all_combinations_{metric_store}{suffix}.csv",
        )
        summarize_metric(
            df,
            metric,
            "rank",
            aggregate_by=["Method"],
            rank_within=["Source-Target Pair"],
            out_path=SUMMARY_DIR,
            fname=f"results_ranks_all_seasonalities_by_method_{metric_store}{suffix}.csv",
        )
        summarize_metric(
            df,
            metric,
            "mean",
            aggregate_by=["Method"],
            out_path=SUMMARY_DIR,
            fname=f"results_all_seasonalities_by_method_{metric_store}{suffix}.csv",
        )


if __name__ == "__main__":
    main()
