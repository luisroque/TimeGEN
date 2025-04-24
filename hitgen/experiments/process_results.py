import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Optional


base_path_load = "assets/results_forecast_out_domain"
results_files_out_domain = [
    f for f in os.listdir(base_path_load) if f.endswith(".json")
]

results_combined = []
for result in results_files_out_domain:
    with open(os.path.join(base_path_load, result), "r") as f:
        result_details = json.load(f)

    results_combined.append(result_details)

results_df = pd.DataFrame(results_combined).reset_index(drop=True)

results_filtered = results_df[
    (results_df["Dataset Source"] != results_df["Dataset"])
].copy()


results_filtered["Source-Target Pair"] = (
    results_filtered["Dataset Source"]
    + " ("
    + results_filtered["Dataset Group Source"]
    + ") → "
    + results_filtered["Dataset"]
    + " ("
    + results_filtered["Group"]
    + ")"
)

results_filtered.rename(
    columns={
        "Forecast SMAPE MEAN (last window) Per Series_out_domain": "SMAPE Mean",
        "Forecast MASE MEAN (last window) Per Series_out_domain": "MASE Mean",
        "Forecast MAE MEAN (last window) Per Series_out_domain": "MAE Mean",
        "Forecast RMSE MEAN (last window) Per Series_out_domain": "RMSE Mean",
        "Forecast RMSSE MEAN (last window) Per Series_out_domain": "RMSSE Mean",
        "Dataset": "Dataset Target",
        "Group": "Dataset Group Target",
    },
    inplace=True,
)
results_filtered = results_filtered[
    [
        "Dataset Source",
        "Dataset Group Source",
        "Source-Target Pair",
        "Dataset Target",
        "Dataset Group Target",
        "Method",
        "SMAPE Mean",
        "MASE Mean",
        "MAE Mean",
        "RMSE Mean",
        "RMSSE Mean",
    ]
]


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
    fname: str | None = None,
    rank_method: str = "min",
    agg_func=np.nanmean,
) -> pd.DataFrame:
    """
    Generic summary / ranking utility for the forecast results grid.

    Parameters
        df  : cleaned results table
        metric : column to aggregate (e.g. "SMAPE Mean")
        mode : "rank" | "mean"
        rank_within : list of columns that define the grouping within which to rank.
                      Ignored when `mode == "mean"`.
        aggregate_by : final grouping columns for the summary table
        filter_same_seasonality : keep only rows where src & tgt seasonalities match
        out_path : directory to write csv (if None → just return df)
    """
    work = df.copy()

    if filter_same_seasonality:
        work = work[work[src_seas_col] == work[tgt_seas_col]]

    if mode == "rank":
        if not rank_within:
            raise ValueError("`rank_within` must be given when mode='rank'")
        work["Rank"] = work.groupby(rank_within)[metric].rank(method=rank_method)
        summary = (
            work.groupby(aggregate_by)["Rank"]
            .apply(agg_func)
            .reset_index()
            .rename(columns={"Rank": "Rank"})
        )
        sort_by = aggregate_by + ["Rank"]
        summary.sort_values(by=sort_by, inplace=True)

    elif mode == "mean":
        summary = (
            work.groupby(aggregate_by)[metric]
            .apply(agg_func)
            .reset_index()
            .rename(columns={metric: metric})
        )
        sort_by = aggregate_by + [metric]
        summary.sort_values(by=sort_by, inplace=True)

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        if fname is None:
            stem = "_".join(aggregate_by)
            fname = f"{mode}_{metric.replace(' ','_').lower()}_{stem}.csv"
        summary.to_csv(out_path / fname, index=False)

    return summary


base_path = Path("assets/results_forecast_out_domain_summary")
out_dir = base_path
metric = "MASE Mean"
metric_store = f"{metric.replace(' ','_').lower()}"

# # rank – all seasonalities, grouped by src-dataset & method
# summarize_metric(
#     results_filtered,
#     metric=metric,
#     mode="rank",
#     rank_within=["Source-Target Pair"],
#     aggregate_by=["Dataset Source", "Dataset Group Source", "Method"],
#     out_path=out_dir,
#     fname="results_ranks_all_seasonalities.csv",
# )

# mean SMAPE – same grouping
summarize_metric(
    results_filtered,
    metric=metric,
    mode="mean",
    aggregate_by=["Dataset Source", "Dataset Group Source", "Method"],
    out_path=out_dir,
    fname=f"results_all_seasonalities_{metric_store}.csv",
)

# mean SMAPE – every individual src–tgt pair
summarize_metric(
    results_filtered,
    metric=metric,
    mode="mean",
    aggregate_by=[
        "Dataset Source",
        "Dataset Group Source",
        "Dataset Target",
        "Dataset Group Target",
        "Method",
    ],
    out_path=out_dir,
    fname=f"results_all_seasonalities_all_combinations_{metric_store}.csv",
)

# rank – by Method only (all seasonalities)
summarize_metric(
    results_filtered,
    metric=metric,
    mode="rank",
    rank_within=["Source-Target Pair"],
    aggregate_by=["Method"],
    out_path=out_dir,
    fname=f"results_ranks_all_seasonalities_by_method_{metric_store}.csv",
)

# mean SMAPE – by Method only
summarize_metric(
    results_filtered,
    metric=metric,
    mode="mean",
    aggregate_by=["Method"],
    out_path=out_dir,
    fname=f"results_all_seasonalities_by_method_{metric_store}.csv",
)

# # rank & mean restricted to same seasonality transfers
# for m in ("rank", "mean"):
#     summarize_metric(
#         results_filtered,
#         metric=metric,
#         mode=m,
#         rank_within=None if m == "mean" else ["Source-Target Pair"],
#         aggregate_by=(
#             ["Dataset Source", "Dataset Group Source", "Method"]
#             if m == "rank"
#             else ["Method"]
#         ),
#         filter_same_seasonality=True,
#         out_path=out_dir,
#         fname=f"results_{m}_same_seasonalities_{'by_method' if m=='mean' else 'by_source'}_{metric_store}.csv",
#     )
