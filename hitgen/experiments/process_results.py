import os
import json
import pandas as pd

base_path = "assets/results_forecast_out_domain"
results_files_out_domain = [f for f in os.listdir(base_path) if f.endswith(".json")]

results_combined = []
for result in results_files_out_domain:
    with open(os.path.join(base_path, result), "r") as f:
        result_details = json.load(f)

    results_combined.append(result_details)

results_df = pd.DataFrame(results_combined).reset_index(drop=True)

results_filtered = results_df[
    (results_df["Dataset Source"] != results_df["Dataset"])
    # & (results_df["Dataset"] != "Traffic")
    # & (results_df["Dataset Source"] != "Traffic")
    # & ~(
    #     (results_df["Dataset Source"] == "M3")
    #     & (results_df["Dataset Group Source"] == "Yearly")
    # )
].copy()

results_filtered.to_csv(os.path.join(base_path, "results_filtered.csv"), index=False)


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
    ]
]

# rank all seasonalities by source dataset and method
results_all_seasonalities = results_filtered.copy()
results_all_seasonalities["Rank"] = results_all_seasonalities.groupby(
    ["Source-Target Pair"]
)["SMAPE Mean"].rank(method="min")

results_all_seasonalities = (
    results_all_seasonalities.groupby(
        ["Dataset Source", "Dataset Group Source", "Method"]
    )["Rank"]
    .mean()
    .reset_index()
)

results_all_seasonalities.to_csv(
    os.path.join(base_path, "results_ranks_all_seasonalities.csv"), index=False
)

# smape all seasonalities by source dataset and method
results_all_seasonalities_smape = results_filtered.copy()
results_all_seasonalities_smape = (
    results_all_seasonalities_smape.groupby(
        ["Dataset Source", "Dataset Group Source", "Method"]
    )["SMAPE Mean"]
    .mean()
    .reset_index()
)

results_all_seasonalities_smape.to_csv(
    os.path.join(base_path, "results_avg_smape_all_seasonalities.csv"), index=False
)

# rank all seasonalities by method
results_all_seasonalities_method = results_filtered.copy()
results_all_seasonalities_method["Rank"] = results_all_seasonalities_method.groupby(
    ["Source-Target Pair"]
)["SMAPE Mean"].rank(method="min")

results_all_seasonalities_method = (
    results_all_seasonalities_method.groupby(["Method"])["Rank"].mean().reset_index()
)

results_all_seasonalities_method.sort_values(by="Rank", inplace=True)

results_all_seasonalities_method.to_csv(
    os.path.join(base_path, "results_ranks_all_seasonalities_method.csv"), index=False
)

# sampe all seasonalities by method
results_all_seasonalities_method_smape = results_filtered.copy()
results_all_seasonalities_method_smape = (
    results_all_seasonalities_method_smape.groupby(["Method"])["SMAPE Mean"]
    .mean()
    .reset_index()
)

results_all_seasonalities_method_smape.sort_values(by="SMAPE Mean", inplace=True)

results_all_seasonalities_method_smape.to_csv(
    os.path.join(base_path, "results_smape_all_seasonalities_method.csv"),
    index=False,
)

# rank same seasonalities by source dataset and method
results_same_seasonalities = results_filtered[
    (
        results_filtered["Dataset Group Source"]
        == results_filtered["Dataset Group Target"]
    )
].copy()

results_same_seasonalities["Rank"] = results_same_seasonalities.groupby(
    ["Source-Target Pair"]
)["SMAPE Mean"].rank(method="min")

results_same_seasonalities = (
    results_same_seasonalities.groupby(
        ["Dataset Source", "Dataset Group Source", "Method"]
    )["Rank"]
    .mean()
    .reset_index()
)

results_same_seasonalities.to_csv(
    os.path.join(base_path, "results_ranks_same_seasonalities.csv"), index=False
)

# sampe same seasonalities by source dataset and method
results_same_seasonalities_smape = results_filtered[
    (
        results_filtered["Dataset Group Source"]
        == results_filtered["Dataset Group Target"]
    )
].copy()

results_same_seasonalities_smape = (
    results_same_seasonalities_smape.groupby(
        ["Dataset Source", "Dataset Group Source", "Method"]
    )["SMAPE Mean"]
    .mean()
    .reset_index()
)

results_same_seasonalities_smape.to_csv(
    os.path.join(base_path, "results_smape_same_seasonalities.csv"), index=False
)


# rank same seasonalities by method
results_same_seasonalities_method = results_filtered[
    (
        results_filtered["Dataset Group Source"]
        == results_filtered["Dataset Group Target"]
    )
].copy()

results_same_seasonalities_method["Rank"] = results_same_seasonalities_method.groupby(
    ["Source-Target Pair"]
)["SMAPE Mean"].rank(method="min")

results_same_seasonalities_method = (
    results_same_seasonalities_method.groupby(["Method"])["Rank"].mean().reset_index()
)

results_same_seasonalities_method.sort_values(by="Rank", inplace=True)


results_same_seasonalities_method.to_csv(
    os.path.join(base_path, "results_ranks_same_seasonalities_method.csv"), index=False
)

# sampe same seasonalities by method
results_same_seasonalities_method_smape = results_filtered[
    (
        results_filtered["Dataset Group Source"]
        == results_filtered["Dataset Group Target"]
    )
].copy()

results_same_seasonalities_method_smape = (
    results_same_seasonalities_method_smape.groupby(["Method"])["SMAPE Mean"]
    .mean()
    .reset_index()
)

results_same_seasonalities_method_smape.sort_values(by="SMAPE Mean", inplace=True)


results_same_seasonalities_method_smape.to_csv(
    os.path.join(base_path, "results_smape_same_seasonalities_method.csv"), index=False
)
