import os
import pandas as pd

base_path = "assets/model_weights_out_domain/hypertuning_final"
plots_dir = "assets/plots"
results_out_dir = "assets/results_forecast_out_domain_summary"

os.makedirs(plots_dir, exist_ok=True)
os.makedirs(results_out_dir, exist_ok=True)

performance_results = [
    f
    for f in os.listdir(base_path)
    if (f.endswith(".csv") and not (f.startswith("MIXED")))
]

results_combined = []
for result in performance_results:
    csv_path = os.path.join(base_path, result)
    df = pd.read_csv(csv_path)

    df["Dataset"] = result.split("_")[0]
    df["Group"] = result.split("_")[1]
    df["Method"] = result.split("_")[2]
    df = df[["Dataset", "Group", "Method", "time_total_s", "loss"]]

    results_combined.append(df)

idx = (
    pd.concat(results_combined, ignore_index=True)
    .groupby(["Dataset", "Group", "Method"])["loss"]
    .idxmin()
)

min_loss_df = pd.concat(results_combined, ignore_index=True).loc[idx]


results_df = min_loss_df.groupby("Method")["time_total_s"].sum().reset_index()

method_order = sorted(results_df["Method"].unique().tolist())

csv_path = os.path.join(results_out_dir, "training_time_stats_per_method.csv")
results_df.to_csv(csv_path, index=False)
