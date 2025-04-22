import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base_path = "assets/model_weights/hypertuning"
plots_dir = "assets/plots"
results_out_dir = "assets/results_forecast_in_domain"

os.makedirs(plots_dir, exist_ok=True)
os.makedirs(results_out_dir, exist_ok=True)

performance_results = [f for f in os.listdir(base_path) if f.endswith(".csv")]

results_combined = []
for result in performance_results:
    csv_path = os.path.join(base_path, result)
    df = pd.read_csv(csv_path)

    df["Dataset"] = result.split("_")[0]
    df["Group"] = result.split("_")[1]
    df["Method"] = result.split("_")[2]
    df = df[["Dataset", "Group", "Method", "time_total_s"]]

    # compute min time per dataset

    results_combined.append(df)

results_df = pd.concat(results_combined, ignore_index=True)

method_order = sorted(results_df["Method"].unique().tolist())


plt.figure(figsize=(12, 6))
sns.boxplot(
    data=results_df, x="Method", y="time_total_s", order=method_order, showfliers=True
)
# plt.yscale("log")
plt.title("Total Training Time per Method (log scale sorted by average training time)")
plt.xlabel("Method")
plt.ylabel("Time (log seconds)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    os.path.join(plots_dir, "boxplot_training_time_per_method_sorted_log.png"), dpi=300
)
plt.show()

csv_path = os.path.join(results_out_dir, "training_time_stats_per_method.csv")
