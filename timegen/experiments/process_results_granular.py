import os
import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd


BASE_OUT_DOMAIN = Path("assets/results_forecast_out_domain")
BASE_IN_DOMAIN = Path("assets/results_forecast_in_domain")
BASE_BASIC_FORECASTING = Path("assets/results_forecast_basic_forecasting")
SUMMARY_DIR = Path("assets/results_summary")


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


def create_dataset_combination_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a dataset combination column for easier identification.
    """
    return df.assign(
        **{
            "Dataset_Combination": df["Dataset Target"]
            + "_"
            + df["Dataset Group Target"]
        }
    )


def process_out_domain_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process out-domain data and separate single-source and multi-source results.
    """
    # Filter out same dataset combinations (in-domain within out-domain)
    df_filtered = df[df["Dataset Source"] != df["Dataset Target"]].copy()

    # Separate single-source (specific source) and multi-source (MIXED source)
    single_source_df = df_filtered[df_filtered["Dataset Source"] != "MIXED"].copy()
    multi_source_df = df_filtered[df_filtered["Dataset Source"] == "MIXED"].copy()

    return single_source_df, multi_source_df


def create_granular_pivot_table(
    df: pd.DataFrame, metric: str = "MASE Mean", regime_name: str = "regime"
) -> pd.DataFrame:
    """
    Create a pivot table with methods as rows and datasets as columns.
    """
    # Clean method names (remove 'Auto' prefix)
    df = df.copy()
    df["Method"] = df["Method"].str.replace("Auto", "", regex=False)

    # Create pivot table
    pivot_df = df.pivot_table(
        index="Method",
        columns="Dataset_Combination",
        values=metric,
        aggfunc="mean",  # In case there are multiple entries
    )

    # Add average column
    pivot_df["Average"] = pivot_df.mean(axis=1, skipna=True)

    # Sort by average performance
    pivot_df = pivot_df.sort_values("Average")

    # Round to 3 decimal places
    pivot_df = pivot_df.round(3)

    return pivot_df


def process_regime_data(
    df: pd.DataFrame,
    regime_name: str,
    domain_suffix: str,
    output_dir: Path,
    metric: str = "MASE Mean",
) -> None:
    """
    Process data for a specific regime and save granular results.
    """
    print(f"Processing {regime_name} regime...")

    # Columns are already renamed in main(), just copy the dataframe
    df_processed = df.copy()

    # Add dataset combination column
    df_processed = create_dataset_combination_column(df_processed)

    # For single-source, average across all source datasets
    # For other regimes, create standard pivot table
    pivot_table = create_granular_pivot_table(df_processed, metric, regime_name)

    # Save the pivot table
    filename = f"{regime_name}_{metric.replace(' ', '_').lower()}.csv"
    filepath = output_dir / filename
    pivot_table.to_csv(filepath)
    print(f"  Saved: {filename}")


def main():
    """
    Main function to process all evaluation regimes and create granular results.
    """
    # Create output directory
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    # Load data from all regimes
    print("Loading data...")
    out_df = load_json_files(BASE_OUT_DOMAIN)
    in_df = load_json_files(BASE_IN_DOMAIN)
    basic_df = load_json_files(BASE_BASIC_FORECASTING)

    print(
        f"Loaded {len(out_df)} out-domain, {len(in_df)} in-domain, {len(basic_df)} basic forecasting results"
    )

    # Rename columns first
    out_df = rename_columns(out_df, "out_domain")
    in_df = rename_columns(in_df, "in_domain")
    basic_df = rename_columns(basic_df, "basic_forecasting")

    # Process out-domain data to separate single-source and multi-source
    single_source_df, multi_source_df = process_out_domain_data(out_df)

    # Add dummy source columns for in-domain and basic forecasting
    in_df["Dataset Source"] = "None"
    in_df["Dataset Group Source"] = "None"
    basic_df["Dataset Source"] = "None"
    basic_df["Dataset Group Source"] = "None"

    # Process each regime
    regimes = [
        (basic_df, "full_shot", "basic_forecasting"),
        (in_df, "in_domain", "in_domain"),
        (single_source_df, "single_source", "out_domain"),
        (multi_source_df, "multi_source", "out_domain"),
    ]

    metric = "MASE Mean"

    for df, regime_name, domain_suffix in regimes:
        if not df.empty:
            process_regime_data(df, regime_name, domain_suffix, SUMMARY_DIR, metric)
        else:
            print(f"Warning: No data found for {regime_name} regime")

    print(f"\nGranular results saved to: {SUMMARY_DIR}")
    print("\nGenerated files:")
    for file in sorted(SUMMARY_DIR.glob("*.csv")):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
