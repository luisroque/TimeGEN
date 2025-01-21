import pandas as pd
from metaforecast.synth import (
    DBA,
    Jittering,
    Scaling,
    MagnitudeWarping,
    TimeWarping,
    SeasonalMBB,
    TSMixup,
)


def workflow_metaforecast_methods(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Applies multiple synthetic/augmentation techniques to a time-series dataset.
    """
    FREQS = {"H": 24, "D": 1, "M": 12, "Q": 4, "W": 1, "Y": 1}
    if freq not in FREQS:
        raise ValueError(
            f"Unsupported frequency '{freq}'. Valid options: {list(FREQS.keys())}"
        )
    per = FREQS[freq]

    print("Starting to generate synthetic series using multiple methods...")
    print(
        "Methods used for this generation: DBA, Jitter, Scaling, MagWarp, TimeWarp, MBB, TSMixup"
    )

    dba_synth = DBA(max_n_uids=10)
    df_dba = dba_synth.transform(df)
    df_dba["method"] = "DBA"
    df_dba["unique_id"] = df["unique_id"]

    jitter_synth = Jittering()
    df_jitter = jitter_synth.transform(df)
    df_jitter["method"] = "Jitter"
    df_jitter["unique_id"] = df["unique_id"]

    scaling_synth = Scaling()
    df_scaling = scaling_synth.transform(df)
    df_scaling["method"] = "Scaling"
    df_scaling["unique_id"] = df["unique_id"]

    magwarp_synth = MagnitudeWarping()
    df_magwarp = magwarp_synth.transform(df)
    df_magwarp["method"] = "MagWarp"
    df_magwarp["unique_id"] = df["unique_id"]

    timewarp_synth = TimeWarping()
    df_timewarp = timewarp_synth.transform(df)
    df_timewarp["method"] = "TimeWarp"
    df_timewarp["unique_id"] = df["unique_id"]

    mbb_synth = SeasonalMBB(seas_period=per)
    df_mbb = mbb_synth.transform(df)
    df_mbb["method"] = "MBB"
    df_mbb["unique_id"] = df["unique_id"]

    # TODO: TS_MIXUP NEEDS TO BE UPDATED TO CREATE SERIES WITH SIMILAR LENGTH AS THE ORIGINAL ONES
    # ts_mixup = TSMixup(max_n_uids=7, min_len=50, max_len=96)
    # df_mixup = ts_mixup.transform(df)
    # df_mixup["method"] = "TSMixup"
    # df_mixup["unique_id"] = df["unique_id"]

    df_synthetic = pd.concat(
        [
            df,
            df_dba,
            df_jitter,
            df_scaling,
            df_magwarp,
            df_timewarp,
            df_mbb,
            # df_mixup,
        ],
        ignore_index=True,
    )

    df_synthetic = df_synthetic.sort_values(by=["unique_id", "ds"]).reset_index(
        drop=True
    )

    print("Synthetic series generation completed.")

    return df_synthetic
