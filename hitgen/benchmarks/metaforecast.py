import pandas as pd
from metaforecast.synth import (
    DBA,
    Jittering,
    Scaling,
    MagnitudeWarping,
    TimeWarping,
    SeasonalMBB,
    TSMixup,
    GaussianDiffusion,
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
        "Methods used for this generation: DBA, Jitter, Scaling, MagWarp, TimeWarp, MBB, TSMixup, GaussDiff"
    )

    dba_synth = DBA(max_n_uids=10)
    df_dba = dba_synth.transform(df)
    df_dba["method"] = "DBA"

    jitter_synth = Jittering()
    df_jitter = jitter_synth.transform(df)
    df_jitter["method"] = "Jitter"

    scaling_synth = Scaling()
    df_scaling = scaling_synth.transform(df)
    df_scaling["method"] = "Scaling"

    magwarp_synth = MagnitudeWarping()
    df_magwarp = magwarp_synth.transform(df)
    df_magwarp["method"] = "MagWarp"

    timewarp_synth = TimeWarping()
    df_timewarp = timewarp_synth.transform(df)
    df_timewarp["method"] = "TimeWarp"

    mbb_synth = SeasonalMBB(seas_period=per)
    df_mbb = mbb_synth.transform(df)
    df_mbb["method"] = "MBB"

    ts_mixup = TSMixup(max_n_uids=7, min_len=50, max_len=96)
    df_mixup = ts_mixup.transform(df)
    df_mixup["method"] = "TSMixup"

    gauss_diff = GaussianDiffusion()
    df_gaussdiff = gauss_diff.transform(df)
    df_gaussdiff["method"] = "GaussDiff"

    df_synthetic = pd.concat(
        [
            df,
            df_dba,
            df_jitter,
            df_scaling,
            df_magwarp,
            df_timewarp,
            df_mbb,
            df_mixup,
            df_gaussdiff,
        ],
        ignore_index=True,
    )

    df_synthetic = df_synthetic.sort_values(by=["unique_id", "ds"]).reset_index(
        drop=True
    )

    print("Synthetic series generation completed.")

    return df_synthetic
