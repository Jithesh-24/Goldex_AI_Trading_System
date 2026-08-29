"""Real measurement of per-column historical data quality -- feeds the
registry's historical_coverage/status metadata (Phase 3 spec section 2).
Never hand-wave these numbers; always re-run this against the real CSV."""
import pandas as pd


def measure_coverage(csv_path: str) -> dict:
    df = pd.read_csv(csv_path, usecols=["time", "tick_volume", "spread", "real_volume"],
                      parse_dates=["time"])
    n = len(df)
    real_volume_nonzero_frac = float((df["real_volume"] != 0).mean())
    tick_volume_nonzero_frac = float((df["tick_volume"] != 0).mean())

    daily = df.set_index("time")["tick_volume"].resample("1D").apply(lambda s: (s != 0).mean())
    trailing30 = daily.rolling(30, min_periods=1).mean()
    degraded = trailing30[trailing30 < 0.05]
    tick_volume_degrades_after = str(degraded.index[0].date()) if len(degraded) else None

    spread_counts = df["spread"].value_counts(normalize=True)
    spread_constant_frac = float(spread_counts.iloc[0])
    spread_unique_values = sorted(df["spread"].unique().tolist())

    return {
        "n_rows": n,
        "real_volume_nonzero_frac": real_volume_nonzero_frac,
        "tick_volume_nonzero_frac": tick_volume_nonzero_frac,
        "tick_volume_degrades_after": tick_volume_degrades_after,
        "spread_constant_frac": spread_constant_frac,
        "spread_unique_values": spread_unique_values,
    }


if __name__ == "__main__":
    import json
    result = measure_coverage("data/gold_seed_merged_full6yr.csv")
    print(json.dumps(result, indent=2, default=str))
