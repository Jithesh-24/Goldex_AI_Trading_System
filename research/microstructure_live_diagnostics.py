"""Diagnostic script for microstructure_live features.

Run TickActivityTracker over synthetic replay ticks, build a DataFrame of 5 outputs,
and call correlation_redundancy to check for redundancy in the live-only family.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

from contracts.tick import Tick
from market.state_engine import StateEngine
from market.synthetic_replay import generate_ticks
from features.microstructure_live import TickActivityTracker
from features.registry.diagnostics import correlation_redundancy


def main():
    # Generate synthetic ticks
    ticks_data = generate_ticks(n=3000, seed=42)

    # Initialize engine and tracker
    engine = StateEngine("GOLD.i#")
    tracker = TickActivityTracker()

    # Collect outputs
    outputs = []

    for tick_dict in ticks_data:
        # Convert dict to Tick object
        market_timestamp = datetime.fromisoformat(tick_dict["market_timestamp"])
        ingestion_timestamp = datetime.fromisoformat(tick_dict["ingestion_timestamp"])
        bid = tick_dict["bid"]
        ask = tick_dict["ask"]
        mid = (bid + ask) / 2
        spread = ask - bid

        tick = Tick(
            symbol=tick_dict["symbol"],
            market_timestamp=market_timestamp,
            ingestion_timestamp=ingestion_timestamp,
            bid=bid,
            ask=ask,
            mid=mid,
            spread=spread,
            source=tick_dict["source"],
            internal_seq=tick_dict["internal_seq"],
            tick_volume=tick_dict.get("tick_volume"),
        )

        # Pass through state engine to get MarketState
        state = engine.on_tick(tick)

        if state is None:
            continue

        # Update tracker and collect output
        out = tracker.update(state)
        outputs.append(out)

    # Build DataFrame from outputs
    df = pd.DataFrame(outputs)

    # Filter to rows with non-None values for all columns
    df_clean = df.dropna(how='any')

    print(f"Generated {len(outputs)} tracker outputs from {len(ticks_data)} ticks")
    print(f"DataFrame shape: {df.shape}")
    print(f"Clean DataFrame (all non-None) shape: {df_clean.shape}")
    print()
    print("DataFrame head:")
    print(df_clean.head(10))
    print()

    # Run correlation_redundancy
    if len(df_clean) > 0:
        pairs = correlation_redundancy(df_clean, threshold=0.95)
        print(f"Correlation redundancy pairs (threshold=0.95):")
        if pairs:
            for a, b, corr in pairs:
                print(f"  {a} <-> {b}: {corr:.4f}")
        else:
            print("  (none)")
        print()

        # Threshold-only reporting hides substantial-but-subthreshold correlations.
        # Print the full matrix so those aren't silently lost.
        print("Full correlation matrix (all pairs, not just those above threshold):")
        print(df_clean.corr())
        print()

        # Coefficient of variation per column. A CV near 0 for a column means it's
        # quasi-constant in THIS run's data -- which weakens any "not redundant"
        # conclusion drawn from its correlations (a feature that barely moves can't
        # correlate strongly with anything, independent of whether it's genuinely
        # independent information in real market data).
        print("Coefficient of variation per column (std / |mean|):")
        for col in df_clean.columns:
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            cv = std / abs(mean) if abs(mean) > 1e-12 else float("nan")
            print(f"  {col}: {cv:.4f}")
        print()
        print(
            "CAVEAT: tick_interarrival_mean_60s, tick_interarrival_std_60s, and "
            "tick_arrival_burstiness_60s all show CV < 0.02 in this run. This is a "
            "limitation of the synthetic data, not evidence of redundancy: "
            "market/synthetic_replay.py's generate_ticks() draws i.i.d. "
            "rng.randint(20, 45) ms interarrival gaps with no clustering/regime "
            "structure, so by law-of-large-numbers each 60s window's rolling "
            "mean/std barely moves tick-to-tick. Real tick arrivals cluster during "
            "volatility (the premise of a 'burstiness' feature); this generator "
            "cannot produce that. Low observed variance for these 3 features here "
            "does not strongly demonstrate independence from each other -- it may "
            "just reflect the generator's lack of clustering structure."
        )
    else:
        print("No complete rows for correlation analysis")


if __name__ == "__main__":
    main()
