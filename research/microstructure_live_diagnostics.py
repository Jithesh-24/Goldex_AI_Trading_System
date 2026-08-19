"""Diagnostic script for microstructure_live features.

Run TickActivityTracker over synthetic replay ticks, build a DataFrame of 5 outputs,
and call correlation_redundancy to check for redundancy in the live-only family.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import pandas as pd

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
        if out is not None:
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
    else:
        print("No complete rows for correlation analysis")


if __name__ == "__main__":
    main()
