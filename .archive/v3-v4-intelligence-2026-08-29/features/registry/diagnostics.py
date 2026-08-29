"""Reusable redundancy/stability diagnostics -- generalizes
research/v3_feature_selection.py's correlation-pruning methodology (spec
section 9). Run fresh against NEW features (e.g. microstructure_live,
Task 21); NOT re-run against the already-OOF-evidenced 92 candidates."""
import numpy as np
import pandas as pd


def correlation_redundancy(df: pd.DataFrame, threshold: float = 0.95) -> list:
    corr = df.corr().abs()
    pairs = []
    cols = list(df.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            val = corr.loc[a, b]
            if pd.notna(val) and val > threshold:
                pairs.append((a, b, float(val)))
    return pairs


def distribution_stability(series_a: pd.Series, series_b: pd.Series) -> dict:
    a, b = series_a.dropna(), series_b.dropna()
    mean_a, mean_b = a.mean(), b.mean()
    std_a, std_b = a.std(), b.std()
    pooled_std = ((std_a ** 2 + std_b ** 2) / 2) ** 0.5
    mean_shift = abs(mean_a - mean_b) / pooled_std if pooled_std > 1e-12 else 0.0
    return {
        "mean_a": float(mean_a), "mean_b": float(mean_b),
        "std_a": float(std_a), "std_b": float(std_b),
        "mean_shift": float(mean_shift),
        "std_ratio": float(std_b / std_a) if std_a > 1e-12 else float("nan"),
    }
