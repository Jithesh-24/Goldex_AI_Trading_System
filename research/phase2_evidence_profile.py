"""research/phase2_evidence_profile.py
Computes a multi-dimensional evidence profile from ONE stored candidate
trajectory (design doc Section 5) -- deliberately NOT a single composite
score. cost_sensitivity/execution_sensitivity/train-validation degradation
are NOT computed here (they need multiple runs/configs) -- that is
research/phase2_tournament.py's job."""
from datetime import datetime

import numpy as np

from research.audit_edge import block_bootstrap


def _parse_ts(value):
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def compute_evidence_profile(records: list, n_subperiods: int = 4) -> dict:
    closed = [r for r in records if r["event_type"] == "POSITION_CLOSED"]
    decides = [r for r in records if r["event_type"] == "DECIDE"]
    closed_sorted = sorted(closed, key=lambda r: _parse_ts(r["timestamp"]))
    pnls = [float(r["realized_pnl"]) for r in closed_sorted]

    n_trades = len(pnls)
    total_pnl = sum(pnls)

    cumulative = np.cumsum(pnls) if pnls else np.array([0.0])
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) else 0.0
    peak_at_max_dd = float(running_max[np.argmax(drawdowns)]) if len(drawdowns) else 0.0
    max_drawdown_pct = (max_drawdown / peak_at_max_dd) if peak_at_max_dd > 0 else 0.0

    sorted_pnls = sorted(pnls)
    decile_count = max(1, len(sorted_pnls) // 10) if sorted_pnls else 0
    worst_decile = sorted_pnls[:decile_count] if decile_count else []
    worst_decile_mean = float(np.mean(worst_decile)) if worst_decile else 0.0

    trades_per_1000_bars = (n_trades / len(decides) * 1000.0) if decides else 0.0

    subperiods = []
    if closed_sorted:
        block_size = max(1, len(closed_sorted) // n_subperiods)
        for b in range(n_subperiods):
            start_idx = b * block_size
            end_idx = (b + 1) * block_size if b < n_subperiods - 1 else len(closed_sorted)
            block = closed_sorted[start_idx:end_idx]
            if not block:
                continue
            subperiods.append({
                "start": str(block[0]["timestamp"]), "end": str(block[-1]["timestamp"]),
                "n_trades": len(block), "total_pnl": sum(float(r["realized_pnl"]) for r in block),
            })

    if pnls:
        lower, middle, upper = block_bootstrap(np.array(pnls), block_size=5, n_boot=1000)
        point = float(np.mean(pnls))
        ci_low, ci_high = float(lower), float(upper)
    else:
        point, ci_low, ci_high = 0.0, 0.0, 0.0

    return {
        "n_trades": n_trades,
        "realized_pnl": {"total": total_pnl, "per_trade_r_like": pnls},
        "drawdown": {"max_drawdown": max_drawdown, "max_drawdown_pct_of_peak": max_drawdown_pct},
        "tail_risk": {"worst_decile_mean": worst_decile_mean},
        "trade_frequency": {"trades_per_1000_bars": trades_per_1000_bars},
        "consistency_across_subperiods": subperiods,
        "confidence_intervals": {"mean_pnl_per_trade": {"point": point, "lower": ci_low, "upper": ci_high}},
    }
