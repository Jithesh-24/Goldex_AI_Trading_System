"""decision/ev_gate.py
Spec section 11: NO_TRADE gate is a fixed, documented min-edge threshold
plus status-gating -- never a bare p_win>0.6 heuristic. Long and short are
evaluated independently (no symmetry assumed)."""
from typing import Optional

from decision.ev_formula import compute_barrier_split, raw_ev, risk_adjusted_ev, DEFAULT_K
from contracts.specialist_output import BarrierOutput

# Set from the known transaction-cost floor plus a minimum-edge buffer
# (documented in docs/ARCHITECTURE.md's Phase 5 section, Task 15) -- not
# curve-fit to any evaluation window.
MIN_EDGE_THRESHOLD = 0.02


def compute_side_ev(barrier: BarrierOutput, direction_gate_ok: bool, p_sl_given_not_win: Optional[float],
                     tp_r: Optional[float], sl_r: Optional[float], timeout_r: Optional[float],
                     cost_r: Optional[float], uncertainty: float, k: float = None) -> Optional[float]:
    if not direction_gate_ok:
        return None
    if k is None:
        k = DEFAULT_K
    split = compute_barrier_split(barrier, p_sl_given_not_win)
    ev_raw = raw_ev(split["p_tp"], split["p_sl"], split["p_timeout"], tp_r, sl_r, timeout_r, cost_r)
    if ev_raw is None:
        return None
    return risk_adjusted_ev(ev_raw, uncertainty, k)


def decide(long_ev_adj: Optional[float], short_ev_adj: Optional[float]) -> tuple[str, Optional[str], str]:
    if long_ev_adj is None and short_ev_adj is None:
        return "NO_TRADE", None, "both sides unavailable -- required specialist(s) missing or gated off"
    long_ok = long_ev_adj is not None and long_ev_adj > MIN_EDGE_THRESHOLD
    short_ok = short_ev_adj is not None and short_ev_adj > MIN_EDGE_THRESHOLD
    if not long_ok and not short_ok:
        return "NO_TRADE", None, f"neither side cleared min_edge_threshold={MIN_EDGE_THRESHOLD}"
    if long_ok and (not short_ok or long_ev_adj >= short_ev_adj):
        return "LONG_CANDIDATE", "long", f"ev_adj={long_ev_adj:.4f} above min_edge_threshold"
    return "SHORT_CANDIDATE", "short", f"ev_adj={short_ev_adj:.4f} above min_edge_threshold"
