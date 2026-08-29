"""decision/ev_formula.py
Spec sections 7a/9: EV_side = p_tp*TP_R - p_sl*SL_R - p_timeout*timeout_R
- cost_R, risk-adjusted by a lower-confidence bound EV_adj = EV_raw -
k*uncertainty. Barrier-primary: p_tp comes straight from the Barrier
role's calibrated win probability; p_sl/p_timeout are derived from the
Task 2 split classifier's P(sl|not-win)."""
from typing import Optional

from contracts.specialist_output import BarrierOutput

EV_FORMULA_VERSION = "v1"
DEFAULT_K = 0.8  # Derived and OOS-validated via research.phase5_uncertainty_k (finer grid): per-horizon chosen_k={h15:0.6, h45:0.7, h90:0.8}; most conservative=0.8 (h90, accuracy=0.6125)
_OK_STATUSES = {"VALIDATED", "CANDIDATE"}


def compute_barrier_split(barrier: BarrierOutput, p_sl_given_not_win: Optional[float]) -> dict:
    if barrier.model_status not in _OK_STATUSES or barrier.p_tp is None or p_sl_given_not_win is None:
        return {"p_tp": None, "p_sl": None, "p_timeout": None}
    p_tp = barrier.p_tp
    p_not_win = 1.0 - p_tp
    p_sl = p_not_win * p_sl_given_not_win
    p_timeout = p_not_win * (1.0 - p_sl_given_not_win)
    return {"p_tp": p_tp, "p_sl": p_sl, "p_timeout": p_timeout}


def raw_ev(p_tp, p_sl, p_timeout, tp_r, sl_r, timeout_r, cost_r) -> Optional[float]:
    inputs = [p_tp, p_sl, p_timeout, tp_r, sl_r, timeout_r, cost_r]
    if any(v is None for v in inputs):
        return None
    return p_tp * tp_r - p_sl * sl_r - p_timeout * timeout_r - cost_r


def risk_adjusted_ev(ev_raw: float, uncertainty: float, k: float) -> float:
    return ev_raw - k * uncertainty
