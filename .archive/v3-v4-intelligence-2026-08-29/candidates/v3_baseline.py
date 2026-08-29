"""candidates/v3_baseline.py
V3 baseline candidate (design doc Section 2, Candidate A) -- a THIN ADAPTER
over V3's already walk-forward-validated OOF predictions, with ZERO
modification to the underlying V3 code (research.phase5_ev_dataset,
decision.ev_formula, decision.ev_gate are all reused unmodified). This
candidate has NO privileged status over any other candidate in the roster --
it is scored by the same evidence-profile harness as every other entrant.

V3's methodology is barrier-based (a fixed SL/TP/timeout evaluated once per
event), not a per-bar discretionary decision -- so manage() always returns
HOLD here: this baseline relies entirely on the SL/TP set at entry, plus
Phase 1's own safety-net checks, never a discretionary exit. This is a
faithful, documented limitation of representing a barrier-style V3 strategy
inside Phase 1's continuous decide()/manage() loop, not a hidden
inconsistency."""
from candidates.base import CandidateMetadata
from research.phase5_ev_dataset import assemble_replay_dataset
from decision.ev_formula import compute_barrier_split, raw_ev
from decision.ev_gate import MIN_EDGE_THRESHOLD

P_SL_GIVEN_NOT_WIN = 0.5


class _FakeBarrierOutput:
    """Lightweight mock of BarrierOutput to pass precomputed p_barrier_win
    through compute_barrier_split."""
    def __init__(self, p_tp):
        self.model_status = "VALIDATED"
        self.p_tp = p_tp


class V3BaselineCandidate:
    def __init__(self, max_holding: int, rows: int = None):
        self.metadata = CandidateMetadata(
            candidate_id="v3_baseline", version="v1",
            description="Thin adapter over V3's OOF Direction/Barrier/MAE/MFE predictions and EV formula.",
            mechanism_family="v3-ensemble",
        )
        data = assemble_replay_dataset(max_holding, rows=rows)
        self._by_timestamp = {}
        for i in range(data["n"]):
            self._by_timestamp[data["timestamp"][i]] = i
        self._data = data

    def _lookup(self, market_state):
        idx = self._by_timestamp.get(market_state.market_timestamp)
        return idx

    def decide(self, market_state, account):
        idx = self._lookup(market_state)
        if idx is None:
            return ("NO_TRADE", None, None)
        data = self._data
        barrier = _FakeBarrierOutput(p_tp=float(data["p_barrier_win"][idx]))
        split = compute_barrier_split(barrier, P_SL_GIVEN_NOT_WIN)
        tp_r = float(data["mfe_r"][idx])
        sl_r = float(data["mae_r"][idx])
        ev = raw_ev(split["p_tp"], split["p_sl"], split["p_timeout"], tp_r, sl_r, 0.0, 0.0)
        if ev is None or ev <= MIN_EDGE_THRESHOLD:
            return ("NO_TRADE", None, None)
        side = data["side"][idx]
        mid = market_state.mid
        if side == 1.0:
            return ("LONG", mid - sl_r * mid, mid + tp_r * mid)
        return ("SHORT", mid + sl_r * mid, mid - tp_r * mid)

    def manage(self, market_state, position_view, account):
        return "HOLD"
