"""research/phase3_real_run.py
Executes the Phase 2 + Phase 3 candidate roster against a bounded
chronological slice of the real 6.7-year Gold dataset. This is a research
execution script (like Batch 1/2's real full-history runs), not part of the
TDD-covered unit test suite -- its own correctness is validated by the unit
tests on every module it calls.

V3BaselineCandidate is deliberately excluded here (see design doc discussion)
-- it needs a ~15-minute assemble_replay_dataset() call whose OOF event
universe is walk-forward-validated on specific historical windows that may
not align with this script's arbitrary training/validation row boundaries.
It needs its own dedicated integration run, not folding into this one.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 research/phase3_real_run.py
"""
import pandas as pd

from candidates.controls import NoTradeCandidate, RandomCandidate
from candidates.statistical_null import MomentumMeanReversionCandidate
from candidates.regime_conditioned import RegimeConditionedCandidate
from candidates.simple_learned import SimpleLearnedCandidate
from candidates.tabular_qlearning import TabularQLearningCandidate
from candidates.bayesian_online import BayesianOnlineCandidate
from candidates.hmm_regime import HMMRegimeCandidate
from candidates.sequence_history import SequenceHistoryCandidate
from simulator.contracts import SimulatedExecutionConfig
from research.phase2_experience_store import ExperienceStore
from research.phase3_tournament import run_phase3_tournament
from research.phase3_representation_research import (
    analyze_return_autocorrelation, analyze_volatility_clustering, analyze_regime_persistence,
)

TRAINING_ROWS = 300_000
VALIDATION_ROWS = 100_000
DATA_PATH = "data/gold_seed_merged_full6yr.csv"


def load_slices():
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    df_training = df.iloc[:TRAINING_ROWS].reset_index(drop=True)
    df_validation = df.iloc[TRAINING_ROWS:TRAINING_ROWS + VALIDATION_ROWS].reset_index(drop=True)
    return df_training, df_validation


def build_roster():
    return [
        NoTradeCandidate(),
        RandomCandidate(seed=42),
        MomentumMeanReversionCandidate(),
        RegimeConditionedCandidate(),
        # Untrained, arbitrary placeholder weights -- no fitting script exists yet for this candidate;
        # documented explicitly, not presented as a trained model.
        SimpleLearnedCandidate(weights={"short_return": 10.0, "medium_return": 10.0, "rsi_like": 5.0}),
        TabularQLearningCandidate(),
        BayesianOnlineCandidate(),
        HMMRegimeCandidate(),
        SequenceHistoryCandidate(),
    ]


def print_representation_findings(df_training):
    closes = df_training["close"].to_numpy()
    print("\n=== Market-Flow Representation Research (Section 4) ===")
    print("Return autocorrelation (lags 1-5):", analyze_return_autocorrelation(closes, max_lag=5))
    print("Volatility clustering (lags 1-5):", analyze_volatility_clustering(closes, max_lag=5))
    hmm_probe = HMMRegimeCandidate(max_em_iterations=10)
    probe_records = [{"event_type": "DECIDE", "market_state_snapshot": {"mid": float(c)}} for c in closes]
    hmm_probe.learn(probe_records)
    if hmm_probe.is_trained:
        print("Regime persistence:", analyze_regime_persistence(hmm_probe, closes))
    else:
        print("Regime persistence: HMM did not train (insufficient data)")


def print_results(result):
    print("\n=== Control Gate ===")
    print(result["control_gate"])
    if not result["control_gate"]["passed"]:
        print("\nCONTROL GATE FAILED -- ranking halted, no candidate results to report.")
        return
    print("\n=== Candidate Results ===")
    for candidate_id, data in result["candidates"].items():
        tp, vp = data["training_profile"], data["validation_profile"]
        print(f"\n{candidate_id} ({data['metadata']['mechanism_family']}) -- verdict: {data['verdict']}")
        print(f"  training:   n_trades={tp['n_trades']}, total_pnl={tp['realized_pnl']['total']:.4f}")
        print(f"  validation: n_trades={vp['n_trades']}, total_pnl={vp['realized_pnl']['total']:.4f}")
        print(f"  validation CI mean_pnl_per_trade: {vp['confidence_intervals']['mean_pnl_per_trade']}")


def main():
    df_training, df_validation = load_slices()
    print_representation_findings(df_training)
    roster = build_roster()
    config = SimulatedExecutionConfig()
    store = ExperienceStore(base_dir="research/phase3_real_run_experience")
    result = run_phase3_tournament(df_training, df_validation, roster, config, store, run_id="phase3_real_run_001")
    print_results(result)


if __name__ == "__main__":
    main()
