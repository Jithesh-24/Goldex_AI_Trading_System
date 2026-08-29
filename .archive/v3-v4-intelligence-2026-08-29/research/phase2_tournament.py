"""research/phase2_tournament.py
The competition harness (design doc Sections 1-7). Runs every candidate
through simulator.replay.run_replay on SIMULATED_TRAINING then
SIMULATED_VALIDATION, persists full trajectories via ExperienceStore, and
computes evidence profiles -- NEVER a composite score. The random control's
profile is checked FIRST as a validity gate on the harness itself; if it
looks persistently profitable, the harness is presumed buggy and ranking
halts before any other candidate's result is trusted."""
from simulator.contracts import EnvironmentTag
from simulator.replay import run_replay
from research.phase2_evidence_profile import compute_evidence_profile

MIN_TRADES_FOR_CONFIDENCE = 30


def _run_one(df, candidate, config, environment_tag, store, run_id):
    recorder = run_replay(df, candidate.decide, candidate.manage, config, environment_tag)
    records = recorder.all_records()
    store.write_run(candidate.metadata.candidate_id, candidate.metadata.version, run_id, environment_tag, records)
    dict_records = [
        {"event_type": r.event_type, "timestamp": r.timestamp, "realized_pnl": r.realized_pnl,
         "cost_amount": r.cost_amount, "outcome": r.outcome}
        for r in records
    ]
    return compute_evidence_profile(dict_records)


def _verdict_for(profile: dict, mechanism_family: str) -> str:
    if mechanism_family == "control":
        return "CONTROL"
    if profile["n_trades"] == 0 or profile["realized_pnl"]["total"] <= 0:
        return "REJECT"
    if profile["n_trades"] < MIN_TRADES_FOR_CONFIDENCE:
        return "NEEDS_MORE_EVIDENCE"
    if profile["confidence_intervals"]["mean_pnl_per_trade"]["lower"] > 0:
        return "KEEP"
    return "NEEDS_MORE_EVIDENCE"


def run_tournament(df_training, df_validation, roster: list, config, store, run_id: str) -> dict:
    random_candidate = None
    for candidate in roster:
        if candidate.metadata.candidate_id == "control_random":
            random_candidate = candidate
            break

    if random_candidate is not None:
        random_val_profile = _run_one(
            df_validation, random_candidate, config, EnvironmentTag.SIMULATED_VALIDATION, store, run_id
        )
        ci_lower = random_val_profile["confidence_intervals"]["mean_pnl_per_trade"]["lower"]
        persistently_profitable = random_val_profile["realized_pnl"]["total"] > 0 and ci_lower > 0
        if persistently_profitable:
            return {
                "control_gate": {
                    "passed": False, "random_candidate_profile": random_val_profile,
                    "reason": "RandomCandidate showed persistent profitability on validation "
                              "(total > 0 and CI lower bound > 0) -- this indicates a simulator/harness "
                              "bug, not a real trading edge. Ranking halted; investigate before trusting "
                              "any other candidate's result.",
                },
                "candidates": {},
            }
        control_gate = {"passed": True, "random_candidate_profile": random_val_profile, "reason": "OK"}
    else:
        control_gate = {"passed": True, "random_candidate_profile": None, "reason": "No random control in roster."}

    results = {}
    for candidate in roster:
        training_profile = _run_one(df_training, candidate, config, EnvironmentTag.SIMULATED_TRAINING, store, run_id)
        if candidate.metadata.candidate_id == "control_random" and random_candidate is not None:
            validation_profile = random_val_profile
        else:
            validation_profile = _run_one(
                df_validation, candidate, config, EnvironmentTag.SIMULATED_VALIDATION, store, run_id
            )
        verdict = _verdict_for(validation_profile, candidate.metadata.mechanism_family)
        results[candidate.metadata.candidate_id] = {
            "metadata": {
                "candidate_id": candidate.metadata.candidate_id, "version": candidate.metadata.version,
                "description": candidate.metadata.description, "mechanism_family": candidate.metadata.mechanism_family,
            },
            "training_profile": training_profile, "validation_profile": validation_profile, "verdict": verdict,
        }

    return {"control_gate": control_gate, "candidates": results}
