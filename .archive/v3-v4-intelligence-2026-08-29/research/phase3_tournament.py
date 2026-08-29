"""research/phase3_tournament.py
Wraps (does not modify) research.phase2_tournament's control-gate and
verdict machinery, adding the optional learn() step between a candidate's
SIMULATED_TRAINING and SIMULATED_VALIDATION runs -- with a mechanical
causality check (design doc Section 6) that learn() never receives anything
but SIMULATED_TRAINING-tagged experience."""
from simulator.contracts import EnvironmentTag
from research.phase2_tournament import _run_one, _verdict_for


def _maybe_learn(candidate, store, run_id):
    if not hasattr(candidate, "learn"):
        return
    records = store.read_run(
        candidate.metadata.candidate_id, candidate.metadata.version, run_id, EnvironmentTag.SIMULATED_TRAINING
    )
    for record in records:
        if record.get("environment_tag") != EnvironmentTag.SIMULATED_TRAINING.value:
            raise ValueError(
                f"learn() causality violation: candidate {candidate.metadata.candidate_id} would have "
                f"received a record tagged {record.get('environment_tag')!r}, not "
                f"{EnvironmentTag.SIMULATED_TRAINING.value!r}."
            )
    candidate.learn(records)


def run_phase3_tournament(df_training, df_validation, roster: list, config, store, run_id: str) -> dict:
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
                    "reason": "RandomCandidate showed persistent profitability -- ranking halted.",
                },
                "candidates": {},
            }
        control_gate = {"passed": True, "random_candidate_profile": random_val_profile, "reason": "OK"}
    else:
        control_gate = {"passed": True, "random_candidate_profile": None, "reason": "No random control in roster."}

    results = {}
    for candidate in roster:
        training_profile = _run_one(df_training, candidate, config, EnvironmentTag.SIMULATED_TRAINING, store, run_id)
        _maybe_learn(candidate, store, run_id)
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
