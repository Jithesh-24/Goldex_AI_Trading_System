"""candidates/base.py
The Candidate protocol imposes NO assumption about internal mechanism -- the
competition harness (research/phase2_tournament.py) only ever calls decide()
and manage() and observes their outputs plus the resulting simulator
experience. A candidate may be rule-based, statistical, a learned model, an
ensemble, or any future architecture; nothing here privileges any of them.
Signatures match simulator.replay.run_replay's decide_fn/manage_fn exactly."""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class CandidateMetadata:
    candidate_id: str
    version: str
    description: str
    mechanism_family: str  # reporting label only, e.g. "rule-based", "learned-linear",
                            # "regime-statistical", "control", "v3-ensemble" -- never used in scoring


@runtime_checkable
class Candidate(Protocol):
    metadata: CandidateMetadata

    def decide(self, market_state, account) -> tuple:
        ...

    def manage(self, market_state, position_view, account) -> str:
        ...
