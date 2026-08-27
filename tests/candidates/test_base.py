"""tests/candidates/test_base.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from candidates.base import CandidateMetadata, Candidate


class _DummyCandidate:
    def __init__(self):
        self.metadata = CandidateMetadata(
            candidate_id="dummy", version="v1", description="test dummy", mechanism_family="control"
        )

    def decide(self, market_state, account):
        return ("NO_TRADE", None, None)

    def manage(self, market_state, position_view, account):
        return "HOLD"


def test_candidate_metadata_fields():
    meta = CandidateMetadata(candidate_id="x", version="v1", description="d", mechanism_family="rule-based")
    assert meta.candidate_id == "x"
    assert meta.version == "v1"
    assert meta.mechanism_family == "rule-based"


def test_dummy_candidate_satisfies_protocol():
    candidate = _DummyCandidate()
    assert isinstance(candidate, Candidate)
    assert candidate.decide(None, None) == ("NO_TRADE", None, None)
    assert candidate.manage(None, None, None) == "HOLD"


"""Add to tests/candidates/test_base.py -- do not remove existing tests."""
from candidates.base import LearningCandidate


class _LearningDummy:
    def __init__(self):
        self.metadata = CandidateMetadata(
            candidate_id="learning_dummy", version="v1", description="test", mechanism_family="control"
        )
        self.learned_from = None

    def decide(self, market_state, account):
        return ("NO_TRADE", None, None)

    def manage(self, market_state, position_view, account):
        return "HOLD"

    def learn(self, training_experience):
        self.learned_from = training_experience


def test_learning_candidate_satisfies_both_protocols():
    candidate = _LearningDummy()
    assert isinstance(candidate, Candidate)
    assert isinstance(candidate, LearningCandidate)
    candidate.learn([{"a": 1}])
    assert candidate.learned_from == [{"a": 1}]


def test_plain_candidate_from_task1_still_satisfies_candidate_but_not_learning_candidate():
    candidate = _DummyCandidate()
    assert isinstance(candidate, Candidate)
    assert not isinstance(candidate, LearningCandidate)


if __name__ == "__main__":
    test_candidate_metadata_fields()
    test_dummy_candidate_satisfies_protocol()
    test_learning_candidate_satisfies_both_protocols()
    test_plain_candidate_from_task1_still_satisfies_candidate_but_not_learning_candidate()
    print("tests/candidates/test_base.py: OK")
