"""DEPRECATED -- moved to features/*.py in Phase 3 (spec section 4).
Kept as a thin re-export so any external reference doesn't hard-break;
new code should import features.replay_engine directly."""
from features.replay_engine import build_candidate_features  # noqa: F401
