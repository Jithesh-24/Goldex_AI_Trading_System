"""python3 tests/test_replay_engine.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
import tempfile
import numpy as np
import pandas as pd

from features.features import build_features
from features.replay_engine import build_candidate_features

# Fixed commit SHA -- last commit in this plan before Task 16, before
# research/features_v3.py was ever touched by this plan (only read, never
# modified). Deliberately NOT "HEAD": once Task 16 Step 5 replaces
# research/features_v3.py with a deprecated shim
# (`from features.replay_engine import build_candidate_features`), HEAD
# becomes a moving pointer that, for any run of this test after that
# commit lands, would `git show` the SHIM's content, exec() it, and get
# back the exact same function object this test is comparing itself
# against -- i.e. the test would become tautological (always trivially
# passes) instead of a real regression guard. Pinning to this SHA keeps
# the comparison meaningful forever, not just during this task's own
# execution.
ORIGINAL_FEATURES_V3_SHA = "3fad4460a771fff175c53673570a900e5be36f13"


def _synthetic_df(n=600):
    rng = np.random.default_rng(0)
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    time = pd.date_range("2026-01-01", periods=n, freq="1min")
    return pd.DataFrame({"time": time, "open": close, "high": close + 0.5,
                          "low": close - 0.5, "close": close,
                          "tick_volume": rng.integers(1, 50, n),
                          "spread": np.full(n, 20.0)})


def _load_original_module():
    """Loads the pinned pre-move git-history copy of
    research/features_v3.py via `git show`, so this equivalence check
    stays real even after Task 16 Step 5 turns the working-tree file into
    a shim. See ORIGINAL_FEATURES_V3_SHA above for why the commit is
    pinned rather than using "HEAD".

    Written to a real file on disk (not exec()'d from an in-memory
    string) because the original module's numba @njit(cache=True)
    kernels need a real file locator to enable on-disk caching --
    exec()'ing from a synthetic filename raises
    "cannot cache function ...: no locator available"."""
    import subprocess
    src = subprocess.run(
        ["git", "show", f"{ORIGINAL_FEATURES_V3_SHA}:research/features_v3.py"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True, text=True, check=True).stdout
    tmp_dir = tempfile.mkdtemp(prefix="original_features_v3_")
    tmp_path = os.path.join(tmp_dir, "original_features_v3.py")
    with open(tmp_path, "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("original_features_v3", tmp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replay_engine_matches_original():
    df = _synthetic_df()
    base = build_features(df)
    original = _load_original_module()
    expected = original.build_candidate_features(df, base)
    actual = build_candidate_features(df, base)
    assert set(actual.columns) - {"time"} == set(expected.columns) - {"time"}
    for col in expected.columns:
        if col == "time":
            continue
        e = expected[col].to_numpy(dtype=np.float64)
        a = actual[col].to_numpy(dtype=np.float64)
        both_nan = np.isnan(e) & np.isnan(a)
        assert np.allclose(e[~both_nan], a[~both_nan], rtol=1e-9, atol=1e-12, equal_nan=True), col


if __name__ == "__main__":
    test_replay_engine_matches_original()
    print("tests/test_replay_engine.py: OK")
