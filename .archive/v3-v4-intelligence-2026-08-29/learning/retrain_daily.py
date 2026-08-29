"""End-of-day retrain: refresh the live seed, re-run the full purged-CV
training pipeline (core/train.py) on the latest 6.7yr+today data, and only
replace the deployed models if the new run's OOF accuracy doesn't regress
past a small tolerance -- an automated nightly retrain that silently degrades
the live model on a bad data day is worse than skipping a night.

Old models are archived (timestamped) before any swap, so a bad promotion is
always reversible by hand.

Run: python3 -m core.retrain_daily   (intended for a nightly cron/systemd timer)
"""
import json
import os
import shutil
import time

from learning import seed_refresh, train
from config.loader import load_config

BASE = train.BASE
MODELS_DIR = os.path.join(BASE, "models", "active")
ARCHIVE_DIR = os.path.join(BASE, "models", "archive", "retrain-snapshots")
ACC_REGRESSION_TOLERANCE = load_config().learning.acc_regression_tolerance
# NOTE (Phase 1 V3): this promotes files into models/active/ but does NOT
# create/update a models/registry/*.json entry -- the registry will go
# stale relative to what's actually in active/ after an automated
# promotion until a later phase wires registry writes into this flow.
# Deliberately out of scope here (full champion/challenger is a later
# phase); flagged so it isn't mistaken for an oversight.


def _read_summary(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def main():
    t_start = time.time()
    print(f"=== retrain_daily start {time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    seed_refresh.main()

    prev_summary = _read_summary(os.path.join(MODELS_DIR, "train_summary.json"))
    prev_acc = prev_summary["mean_oof_acc"] if prev_summary else None

    staging_dir = os.path.join(MODELS_DIR, ".staging")
    os.makedirs(staging_dir, exist_ok=True)
    import sys
    sys.argv = ["core.train", "--out-dir", staging_dir]
    train.main()

    new_summary = _read_summary(os.path.join(staging_dir, "train_summary.json"))
    new_acc = new_summary["mean_oof_acc"] if new_summary else None

    if prev_acc is not None and new_acc is not None and new_acc < prev_acc - ACC_REGRESSION_TOLERANCE:
        print(f"REFUSED: new OOF acc {new_acc:.4f} regressed vs current {prev_acc:.4f} "
              f"by more than {ACC_REGRESSION_TOLERANCE} -- keeping existing models")
        shutil.rmtree(staging_dir)
        return

    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for fname in ("primary.cbm", "meta.cbm", "feature_cols.json", "train_summary.json"):
        src = os.path.join(MODELS_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(ARCHIVE_DIR, f"{stamp}_{fname}"))
        shutil.move(os.path.join(staging_dir, fname), src)
    shutil.rmtree(staging_dir)

    print(f"PROMOTED: new OOF acc {new_acc} (was {prev_acc}) -- archived previous models as {stamp}_*")
    print(f"=== retrain_daily done in {time.time() - t_start:.1f}s ===")


if __name__ == "__main__":
    main()
