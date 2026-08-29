"""Phase 4 Task 13: prints every registry entry grouped by family/role,
for Task 16's completion-report section O. Read-only -- makes no
decisions, promotes nothing.

Run: /home/jith/.hermes/hermes-agent/venv/bin/python3 -m research.phase4_model_inventory
"""
import glob
import json
import os
from collections import defaultdict

from contracts.model_registry import ModelRegistryEntry

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_DIR = os.path.join(BASE, "models", "registry")

HEADLINE_METRIC = {
    "direction": "mean_oof_acc", "opportunity_meta": "meta_win_rate", "regime": "mean_run_length",
    "mae_quantile": "global_coverage", "mfe_quantile": "global_coverage",
    "barrier_probability": "log_loss", "execution_decay": "data_limited",
}


def main():
    by_family = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(REGISTRY_DIR, "*.json"))):
        with open(path) as f:
            entry = ModelRegistryEntry(**json.load(f))
        by_family[entry.family].append(entry)

    for family, entries in sorted(by_family.items()):
        print(f"\n== {family} ==")
        for e in sorted(entries, key=lambda x: x.model_id):
            metric_key = HEADLINE_METRIC.get(family)
            metric_val = e.metrics.get(metric_key) if metric_key else None
            champion = " [CHAMPION]" if e.is_champion else ""
            print(f"  {e.model_id:45s} status={e.status:10s} {metric_key}={metric_val}{champion}")


if __name__ == "__main__":
    main()
