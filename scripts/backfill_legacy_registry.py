"""One-time backfill: register every archived legacy-v7 model artifact
with whatever metadata is honestly derivable from its filename and
sibling files -- no fabricated training_config. Run once:
python3 scripts/backfill_legacy_registry.py
Idempotent: re-running overwrites the same registry files with the same
content (derived purely from what's on disk), it does not duplicate."""
import glob
import json
import os
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_DIR = os.path.join(BASE, "models", "archive", "legacy-v7")
REGISTRY_DIR = os.path.join(BASE, "models", "registry")

FAMILY_GUESS = {
    "gold_lgb_model": "direction",
    "direction_s": "direction",
    "quant_lgb_s": "opportunity_meta",
    "real_ai_s": "opportunity_meta",
    "regime_transition_s": "regime",
    "spec_": "regime",
}


def guess_family(stem: str) -> str:
    for prefix, family in FAMILY_GUESS.items():
        if stem.startswith(prefix):
            return family
    return "regime"  # legacy regime-specialist files that don't match above


def main():
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    written = 0
    for path in sorted(glob.glob(os.path.join(LEGACY_DIR, "*.txt"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        model_id = f"legacy_v7_{stem}"
        mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
        entry = {
            "model_id": model_id,
            "family": guess_family(stem),
            "algorithm": "lightgbm",
            "artifact_path": f"archive/legacy-v7/{os.path.basename(path)}",
            "feature_schema_version": None,
            "feature_cols": [],
            "target_definition": None,
            "training_config": {},
            "training_period": None,
            "validation_period": None,
            "created_at": mtime.isoformat(),
            "status": "archived",
            "is_champion": False,
            "metrics": {},
            "lineage": {},
        }
        out_path = os.path.join(REGISTRY_DIR, f"{model_id}.json")
        with open(out_path, "w") as f:
            json.dump(entry, f, indent=2)
        written += 1
    print(f"backfilled {written} legacy-v7 registry entries into {REGISTRY_DIR}")


if __name__ == "__main__":
    main()
