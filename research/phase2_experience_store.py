"""research/phase2_experience_store.py
Persists FULL experience trajectories (every DECIDE/MANAGE/POSITION_CLOSED
record, not a summary) keyed by (candidate_id, version, run_id,
environment_tag) -- this is what lets a future candidate learn from a prior
run's actual experience instead of only fresh replay (design doc Section 9).
Records round-trip through plain JSON dicts, not reconstructed dataclasses --
downstream evidence-profile code only needs field access."""
import dataclasses
import json
import os
from datetime import datetime

from simulator.contracts import EnvironmentTag


def _serialize_value(value):
    if hasattr(value, "value") and hasattr(value, "name"):  # enum
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if dataclasses.is_dataclass(value):
        return {k: _serialize_value(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    return value


def _record_to_dict(record) -> dict:
    return {field.name: _serialize_value(getattr(record, field.name)) for field in dataclasses.fields(record)}


class ExperienceStore:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def _run_dir(self, candidate_id: str, version: str, run_id: str) -> str:
        return os.path.join(self.base_dir, candidate_id, version, run_id)

    def write_run(self, candidate_id: str, version: str, run_id: str,
                  environment_tag: EnvironmentTag, records: list) -> str:
        run_dir = self._run_dir(candidate_id, version, run_id)
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, f"{environment_tag.value}.jsonl")
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(_record_to_dict(record)) + "\n")
        return path

    def read_run(self, candidate_id: str, version: str, run_id: str, environment_tag: EnvironmentTag) -> list:
        path = os.path.join(self._run_dir(candidate_id, version, run_id), f"{environment_tag.value}.jsonl")
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def list_runs(self, candidate_id: str = None) -> list:
        results = []
        if not os.path.isdir(self.base_dir):
            return results
        candidate_ids = [candidate_id] if candidate_id else os.listdir(self.base_dir)
        for cid in candidate_ids:
            cid_path = os.path.join(self.base_dir, cid)
            if not os.path.isdir(cid_path):
                continue
            for version in os.listdir(cid_path):
                version_path = os.path.join(cid_path, version)
                for run_id in os.listdir(version_path):
                    run_path = os.path.join(version_path, run_id)
                    for fname in os.listdir(run_path):
                        if fname.endswith(".jsonl"):
                            results.append({
                                "candidate_id": cid, "version": version, "run_id": run_id,
                                "environment_tag": fname[: -len(".jsonl")],
                            })
        return results
