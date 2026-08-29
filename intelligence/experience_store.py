"""intelligence/experience_store.py
A thin read-access layer over Phase 1's `simulator.experience.ExperienceRecorder`
(mandate Section 12 item 10). This module does not record or store anything itself
-- Phase 1's `ExperienceRecorder`/`write_tag_guard` already own writing and
partitioning by `environment_tag` (see `simulator/experience.py`). All this module
adds is a READ guard: a hard, code-level refusal to read the untouched final
out-of-sample partition by name.

Naming convention: `EnvironmentTag.SIMULATED_OOS_TEST` (see
`simulator/contracts.py`) is Phase 1's existing, already-established tag for the
untouched final OOS partition -- this module does not invent a new tag, it reads
Phase 1's real enum value.
"""
from __future__ import annotations

from simulator.contracts import EnvironmentTag
from simulator.experience import ExperienceRecord, ExperienceRecorder

# The set of environment tags this store hard-refuses to read from. Established
# here as PROTECTED_TAGS: currently just the untouched final OOS partition
# (`SIMULATED_OOS_TEST`). Any candidate/researcher code path that reads training
# results must not be able to see this partition, even by accident -- so the
# refusal happens at construction time, not at query time.
PROTECTED_TAGS = frozenset({EnvironmentTag.SIMULATED_OOS_TEST})


class ProtectedPartitionError(RuntimeError):
    """Raised when code attempts to read a protected (untouched OOS) partition."""


class ExperienceStore:
    """Read-only accessor over an `ExperienceRecorder`, scoped to one
    `environment_tag`. Refuses immediately (in the constructor) if the
    requested tag is a protected partition -- this is a hard assertion, not a
    soft warning or an empty-result return.
    """

    def __init__(self, recorder: ExperienceRecorder, environment_tag: EnvironmentTag):
        if environment_tag in PROTECTED_TAGS:
            raise ProtectedPartitionError(
                f"Refusing to construct an ExperienceStore for protected partition "
                f"{environment_tag!r} -- the untouched final OOS partition must not "
                f"be read outside of its designated final-evaluation procedure."
            )
        self._recorder = recorder
        self._environment_tag = environment_tag

    def records(self) -> list[ExperienceRecord]:
        """Return this partition's records, ordered by `decision_id`.

        Records with `decision_id is None` (e.g. NO_TRADE decides) sort first,
        using the empty string as their sort key, and preserve their original
        relative order (Python's sort is stable).
        """
        matching = [
            r for r in self._recorder.all_records()
            if r.environment_tag == self._environment_tag
        ]
        return sorted(matching, key=lambda r: r.decision_id or "")
