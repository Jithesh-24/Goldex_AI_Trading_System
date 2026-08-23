"""Opt-in real-tick capture-to-disk. Off by default everywhere it is
constructed (Phase 4 Global Constraints: production behavior must not
change). Exists solely so the 5 Task-21 live-only microstructure features
can eventually be validated against real XM ticks (spec section 22) --
synthetic replay is explicitly not acceptable evidence for that
validation. Never imported by app/engine.py's decision path; wiring this
into market/feed_listener.py is this task's own Step 5, gated by a config
flag defaulting to disabled."""
import csv
import os


class TickCapture:
    def __init__(self, out_path: str, enabled: bool = False):
        self.out_path = out_path
        self.enabled = enabled
        self._fields = None
        self._fh = None
        self._writer = None

    def on_tick(self, tick: dict) -> None:
        if not self.enabled:
            return
        if self._writer is None:
            self._fields = list(tick.keys())
            os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
            self._fh = open(self.out_path, "w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self._fields)
            self._writer.writeheader()
        self._writer.writerow(tick)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
