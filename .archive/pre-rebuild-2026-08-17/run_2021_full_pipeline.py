#!/usr/bin/env python3
"""v7.12 one-shot: append 2021 → FULL matrix rebuild → warm-start retrain chain.

Runs the complete data-completeness pipeline once (2026-08-05):
  1. append_2021_to_rally.py  — 2021 M1 → features → rally cache (fills the gap)
  2. merge_seed.py            — refresh XM seed with latest MT5/ticker bars
  3. build_full_matrix.py --full — FULL rebuild (2021 now in cache → matrix covers
     2020→2026 continuously; ~30-60 min on 8 cores)
  4. retrain_loop.py          — warm-start continuation (preserves Aug 3/4 models),
     direction model, calibration (honest min-support curves), regime journal,
     transition model, 8 regime specialists. The incremental build inside
     retrain_loop is a no-op after the fresh full build (matrix current).
  5. space_guard             — cleans temp + audits disk (inside retrain_loop)

After this, the nightly 03:00 IST EOD runs incremental (fast path) — day-by-day
learning curve with zero data loss, warm-started from today's models.
"""
import subprocess, sys, os, time

BASE = "/home/jith/.hermes/profiles/trading/scripts"
PY = "/home/jith/.hermes/hermes-agent/venv/bin/python3"
t0 = time.time()

def step(name, script, timeout, tail_n=15):
    print(f"\n{'='*70}\n[{time.strftime('%H:%M:%S')}] STEP: {name}\n{'='*70}", flush=True)
    r = subprocess.run([PY, "-u", f"{BASE}/{script}"], capture_output=True, text=True, timeout=timeout)
    out = r.stdout.strip().splitlines()
    for ln in out[-tail_n:]:
        print(ln, flush=True)
    if r.returncode != 0:
        print(f"!! {name} FAILED rc={r.returncode}\n{r.stderr[-1500:]}", flush=True)
        sys.exit(1)
    print(f"→ {name} OK ({time.time()-t0:.0f}s elapsed)", flush=True)
    return out

# guard: 2021 file must exist
if not os.path.exists(f"{BASE}/gold_m1_2021.csv"):
    print("!! gold_m1_2021.csv missing — download 2021 first")
    sys.exit(2)

step("Append 2021 features to rally cache", "append_2021_to_rally.py", timeout=7200, tail_n=6)
step("Refresh XM seed (MT5 + live bars)", "merge_seed.py", timeout=600, tail_n=8)
step("FULL matrix rebuild (2020→2026 continuous)", "build_full_matrix.py", timeout=7200, tail_n=10)
# full retrain chain (warm-start, preserves Aug 3/4 knowledge)
step("Warm-start retrain chain", "retrain_loop.py", timeout=14400, tail_n=25)

print(f"\n{'='*70}\n✅ ALL STEPS COMPLETE in {time.time()-t0:.0f}s\n{'='*70}")
