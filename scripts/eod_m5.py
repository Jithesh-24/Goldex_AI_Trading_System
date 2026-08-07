"""v8 M5 EOD LEARNING LOOP — daily 03:00 IST retrain on the M5 pipeline.

Replaces retrain_loop.py in the EOD cron (job b0c10005e3c0). Full loop:
  1. merge_seed.py          — fresh M1 seed (MT5 history + live bars)
  2. build_m5_matrix.py      — M5 matrix (incremental: appends only new bars)
  3. merge_live_outcomes     — closed loop: append real TP/SL outcome rows
  4. retrain_m5.py           — placement prior (MFE/MFA) → signal rating →
                             base ensemble → direction → calibration →
                             8 specialists → direction prior (all at M5)

Everything writes into models/ atomically; the engine hot-reloads via
regime_specialists.json / ensemble.json mtimes (blocked only mid-trade,
by design — no mid-trade interference).

Usage: python eod_m5.py
"""
import os, sys, time, subprocess

BASE = "/home/jith/.hermes/profiles/trading/scripts"
PY = "/home/jith/.hermes/hermes-agent/venv/bin/python"
LOG = "/tmp/eod_m5.log"
M5_CSV = f"{BASE}/gold_features_m5.csv"

def run(cmd, timeout=36000):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    with open(LOG, "a") as f:
        f.write(f"\n--- {' '.join(cmd[1:])} {time.strftime('%H:%M:%S')} rc={r.returncode} ({time.time()-t0:.0f}s) ---\n")
        f.write(r.stdout[-3000:] + r.stderr[-1500:])
    print(f"[{time.strftime('%H:%M:%S')}] {' '.join(os.path.basename(c) for c in cmd[1:])} "
          f"rc={r.returncode} ({time.time()-t0:.0f}s)")
    if r.returncode != 0:
        print(r.stdout[-800:]); print(r.stderr[-800:])
    return r.returncode == 0

def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] ═══ v8 M5 EOD LEARNING LOOP ═══", flush=True)
    steps = [
        (["merge_seed.py"], 600, "fresh M1 seed"),
        (["build_m5_matrix.py", "--incremental"], 1800, "M5 matrix (incremental)"),
    ]
    for script, timeout, label in steps:
        print(f"→ {label}...", flush=True)
        ok = run([PY, "-u", f"{BASE}/{script[0]}"] + script[1:], timeout)
        if not ok:
            print(f"❌ {label} FAILED — see {LOG}", flush=True)
            sys.exit(1)
    # closed-loop outcomes merge into M5 matrix (before training!)
    print("→ merging live outcomes into M5 matrix...", flush=True)
    env = dict(os.environ); env["FEAT_CSV"] = M5_CSV
    r = subprocess.run([PY, "-c", f"""
import sys; sys.path.insert(0, {BASE!r})
from retrain_loop import merge_live_outcomes_appended
n, tot = merge_live_outcomes_appended({M5_CSV!r})
print(f'merged {{n}} live outcome rows | matrix ~{{tot:,}} rows')
"""], capture_output=True, text=True, env=env, timeout=300)
    with open(LOG, "a") as f:
        f.write(r.stdout + r.stderr)
    print(r.stdout.strip() or r.stderr.strip()[-500:])
    print("→ full M5 retrain...", flush=True)
    ok = run([PY, "-u", f"{BASE}/retrain_m5.py"], 36000)
    if not ok:
        print(f"❌ retrain FAILED — see {LOG}", flush=True)
        sys.exit(1)
    print(f"🎉 v8 M5 EOD LOOP COMPLETE — {time.time()-t0:.0f}s total", flush=True)

if __name__ == "__main__":
    main()
