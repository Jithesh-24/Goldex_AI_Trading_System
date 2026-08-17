"""v7.3d pipeline (2026-08-02) — LABEL FIX: BUY SL/TP levels now match the
backtest/live engine exactly (old label was 0.20 optimistic on BUY → model
learned a fictitious market → SELL-bias + negative real EV).
Chains: rebuild matrix (honest targets) → train placement → per-RR
calibration → backtest (neutral direction prior).
Run under systemd-run --user --unit=v73d (own cgroup, survives restarts).
"""
import subprocess, sys, os, time

BASE = "/home/jith/.hermes/profiles/trading/scripts"
PY = "/home/jith/.hermes/hermes-agent/venv/bin/python3"
LOG = "/home/jith/.hermes/profiles/trading/scripts/v7p3d_pipeline.log"

def run(cmd, timeout=28800, tag=""):
    t0 = time.time()
    print(f"\n=== {tag} === [{time.strftime('%H:%M:%S')}]", flush=True)
    env = dict(os.environ, TMPDIR=f"{BASE}/tmp")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    out = (r.stdout or "")[-5000:]
    err = (r.stderr or "")[-1500:]
    print(out, flush=True)
    if err:
        print("STDERR:", err, flush=True)
    print(f"-- {tag} done in {time.time()-t0:.0f}s (exit {r.returncode})", flush=True)
    return r.returncode

def main():
    os.chdir(BASE)
    with open(LOG, "a") as f:
        f.write(f"\n=== v7.3d PIPELINE START {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    # 1. rebuild matrix with HONEST labels (BUY levels fixed in features.py)
    if run([PY, "build_full_matrix.py"], tag="REBUILD MATRIX (honest labels)") != 0:
        raise SystemExit("matrix rebuild failed")
    # 2. retrain placement on honest labels
    if run([PY, "train_ai.py"], tag="TRAIN PLACEMENT (honest labels)") != 0:
        raise SystemExit("placement training failed")
    # 3. per-RR calibration (adaptive trade management)
    if run([PY, "fit_calibration_by_rr.py"], tag="PER-RR CALIBRATION") != 0:
        print("WARN per-RR calibration failed — continuing", flush=True)
    # 4. backtest with neutral direction prior (direction model = coin flip)
    env = dict(os.environ, TMPDIR=f"{BASE}/tmp", NEUTRAL_PRIOR="1")
    run([PY, "backtest_v7.py"], tag="BACKTEST v7.3d (neutral prior)") 
    print("\n== v7.3d PIPELINE COMPLETE ==", flush=True)
    with open(LOG, "a") as f:
        f.write(f"=== v7.3d PIPELINE DONE {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

if __name__ == "__main__":
    main()
