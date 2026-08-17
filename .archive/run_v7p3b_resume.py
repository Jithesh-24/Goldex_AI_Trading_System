"""v7.3 RESUME pipeline (2026-08-02) — rally + matrix already built, OOM-fixed.
Chains: train placement → train direction → backtest.
Run under systemd-run --user --unit=v7p3b (own cgroup, survives gateway restarts).
"""
import subprocess, sys, os, time

BASE = "/home/jith/.hermes/profiles/trading/scripts"
PY = "/home/jith/.hermes/hermes-agent/venv/bin/python3"
LOG = "/home/jith/.hermes/profiles/trading/scripts/v7p3b_pipeline.log"

def run(cmd, timeout=28800, tag=""):
    t0 = time.time()
    print(f"\n═══ {tag} ═══ [{time.strftime('%H:%M:%S')}]", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    out = (r.stdout or "")[-4000:]
    err = (r.stderr or "")[-1500:]
    print(out, flush=True)
    if err:
        print("STDERR:", err, flush=True)
    print(f"── {tag} done in {time.time()-t0:.0f}s (exit {r.returncode})", flush=True)
    return r.returncode

def main():
    os.chdir(BASE)
    with open(LOG, "a") as f:
        f.write(f"\n=== RESUME PIPELINE START {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    if run([PY, "train_ai.py"], tag="TRAIN PLACEMENT ENSEMBLE (3 seeds, OOM-fixed)") != 0:
        raise SystemExit("placement training failed")
    if run([PY, "train_direction.py"], tag="TRAIN DIRECTION MODEL (3 seeds)") != 0:
        raise SystemExit("direction training failed")
    run([PY, "backtest_v7.py"], tag="BACKTEST v7.3")
    print("\n✅ RESUME PIPELINE COMPLETE", flush=True)
    with open(LOG, "a") as f:
        f.write(f"=== RESUME PIPELINE DONE {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

if __name__ == "__main__":
    main()
