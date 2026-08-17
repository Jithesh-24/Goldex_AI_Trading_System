"""v7.3 STRATEGY PLAYBOOK — full offline pipeline (2026-08-02).
Chains: rally rebuild → full matrix → placement train → direction train → backtest.
features.py changed (13 playbook feats + doubled grid) → rally cache MUST rebuild.
Run under systemd-run --user --unit=v7p3 (own cgroup, survives gateway restarts).
"""
import subprocess, sys, os, time

BASE = "/home/jith/.hermes/profiles/trading/scripts"
PY = "/home/jith/.hermes/hermes-agent/venv/bin/python3"
LOG = "/home/jith/.hermes/profiles/trading/scripts/v7p3_pipeline.log"

def run(cmd, timeout=28800, tag=""):
    t0 = time.time()
    line = f"\n═══ {tag} ═══ [{time.strftime('%H:%M:%S')}]"
    print(line, flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    out = (r.stdout or "")[-4000:]
    err = (r.stderr or "")[-1500:]
    print(out, flush=True)
    if err:
        print("STDERR:", err, flush=True)
    done = f"── {tag} done in {time.time()-t0:.0f}s (exit {r.returncode})"
    print(done, flush=True)
    return r.returncode

def main():
    os.chdir(BASE)
    with open(LOG, "a") as f:
        f.write(f"\n=== PIPELINE START {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    # 0. rally cache rebuild (features.py changed — REQUIRED for v7.3 cols)
    if run([PY, "build_rally_features.py"], tag="REBUILD RALLY CACHE (playbook feats)") != 0:
        raise SystemExit("rally rebuild failed")
    # 1. full matrix (streamed; rally sub + fresh XM + GNU sort + float32)
    if run([PY, "build_full_matrix.py"], tag="BUILD FULL MATRIX v7.3 (108 cols)") != 0:
        raise SystemExit("matrix build failed")
    # 2. placement ensemble (3 seeds, recency-weighted, walk-forward, PAVA)
    if run([PY, "train_ai.py"], tag="TRAIN PLACEMENT ENSEMBLE (3 seeds)") != 0:
        raise SystemExit("placement training failed")
    # 3. direction model (3 seeds, 1 row/bar)
    if run([PY, "train_direction.py"], tag="TRAIN DIRECTION MODEL (3 seeds)") != 0:
        raise SystemExit("direction training failed")
    # 4. backtest (mirrors engine exactly)
    run([PY, "backtest_v7.py"], tag="BACKTEST v7.3")
    print("\n✅ PIPELINE COMPLETE", flush=True)
    with open(LOG, "a") as f:
        f.write(f"=== PIPELINE DONE {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

if __name__ == "__main__":
    main()
