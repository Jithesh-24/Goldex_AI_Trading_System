"""v7 FINAL EDITION — full offline pipeline (2026-08-02).
Chains: build_full_matrix → train placement → train direction → backtest.
Run AFTER build_rally_features.py finishes (gold_features_rally.csv present).
Engine/ticker/watchdog are intentionally down (2GB cgroup safety).
"""
import subprocess, sys, os, time

BASE = "/home/jith/.hermes/profiles/trading/scripts"
PY = "/home/jith/.hermes/hermes-agent/venv/bin/python3"

def run(cmd, timeout=14400, tag=""):
    t0 = time.time()
    print(f"\n═══ {tag} ═══ [{time.strftime('%H:%M:%S')}]", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    out = (r.stdout or "")[-3000:]
    err = (r.stderr or "")[-1000:]
    print(out, flush=True)
    if err:
        print("STDERR:", err, flush=True)
    print(f"── {tag} done in {time.time()-t0:.0f}s (exit {r.returncode})", flush=True)
    return r.returncode

def main():
    os.chdir(BASE)
    # 1. full matrix (streamed; rally sub + fresh XM + GNU sort + float32)
    if run([PY, "build_full_matrix.py"], tag="BUILD FULL MATRIX v7 (95 cols)") != 0:
        raise SystemExit("matrix build failed")
    # 2. placement ensemble (3 seeds, recency-weighted, walk-forward, PAVA)
    if run([PY, "train_ai.py"], tag="TRAIN PLACEMENT ENSEMBLE (3 seeds)") != 0:
        raise SystemExit("placement training failed")
    # 3. direction model (3 seeds, 1 row/bar)
    if run([PY, "train_direction.py"], tag="TRAIN DIRECTION MODEL (3 seeds)") != 0:
        raise SystemExit("direction training failed")
    # 4. backtest (mirrors engine exactly)
    run([PY, "backtest_v7.py"], tag="BACKTEST v7")
    print("\n✅ PIPELINE COMPLETE", flush=True)

if __name__ == "__main__":
    main()
