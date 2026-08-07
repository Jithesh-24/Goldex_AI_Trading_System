"""v8 M5 RETRAIN ORCHESTRATOR — full retrain at M5 base timeframe.

Runs the complete v8 M5 pipeline AFTER build_m5_matrix.py finishes:
  1. fit_placement_prior.py   — learned SL/TP from MFE/MFA (6yr excursions)
  2. fit_signal_rating.py     — rating weights + fire threshold (learned)
  3. train_ai.py              — base ensemble at M5 (FULL 6yr cold start:
                               feature count changed 108→116, so no warm-start;
                               walk-forward OOF + calibration. Subsequent daily
                               EOD uses train_continue.py warm-start on 180d.)
  4. train_direction_htf.py   — direction model at M5
  5. fit_calibration_by_rr.py — base per-dir×RR calibration at M5
  6. train_regime_spec.py     — all 8 regime specialists at M5
  7. regenerate_dir_prior.py  — direction prior at M5 horizon (36 bars)

Each step writes into models/ (atomic). The engine hot-reloads via
regime_specialists.json mtime. Run manually OR by cron EOD (the EOD loop
will be switched to M5 in the same commit — matrix is gold_features_m5.csv).

Usage: python retrain_m5.py [--skip-specialists]
"""
import os, sys, time, subprocess, json

BASE = "/home/jith/.hermes/profiles/trading/scripts"
PY = "/home/jith/.hermes/hermes-agent/venv/bin/python"
LOG = "/tmp/retrain_m5.log"

STEPS = [
    ("fit_placement_prior.py",      "placement_prior.json",  "learned SL/TP from MFE/MFA"),
    ("fit_signal_rating.py",        "signal_rating.json",    "rating weights + fire threshold"),
    ("train_ai.py",                 "ensemble.json",         "base ensemble (M5, full 6yr cold)"),
    ("train_direction_htf.py",      "direction_metrics.json","direction model (M5)"),
    ("fit_calibration_by_rr.py",    "calibration_by_drr.json","base per-dir×RR calibration"),
    ("train_regime_spec.py",        "regime_specialists.json","8 regime specialists (M5)"),
    ("regenerate_dir_prior.py",     "regime_dir_prior.json", "direction prior @ 36 M5 bars"),
]

def run_step(script, artifact, label, skip_specialists=False):
    if skip_specialists and script == "train_regime_spec.py":
        print(f"⏭ skip {script} (--skip-specialists)", flush=True)
        return True
    t0 = time.time()
    print(f"▶ {label} — {script} ({time.time()-t0:.0f}s)", flush=True)
    with open(LOG, "a") as f:
        f.write(f"\n===== {script} {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    env = dict(os.environ)
    env["FEAT_CSV"] = f"{BASE}/gold_features_m5.csv"
    env["PRIOR_BAR_SECS"] = "300"        # M5 bars = 300s
    env["PRIOR_HORIZONS"] = "3,12,36"    # 15/60/180 min at M5
    env["DIR_HORIZON_BARS"] = "36"       # direction model: 36 M5 bars = 180 min
    r = subprocess.run([PY, "-u", f"{BASE}/{script}"],
                       capture_output=True, text=True, timeout=36000, env=env)
    with open(LOG, "a") as f:
        f.write(r.stdout[-4000:])
        f.write(r.stderr[-2000:])
    if r.returncode != 0:
        print(f"❌ {script} FAILED rc={r.returncode} — see {LOG}", flush=True)
        print(r.stdout[-1500:], flush=True)
        print(r.stderr[-1500:], flush=True)
        return False
    art = f"{BASE}/models/{artifact}"
    if not os.path.exists(art):
        print(f"⚠ {script} ok but artifact missing: {art}", flush=True)
    else:
        mt = time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(art)))
        print(f"✅ {script} → {artifact} (mtime {mt}, {time.time()-t0:.0f}s)", flush=True)
    return True

def main():
    skip = "--skip-specialists" in sys.argv
    t0 = time.time()
    print("═══ v8 M5 RETRAIN ═══", flush=True)
    for script, artifact, label in STEPS:
        if not run_step(script, artifact, label, skip):
            print("❌ pipeline aborted — fix and re-run (steps are idempotent)", flush=True)
            sys.exit(1)
    print(f"🎉 M5 RETRAIN COMPLETE — {time.time()-t0:.0f}s total", flush=True)

if __name__ == "__main__":
    main()
