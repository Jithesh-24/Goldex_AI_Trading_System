#!/bin/bash
# chain_completion_watch.sh — silent watchdog for the v8.8 fast-path chain.
# no_agent cron: prints NOTHING until "TRANSITION COMPLETE" appears in the
# chain log (quiet = no spam), then prints the post-trade verification.
#
# 2026-08-12 UPDATE (tick directive): M5 COMPLETE is NOT the final state —
# the TICK retrain (Dukascopy M1 features) follows. The engine MUST stay OFF
# until the tick models are live and verified. This script therefore:
#   - reports the M5 baseline as "phase 1 done, phase 2 (tick) next"
#   - does NOT emit @@RESUME_CRONS@@ — engine watchdog stays PAUSED
#   - only the TICK COMPLETE marker (transition_tick.log) triggers the
#     full green-light report + @@RESUME_CRONS@@
LOG=/home/jith/.hermes/profiles/trading/scripts/transition_v88.log
TICKLOG=/home/jith/.hermes/profiles/trading/scripts/transition_tick.log
MARK=/home/jith/.hermes/profiles/trading/scripts/.chain_report_sent

# Phase 2 (tick) complete → full green-light report.
if grep -q "TICK TRANSITION COMPLETE" "${TICKLOG}" 2>/dev/null; then
    [ -f "${MARK}.tick" ] && exit 0
    touch "${MARK}.tick"
    echo "🏆🏆 TICK TRANSITION COMPLETE — the beast is live"
    echo "── tick OOF / walk-forward results ──"
    awk '/TICK TRANSITION RESUME/{f=1} f' "${TICKLOG}" | grep -E "windows done|AGGREGATE|accuracy|precision|recall|auc|AUC|logloss|OOF saved|calibration|rating|prior|lessons" | tail -25
    echo "── engine state ──"
    systemctl --user is-active ai-engine.service
    TPID=$(pgrep -f ai_signal_engine | head -1)
    [ -n "${TPID}" ] && ps -o etime= -p "${TPID}" | awk '{print "engine uptime:", $1}' || echo "engine NOT RUNNING"
    echo "── model artifacts (96-feat M5 + 100-feat tick) ──"
    MD=/home/jith/.hermes/profiles/trading/scripts/models
    ls -la --time-style=+%m-%d_%H:%M "${MD}"/gold_lgb_model_s*.txt "${MD}"/direction_model* "${MD}"/calibration_by_drr.json "${MD}"/signal_rating.json "${MD}"/regime_dir_prior.json 2>/dev/null | awk '{print $6, $5, $NF}'
    python3 -c "import json; f=json.load(open('${MD}/features.json')); print('features.json count:', len(f.get('features', f)) if isinstance(f, dict) else len(f))" 2>/dev/null
    echo "── tick columns in features.json ──"
    python3 -c "
import json
f=json.load(open('${MD}/features.json'))
feats=f if isinstance(f,list) else f.get('features',[])
print([x for x in feats if 'tick' in x.lower() or 'imb' in x.lower() or 'vol_rel' in x.lower() or 'cvd' in x.lower()])
" 2>/dev/null
    echo "@@RESUME_CRONS@@"
    exit 0
fi

# Phase 1 (M5 baseline) complete → status only, engine STAYS OFF.
if grep -q "v8.8 TRANSITION COMPLETE" "${LOG}" 2>/dev/null; then
    [ -f "${MARK}" ] && exit 0
    touch "${MARK}"
    echo "✅ PHASE 1 DONE — M5 baseline trained (96-feat)"
    echo "── OOF / walk-forward results ──"
    awk '/TRANSITION RESUME FAST-PATH/{f=1} f' "${LOG}" | grep -E "windows done|AGGREGATE|accuracy|precision|recall|OOF saved" | tail -12
    echo
    echo "⏳ PHASE 2 (TICK RETRAIN) in progress — engine intentionally OFF until tick models verified. Engine watchdog cron remains PAUSED."
    exit 0
fi

# Not complete yet → silent
exit 0