#!/bin/bash
# Hermes System Health Check — verify ALL layers after updates/restarts (Linux)

HERMES_HOME="$HOME/.hermes"
CONFIG="$HERMES_HOME/config.yaml"
ENV_FILE="$HERMES_HOME/.env"
LOG_DIR="$HERMES_HOME/logs"
mkdir -p "$LOG_DIR"

HEALTH_LOG="$LOG_DIR/system-health.log"
ERRORS=0

log() { echo "[$(date +%H:%M:%S)] $*"; }
FATALS=0
error() { log "❌ $*"; ERRORS=$((ERRORS + 1)); }
fatal() { log "❌ FATAL: $*"; FATALS=$((FATALS + 1)); ERRORS=$((ERRORS + 1)); }
ok()    { log "✅ $*"; }
warn()  { log "⚠️  $*"; }

echo "=== Hermes System Health Check ===" >> "$HEALTH_LOG"
echo "Timestamp: $(date)" >> "$HEALTH_LOG"

# Layer 1: Config
log "--- Layer 1: Config ---"
if [ -f "$CONFIG" ]; then
    grep -qE '(engine: camofox|cloud_provider: camofox)' "$CONFIG" 2>/dev/null && ok "browser.engine = camofox" || warn "browser.engine not camofox"
    grep -q 'mode: false' "$CONFIG" 2>/dev/null && ok "approvals.mode = false" || warn "Check approvals.mode"
    ok "Config file exists"
else
    fatal "Config MISSING!"
fi

# Layer 2: Environment
log "--- Layer 2: Env ---"
if [ -f "$ENV_FILE" ]; then
    for var in XIAOMI_API_KEY TELEGRAM_BOT_TOKEN CAMOFOX_URL; do
        grep -q "^$var=" "$ENV_FILE" 2>/dev/null && ok ".env has $var" || warn ".env missing $var"
    done
else
    fatal ".env MISSING!"
fi

# Layer 3: All Gateways (systemd user services)
log "--- Layer 3: Gateways ---"
for GW in hermes-gateway hermes-gateway-job hermes-gateway-trading; do
    if systemctl --user is-active "${GW}.service" >/dev/null 2>&1; then
        GW_PID=$(systemctl --user show "${GW}.service" --property=MainPID --value 2>/dev/null)
        [ -n "$GW_PID" ] && [ "$GW_PID" != "0" ] && ok "${GW} running (PID: $GW_PID)" || error "${GW} PID not set!"
    else
        fatal "${GW} NOT active in systemd!"
    fi
done

# Check per-profile .env and config
for PROFILE_DIR in "$HOME/.hermes/profiles/job" "$HOME/.hermes/profiles/trading"; do
    PROFILE_NAME=$(basename "$PROFILE_DIR")
    if [ -f "$PROFILE_DIR/.env" ]; then
        ok "Profile $PROFILE_NAME has .env"
    else
        error "Profile $PROFILE_NAME MISSING .env!"
    fi
    if [ -f "$PROFILE_DIR/config.yaml" ]; then
        ok "Profile $PROFILE_NAME has config.yaml"
    else
        error "Profile $PROFILE_NAME MISSING config.yaml!"
    fi
done

# Layer 4: Camofox
log "--- Layer 4: Camofox ---"
if curl -sf http://localhost:9377/health >/dev/null 2>&1; then
    ok "Camofox responding"
    ENGINE=$(curl -sf http://localhost:9377/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('ok','unknown'))" 2>/dev/null)
    [ "$ENGINE" = "True" ] && ok "Engine healthy" || warn "Engine not healthy"
else
    error "Camofox NOT responding!"
    if systemctl --user is-enabled camofox-browser.service >/dev/null 2>&1; then
        log "Restarting via systemd..."
        systemctl --user restart camofox-browser.service 2>/dev/null && sleep 5 && \
            curl -sf http://localhost:9377/health >/dev/null 2>&1 && ok "Camofox restarted" || error "Failed to restart"
    fi
fi

# Layer 5: Patches (check if custom patches exist)
log "--- Layer 5: Patches ---"
PATCH_FILE="$HERMES_HOME/patches/hermes-customizations.patch"
if [ -f "$PATCH_FILE" ] && [ -d "$HERMES_HOME/hermes-agent" ]; then
    cd "$HERMES_HOME/hermes-agent"
    git apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1 && ok "Patches applied" || { warn "Reapplying..."; git apply "$PATCH_FILE" 2>/dev/null && ok "Patches reapplied" || error "Patch FAILED"; }
else
    warn "No patch file found (optional)"
fi

# Layer 6: Disk
log "--- Layer 6: Disk ---"
DISK_MB=$(($(du -sk "$HERMES_HOME" 2>/dev/null | cut -f1) / 1024))
ok "Hermes: ${DISK_MB}MB"

# Layer 7: Scripts
log "--- Layer 7: Scripts ---"
for script in system-health.sh reapply-patches.sh camofox-watchdog.sh recovery.sh; do
    [ -x "$HERMES_HOME/scripts/$script" ] && ok "Script: $script" || warn "Script missing: $script"
done

# Layer 8: Post-merge hook
log "--- Layer 8: Post-merge hook ---"
[ -x "$HERMES_HOME/hermes-agent/.git/hooks/post-merge" ] && ok "Post-merge hook" || warn "Post-merge hook missing"

# Summary
echo ""
if [ $FATALS -gt 0 ]; then
    echo "❌ $FATALS FATAL error(s), $((ERRORS - FATALS)) warning(s)"
    exit 1
elif [ $ERRORS -gt 0 ]; then
    echo "⚠️  $ERRORS warning(s) — no fatals"
    exit 0
else
    echo "✅ ALL SYSTEMS HEALTHY"
    exit 0
fi
