#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  Hermes Self-Healing Loop — Fully autonomous background maintenance
#  Checks updates, applies patches, runs doctor, fixes issues silently
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
HERMES_REPO="$HERMES_HOME/hermes-agent"
LOG_DIR="$HERMES_HOME/logs"
LOG_FILE="$LOG_DIR/self-heal.log"
LOCK_FILE="/tmp/hermes-self-heal.lock"
STATE_FILE="$LOG_DIR/self-heal-state.json"

mkdir -p "$LOG_DIR"

# ── Lock: prevent concurrent runs ────────────────────────────────
exec 200>"$LOCK_FILE"
flock -n 200 || { echo "[$(date -Iseconds)] SKIP: another self-heal running" >> "$LOG_FILE"; exit 0; }

# ── Logging ──────────────────────────────────────────────────────
log()   { echo "[$(date -Iseconds)] $*" >> "$LOG_FILE"; }
info()  { log "INFO  $*"; }
warn()  { log "WARN  $*"; }
error() { log "ERROR $*"; }
fix()   { log "FIX   $*"; }
ok()    { log "OK    $*"; }

ACTIONS_TAKEN=0
ISSUES_FOUND=0
UPDATED=false

# ═══════════════════════════════════════════════════════════════════
#  PHASE 1: Auto-Update (check for new Hermes version)
# ═══════════════════════════════════════════════════════════════════
info "=== Self-Heal Cycle Starting ==="

if [ -d "$HERMES_REPO/.git" ]; then
    CURRENT_VERSION=$(cd "$HERMES_REPO" && git describe --tags --always 2>/dev/null || echo "unknown")
    info "Current version: $CURRENT_VERSION"
    
    # Fetch latest from upstream (non-destructive)
    if cd "$HERMES_REPO" && git fetch origin --quiet 2>/dev/null; then
        LOCAL_SHA=$(git rev-parse HEAD 2>/dev/null)
        REMOTE_SHA=$(git rev-parse origin/main 2>/dev/null || git rev-parse origin/master 2>/dev/null)
        
        if [ -n "$REMOTE_SHA" ] && [ "$LOCAL_SHA" != "$REMOTE_SHA" ]; then
            warn "Update available: $CURRENT_VERSION → $REMOTE_SHA"
            
            BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
            
            # Capture pre-pull SHA for rollback if syntax breaks
            PRE_PULL_SHA=$(git rev-parse HEAD 2>/dev/null)
            
            # Snapshot dependency state before pull (for smart reinstall)
            DEPS_HASH=""
            if [ -f "pyproject.toml" ]; then
                DEPS_HASH=$(md5sum pyproject.toml 2>/dev/null | cut -d' ' -f1)
            elif [ -f "requirements.txt" ]; then
                DEPS_HASH=$(md5sum requirements.txt 2>/dev/null | cut -d' ' -f1)
            fi
            
            # Strategy: stash local patches, reset to upstream, re-apply patches
            # 1. Stash everything
            STASHED=false
            if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
                git stash push -m "self-heal auto-stash $(date -Iseconds)" --include-untracked --quiet 2>/dev/null && STASHED=true
            fi
            
            # 2. Reset to upstream
            if git reset --hard "origin/$BRANCH" --quiet 2>/dev/null; then
                NEW_VERSION=$(git describe --tags --always 2>/dev/null)
                fix "Updated Hermes: $CURRENT_VERSION → $NEW_VERSION"
                ACTIONS_TAKEN=$((ACTIONS_TAKEN + 1))
                UPDATED=true
                
                # 3. Validate syntax of critical files before proceeding
                SYNTAX_OK=true
                for critical_file in agent/conversation_loop.py agent/error_classifier.py hermes_cli/main.py hermes_cli/config.py; do
                    if [ -f "$critical_file" ]; then
                        if ! python3 -c "import py_compile; py_compile.compile('$critical_file', doraise=True)" 2>/dev/null; then
                            error "SYNTAX ERROR in $critical_file after update!"
                            SYNTAX_OK=false
                            break
                        fi
                    fi
                done
                
                if [ "$SYNTAX_OK" = false ]; then
                    warn "Rolling back to $PRE_PULL_SHA due to syntax errors"
                    git reset --hard "$PRE_PULL_SHA" --quiet 2>/dev/null
                    [ "$STASHED" = true ] && git stash pop --quiet 2>/dev/null || true
                    error "Update ABROLLED — upstream has syntax errors. Will retry next cycle."
                    UPDATED=false
                    ACTIONS_TAKEN=$((ACTIONS_TAKEN - 1))
                else
                    # 4. Clear stale bytecode cache (prevents ImportError on restart)
                    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
                    
                    # 5. Reinstall dependencies only if they actually changed
                    NEW_DEPS_HASH=""
                    if [ -f "pyproject.toml" ]; then
                        NEW_DEPS_HASH=$(md5sum pyproject.toml 2>/dev/null | cut -d' ' -f1)
                    elif [ -f "requirements.txt" ]; then
                        NEW_DEPS_HASH=$(md5sum requirements.txt 2>/dev/null | cut -d' ' -f1)
                    fi
                    
                    if [ "$DEPS_HASH" != "$NEW_DEPS_HASH" ] || [ -n "$DEPS_HASH" ]; then
                        info "Reinstalling Python dependencies..."
                        if [ -f "pyproject.toml" ]; then
                            pip install -q -e . 2>/dev/null || true
                        elif [ -f "requirements.txt" ]; then
                            pip install -q -r requirements.txt 2>/dev/null || true
                        fi
                    else
                        ok "Dependencies unchanged — skipping reinstall"
                    fi
                    
                    # Reinstall node deps if needed
                    if [ -f "package.json" ] && [ ! -d "node_modules" ]; then
                        npm install --quiet 2>/dev/null || true
                    fi
                    
                    # Post-update: reinstall plugin deps
                    if [ -d "plugins" ]; then
                        find plugins/ -name "requirements.txt" -exec pip install -q -r {} \; 2>/dev/null || true
                    fi
                    
                    fix "Post-update deps installed: $NEW_VERSION"
                fi
            else
                warn "Reset failed — restoring stash"
                [ "$STASHED" = true ] && git stash pop --quiet 2>/dev/null || true
            fi
            
            # 6. Pop stash to restore local changes (Phase 2 will re-apply
            #    hermes-customizations.patch for any changes that didn't survive)
            if [ "$STASHED" = true ] && [ -d "$HERMES_REPO" ]; then
                cd "$HERMES_REPO"
                if ! git stash pop --quiet 2>/dev/null; then
                    warn "Stash pop had conflicts — dropping stash (patch file will re-apply)"
                    git checkout -- . 2>/dev/null || true
                    git clean -fd 2>/dev/null || true
                fi
            fi
        else
            ok "Hermes is up to date ($CURRENT_VERSION)"
        fi
    else
        warn "Git fetch failed (offline or no remote)"
    fi
else
    warn "Hermes repo not found at $HERMES_REPO"
fi

# ═══════════════════════════════════════════════════════════════════
#  PHASE 2: Reapply Persistent Patches
# ═══════════════════════════════════════════════════════════════════
PATCH_FILE="$HERMES_HOME/patches/hermes-customizations.patch"
if [ -f "$PATCH_FILE" ] && [ -d "$HERMES_REPO" ]; then
    cd "$HERMES_REPO"
    if git apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
        ok "Patches already applied"
    else
        if git apply --check "$PATCH_FILE" 2>/dev/null; then
            git apply "$PATCH_FILE" 2>/dev/null
            fix "Custom patches reapplied"
            ACTIONS_TAKEN=$((ACTIONS_TAKEN + 1))
        else
            error "Patch apply failed — may need regeneration"
            ISSUES_FOUND=$((ISSUES_FOUND + 1))
        fi
    fi
fi

# ═══════════════════════════════════════════════════════════════════
#  PHASE 2b: Gateway restart REMOVED
# ═══════════════════════════════════════════════════════════════════
# REASON: Restarting all 3 gateways in sequence causes cascading
# failures.  Gateways are self-healing via Phase 4 (individual
# health checks).  If a gateway has stale modules after an update,
# it will recover on its own next restart or the user can
# manually restart it.  Do NOT auto-restart gateways here.

# ═══════════════════════════════════════════════════════════════════
#  PHASE 3: Hermes Doctor --fix (auto-repair config & deps)
# ═══════════════════════════════════════════════════════════════════
if command -v hermes &>/dev/null; then
    DOCTOR_OUTPUT=$(hermes doctor --fix 2>&1) || true
    DOCTOR_EXIT=$?
    
    if [ $DOCTOR_EXIT -ne 0 ]; then
        warn "Doctor found issues (exit=$DOCTOR_EXIT)"
        # Extract actionable fixes from doctor output
        echo "$DOCTOR_OUTPUT" | grep -i "fix\|repair\|install\|missing\|broken" | head -5 | while read -r line; do
            info "Doctor: $line"
        done
        ISSUES_FOUND=$((ISSUES_FOUND + 1))
    else
        ok "Doctor check passed"
    fi
    
    # Check if config is valid
    if ! hermes config check &>/dev/null; then
        warn "Config check failed, attempting migration"
        hermes config migrate &>/dev/null || true
        ACTIONS_TAKEN=$((ACTIONS_TAKEN + 1))
    fi
else
    warn "hermes command not found"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi

# ═══════════════════════════════════════════════════════════════════
#  PHASE 4: Gateway Health & Auto-Restart (all three bots)
# ═══════════════════════════════════════════════════════════════════
for svc in hermes-gateway hermes-gateway-job hermes-gateway-trading; do
    if systemctl --user is-active "$svc" &>/dev/null; then
        ok "$svc running"
    else
        warn "$svc down — restarting"
        systemctl --user start "$svc" 2>/dev/null || true
        sleep 3
        if systemctl --user is-active "$svc" &>/dev/null; then
            fix "$svc restarted via systemd"
            ACTIONS_TAKEN=$((ACTIONS_TAKEN + 1))
        else
            error "$svc failed to restart"
            ISSUES_FOUND=$((ISSUES_FOUND + 1))
        fi
    fi
done

# Phase 4b: Version mismatch check — REMOVED
# Version checking is handled by version-check-restart.sh via hermes-version-check.timer
# and restart-guard.timer handles the actual restart safely.
# Do NOT add version mismatch detection here — grep on gateway.log picks up
# user-pasted error messages ("running code from f2b8a5d541") which are NOT
# actual version metadata, causing infinite false-positive restart loops.

# ═══════════════════════════════════════════════════════════════════
#  PHASE 4c: Gateway Health Verification (after any restarts)
# ═══════════════════════════════════════════════════════════════════
if [ $ACTIONS_TAKEN -gt 0 ]; then
    # Run health check if gateways were restarted
    if [ -f "$HERMES_HOME/scripts/gateway-health-check.sh" ]; then
        if bash "$HERMES_HOME/scripts/gateway-health-check.sh" 2>/dev/null; then
            ok "Gateway health check passed after restart"
        else
            warn "Gateway health check found issues after restart"
        fi
    fi
fi

# ═══════════════════════════════════════════════════════════════════
#  PHASE 5: Camofox Browser Health & Auto-Restart
# ═══════════════════════════════════════════════════════════════════
if curl -sf http://localhost:9377/health &>/dev/null; then
    ok "Camofox healthy"
else
    warn "Camofox down — restarting"
    
    # Kill orphans
    pkill -9 -f "camofox-browser" 2>/dev/null || true
    pkill -9 -f "camoufox" 2>/dev/null || true
    sleep 2
    
    # Try systemd first
    if systemctl --user is-enabled camofox-browser.service &>/dev/null; then
        systemctl --user restart camofox-browser.service 2>/dev/null || true
        sleep 8
    fi
    
    # Verify
    if curl -sf http://localhost:9377/health &>/dev/null; then
        fix "Camofox restarted"
        ACTIONS_TAKEN=$((ACTIONS_TAKEN + 1))
    else
        # Direct start fallback
        CAMOFOX_DIR="$HERMES_REPO/node_modules/@askjo/camofox-browser"
        if [ -d "$CAMOFOX_DIR" ]; then
            cd "$CAMOFOX_DIR" && nohup npm start &>/dev/null &
            sleep 8
            if curl -sf http://localhost:9377/health &>/dev/null; then
                fix "Camofox started directly"
                ACTIONS_TAKEN=$((ACTIONS_TAKEN + 1))
            else
                error "Camofox failed to start"
                ISSUES_FOUND=$((ISSUES_FOUND + 1))
            fi
        else
            error "Camofox not installed"
            ISSUES_FOUND=$((ISSUES_FOUND + 1))
        fi
    fi
fi

# ═══════════════════════════════════════════════════════════════════
#  PHASE 6: Post-Merge Hook Integrity
# ═══════════════════════════════════════════════════════════════════
HOOK="$HERMES_REPO/.git/hooks/post-merge"
if [ -d "$HERMES_REPO/.git" ]; then
    if [ -f "$HOOK" ]; then
        if [ -x "$HOOK" ]; then
            ok "Post-merge hook intact"
        else
            warn "Post-merge hook not executable — fixing"
            chmod +x "$HOOK"
            fix "Post-merge hook made executable"
            ACTIONS_TAKEN=$((ACTIONS_TAKEN + 1))
        fi
    else
        warn "Post-merge hook missing — creating"
        mkdir -p "$(dirname "$HOOK")"
        cat > "$HOOK" << 'HOOK_EOF'
#!/bin/bash
set -o pipefail
PATCH_FILE="$HOME/.hermes/patches/hermes-customizations.patch"
GIT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

if [ -f "$PATCH_FILE" ]; then
    cd "$GIT_DIR" || exit 1
    if git apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
        echo "  ✅ Custom patches already applied"
    else
        git apply "$PATCH_FILE" 2>/dev/null && echo "  ✅ Patches reapplied" || echo "  ⚠️ Patch failed"
    fi
fi
HOOK_EOF
        chmod +x "$HOOK"
        fix "Post-merge hook created"
        ACTIONS_TAKEN=$((ACTIONS_TAKEN + 1))
    fi
fi

# ═══════════════════════════════════════════════════════════════════
#  PHASE 7: Disk & Session Cleanup
# ═══════════════════════════════════════════════════════════════════
# Prune old sessions (>30 days)
hermes sessions prune --older-than 30 &>/dev/null || true

# Prune large log files (>10MB, keep last 1000 lines)
for logfile in "$LOG_DIR"/*.log; do
    if [ -f "$logfile" ] && [ "$(stat -c%s "$logfile" 2>/dev/null || echo 0)" -gt 10485760 ]; then
        tail -1000 "$logfile" > "$logfile.tmp" && mv "$logfile.tmp" "$logfile"
        info "Pruned $(basename "$logfile")"
    fi
done

# ═══════════════════════════════════════════════════════════════════
#  PHASE 8: Memory Compression
# ═══════════════════════════════════════════════════════════════════
if [ -f "$HERMES_HOME/scripts/memory-compressor.py" ]; then
    if python3 "$HERMES_HOME/scripts/memory-compressor.py" 2>/dev/null; then
        ok "Memory compressor ran"
    else
        warn "Memory compressor failed"
    fi
else
    warn "memory-compressor.py not found"
fi

# ═══════════════════════════════════════════════════════════════════
#  PHASE 9: Write State (for external monitoring)
# ═══════════════════════════════════════════════════════════════════
cat > "$STATE_FILE" << EOF
{
    "last_run": "$(date -Iseconds)",
    "actions_taken": $ACTIONS_TAKEN,
    "issues_found": $ISSUES_FOUND,
    "version": "$(cd "$HERMES_REPO" && git describe --tags --always 2>/dev/null || echo 'unknown')"
}
EOF

# ═══════════════════════════════════════════════════════════════════
#  Summary (silent unless issues)
# ═══════════════════════════════════════════════════════════════════
if [ $ISSUES_FOUND -gt 0 ] || [ $ACTIONS_TAKEN -gt 0 ]; then
    info "=== Self-Heal Complete: $ACTIONS_TAKEN action(s), $ISSUES_FOUND unresolved issue(s) ==="
else
    ok "=== Self-Heal Complete: all systems nominal ==="
fi

# Release lock
flock -u 200 2>/dev/null || true
exit 0
