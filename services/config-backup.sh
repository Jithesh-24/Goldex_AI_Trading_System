#!/usr/bin/env bash
# Daily backup of Hermes config, .env, skills, scripts, and ALL profiles
BACKUP_DIR="$HOME/.hermes/backups"
DATE=$(date +%Y-%m-%d)
mkdir -p "$BACKUP_DIR"

# Main config
tar czf "$BACKUP_DIR/hermes-config-$DATE.tar.gz" \
  -C "$HOME/.hermes" \
  config.yaml .env skills/ scripts/ auth.json

# Per-profile backups
for PROFILE_DIR in "$HOME/.hermes/profiles/job" "$HOME/.hermes/profiles/trading"; do
    PROFILE_NAME=$(basename "$PROFILE_DIR")
    if [ -d "$PROFILE_DIR" ]; then
        tar czf "$BACKUP_DIR/hermes-${PROFILE_NAME}-config-$DATE.tar.gz" \
          -C "$PROFILE_DIR" \
          config.yaml .env auth.json 2>/dev/null
        echo "✓ Profile $PROFILE_NAME backed up"
    fi
done

# Keep only last 7 days of backups
find "$BACKUP_DIR" -name "hermes-*-config-*.tar.gz" -mtime +7 -delete 2>/dev/null

echo "✓ Main config backup: $BACKUP_DIR/hermes-config-$DATE.tar.gz"
ls -lh "$BACKUP_DIR/hermes-config-$DATE.tar.gz" 2>/dev/null | awk '{print $5}'
