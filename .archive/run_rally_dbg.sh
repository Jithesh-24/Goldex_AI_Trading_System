#!/bin/bash
# Debug: run rally build directly, capture full error
cd /home/jith/.hermes/profiles/trading/scripts
export TMPDIR=/home/jith/.hermes/profiles/trading/tmp
exec /home/jith/.hermes/hermes-agent/venv/bin/python3 -u build_rally_features.py 2>&1 | tee /tmp/rally_dbg.log
