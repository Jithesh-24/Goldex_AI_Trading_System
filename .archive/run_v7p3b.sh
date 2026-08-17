#!/bin/bash
# v7.3 RESUME launcher — systemd-owned, OOM-fixed training.
cd /home/jith/.hermes/profiles/trading/scripts
export TMPDIR=/home/jith/.hermes/profiles/trading/tmp
mkdir -p "$TMPDIR"
exec /home/jith/.hermes/hermes-agent/venv/bin/python3 -u run_v7p3b_resume.py 2>&1 | tee -a v7p3b_pipeline.log
