#!/bin/bash
export FEAT_CSV=/home/jith/.hermes/profiles/trading/scripts/gold_features_m5_full.csv
cd /home/jith/.hermes/profiles/trading/scripts
/home/jith/.hermes/hermes-agent/venv/bin/python3 -u train_ai.py 2>&1
