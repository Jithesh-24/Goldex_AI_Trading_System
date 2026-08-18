#!/bin/bash
export DISPLAY=:0
export XAUTHORITY=/home/jith/.Xauthority
exec /home/jith/.cua-driver/packages/releases/0.6.8-x86_64-unknown-linux-gnu/cua-driver mcp "$@"
