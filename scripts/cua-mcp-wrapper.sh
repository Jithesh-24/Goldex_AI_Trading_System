#!/bin/bash
unset HTTPS_PROXY HTTP_PROXY https_proxy http_proxy
export DISPLAY=:99
export XAUTHORITY=/home/jith/.Xauthority
export CUA_DRIVER_CDP_PORT=9222
exec /home/jith/.cua-driver/packages/current/cua-driver mcp "$@"
