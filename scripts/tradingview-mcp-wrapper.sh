#!/bin/bash
# Wrapper that clears HTTPS_PROXY so tradingview-mcp can reach Yahoo Finance
unset HTTPS_PROXY
unset https_proxy
exec tradingview-mcp "$@"
