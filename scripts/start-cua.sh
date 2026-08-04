#!/bin/bash
exec env -u HTTPS_PROXY -u HTTP_PROXY -u https_proxy -u http_proxy \
  DISPLAY=:0 XAUTHORITY=/home/jith/.Xauthority \
  /home/jith/.cua-driver/packages/releases/0.6.8-x86_64-unknown-linux-gnu/cua-driver mcp
