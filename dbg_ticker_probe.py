#!/usr/bin/env python3
"""Debug probe: replicate xm_ticker.py's EXACT connect + poll cycle (v2026-08-10)."""
import os, sys, time, json
import MetaTrader5 as mt5

SYM = "GOLD.i#"

def connect():
    try:
        if not mt5.initialize():
            print("DBG init FAILED:", mt5.last_error(), flush=True)
            return False
        sel = mt5.symbol_select(SYM, True)
        print(f"DBG select -> {sel} err {mt5.last_error()}", flush=True)
        return True
    except Exception as e:
        print("DBG connect err:", e, flush=True)
        return False

def main():
    print("DBG starting", flush=True)
    connected = connect()
    print(f"DBG connected={connected}", flush=True)
    # emulate the ticker's warm-up
    for i in range(20):
        t = mt5.symbol_info_tick(SYM)
        print(f"DBG warmup{i} tick={'OK' if t is not None else 'None'}", flush=True)
        if t is not None:
            break
        time.sleep(0.25)
    # emulate main loop first polls
    for i in range(5):
        t = mt5.symbol_info_tick(SYM)
        print(f"DBG mainpoll{i} tick={'OK' if t is not None else 'None'}", flush=True)
        time.sleep(1)
    mt5.shutdown()

if __name__ == "__main__":
    main()
