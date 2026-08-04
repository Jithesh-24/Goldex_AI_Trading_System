"""PHASE 0 — Bulk download real XAUUSD spot M1 bars from MT5 (XM broker).
Writes to gold_m1_history.csv in the trading profile scripts dir."""
import MetaTrader5 as mt5
import datetime
import sys
import os

SYM = "GOLD.i#"
OUT = "/home/jith/.hermes/profiles/trading/scripts/gold_m1_history.csv"

def main():
    if not mt5.initialize():
        print(f"INIT FAIL: {mt5.last_error()}"); sys.exit(1)
    acc = mt5.account_info()
    print(f"Connected: {acc.login} | {acc.server} | Balance ${acc.balance:.2f}")
    
    if not mt5.symbol_select(SYM, True):
        print(f"SELECT FAIL: {mt5.last_error()}"); sys.exit(1)
    
    # Try to get maximum history — use naive UTC datetimes (tz-aware fails)
    utc_now = datetime.datetime.utcnow()
    # 60 days = server max for M1 (~60k bars)
    from_dt = utc_now - datetime.timedelta(days=60)
    rates = mt5.copy_rates_range(SYM, mt5.TIMEFRAME_M1, from_dt, utc_now)
    
    if rates is None or len(rates) == 0:
        # Fallback: from_pos with max count (terminal MaxBars=100000)
        print(f"copy_rates_range empty ({mt5.last_error()}), trying from_pos...")
        rates = mt5.copy_rates_from_pos(SYM, mt5.TIMEFRAME_M1, 0, 100000)
    
    if rates is None or len(rates) == 0:
        print(f"NO DATA: {mt5.last_error()}"); sys.exit(1)
    
    print(f"Downloaded {len(rates)} M1 bars")
    first = datetime.datetime.utcfromtimestamp(rates[0][0])
    last = datetime.datetime.utcfromtimestamp(rates[-1][0])
    print(f"Range: {first} -> {last}")
    
    # Write CSV
    with open(OUT, "w") as f:
        f.write("time,open,high,low,close,tick_volume,spread,real_volume\n")
        for r in rates:
            t = datetime.datetime.utcfromtimestamp(r[0]).strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{t},{r[1]:.2f},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]},{r[6]},{r[7]}\n")
    
    sz = os.path.getsize(OUT) / 1024
    print(f"Saved: {OUT} ({sz:.0f} KB)")
    mt5.shutdown()

if __name__ == "__main__":
    main()
