"""
Collect REAL XAUUSD spot data via TV scanner, then train the learning system.
This replaces the GC=F training data with actual spot market data.
"""
import json, urllib.request, time, sys, os, math
from datetime import datetime, timezone
from collections import deque

BASE = os.path.expanduser("~/.hermes/profiles/trading/cron/output")
JOURNAL = f"{BASE}/trade_journal.jsonl"

# ── DATA COLLECTOR ──
def scan():
    try:
        p = {"symbols":{"tickers":["TVC:GOLD"]},
             "columns":["close","high","low","open","volume","RSI","ATR","Recommend.All",
                        "BB.upper","BB.lower","MACD.macd","MACD.signal","SMA20","SMA50",
                        "change","change_abs","Volatility.DR","Pivot.M.Classic.Resistance",
                        "Pivot.M.Classic.Support"]}
        r = urllib.request.Request("https://scanner.tradingview.com/cfd/scan",
            data=json.dumps(p).encode(),
            headers={"User-Agent":"Mozilla/5.0","Content-Type":"application/json",
                     "Origin":"https://www.tradingview.com"}, method="POST")
        with urllib.request.urlopen(r, timeout=10) as resp:
            d = json.loads(resp.read())
        if d.get("data"):
            raw = d["data"][0]["d"]
            return {
                "price": raw[0], "high": raw[1], "low": raw[2], "open": raw[3],
                "volume": raw[4], "rsi": raw[5], "atr": raw[6],
                "bb_u": raw[8], "bb_l": raw[9],
                "macd": raw[10], "signal": raw[11],
                "sma20": raw[12], "sma50": raw[13],
                "change": raw[14], "change_abs": raw[15]
            } if all(raw[0:4]) else None
    except: return None

# Build 1-minute OHLC
class CollectOHLC:
    def __init__(self):
        self.bars = []
        self.current_min = None
        self.current_bar = None
    
    def tick(self, p, ts):
        min_key = int(ts // 60)
        if self.current_min != min_key:
            # Save completed bar
            if self.current_bar:
                self.bars.append(self.current_bar)
                self._save_bar(self.current_bar)
            self.current_min = min_key
            self.current_bar = {"t": min_key * 60, "o": p, "h": p, "l": p, "c": p, "v": 0}
        else:
            self.current_bar["h"] = max(self.current_bar["h"], p)
            self.current_bar["l"] = min(self.current_bar["l"], p)
            self.current_bar["c"] = p
            self.current_bar["v"] += 1
    
    def _save_bar(self, bar):
        with open(f"{BASE}/spot_bars.jsonl", "a") as f:
            f.write(json.dumps(bar) + "\n")

def load_spot_bars():
    bars = []
    try:
        with open(f"{BASE}/spot_bars.jsonl") as f:
            for line in f:
                if line.strip():
                    bars.append(json.loads(line))
    except: pass
    return bars

# ── TRAINING ──
print("=" * 60)
print("REAL XAUUSD SPOT DATA COLLECTOR + TRAINER")
print("=" * 60)

# Load bot modules
with open('/home/jith/.hermes/profiles/trading/scripts/hermes_scalper.py') as f:
    bot_code = f.read()
namespace = {}
exec(bot_code.split('def main()')[0], namespace)

calc_atr = namespace['calc_atr']
calc_bb = namespace['calc_bb']
detect_regime_raw = namespace['detect_regime']
session_adj = namespace['session_adj']
get_vol_state = namespace['get_vol_state']
compute_sl_tp = namespace['compute_sl_tp']
daily_context = namespace['daily_context']
compute_momentum = namespace['compute_momentum']
local_range_position = namespace['local_range_position']
compute_streak = namespace['compute_streak']
bucket = namespace['bucket']
MOM_TH = namespace['MOM_TH']; POS_TH = namespace['POS_TH']; STR_TH = namespace['STR_TH']
m1_momentum = namespace['m1_momentum']
m2_rsi = namespace['m2_rsi']
m3_bb = namespace['m3_bb']
m4_structure = namespace['m4_structure']
m5_divergence = namespace['m5_divergence']
m6_patterns = namespace['m6_patterns']
m7_stats = namespace['m7_stats']
m8_macd = namespace['m8_macd']

SPREAD = 0.2

class MiniOHLC:
    def __init__(self, bs):
        self.bars = bs
    def close_prices(self, n=0):
        arr = [b['c'] for b in self.bars]
        return arr[-n:] if n and arr else arr
    def highs(self, n=0):
        arr = [b['h'] for b in self.bars]
        return arr[-n:] if n and arr else arr
    def lows(self, n=0):
        arr = [b['l'] for b in self.bars]
        return arr[-n:] if n and arr else arr

class MiniState:
    def __init__(self, ohlc, p):
        self.ohlc = ohlc
        self.p = p
        self.raw = {}
        cl = ohlc.close_prices()
        if len(cl) >= 20:
            bb_u, bb_m, bb_l = calc_bb(cl)
            self.raw["BB.upper"] = bb_u
            self.raw["BB.lower"] = bb_l
        if len(cl) >= 14:
            gains = losses = 0
            for i in range(-14, 0):
                diff = cl[i] - cl[i-1]
                if diff > 0: gains += diff
                else: losses -= diff
            avg_g = gains / 14
            avg_l = losses / 14 if losses else 0.01
            self.raw['RSI'] = 100 - 100 / (1 + avg_g / avg_l)

# ── COLLECT DATA ──
print("\nPhase 1: Collecting live XAUUSD spot data...")
collector = CollectOHLC()
t0 = time.time()
duration = 300  # 5 minutes
spot_bars_count = len(load_spot_bars())
print(f"Existing spot bars: {spot_bars_count}")

while time.time() - t0 < duration:
    data = scan()
    if data and data['price'] > 0:
        collector.tick(data['price'], time.time())
        bars_now = len(load_spot_bars())
        if bars_now > spot_bars_count:
            spot_bars_count = bars_now
            print(f"  {bars_now} bars collected (${data['price']:.2f})", end="\r")
    time.sleep(4.8 - ((time.time() - t0) % 4.8))

print(f"\nCollected {spot_bars_count} spot bars")

# ── SIMULATE SIGNALS ──
print("\nPhase 2: Simulating signals on real spot data...")
bars = load_spot_bars()
if not bars:
    print("❌ No bars collected!")
    sys.exit(1)

# Group by day
days = {}
for b in bars:
    day_str = datetime.fromtimestamp(b['t']).strftime('%Y-%m-%d')
    if day_str not in days:
        days[day_str] = []
    days[day_str].append(b)

print(f"Processing {len(days)} days of spot data ({len(bars)} bars)")

WINDOW = 60
sim_trades = []

for day_str in sorted(days.keys()):
    day_bars = days[day_str]
    if len(day_bars) < WINDOW:
        continue
    
    # Daily context
    dh = max(b['h'] for b in day_bars)
    dl = min(b['l'] for b in day_bars)
    dc = day_bars[-1]['c']
    daily_raw = {'high': dh, 'low': dl, 'close': dc, 'RSI': 50,
                 'change_abs': dc - day_bars[0]['o'], 'Recommend.All': 0,
                 'ATR': (dh - dl) * 0.6}
    
    for i in range(WINDOW, len(day_bars) - 5):
        ctx = day_bars[i-WINDOW:i+1]
        p = day_bars[i]['c']
        if not p or p <= 0:
            continue
        
        o = MiniOHLC(ctx)
        s = MiniState(o, p)
        daily = daily_context(p, daily_raw)
        
        regime, _ = detect_regime_raw(s)
        if not regime:
            continue
        
        cl = o.close_prices()
        if len(cl) < 20:
            continue
        
        # Run modules
        modules = [
            ("M1", *m1_momentum(o, daily)),
            ("M2", *m2_rsi(o, s.raw.get('RSI',50), daily)),
            ("M3", *m3_bb(o, s.raw.get("BB.upper",0) or 0, s.raw.get("BB.lower",0) or 0, p, daily)),
            ("M4", *m4_structure(o, daily)),
            ("M5", *m5_divergence(o, daily)),
            ("M6", *m6_patterns(o, daily)),
            ("M7", *m7_stats(o, daily)),
            ("M8", *m8_macd(o, 0, 0, daily)),
        ]
        active = [(n, d, c, r) for n, d, c, r in modules if d != 0]
        if not active:
            continue
        
        w_map = {"SQUEEZE":{"M1":0.20,"M2":0.10,"M3":0.25,"M4":0.05,"M5":0.05,"M6":0.10,"M7":0.10,"M8":0.15},
                 "TRENDING":{"M1":0.25,"M2":0.10,"M3":0.05,"M4":0.15,"M5":0.05,"M6":0.10,"M7":0.10,"M8":0.20},
                 "VOLATILE":{"M1":0.15,"M2":0.20,"M3":0.10,"M4":0.05,"M5":0.15,"M6":0.05,"M7":0.10,"M8":0.20},
                 "RANGING":{"M1":0.10,"M2":0.20,"M3":0.20,"M4":0.05,"M5":0.10,"M6":0.10,"M7":0.10,"M8":0.15}}
        w = w_map.get(regime, w_map["RANGING"])
        tw = sum(w.get(n, 0.10) for n, d, c, r in active)
        bv = sum(w.get(n, 0.10)*(c/100) for n,d,c,r in active if d==1)
        sv = sum(w.get(n, 0.10)*(c/100) for n,d,c,r in active if d==-1)
        bp = bv/tw*100 if tw > 0 else 0
        sp = sv/tw*100 if tw > 0 else 0
        direction = "BUY" if bp >= sp else ("SELL" if sp > bp else None)
        if not direction:
            continue
        
        atr_v = max(calc_atr(cl, o.highs(), o.lows()), 2.0)
        vs, vr = get_vol_state(o)
        is_ct = (direction=="BUY" and daily['bearish'] and daily['position']>20) or \
                (direction=="SELL" and not daily['bearish'] and daily['position']<80)
        
        conf = 70
        sl, tp, rr, sl_m, tp_m, atr_fire, sl_dist = compute_sl_tp(
            atr_v, regime, p, direction, vs, vr, daily['daily_atr'], conf, len([1 for n,d,c,r in active if d==(1 if direction=="BUY" else -1)]), is_ct)
        
        # Look ahead (max 30 bars)
        result = 'pending'
        for j in range(i+1, min(i+30, len(day_bars))):
            if direction == "BUY":
                if day_bars[j]['l'] <= sl:
                    result = 'sl'; break
                if day_bars[j]['h'] >= tp:
                    result = 'tp'; break
            else:
                if day_bars[j]['h'] >= sl:
                    result = 'sl'; break
                if day_bars[j]['l'] <= tp:
                    result = 'tp'; break
        
        if result == 'pending':
            continue
        
        mom = compute_momentum(cl, 5)
        rp_local = local_range_position(cl, 20)
        
        sim_trades.append({
            'direction': direction, 'entry_price': round(p, 2),
            'sl': round(sl, 2), 'tp': round(tp, 2),
            'result': result, 'exit_price': round((sl if result=='sl' else tp), 2),
            'pnl': round(abs((sl if result=='sl' else tp) - p), 2),
            'r_multiple': round(abs((sl if result=='sl' else tp) - p)/(sl_dist+SPREAD), 1),
            'signal_type': 'REAL_TRAINING',
            'regime': regime, 'session': session_adj(p),
            'confidence': conf,
            'daily_position': round(daily['position'], 1),
            'momentum': round(mom, 1), 'range_pos': round(rp_local, 1),
            'streak': 0,
            'sl_distance': round(sl_dist, 2),
            'day': day_str, 'entry_bar': i,
        })
        
        if len(sim_trades) % 25 == 0:
            print(f"  ... {len(sim_trades)} simulated trades", end="\r")

print(f"\n\nResults: {len(sim_trades)} simulated XAUUSD spot trades")

# ── CLEAR OLD GC=F TRAINING DATA ──
print("\nPhase 3: Replacing training data...")
with open(JOURNAL, 'r') as f:
    all_lines = f.readlines()

# Keep only LIVE trades (not TRAINING, not REAL_TRAINING yet)
live_lines = [l for l in all_lines if l.strip() and json.loads(l).get('signal_type') not in ('TRAINING', 'REAL_TRAINING', 'RESTORED')]
print(f"Keeping {len(live_lines)} live trade entries")

# Write back live trades only
with open(JOURNAL, 'w') as f:
    for l in live_lines:
        f.write(l)
    # Append new REAL training trades
    for t in sim_trades:
        f.write(json.dumps(t) + '\n')

print(f"Written {len(sim_trades)} new REAL training trades")
print(f"Total journal: {len(live_lines) + len(sim_trades)} entries")

# ── VERIFY ──
print("\nPhase 4: Verify learning system sees the right data...")
exec(bot_code.split('def main()')[0], namespace)
loaded = namespace['load_all_trades']()
print(f"Loaded {len(loaded)} trades by learning system")

# Test various scenarios
tests = [
    ('SELL', 'RANGING', 85, 'NY', 80, 80, 3),
    ('BUY', 'RANGING', 30, 'NY', 30, 20, 0),
    ('BUY', 'SQUEEZE', 50, 'NY', 50, 50, 0),
    ('BUY', 'SQUEEZE', 90, 'NY', 90, 90, 5),
    ('SELL', 'SQUEEZE', 85, 'NY', 80, 80, 3),
    ('SELL', 'SQUEEZE', 10, 'NY', 20, 10, -3),
]
ql = namespace['query_learning6']
print(f"\n{'Scenario':<55s} {'WR':>5s} {'n':>5s}")
print("-"*65)
for d, r, dp, sess, mom, rp, strk in tests:
    wr, n, feat = ql(loaded, d, r, dp, sess, mom, rp, strk)
    print(f'{d:5s} {r:10s} dp={dp:2.0f} {sess} | mom={mom:3.0f} rp={rp:3.0f} str={strk:+d}  → {wr:5.0f}% (n={n})')

print("\n✅ Training complete! Bot now uses REAL XAUUSD spot data.")
print(f"Saved to {JOURNAL}")
