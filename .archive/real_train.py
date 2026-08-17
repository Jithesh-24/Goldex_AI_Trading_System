"""
Fast training: simulates signals on persisted XAUUSD spot OHLC bars
and writes REAL training data to the journal.
"""
import json, os, sys, time, math
from datetime import datetime, timezone

BASE = os.path.expanduser("~/.hermes/profiles/trading/cron/output")
BAR_FILE = f"{BASE}/spot_ohlc.jsonl"
JOURNAL = f"{BASE}/trade_journal.jsonl"

# Load REAL spot bars
bars = []
try:
    with open(BAR_FILE) as f:
        for line in f:
            if line.strip():
                bars.append(json.loads(line))
except: pass
print(f"Loaded {len(bars)} real XAUUSD spot bars")

if len(bars) < 10:
    print("⚠️ Too few bars. Starting bot for data collection...")

# Load bot modules
with open('/home/jith/.hermes/profiles/trading/scripts/hermes_scalper.py') as f:
    code = f.read()
ns = {}
exec(code.split('def main()')[0], ns)

calc_atr = ns['calc_atr']
calc_bb = ns['calc_bb']
detect_regime_raw = ns['detect_regime']
get_vol_state = ns['get_vol_state']
compute_sl_tp = ns['compute_sl_tp']
daily_context = ns['daily_context']
compute_momentum = ns['compute_momentum']
local_range_position = ns['local_range_position']
compute_streak = ns['compute_streak']
session_adj = ns['session_adj']
SPREAD = 0.2

# Module functions
m1_momentum = ns['m1_momentum']
m2_rsi = ns['m2_rsi']
m3_bb = ns['m3_bb']
m4_structure = ns['m4_structure']
m5_divergence = ns['m5_divergence']
m6_patterns = ns['m6_patterns']
m7_stats = ns['m7_stats']
m8_macd = ns['m8_macd']

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
    def __init__(self, o, p):
        self.ohlc = o; self.p = p; self.raw = {}
        cl = o.close_prices()
        if len(cl) >= 20:
            bb_u, bb_m, bb_l = calc_bb(cl)
            self.raw["BB.upper"] = bb_u; self.raw["BB.lower"] = bb_l
        if len(cl) >= 14:
            gains = losses = 0
            for i in range(-14, 0):
                diff = cl[i] - cl[i-1]
                if diff > 0: gains += diff; continue
                losses -= diff
            avg_g = gains / 14; avg_l = losses / 14 if losses else 0.01
            self.raw['RSI'] = 100 - 100 / (1 + avg_g / avg_l)

# Simulate signals on the bars
WINDOW = 40
trades = []

# Group by date
from collections import defaultdict
days = defaultdict(list)
for b in bars:
    ds = datetime.fromtimestamp(b.get('t', b.get('minute',0) * 60)).strftime('%Y-%m-%d')
    days[ds].append(b)

for day_str, day_bars in sorted(days.items()):
    if len(day_bars) < WINDOW:
        continue
    dh = max(b['h'] for b in day_bars)
    dl = min(b['l'] for b in day_bars)
    dc = day_bars[-1]['c']
    dr = {'high': dh, 'low': dl, 'close': dc, 'RSI': 50,
          'change_abs': dc - day_bars[0]['o'], 'Recommend.All': 0,
          'ATR': max((dh - dl) * 0.6, 10)}

    for i in range(WINDOW, len(day_bars) - 5):
        ctx = day_bars[i-WINDOW:i+1]
        p = day_bars[i]['c']
        if not p or p <= 0: continue

        o = MiniOHLC(ctx)
        s = MiniState(o, p)
        daily = daily_context(p, dr)

        regime, _ = detect_regime_raw(s)
        if not regime: continue

        cl = o.close_prices()
        if len(cl) < 15: continue

        modules = [("M1", *m1_momentum(o, daily)), ("M2", *m2_rsi(o, s.raw.get('RSI',50), daily)),
                   ("M3", *m3_bb(o, s.raw.get("BB.upper",0) or 0, s.raw.get("BB.lower",0) or 0, p, daily)),
                   ("M4", *m4_structure(o, daily)), ("M5", *m5_divergence(o, daily)),
                   ("M6", *m6_patterns(o, daily)), ("M7", *m7_stats(o, daily)),
                   ("M8", *m8_macd(o, 0, 0, daily))]
        active = [(n, d, c, r) for n, d, c, r in modules if d != 0]
        if not active: continue

        tw = sum(w.get(n, 0.10) for n, d, c, r in active)
        bv = sum(w.get(n, 0.10)*(c/100) for n,d,c,r in active if d==1)
        sv = sum(w.get(n, 0.10)*(c/100) for n,d,c,r in active if d==-1)
        bp = bv/tw*100 if tw > 0 else 0
        sp = sv/tw*100 if tw > 0 else 0
        direction = "BUY" if bp >= sp else ("SELL" if sp > bp else None)
        if not direction: continue

        atr_v = max(calc_atr(cl, o.highs(), o.lows()), 2.0)
        vs, vr = get_vol_state(o)
        is_ct = (direction=="BUY" and daily['bearish'] and daily['position']>20) or \
                (direction=="SELL" and not daily['bearish'] and daily['position']<80)

        sl, tp, rr, sl_m, tp_m, atr_fire, sl_dist = compute_sl_tp(
            atr_v, regime, p, direction, vs, vr, daily['daily_atr'],
            70, len([1 for n,d,c,r in active if d==(1 if direction=="BUY" else -1)]), is_ct)

        # Check outcome (max 30 bars ahead)
        result = 'pending'
        for j in range(i+1, min(i+30, len(day_bars))):
            if direction == "BUY":
                if day_bars[j]['l'] <= sl: result = 'sl'; break
                if day_bars[j]['h'] >= tp: result = 'tp'; break
            else:
                if day_bars[j]['h'] >= sl: result = 'sl'; break
                if day_bars[j]['l'] <= tp: result = 'tp'; break
        if result == 'pending': continue

        mom = compute_momentum(cl, 5)
        rp = local_range_position(cl, 20)

        trades.append({
            'direction': direction, 'entry_price': round(p, 2),
            'sl': round(sl, 2), 'tp': round(tp, 2),
            'result': result,
            'exit_price': round((sl if result=='sl' else tp), 2),
            'pnl': round(abs((sl if result=='sl' else tp) - p), 2),
            'r_multiple': round(abs((sl if result=='sl' else tp) - p)/(sl_dist+SPREAD), 1),
            'signal_type': 'REAL_TRAINING',
            'regime': regime, 'session': session_adj(p),
            'confidence': 70,
            'daily_position': round(daily['position'], 1),
            'momentum': round(mom, 1), 'range_pos': round(rp, 1),
            'streak': 0, 'sl_distance': round(sl_dist, 2),
            'day': day_str, 'entry_bar': i,
        })

print(f"Simulated {len(trades)} trades from {len(bars)} spot bars")

# Merge into journal (dedup by day+bar+direction)
existing = {}
try:
    with open(JOURNAL) as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                k = f"{t.get('day','')}_{t.get('entry_bar',0)}_{t.get('direction','')}"
                existing[k] = line
except: pass

print(f"Existing journal entries: {len(existing)}")

added = 0
for t in trades:
    k = f"{t['day']}_{t['entry_bar']}_{t['direction']}"
    if k not in existing:
        existing[k] = json.dumps(t)
        added += 1

with open(JOURNAL, 'w') as f:
    for line in existing.values():
        f.write(line + '\n')

print(f"Added {added} new trades from spot bars")
print(f"Total journal: {len(existing)} entries")

# Verify learning system
exec(code.split('def main()')[0], ns)
all_t = ns['load_all_trades']()
print(f"Learning system loads: {len(all_t)} trades")

ql = ns['query_learning6']
tests = [
    ('SELL', 'RANGING', 85, 'NY', 80, 80, 3),
    ('BUY', 'RANGING', 30, 'NY', 30, 20, 0),
    ('BUY', 'SQUEEZE', 50, 'NY', 50, 50, 0),
    ('BUY', 'SQUEEZE', 90, 'NY', 90, 90, 5),
    ('SELL', 'SQUEEZE', 85, 'NY', 80, 80, 3),
    ('SELL', 'SQUEEZE', 10, 'NY', 20, 10, -3),
]
print(f"\n{'Scenario':<55s} {'WR':>5s} {'n':>5s}")
print("-"*65)
for d, r, dp, sess, mom, rp, strk in tests:
    wr, n, feat = ql(all_t, d, r, dp, sess, mom, rp, strk)
    print(f'{d:5s} {r:10s} dp={dp:2.0f} {sess} | mom={mom:3.0f} rp={rp:3.0f} str={strk:+d}  → {wr:5.0f}% (n={n})')
