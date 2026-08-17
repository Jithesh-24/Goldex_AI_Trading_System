"""
Train the learning system with historical GC=F data.
Runs the bot's 8 modules on every 1-minute bar, simulates SL/TP outcomes,
and populates the learning database with thousands of labeled trades.
"""
import json, sys, math, random
from datetime import datetime, timezone

# Load training data
data = json.load(open('/tmp/gcf_training_data.json'))
bars_1m = data['1m']
print(f'Loaded {len(bars_1m)} 1m bars')

# Import bot modules (extract from scalper)
sys.path.insert(0, '/home/jith/.hermes/profiles/trading/scripts')
bot_file = '/home/jith/.hermes/profiles/trading/scripts/hermes_scalper.py'

# Read bot code and extract needed functions
with open(bot_file) as f:
    bot_code = f.read()

# Extract everything before def main()
# The bot has all modules defined before main
namespace = {}
exec(bot_code.split('def main()')[0], namespace)

# Also need the constants
SPREAD = 0.2
MIN_SL = 3.0

# Access needed functions
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
MOM_TH = namespace['MOM_TH']
POS_TH = namespace['POS_TH']
STR_TH = namespace['STR_TH']

# Module functions
m1_momentum = namespace['m1_momentum']
m2_rsi = namespace['m2_rsi']
m3_bb = namespace['m3_bb']
m4_structure = namespace['m4_structure']
m5_divergence = namespace['m5_divergence']
m6_patterns = namespace['m6_patterns']
m7_stats = namespace['m7_stats']
m8_macd = namespace['m8_macd']

# Build a simple State-like object for the detectors
class TickManager:
    """Mini version of the bot's OHLC that works with pre-loaded bars."""
    def __init__(self, bars_window):
        self.bars = bars_window
    
    def close_prices(self, n=0):
        arr = [b['c'] for b in self.bars]
        return arr[-n:] if n and arr else arr
    
    def highs(self, n=0):
        arr = [b['h'] for b in self.bars]
        return arr[-n:] if n and arr else arr
    
    def lows(self, n=0):
        arr = [b['l'] for b in self.bars]
        return arr[-n:] if n and arr else arr

class SimState:
    """Simulates the bot's State object."""
    def __init__(self, bars_window):
        self.ohlc = TickManager(bars_window)
        self.p = bars_window[-1]['c']
        self.raw = {}
        self.ck = len(bars_window)
        # Compute some raw values from bars
        cl = self.ohlc.close_prices()
        if len(cl) >= 20:
            bb_u, bb_m, bb_l = calc_bb(cl)
            self.raw['BB.upper'] = bb_u
            self.raw['BB.lower'] = bb_l
        # RSI approximation from close prices
        if len(cl) >= 14:
            rsi = calc_rsi(cl)
            self.raw['RSI'] = rsi

def calc_rsi(cl, period=14):
    """Simple RSI calculation."""
    if len(cl) < period + 1:
        return 50
    gains = losses = 0
    for i in range(-period, 0):
        diff = cl[i] - cl[i-1]
        if diff > 0: gains += diff
        else: losses -= diff
    avg_g = gains / period
    avg_l = losses / period if losses else 0.01
    return 100 - 100 / (1 + avg_g / avg_l)

# ── MAIN TRAINING LOOP ──
simulated_trades = []
window = 120  # bars of context for modules

# Group bars into days
days = {}
for b in bars_1m:
    day_str = datetime.fromtimestamp(b['t']).strftime('%Y-%m-%d')
    if day_str not in days:
        days[day_str] = []
    days[day_str].append(b)

print(f'Processing {len(days)} days of data...')

MIN_SL_DYNAMIC = 3.0  # Will be adjusted per day

for day_str in sorted(days.keys()):
    day_bars = days[day_str]
    if len(day_bars) < window:
        continue
    
    # Compute daily context for this day
    day_high = max(b['h'] for b in day_bars)
    day_low = min(b['l'] for b in day_bars)
    day_close = day_bars[-1]['c']
    day_rng = day_high - day_low
    
    # Simulated daily data (as fetch_daily would return)
    daily_raw = {
        'high': day_high, 'low': day_low, 'close': day_close,
        'RSI': 50, 'change_abs': day_close - day_bars[0]['o'],
        'Recommend.All': 0, 'ATR': day_rng * 0.6  # simplified
    }
    
    print(f'\n{day_str}: {len(day_bars)} bars, range ${day_low:.2f}-${day_high:.2f}')
    
    # For each potential signal point
    for i in range(window, len(day_bars) - 5):  # leave 5 bars ahead for SL/TP check
        # Build context window (last 120 bars)
        context = day_bars[i-window:i+1]
        p = day_bars[i]['c']
        if not p or p <= 0:
            continue
        
        # Build state
        s = SimState(context)
        
        # Daily context
        daily = daily_context(p, daily_raw)
        
        # Get regime
        regime, _ = detect_regime_raw(s)
        if not regime:
            continue
        
        # Run modules
        cl = s.ohlc.close_prices()
        if len(cl) < 20:
            continue
        
        modules = [
            ("M1", *m1_momentum(s.ohlc, daily)),
            ("M2", *m2_rsi(s.ohlc, s.raw.get('RSI',50), daily)),
            ("M3", *m3_bb(s.ohlc, s.raw.get("BB.upper",0) or 0, s.raw.get("BB.lower",0) or 0, p, daily)),
            ("M4", *m4_structure(s.ohlc, daily)),
            ("M5", *m5_divergence(s.ohlc, daily)),
            ("M6", *m6_patterns(s.ohlc, daily)),
            ("M7", *m7_stats(s.ohlc, daily)),
            ("M8", *m8_macd(s.ohlc, 0, 0, daily)),
        ]
        active = [(n, d, c, r) for n, d, c, r in modules if d != 0]
        if not active:
            continue
        
        # Regime-weighted voting
        w_map = {"SQUEEZE":{"M1":0.20,"M2":0.10,"M3":0.25,"M4":0.05,"M5":0.05,"M6":0.10,"M7":0.10,"M8":0.15},
                 "TRENDING":{"M1":0.25,"M2":0.10,"M3":0.05,"M4":0.15,"M5":0.05,"M6":0.10,"M7":0.10,"M8":0.20},
                 "VOLATILE":{"M1":0.15,"M2":0.20,"M3":0.10,"M4":0.05,"M5":0.15,"M6":0.05,"M7":0.10,"M8":0.20},
                 "RANGING":{"M1":0.10,"M2":0.20,"M3":0.20,"M4":0.05,"M5":0.10,"M6":0.10,"M7":0.10,"M8":0.15}}
        w = w_map.get(regime, w_map["RANGING"])
        tw = sum(w.get(n, 0.10) for n, d, c, r in active)
        bv = sum(w.get(n, 0.10) * (c/100) for n, d, c, r in active if d == 1)
        sv = sum(w.get(n, 0.10) * (c/100) for n, d, c, r in active if d == -1)
        bp = bv / tw * 100 if tw > 0 else 0
        sp = sv / tw * 100 if tw > 0 else 0
        
        direction = "BUY" if bp >= sp else ("SELL" if sp > bp else None)
        if not direction:
            continue
        
        # Check if we already have a trade in this direction active
        if any(t['entry_bar'] == i for t in simulated_trades[-3:] if t.get('day') == day_str):
            continue
        
        reasons = [r for n, d, c, r in active if d == (1 if direction == "BUY" else -1)]
        if len(reasons) < 1:
            continue
        
        # Compute SL/TP
        atr_v = max(calc_atr(cl, s.ohlc.highs(), s.ohlc.lows()), 2.0)
        vol_state, vol_ratio = get_vol_state(s.ohlc)
        is_ct = (direction == "BUY" and daily['bearish'] and daily['position'] > 20) or \
                (direction == "SELL" and not daily['bearish'] and daily['position'] < 80)
        
        # Default confidence for simulated trades (not yet learned)
        conf = 70
        
        sl, tp, rr, sl_m, tp_m, atr_fire, sl_dist = compute_sl_tp(
            atr_v, regime, p, direction, vol_state, vol_ratio, daily['daily_atr'],
            conf, len(reasons), is_ct)
        
        tp_dist = abs(p - tp)
        total_sl = sl_dist + 0.2
        
        # Look ahead to check outcome (max 60 bars = ~1 hour)
        result = 'pending'
        exit_price = 0
        hit_bar = 0
        
        for j in range(i+1, min(i+60, len(day_bars))):
            if direction == "BUY":
                if day_bars[j]['l'] <= sl:
                    result = 'sl'
                    exit_price = sl
                    hit_bar = j
                    break
                if day_bars[j]['h'] >= tp:
                    result = 'tp'
                    exit_price = tp
                    hit_bar = j
                    break
            else:  # SELL
                if day_bars[j]['h'] >= sl:
                    result = 'sl'
                    exit_price = sl
                    hit_bar = j
                    break
                if day_bars[j]['l'] <= tp:
                    result = 'tp'
                    exit_price = tp
                    hit_bar = j
                    break
        
        if result == 'pending':
            continue  # Not resolved within 60 bars
        
        # Compute dimensions
        mom = compute_momentum(cl, 5)
        rp_local = local_range_position(cl, 20)
        
        trade = {
            'direction': direction,
            'entry_price': round(p, 2),
            'sl': round(sl, 2),
            'tp': round(tp, 2),
            'result': result,
            'exit_price': round(exit_price, 2),
            'pnl': round(abs(exit_price - p), 2),
            'r_multiple': round(abs(exit_price - p) / total_sl, 1),
            'signal_type': 'TRAINING',
            'regime': regime,
            'session': 'BACKTEST',
            'confidence': conf,
            'daily_position': round(daily['position'], 1),
            'momentum': round(mom, 1),
            'range_pos': round(rp_local, 1),
            'streak': 0,
            'sl_distance': round(sl_dist, 2),
            'day': day_str,
            'entry_bar': i,
        }
        simulated_trades.append(trade)
        
        if len(simulated_trades) % 100 == 0:
            print(f'  ... {len(simulated_trades)} simulated trades so far')

# Save training data
json.dump(simulated_trades, open('/tmp/training_trades.json','w'), indent=2)

# Stats
wins = sum(1 for t in simulated_trades if t['result'] == 'tp')
losses = sum(1 for t in simulated_trades if t['result'] == 'sl')
print(f'\n{"="*50}')
print(f'Training complete: {len(simulated_trades)} simulated trades')
print(f'Wins: {wins} ({wins*100//max(1,len(simulated_trades))}%)')
print(f'Losses: {losses} ({losses*100//max(1,len(simulated_trades))}%)')
print(f'Net PnL: ${sum(t["pnl"] for t in simulated_trades if t["result"]=="tp") - sum(t["pnl"] for t in simulated_trades if t["result"]=="sl"):.1f}')

# Per-regime breakdown
for regime in ['SQUEEZE','RANGING','TRENDING','VOLATILE']:
    reg_trades = [t for t in simulated_trades if t['regime'] == regime]
    if reg_trades:
        rw = sum(1 for t in reg_trades if t['result'] == 'tp')
        print(f'  {regime}: {len(reg_trades)} trades, {rw*100//len(reg_trades)}% WR')
