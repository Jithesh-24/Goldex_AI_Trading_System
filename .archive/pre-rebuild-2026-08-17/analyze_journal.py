import json
from collections import Counter
from datetime import datetime

trades = []
with open('/home/jith/.hermes/profiles/trading/cron/output/trade_journal_ai.jsonl') as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            trades.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Line {i} parse error: {e}")

print(f"Total trades parsed: {len(trades)}")
if not trades:
    exit()

wins = sum(1 for t in trades if t.get('result') == 'TP')
losses = sum(1 for t in trades if t.get('result') == 'SL')
others = len(trades) - wins - losses
print(f"Wins (TP): {wins}")
print(f"Losses (SL): {losses}")
print(f"Other: {others}")
print(f"Win rate: {wins/len(trades)*100:.1f}%")

# Break down by source
sources = Counter(t.get('src', 'unknown') for t in trades)
print(f"\n=== TRADES BY SOURCE ===")
for src, count in sources.most_common():
    subset = [t for t in trades if t.get('src', 'unknown') == src]
    sub_wins = sum(1 for t in subset if t.get('result') == 'TP')
    print(f"  {src}: {count} trades, {sub_wins} wins ({sub_wins/len(subset)*100:.1f}% win rate)")

# Confidence stats
confs = [t.get('conf', 0) for t in trades]
print(f"\n=== CONFIDENCE STATS ===")
print(f"Mean: {sum(confs)/len(confs):.4f}")
print(f"Min: {min(confs):.4f}")
print(f"Max: {max(confs):.4f}")

# Win rate by confidence bucket
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    high_conf = [t for t in trades if t.get('conf', 0) >= threshold]
    if high_conf:
        hw = sum(1 for t in high_conf if t.get('result') == 'TP')
        print(f"  conf >= {threshold}: {len(high_conf)} trades, {hw} wins ({hw/len(high_conf)*100:.1f}%)")

# PnL stats
pnls = [t.get('pnl', 0) for t in trades]
print(f"\n=== PNL STATS ===")
print(f"Total PnL: {sum(pnls):.2f}")
print(f"Mean PnL: {sum(pnls)/len(pnls):.2f}")
print(f"Max win: {max(pnls):.2f}")
print(f"Max loss: {min(pnls):.2f}")
avg_w = [p for p in pnls if p > 0]
avg_l = [p for p in pnls if p < 0]
if avg_w:
    print(f"Avg win: {sum(avg_w)/len(avg_w):.2f}")
if avg_l:
    print(f"Avg loss: {sum(avg_l)/len(avg_l):.2f}")

# Time range
ts_list = [t.get('t', 0) for t in trades]
print(f"\n=== TIME RANGE ===")
print(f"First trade: {datetime.fromtimestamp(ts_list[0]).strftime('%Y-%m-%d %H:%M')}")
print(f"Last trade: {datetime.fromtimestamp(ts_list[-1]).strftime('%Y-%m-%d %H:%M')}")

# Direction breakdown
dirs = Counter(t.get('dir', '?') for t in trades)
print(f"\n=== DIRECTION BREAKDOWN ===")
for d, count in dirs.most_common():
    subset = [t for t in trades if t.get('dir') == d]
    sub_wins = sum(1 for t in subset if t.get('result') == 'TP')
    print(f"  {d}: {count} trades, {sub_wins} wins ({sub_wins/len(subset)*100:.1f}%)")

# Last 30 trades win rate
last30 = trades[-30:]
l30w = sum(1 for t in last30 if t.get('result') == 'TP')
print(f"\n=== LAST 30 TRADES ===")
print(f"Wins: {l30w}/{len(last30)} ({l30w/len(last30)*100:.1f}%)")

# Check for streaks
streak = 0
max_streak = 0
for t in trades:
    if t.get('result') == 'SL':
        streak += 1
        max_streak = max(max_streak, streak)
    else:
        streak = 0
print(f"\nMax consecutive losses: {max_streak}")

# Recent confidence trend
print(f"\n=== RECENT CONFIDENCE TREND (last 20) ===")
for t in trades[-20:]:
    ts_dt = datetime.fromtimestamp(t.get('t', 0)).strftime('%m-%d %H:%M')
    print(f"  {ts_dt} {t.get('dir','?'):4s} conf={t.get('conf',0):.3f} result={t.get('result','?')} pnl={t.get('pnl',0):+.2f}")

# Win rate over time (first half vs second half)
mid = len(trades) // 2
first_half = trades[:mid]
second_half = trades[mid:]
fw = sum(1 for t in first_half if t.get('result') == 'TP')
sw = sum(1 for t in second_half if t.get('result') == 'TP')
print(f"\n=== WIN RATE TREND ===")
print(f"First half ({len(first_half)} trades): {fw/len(first_half)*100:.1f}%")
print(f"Second half ({len(second_half)} trades): {sw/len(second_half)*100:.1f}%")

# Check confidence values that are identical (possible model stuck)
conf_counts = Counter(round(c, 4) for c in confs)
print(f"\n=== UNIQUE CONFIDENCE VALUES ===")
for cv, cnt in conf_counts.most_common(10):
    print(f"  conf={cv}: {cnt} times")
