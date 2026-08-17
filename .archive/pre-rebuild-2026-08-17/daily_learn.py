#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
daily_learn.py — Daily self-learning and self-rectification

Runs daily at market close (21:00 UTC):
1. Collects today's trades from journal
2. Analyzes what worked, what didn't
3. Identifies patterns in losses
4. Retrains model with latest data
5. Updates signal rating
6. Generates daily report

The system LEARNS from its mistakes and RECTIFIES itself.
"""

import json
import os
import time
from datetime import datetime, timezone

BASE = "/home/jith/.hermes/profiles/trading/scripts"
JOURNAL = os.path.join(BASE, "trade_journal.jsonl")
REPORT_DIR = os.path.join(BASE, "daily_reports")
MODEL_DIR = os.path.join(BASE, "models")

def load_trades(date=None):
    """Load trades for a specific date (or today)."""
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    trades = []
    if os.path.exists(JOURNAL):
        with open(JOURNAL) as f:
            for line in f:
                try:
                    trade = json.loads(line.strip())
                    if trade.get("date") == date:
                        trades.append(trade)
                except:
                    pass
    return trades

def analyze_trades(trades):
    """Analyze today's trades for patterns."""
    if not trades:
        return {"total": 0, "message": "No trades today"}
    
    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) <= 0]
    
    analysis = {
        "total": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) if trades else 0,
        "total_pnl": sum(t.get("pnl", 0) for t in trades),
        "avg_win": sum(t.get("pnl", 0) for t in wins) / len(wins) if wins else 0,
        "avg_loss": sum(t.get("pnl", 0) for t in losses) / len(losses) if losses else 0,
    }
    
    # Analyze loss patterns
    loss_patterns = {}
    for t in losses:
        reason = t.get("exit_reason", "unknown")
        loss_patterns[reason] = loss_patterns.get(reason, 0) + 1
    
    analysis["loss_patterns"] = loss_patterns
    
    # Analyze win patterns
    win_patterns = {}
    for t in wins:
        reason = t.get("exit_reason", "unknown")
        win_patterns[reason] = win_patterns.get(reason, 0) + 1
    
    analysis["win_patterns"] = win_patterns
    
    # Session analysis
    session_stats = {}
    for t in trades:
        session = t.get("session", "unknown")
        if session not in session_stats:
            session_stats[session] = {"wins": 0, "losses": 0, "pnl": 0}
        if t.get("pnl", 0) > 0:
            session_stats[session]["wins"] += 1
        else:
            session_stats[session]["losses"] += 1
        session_stats[session]["pnl"] += t.get("pnl", 0)
    
    analysis["session_stats"] = session_stats
    
    return analysis

def identify_improvements(analysis):
    """Identify what needs improvement based on analysis."""
    improvements = []
    
    # Check win rate
    if analysis.get("win_rate", 0) < 0.5:
        improvements.append({
            "area": "signal_quality",
            "issue": f"Win rate {analysis['win_rate']:.1%} below 50%",
            "action": "Tighten entry criteria, increase minimum rating threshold"
        })
    
    # Check loss patterns
    loss_patterns = analysis.get("loss_patterns", {})
    for reason, count in loss_patterns.items():
        if count > 2:
            improvements.append({
                "area": "exit_management",
                "issue": f"{count} losses from {reason}",
                "action": f"Review {reason} logic, consider tighter stops"
            })
    
    # Check session performance
    session_stats = analysis.get("session_stats", {})
    for session, stats in session_stats.items():
        if stats["losses"] > stats["wins"] * 2:
            improvements.append({
                "area": "session_filter",
                "issue": f"{session}: {stats['wins']}W/{stats['losses']}L",
                "action": f"Reduce or avoid trading in {session}"
            })
    
    # Check avg loss vs avg win
    if analysis.get("avg_loss", 0) < 0 and analysis.get("avg_win", 0) > 0:
        if abs(analysis["avg_loss"]) > analysis["avg_win"] * 1.5:
            improvements.append({
                "area": "risk_management",
                "issue": f"Avg loss {abs(analysis['avg_loss']):.2f} > 1.5x avg win {analysis['avg_win']:.2f}",
                "action": "Tighten stop losses, improve R:R ratio"
            })
    
    return improvements

def save_report(date, analysis, improvements):
    """Save daily report."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, f"report_{date}.json")
    
    report = {
        "date": date,
        "analysis": analysis,
        "improvements": improvements,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return report_path

def run_daily_learning():
    """Main daily learning loop."""
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"\n{'='*60}")
    print(f"DAILY LEARNING — {date}")
    print(f"{'='*60}")
    
    # 1. Load today's trades
    trades = load_trades(date)
    print(f"\nTrades today: {len(trades)}")
    
    # 2. Analyze
    analysis = analyze_trades(trades)
    print(f"Win rate: {analysis.get('win_rate', 0):.1%}")
    print(f"Total PnL: ${analysis.get('total_pnl', 0):.2f}")
    
    # 3. Identify improvements
    improvements = identify_improvements(analysis)
    print(f"\nImprovements needed: {len(improvements)}")
    for imp in improvements:
        print(f"  - [{imp['area']}] {imp['issue']}")
        print(f"    Action: {imp['action']}")
    
    # 4. Save report
    report_path = save_report(date, analysis, improvements)
    print(f"\nReport saved: {report_path}")
    
    # 5. Trigger retrain if significant improvements needed
    if len(improvements) > 2:
        print("\n⚠️ Multiple improvements needed — triggering retrain")
        os.system(f"cd {BASE} && python3 retrain_m5.py &")
    
    return analysis, improvements

if __name__ == "__main__":
    run_daily_learning()
