#!/usr/bin/env python3
"""Disk usage monitor — alerts when disk or RAM thresholds are crossed."""
import shutil
import json

THRESHOLD_DISK = 85    # alert if disk usage % exceeds this
THRESHOLD_RAM  = 90    # alert if RAM usage % exceeds this

errors = []

# Disk check
disk = shutil.disk_usage("/")
disk_pct = disk.used / disk.total * 100
if disk_pct > THRESHOLD_DISK:
    errors.append(f"⚠️ DISK: {disk_pct:.0f}% used ({disk.used//1024**3}G/{disk.total//1024**3}G)")

# RAM check
try:
    with open("/proc/meminfo") as f:
        meminfo = {k: int(v.split()[0]) for k, v in (line.split(":") for line in f if ":" in line)}
    total = meminfo.get("MemTotal", 0)
    avail = meminfo.get("MemAvailable", total)
    ram_pct = (1 - avail / total) * 100 if total else 0
    if ram_pct > THRESHOLD_RAM:
        errors.append(f"⚠️ RAM: {ram_pct:.0f}% used ({total//1024**3}G total, {avail//1024**3}G free)")
except Exception:
    pass

if errors:
    print("\n".join(errors))
