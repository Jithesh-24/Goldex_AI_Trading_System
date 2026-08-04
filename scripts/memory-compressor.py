#!/usr/bin/env python3
"""
Hermes Memory Compressor — auto-compress MEMORY.md and USER.md when near limits.

Thresholds (80% of configured limits):
  MEMORY.md: 1760 bytes (of 2200 limit)
  USER.md:   1100 bytes (of 1375 limit)

Strategies:
  1. Remove duplicate lines (exact + fuzzy)
  2. Merge similar entries (shared prefix/suffix)
  3. Shorten verbose entries (strip filler words, collapse whitespace)
  4. Remove stale/redundant phrases

Silent when nothing to compress. Logs actions to self-heal.log.
"""

import os
import re
import sys
from pathlib import Path
from difflib import SequenceMatcher

HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
LOG_FILE = HERMES_HOME / "logs" / "self-heal.log"

# Thresholds: 80% of configured limits
MEMORY_LIMIT = 2200

# Collect all memory directories: main + all profiles
MEMORY_DIRS = [HERMES_HOME / "memories"]
profiles_dir = HERMES_HOME / "profiles"
if profiles_dir.exists():
    for profile_dir in profiles_dir.iterdir():
        if profile_dir.is_dir() and (profile_dir / "memories").exists():
            MEMORY_DIRS.append(profile_dir / "memories")
MEMORY_THRESHOLD = int(MEMORY_LIMIT * 0.80)  # 1760

USER_LIMIT = 1375
USER_THRESHOLD = int(USER_LIMIT * 0.80)  # 1100

# Filler words/phrases to strip during compression
FILLER_PATTERNS = [
    r"\bthis is\b",
    r"\bthat is\b",
    r"\bit is\b",
    r"\bthe user\b",
    r"\bthe agent\b",
    r"\bhas been\b",
    r"\bcan be\b",
    r"\bwill be\b",
    r"\bshould be\b",
    r"\bmust be\b",
    r"\bwhich is\b",
    r"\bwho is\b",
    r"\bwhere is\b",
    r"\bwhen is\b",
    r"\bwas previously\b",
    r"\bpreviously was\b",
    r"\bin order to\b",
    r"\bfor the purpose of\b",
    r"\bdue to the fact that\b",
    r"\bat this point in time\b",
    r"\bfor the time being\b",
]


def log(msg: str):
    """Append to self-heal.log."""
    from datetime import datetime
    ts = datetime.now().isoformat(timespec="seconds")
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"[{ts}] MEMCOMP {msg}\n")


def read_file(path: Path) -> str | None:
    """Read file, return None if missing/empty."""
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8").strip()
    return content if content else None


def write_file(path: Path, content: str):
    """Write content atomically (write-to-temp + rename)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.rename(path)


def remove_exact_duplicates(lines: list[str]) -> list[str]:
    """Remove exact duplicate lines, preserving order."""
    seen = set()
    result = []
    for line in lines:
        normalized = line.strip().lower()
        if normalized and normalized in seen:
            continue
        if normalized:
            seen.add(normalized)
        result.append(line)
    return result


def remove_fuzzy_duplicates(lines: list[str], threshold: float = 0.85) -> list[str]:
    """Remove near-duplicate lines (fuzzy match)."""
    if len(lines) <= 1:
        return lines

    result = []
    seen_normalized = []

    for line in lines:
        normalized = line.strip().lower()
        if not normalized:
            result.append(line)
            continue

        is_dup = False
        for seen in seen_normalized:
            ratio = SequenceMatcher(None, normalized, seen).ratio()
            if ratio >= threshold:
                is_dup = True
                break

        if not is_dup:
            result.append(line)
            seen_normalized.append(normalized)

    return result


def strip_filler(text: str) -> str:
    """Remove filler words and phrases."""
    result = text
    for pattern in FILLER_PATTERNS:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    # Collapse multiple spaces
    result = re.sub(r"  +", " ", result)
    return result.strip()


def collapse_whitespace(text: str) -> str:
    """Normalize whitespace: collapse runs, trim lines."""
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            result.append(stripped)
        elif result and result[-1] != "":
            result.append("")  # keep single blank line
    return "\n".join(result)


def merge_similar_entries(lines: list[str]) -> list[str]:
    """Merge entries that share >70% prefix."""
    if len(lines) <= 2:
        return lines

    result = []
    skip = set()

    for i, line in enumerate(lines):
        if i in skip or not line.strip():
            result.append(line)
            continue

        merged = False
        for j in range(i + 1, len(lines)):
            if j in skip or not lines[j].strip():
                continue

            # Check if one is a prefix/subset of the other
            shorter = min(line, lines[j], key=len)
            longer = max(line, lines[j], key=len)

            if shorter.lower() in longer.lower():
                skip.add(j)
                merged = True
                break

            # Check similarity
            ratio = SequenceMatcher(None, line.lower(), lines[j].lower()).ratio()
            if ratio > 0.70:
                # Keep the longer (more informative) one
                if len(lines[j]) > len(line):
                    skip.add(i)
                    result.append(lines[j])
                else:
                    skip.add(j)
                merged = True
                break

        if not merged:
            result.append(line)

    return result


def compress_memory(content: str) -> str:
    """Apply all compression strategies to memory content."""
    result = content

    # Step 1: Normalize whitespace
    result = collapse_whitespace(result)

    # Step 2: Split into lines, remove exact dupes
    lines = result.split("\n")
    lines = remove_exact_duplicates(lines)

    # Step 3: Remove fuzzy dupes
    lines = remove_fuzzy_duplicates(lines, threshold=0.85)

    # Step 4: Merge similar entries
    lines = merge_similar_entries(lines)

    # Step 5: Strip filler from each line
    lines = [strip_filler(line) for line in lines]

    # Step 6: Final cleanup
    result = "\n".join(lines)
    result = collapse_whitespace(result)

    return result


def process_file(path: Path, limit: int, threshold: int, label: str) -> bool:
    """Process one memory file. Returns True if compressed."""
    content = read_file(path)
    if content is None:
        return False

    current_size = len(content.encode("utf-8"))

    if current_size < threshold:
        log(f"{label}: {current_size}B < {threshold}B threshold — OK")
        return False

    log(f"{label}: {current_size}B >= {threshold}B threshold — compressing")

    compressed = compress_memory(content)
    new_size = len(compressed.encode("utf-8"))

    if new_size >= current_size:
        log(f"{label}: compression had no effect ({current_size}B → {new_size}B) — skipping")
        return False

    # Safety: don't compress below 50% (something might be wrong)
    min_size = int(limit * 0.50)
    if new_size < min_size:
        log(f"{label}: compressed too aggressively ({new_size}B < {min_size}B) — skipping")
        return False

    write_file(path, compressed)
    saved = current_size - new_size
    pct = (saved / current_size) * 100
    log(f"{label}: compressed {current_size}B → {new_size}B (saved {saved}B, {pct:.0f}%)")
    return True


def main():
    """Main entry point — compress across all profiles."""
    any_compressed = False

    for mem_dir in MEMORY_DIRS:
        profile_name = str(mem_dir.parent.name) if mem_dir.parent.name != ".hermes" else "main"
        mem_dir.mkdir(parents=True, exist_ok=True)

        memory_file = mem_dir / "MEMORY.md"
        user_file = mem_dir / "USER.md"

        if process_file(memory_file, MEMORY_LIMIT, MEMORY_THRESHOLD, f"[{profile_name}] MEMORY.md"):
            any_compressed = True
        if process_file(user_file, USER_LIMIT, USER_THRESHOLD, f"[{profile_name}] USER.md"):
            any_compressed = True

    if not any_compressed:
        # Silent — nothing to compress
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
