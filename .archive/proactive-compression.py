#!/usr/bin/env python3
"""
Proactive Session Cleanup
=========================
Silently cleans up stale sessions and compressed data from state.db.
Compression itself is handled by the conversation loop inside Hermes.
This script keeps the database healthy so compression stays fast.

Runs every 5m via Hermes cron (no_agent, deliver:local — fully silent).
"""

import os
import sqlite3
import time

HERMES_HOME = os.path.expanduser("~/.hermes")
DB_PATH = os.path.join(HERMES_HOME, "state.db")


def get_all_dbs():
    """Return (name, path) for main + all profile state.dbs."""
    dbs = [("main", DB_PATH)]
    profiles_dir = os.path.join(HERMES_HOME, "profiles")
    if os.path.isdir(profiles_dir):
        for name in os.listdir(profiles_dir):
            pdb = os.path.join(profiles_dir, name, "state.db")
            if os.path.isfile(pdb):
                dbs.append((name, pdb))
    return dbs


def cleanup_db(db_path, name):
    """Silently clean up a database."""
    try:
        db = sqlite3.connect(db_path, timeout=10)
        db.execute("PRAGMA journal_mode=WAL")
    except Exception:
        return 0

    total = 0
    try:
        # 1. Remove abandoned sessions (ended >24h ago, <100 tokens)
        try:
            abandoned = db.execute("""
                SELECT id FROM sessions
                WHERE input_tokens < 100
                AND ended_at IS NOT NULL
                AND ended_at < ?
            """, (time.time() - 86400,)).fetchall()

            for (sid,) in abandoned:
                db.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
                db.execute("DELETE FROM sessions WHERE id = ?", (sid,))
            total += len(abandoned)
        except Exception:
            pass

        # 2. Remove orphaned compression locks
        try:
            db.execute("""
                DELETE FROM compression_locks
                WHERE session_id NOT IN (SELECT id FROM sessions)
            """)
        except Exception:
            pass

        # 3. Remove stale locks older than 1 hour
        try:
            db.execute("""
                DELETE FROM compression_locks
                WHERE expires_at < ?
            """, (time.time(),))
        except Exception:
            pass

        # 4. Trim bloated sessions (>500 messages) to last 200
        try:
            bloated = db.execute("""
                SELECT session_id, COUNT(*) as cnt
                FROM messages
                GROUP BY session_id
                HAVING cnt > 500
            """).fetchall()

            for (sid, cnt) in bloated:
                db.execute("""
                    DELETE FROM messages
                    WHERE session_id = ?
                    AND id NOT IN (
                        SELECT id FROM messages WHERE session_id = ?
                        ORDER BY timestamp DESC LIMIT 200
                    )
                """, (sid, sid))
                db.execute("UPDATE sessions SET message_count = 200 WHERE id = ?", (sid,))
            total += len(bloated)
        except Exception:
            pass

        db.commit()
    except Exception:
        pass
    finally:
        db.close()

    # VACUUM outside transaction (only if we did work)
    if total:
        try:
            vdb = sqlite3.connect(db_path, timeout=10)
            vdb.execute("VACUUM")
            vdb.close()
        except Exception:
            pass

    return total


def main():
    """Clean up all databases silently."""
    for name, db_path in get_all_dbs():
        if os.path.exists(db_path):
            cleanup_db(db_path, name)
    # Silent — empty stdout = no delivery


if __name__ == "__main__":
    main()
