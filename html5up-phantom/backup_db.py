"""Back up the live SQLite database to a timestamped copy, then prune old ones.

Uses sqlite3's own backup API rather than copying the file directly — a plain
file copy can grab a half-written page if something is writing to the database
at that exact moment, producing a corrupt backup. The backup API takes a
consistent snapshot regardless.

Run manually:
    python backup_db.py

Schedule it (keeps the last 14 backups by default):
    Linux/macOS (cron), daily at 2am — crontab -e:
        0 2 * * * cd /path/to/html5up-phantom && /path/to/venv/bin/python backup_db.py >> logs/backup.log 2>&1

    Windows — Task Scheduler, daily trigger, action:
        program: C:\\path\\to\\python.exe
        arguments: backup_db.py
        start in: G:\\code\\Data-Science-Professional-summary\\html5up-phantom
"""
import os
import sqlite3
import sys
from datetime import datetime

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, 'instance', 'users.db')
BACKUP_DIR = os.path.join(APP_DIR, 'backups')
KEEP_LAST = 14


def backup_database():
    if not os.path.exists(DB_PATH):
        print(f"No database found at {DB_PATH} — nothing to back up.")
        return

    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    dest_path = os.path.join(BACKUP_DIR, f'users-{stamp}.db')

    source = sqlite3.connect(DB_PATH)
    dest = sqlite3.connect(dest_path)
    with dest:
        source.backup(dest)
    dest.close()
    source.close()

    size_kb = os.path.getsize(dest_path) / 1024
    print(f"Backed up database to {dest_path} ({size_kb:.0f} KB)")

    prune_old_backups()


def prune_old_backups():
    backups = sorted(
        (f for f in os.listdir(BACKUP_DIR) if f.startswith('users-') and f.endswith('.db')),
        reverse=True,
    )
    for old in backups[KEEP_LAST:]:
        os.remove(os.path.join(BACKUP_DIR, old))
        print(f"Removed old backup: {old}")


if __name__ == '__main__':
    try:
        backup_database()
    except Exception as e:
        print(f"Backup failed: {e}", file=sys.stderr)
        sys.exit(1)
