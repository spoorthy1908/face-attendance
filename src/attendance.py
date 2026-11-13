"""Milestone 5: store and read attendance records in a SQLite database.

Usage:  python src/attendance.py              # show today's attendance
        python src/attendance.py 2026-11-12   # show attendance for a date
"""
import sqlite3
import sys
from contextlib import closing
from datetime import date, datetime

from config import ATTENDANCE_DB


def connect():
    """Open the database, creating the file and table the first time."""
    ATTENDANCE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ATTENDANCE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            UNIQUE(name, date)
        )
    """)
    return conn


def mark_present(name):
    """Record that a person is present today. Returns True if newly marked,
    False if they were already marked today."""
    now = datetime.now()
    with closing(connect()) as conn, conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO attendance (name, date, time) VALUES (?, ?, ?)",
            (name, now.date().isoformat(), now.strftime("%H:%M:%S")),
        )
        return cursor.rowcount == 1


def get_attendance(day):
    """Return a list of (name, time) for everyone present on a day (YYYY-MM-DD)."""
    with closing(connect()) as conn:
        return conn.execute(
            "SELECT name, time FROM attendance WHERE date = ? ORDER BY time",
            (day,),
        ).fetchall()


def main():
    day = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    records = get_attendance(day)
    print(f"Attendance for {day}: {len(records)} present")
    for name, time in records:
        print(f"  {time}  {name.replace('_', ' ')}")


if __name__ == "__main__":
    main()
