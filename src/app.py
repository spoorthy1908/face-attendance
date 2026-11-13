"""Milestone 6: web dashboard to view attendance in the browser.

Usage:  python src/app.py
Then open http://127.0.0.1:5001 in your browser. Press Ctrl + C to stop.
"""
import csv
import io
from contextlib import closing
from datetime import date

from flask import Flask, Response, render_template, request

from attendance import connect, get_attendance
from config import EMBEDDINGS_DIR

app = Flask(__name__)


@app.template_filter("pretty")
def pretty(name):
    """Turn 'Ammu_K' into 'Ammu K' for display."""
    return name.replace("_", " ")


def selected_day():
    """Read ?date=YYYY-MM-DD from the URL, falling back to today."""
    value = request.args.get("date", "")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        return date.today().isoformat()


def registered_people():
    return sorted(path.stem for path in EMBEDDINGS_DIR.glob("*.npy"))


def build_register(day):
    """Return one row per registered person: (name, time or None)."""
    times = dict(get_attendance(day))
    people = set(registered_people()) | set(times)   # include anyone later unregistered
    return sorted(((name, times.get(name)) for name in people),
                  key=lambda row: (row[1] is None, row[1] or "", row[0]))


def recent_days(limit=10):
    with closing(connect()) as conn:
        return conn.execute(
            "SELECT date, COUNT(*) FROM attendance GROUP BY date ORDER BY date DESC LIMIT ?",
            (limit,),
        ).fetchall()


@app.route("/")
def dashboard():
    day = selected_day()
    register = build_register(day)
    present = sum(1 for _, time in register if time)
    return render_template(
        "dashboard.html",
        day=day,
        day_label=date.fromisoformat(day).strftime("%A %-d %B %Y"),
        register=register,
        present=present,
        total=len(register),
        history=recent_days(),
        today=date.today().isoformat(),
    )


@app.route("/export.csv")
def export_csv():
    day = selected_day()
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["name", "date", "status", "time"])
    for name, time in build_register(day):
        writer.writerow([pretty(name), day, "present" if time else "absent", time or ""])
    return Response(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=attendance-{day}.csv"},
    )


if __name__ == "__main__":
    # Port 5001 because macOS uses port 5000 for AirPlay Receiver
    app.run(host="127.0.0.1", port=5001, debug=True)
