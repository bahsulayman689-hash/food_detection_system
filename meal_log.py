"""
Meal history storage. Uses SQLite (food_log.db, created automatically next
to this file) so the app works with zero external setup.

You already use Supabase elsewhere — swapping this for a Supabase table
later is a small change: same four functions, same shape of data in/out,
just replace the sqlite3 calls with supabase-py calls against a `meals`
table with the same columns (timestamp, items_json, total_calories).
Worth doing once you want the log to sync across devices instead of
living on one machine.
"""
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "food_log.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            items_json TEXT NOT NULL,
            total_calories INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def log_meal(items: list[dict], total_calories: int, when: datetime | None = None) -> int:
    """
    items: list of {"food": str, "count": int, "calories": int} — one entry
    per detected food type in the photo (matches app.py's known_rows shape).
    Returns the new row's id.
    """
    when = when or datetime.now()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "INSERT INTO meals (timestamp, items_json, total_calories) VALUES (?, ?, ?)",
        (when.isoformat(), json.dumps(items), total_calories),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def delete_meal(meal_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM meals WHERE id = ?", (meal_id,))
    conn.commit()
    conn.close()


def get_today_total() -> int:
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT COALESCE(SUM(total_calories), 0) FROM meals WHERE timestamp >= ?",
        (today_start.isoformat(),),
    ).fetchone()
    conn.close()
    return row[0]


def get_daily_totals(days: int = 7) -> list[dict]:
    """Returns [{"date": "2026-09-25", "calories": 1840}, ...] for the last N days, oldest first."""
    start = (datetime.now() - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT timestamp, total_calories FROM meals WHERE timestamp >= ? ORDER BY timestamp",
        (start.isoformat(),),
    ).fetchall()
    conn.close()

    totals = {}
    cursor_date = start.date()
    end_date = datetime.now().date()
    while cursor_date <= end_date:
        totals[cursor_date.isoformat()] = 0
        cursor_date += timedelta(days=1)

    for timestamp_str, calories in rows:
        day = timestamp_str[:10]
        totals[day] = totals.get(day, 0) + calories

    return [{"date": d, "calories": c} for d, c in totals.items()]


def get_recent_meals(limit: int = 20) -> list[dict]:
    """Returns most-recent-first list of {"id", "timestamp", "items", "total_calories"}."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, timestamp, items_json, total_calories FROM meals ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [
        {
            "id": row_id,
            "timestamp": timestamp_str,
            "items": json.loads(items_json),
            "total_calories": total_calories,
        }
        for row_id, timestamp_str, items_json, total_calories in rows
    ]
