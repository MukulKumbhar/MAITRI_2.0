"""
MAITRI 2.0 — M6: Database Module
SQLite-backed telemetry logging with graceful schema migration.
"""

import datetime
import sqlite3

import pandas as pd

DB_FILE = "maitri_logs.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS stress_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT,
    face_emotion    TEXT,
    fused_emotion   TEXT,
    voice_state     TEXT,
    heart_rate      REAL,
    temperature     REAL,
    spo2            REAL,
    blink_rate      REAL,
    fatigue_label   TEXT,
    stress_score    REAL,
    stress_level    TEXT,
    alert_triggered TEXT,
    response_msg    TEXT
)
"""

# Columns added after initial schema — migrated gracefully
_MIGRATIONS = [
    ("fused_emotion",   "TEXT DEFAULT 'Neutral'"),
    ("spo2",            "REAL DEFAULT 98.0"),
    ("blink_rate",      "REAL DEFAULT 0.0"),
    ("fatigue_label",   "TEXT DEFAULT 'Normal'"),
    ("stress_level",    "TEXT DEFAULT 'NOMINAL'"),
]


def init_db() -> None:
    """Create tables and apply any pending column migrations."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(_SCHEMA)

    # Graceful migrations — skip if column already exists
    c.execute("PRAGMA table_info(stress_logs)")
    existing = {row[1] for row in c.fetchall()}
    for col_name, col_def in _MIGRATIONS:
        if col_name not in existing:
            c.execute(f"ALTER TABLE stress_logs ADD COLUMN {col_name} {col_def}")

    conn.commit()
    conn.close()


def log_event(
    face_emotion: str,
    fused_emotion: str,
    voice_state: str,
    heart_rate: float,
    temperature: float,
    spo2: float,
    blink_rate: float,
    fatigue_label: str,
    stress_score: float,
    stress_level: str,
    alert_triggered: str,
    response_msg: str,
) -> None:
    """Insert one telemetry record."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """INSERT INTO stress_logs
           (timestamp, face_emotion, fused_emotion, voice_state,
            heart_rate, temperature, spo2, blink_rate, fatigue_label,
            stress_score, stress_level, alert_triggered, response_msg)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            face_emotion,
            fused_emotion,
            voice_state,
            heart_rate,
            temperature,
            spo2,
            blink_rate,
            fatigue_label,
            stress_score,
            stress_level,
            alert_triggered,
            response_msg,
        ),
    )
    conn.commit()
    conn.close()


def get_logs() -> pd.DataFrame:
    """Return all logs, newest first."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(
        "SELECT * FROM stress_logs ORDER BY id DESC", conn
    )
    conn.close()
    return df


def get_recent_trend(n: int = 50) -> pd.DataFrame:
    """Return the last n records for trend charts."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(
        f"SELECT timestamp, stress_score, stress_level FROM stress_logs ORDER BY id DESC LIMIT {n}",
        conn,
    )
    conn.close()
    return df.iloc[::-1].reset_index(drop=True)  # chronological order

