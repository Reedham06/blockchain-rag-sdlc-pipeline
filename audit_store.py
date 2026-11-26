"""Local SQLite audit trail for masked pipeline inputs, evidence, decisions and approvals."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from typing import Any

from project_engine import sanitize_prompt_text


def _safe(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_prompt_text(value)
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_safe(item) for item in value]
    if isinstance(value, dict):
        secret_fields = ("api_key", "role_token", "mfa_code", "password", "secret")
        return {
            str(key): _safe(item) for key, item in value.items()
            if not any(field in str(key).lower() for field in secret_fields)
        }
    return value


def _database_path() -> str:
    configured = os.environ.get("BLOCKCHAIN_RAG_AUDIT_DB")
    path = configured or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "audit.sqlite3")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path


def create_run_id() -> str:
    return uuid.uuid4().hex


def record_event(run_id: str, event_type: str, actor: str, payload: dict) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id or ""):
        raise ValueError("A valid run ID is required for audit events.")
    event = {
        "event_id": uuid.uuid4().hex,
        "run_id": run_id,
        "event_type": event_type,
        "actor": _safe(actor or "Coordinator"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": _safe(payload),
    }
    try:
        retention_days = int(os.environ.get("BLOCKCHAIN_RAG_AUDIT_RETENTION_DAYS", "365"))
    except ValueError as exc:
        raise ValueError("BLOCKCHAIN_RAG_AUDIT_RETENTION_DAYS must be an integer.") from exc
    if not 1 <= retention_days <= 3650:
        raise ValueError("BLOCKCHAIN_RAG_AUDIT_RETENTION_DAYS must be between 1 and 3650.")
    with closing(sqlite3.connect(_database_path(), timeout=10)) as connection:
        with connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )"""
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_run ON audit_events(run_id, timestamp)")
            cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
            connection.execute("DELETE FROM audit_events WHERE timestamp < ?", (cutoff,))
            connection.execute(
                "INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?)",
                (event["event_id"], run_id, event_type, event["actor"], event["timestamp"],
                 json.dumps(event["payload"], ensure_ascii=False)),
            )
    return event


def get_run_events(run_id: str) -> list:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id or ""):
        raise ValueError("A valid run ID is required.")
    with closing(sqlite3.connect(_database_path(), timeout=10)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT event_id, event_type, actor, timestamp, payload_json FROM audit_events WHERE run_id = ? ORDER BY timestamp",
            (run_id,),
        ).fetchall()
    return [{
        "event_id": row["event_id"], "event_type": row["event_type"], "actor": row["actor"],
        "timestamp": row["timestamp"], "payload": json.loads(row["payload_json"]),
    } for row in rows]
