import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATABASE_PATH = Path(os.getenv("AGENTBENCH_DB_PATH", "data/agentbench.db"))


def _connect() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            task_success_rate REAL NOT NULL,
            average_latency_ms REAL NOT NULL,
            total_tasks INTEGER NOT NULL,
            successful_tasks INTEGER NOT NULL,
            result_json TEXT NOT NULL
        )
    """)
    connection.commit()
    return connection


def save_run(result: dict[str, Any]) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        cursor = connection.execute(
            """INSERT INTO benchmark_runs
            (created_at, provider, model, task_success_rate, average_latency_ms,
             total_tasks, successful_tasks, result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                created_at,
                result["provider"],
                result["model"],
                result["task_success_rate"],
                result["average_latency_ms"],
                result["total_tasks"],
                result["successful_tasks"],
                json.dumps(result),
            ),
        )
        run_id = cursor.lastrowid
        connection.execute(
            "UPDATE benchmark_runs SET result_json = ? WHERE id = ?",
            (json.dumps({**result, "run_id": run_id, "created_at": created_at}), run_id),
        )
    return {**result, "run_id": run_id, "created_at": created_at}


def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 200))
    with _connect() as connection:
        rows = connection.execute(
            """SELECT id AS run_id, created_at, provider, model, task_success_rate,
                      average_latency_ms, total_tasks, successful_tasks
               FROM benchmark_runs ORDER BY id DESC LIMIT ?""",
            (safe_limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_run(run_id: int) -> dict[str, Any] | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT result_json FROM benchmark_runs WHERE id = ?", (run_id,)
        ).fetchone()
    return json.loads(row["result_json"]) if row else None
