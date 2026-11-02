"""Thin SQLite wrapper. Autocommit plus explicit `transaction()` blocks, serialised by one lock."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path


class Database:
    def __init__(self, path: str | Path, schema: str):
        self.path = str(path)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        with self.lock:
            self.conn.executescript(schema)

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self.lock:
            return self.conn.execute(sql, params)

    def one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self.lock:
            return self.conn.execute(sql, params).fetchone()

    def all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    @contextmanager
    def transaction(self):
        """Re-entrant: a nested block joins the outer transaction, which commits or rolls back all of it."""
        with self.lock:
            if self.conn.in_transaction:
                yield self
                return
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            self.conn.execute("COMMIT")

    def close(self) -> None:
        with self.lock:
            self.conn.close()


AGENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    run_id TEXT,
    private INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, id);

CREATE TABLE IF NOT EXISTS workflows (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    kind TEXT,
    state TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_workflows_user ON workflows(user_id, updated_at);

CREATE TABLE IF NOT EXISTS proposals (
    id TEXT PRIMARY KEY,
    code TEXT NOT NULL,
    user_id TEXT NOT NULL,
    workflow_id TEXT NOT NULL,
    action TEXT NOT NULL,
    details TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    status TEXT NOT NULL,
    shown_msg_id INTEGER,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_proposals_user ON proposals(user_id, status);

CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    user_message_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS executions (
    idempotency_key TEXT PRIMARY KEY,
    approval_id TEXT NOT NULL,
    proposal_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    workflow_id TEXT NOT NULL,
    action TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    hold_id TEXT,
    result TEXT,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_executions_user ON executions(user_id, status);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    category TEXT NOT NULL,
    source TEXT NOT NULL,
    valid_until TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id, status);

CREATE TABLE IF NOT EXISTS booking_snapshots (
    booking_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL,
    snapshot TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS followups (
    booking_id TEXT NOT NULL,
    event TEXT NOT NULL,
    user_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    detail TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (booking_id, event)
);

CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    message_id INTEGER NOT NULL,
    channel TEXT NOT NULL,
    delivered INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
"""
