"""SQLite storage in DATA_DIR: append-only hash-chained audit log, sandbox photos, LLM call log."""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

GENESIS_HASH = "0" * 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    sandbox_id TEXT NOT NULL,
    loan_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    ai_recommendation TEXT,
    model_version TEXT,
    inputs_hash TEXT NOT NULL,
    comment TEXT NOT NULL DEFAULT '',
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS audit_sandbox_loan ON audit (sandbox_id, loan_id);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;

CREATE TABLE IF NOT EXISTS photos (
    photo_id TEXT PRIMARY KEY,
    sandbox_id TEXT NOT NULL,
    loan_id TEXT NOT NULL,
    server_ts TEXT NOT NULL,
    client_ts TEXT,
    lat REAL,
    lon REAL,
    result_json TEXT NOT NULL,
    image_file TEXT
);
CREATE INDEX IF NOT EXISTS photos_sandbox_loan ON photos (sandbox_id, loan_id);

CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    loan_id TEXT NOT NULL,
    model TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    ok INTEGER NOT NULL,
    error TEXT
);
"""

AUDIT_FIELDS = ("ts", "sandbox_id", "loan_id", "actor", "action", "ai_recommendation",
                "model_version", "inputs_hash", "comment")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode()


def row_hash(prev_hash: str, row: dict) -> str:
    """hash = sha256(prev_hash + canonical(row))"""
    return sha256_hex(prev_hash.encode() + canonical({k: row[k] for k in AUDIT_FIELDS}))


class Database:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.photo_dir = data_dir / "photos"
        self.photo_dir.mkdir(parents=True, exist_ok=True)
        self.path = data_dir / "maqsad.sqlite3"
        self._write_lock = threading.Lock()
        with self.conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        c.row_factory = sqlite3.Row
        try:
            yield c
        finally:
            c.close()

    # ---------- audit (append-only, hash chain) ----------

    def audit_append(self, *, sandbox_id: str, loan_id: str, actor: str, action: str,
                     ai_recommendation: Optional[str], model_version: Optional[str],
                     inputs_hash: str, comment: str = "") -> int:
        row = {"ts": utc_now(), "sandbox_id": sandbox_id, "loan_id": loan_id, "actor": actor,
               "action": action, "ai_recommendation": ai_recommendation,
               "model_version": model_version, "inputs_hash": inputs_hash, "comment": comment}
        with self._write_lock, self.conn() as c:
            c.execute("BEGIN IMMEDIATE")
            last = c.execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
            prev = last["hash"] if last else GENESIS_HASH
            h = row_hash(prev, row)
            cur = c.execute(
                f"INSERT INTO audit ({', '.join(AUDIT_FIELDS)}, prev_hash, hash) "
                f"VALUES ({', '.join('?' * (len(AUDIT_FIELDS) + 2))})",
                [row[k] for k in AUDIT_FIELDS] + [prev, h],
            )
            c.execute("COMMIT")
            return int(cur.lastrowid)

    def audit_list(self, sandbox_id: str, loan_id: str) -> list:
        with self.conn() as c:
            rows = c.execute(
                "SELECT ts, actor, action, ai_recommendation, model_version, inputs_hash "
                "FROM audit WHERE sandbox_id = ? AND loan_id = ? ORDER BY id",
                (sandbox_id, loan_id),
            ).fetchall()
        return [dict(r) for r in rows]

    def audit_verify(self) -> bool:
        """Recompute the whole chain; False if any row was tampered with."""
        prev = GENESIS_HASH
        with self.conn() as c:
            for r in c.execute("SELECT * FROM audit ORDER BY id"):
                r = dict(r)
                if r["prev_hash"] != prev or r["hash"] != row_hash(prev, r):
                    return False
                prev = r["hash"]
        return True

    # ---------- sandbox photos ----------

    def photo_add(self, *, photo_id: str, sandbox_id: str, loan_id: str, server_ts: str,
                  client_ts: Optional[str], lat, lon, result: dict, blurred_image: Optional[bytes]) -> None:
        image_file = None
        if blurred_image is not None:  # the original image is never written
            image_file = f"{photo_id}.img"
            (self.photo_dir / image_file).write_bytes(blurred_image)
        with self._write_lock, self.conn() as c:
            c.execute(
                "INSERT INTO photos VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (photo_id, sandbox_id, loan_id, server_ts, client_ts, lat, lon,
                 json.dumps(result, ensure_ascii=False), image_file),
            )

    def photo_list(self, sandbox_id: str, loan_id: str) -> list:
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM photos WHERE sandbox_id = ? AND loan_id = ? ORDER BY server_ts",
                (sandbox_id, loan_id),
            ).fetchall()
        return [dict(r) for r in rows]

    def photo_file(self, sandbox_id: str, photo_id: str) -> Optional[Path]:
        with self.conn() as c:
            r = c.execute("SELECT image_file FROM photos WHERE sandbox_id = ? AND photo_id = ?",
                          (sandbox_id, photo_id)).fetchone()
        if not r or not r["image_file"]:
            return None
        p = self.photo_dir / r["image_file"]
        return p if p.is_file() else None

    def reset_sandbox_data(self) -> None:
        """Admin reset: removes sandbox photos. The audit log is append-only and is kept."""
        with self._write_lock, self.conn() as c:
            c.execute("DELETE FROM photos")
        shutil.rmtree(self.photo_dir, ignore_errors=True)
        self.photo_dir.mkdir(parents=True, exist_ok=True)

    # ---------- LLM call log ----------

    def llm_log(self, *, loan_id: str, model: str, latency_ms: float, input_tokens, output_tokens,
                cost_usd, ok: bool, error: Optional[str] = None) -> None:
        with self._write_lock, self.conn() as c:
            c.execute(
                "INSERT INTO llm_calls (ts, loan_id, model, latency_ms, input_tokens, output_tokens, "
                "cost_usd, ok, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (utc_now(), loan_id, model, latency_ms, input_tokens, output_tokens, cost_usd, int(ok), error),
            )
