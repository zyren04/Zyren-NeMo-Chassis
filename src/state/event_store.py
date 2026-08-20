"""
Event Store - Append-only SQLite Event Sourcing Logger (async-native)
Records all state mutations, timestamps, and node execution metrics.
Includes durable session recovery (nemo-rl-session-memory patterns).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite


@dataclass
class EventRecord:
    """Single event record in the event store."""

    event_id: str
    execution_id: str
    event_type: str
    node_name: str | None
    payload: dict[str, Any]
    timestamp: datetime
    iteration: int

    # NEW: Telemetry correlation fields for NeMo Relay observability
    trace_id: str | None = None
    span_id: str | None = None
    relay_uuid: str | None = None
    relay_parent_uuid: str | None = None


@dataclass
class NodeMetricRecord:
    """Node execution metrics record."""

    metric_id: str
    execution_id: str
    node_name: str
    start_time: datetime
    end_time: datetime | None
    duration_ms: float | None
    exit_code: int | None
    tokens_consumed: int
    api_calls: int
    success: bool
    error_message: str | None


@dataclass
class SessionRecord:
    """Durable session metadata for cross-disconnect recovery (nemo-rl-session-memory pattern)."""

    session_id: str
    created_at: datetime
    updated_at: datetime
    goal: str
    current_subtask: str
    loaded_skills: list[str]
    status: str
    plan: list[str]
    assumptions: list[str]
    blockers: list[str]
    handoff_summary: str
    next_actions: list[str]
    watch_outs: list[str]

    def to_markdown(self) -> str:
        """Export as session_state.md format."""
        lines = [
            f"# Session State: {self.session_id}",
            "",
            f"**Created:** {self.created_at.isoformat()}",
            f"**Updated:** {self.updated_at.isoformat()}",
            "",
            "## Goal",
            self.goal,
            "",
            "## Current Subtask",
            self.current_subtask,
            "",
            "## Loaded Skills",
        ]
        if self.loaded_skills:
            lines.extend(f"- {s}" for s in self.loaded_skills)
        else:
            lines.append("- None")
        lines.extend(
            [
                "",
                "## Status",
                self.status,
                "",
                "## Plan",
            ]
        )
        if self.plan:
            lines.extend(f"- [ ] {p}" for p in self.plan)
        else:
            lines.append("- None")
        lines.extend(
            [
                "",
                "## Assumptions",
            ]
        )
        if self.assumptions:
            lines.extend(f"- {a}" for a in self.assumptions)
        else:
            lines.append("- None")
        lines.extend(
            [
                "",
                "## Blockers",
            ]
        )
        if self.blockers:
            lines.extend(f"- {b}" for b in self.blockers)
        else:
            lines.append("- None known")
        lines.extend(
            [
                "",
                "## Handoff Summary",
                self.handoff_summary,
                "",
                "## Next Actions",
            ]
        )
        if self.next_actions:
            lines.extend(f"- {a}" for a in self.next_actions)
        else:
            lines.append("- None")
        lines.extend(
            [
                "",
                "## Watch Outs",
            ]
        )
        if self.watch_outs:
            lines.extend(f"- {w}" for w in self.watch_outs)
        else:
            lines.append("- None")
        lines.append("")
        return "\n".join(lines)


@dataclass
class TimelineEntry:
    """Append-only timeline entry (mirrors timeline.md from nemo-rl-session-memory)."""

    entry_id: str
    session_id: str
    timestamp: datetime
    user_request: str
    context_gathered: str
    decision: str
    result: str

    def to_markdown(self) -> str:
        """Export as timeline.md entry format."""
        return (
            f"## {self.timestamp.isoformat()} — {self.user_request}\n\n"
            f"**Context Gathered:** {self.context_gathered}\n\n"
            f"**Decision:** {self.decision}\n\n"
            f"**Result:** {self.result}\n\n"
            "---\n"
        )


class EventStore:
    """
    Append-only SQLite event store for event sourcing.
    Fully async-native with a single connection and write locking for SQLite concurrency safety.
    """

    def __init__(self, db_path: str = "data/events.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: aiosqlite.Connection | None = None
        self._write_lock: asyncio.Lock | None = None
        self._initialized = False

    def _get_lock(self) -> asyncio.Lock:
        """Get or create the write lock in the current event loop."""
        # Always create a new lock for the current event loop to avoid
        # "bound to a different event loop" errors when the store is shared
        # across event loops (e.g., in thread pools or subprocesses).
        loop = asyncio.get_running_loop()
        if self._write_lock is None or self._write_lock._loop is not loop:
            self._write_lock = asyncio.Lock()
        return self._write_lock

    async def _initialize(self) -> None:
        """Initialize database schema and connection."""
        if self._initialized:
            return
        async with self._get_lock():
            if self._initialized:
                return
            self._conn = await aiosqlite.connect(str(self.db_path))
            self._conn.row_factory = aiosqlite.Row
            # Enable WAL mode for better concurrent access
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.execute("PRAGMA busy_timeout=30000")
            await self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    node_name TEXT,
                    payload TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_events_execution_id ON events(execution_id);
                CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
                CREATE INDEX IF NOT EXISTS idx_events_node ON events(node_name);
                CREATE TABLE IF NOT EXISTS node_metrics (
                    metric_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL,
                    node_name TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration_ms REAL,
                    exit_code INTEGER,
                    tokens_consumed INTEGER DEFAULT 0,
                    api_calls INTEGER DEFAULT 0,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_metrics_execution_id ON node_metrics(execution_id);
                CREATE INDEX IF NOT EXISTS idx_metrics_node ON node_metrics(node_name);
                CREATE TABLE IF NOT EXISTS state_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_snapshots_execution_id ON state_snapshots(execution_id);
                CREATE INDEX IF NOT EXISTS idx_snapshots_iteration ON state_snapshots(iteration);
                -- Session recovery tables (nemo-rl-session-memory pattern)
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    current_subtask TEXT NOT NULL,
                    loaded_skills TEXT NOT NULL,        -- JSON array
                    status TEXT NOT NULL,
                    plan TEXT NOT NULL,                 -- JSON array
                    assumptions TEXT NOT NULL,          -- JSON array
                    blockers TEXT NOT NULL,             -- JSON array
                    handoff_summary TEXT NOT NULL,
                    next_actions TEXT NOT NULL,         -- JSON array
                    watch_outs TEXT NOT NULL            -- JSON array
                );
                CREATE TABLE IF NOT EXISTS timeline (
                    entry_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_request TEXT NOT NULL,
                    context_gathered TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    result TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_timeline_session ON timeline(session_id);
                CREATE INDEX IF NOT EXISTS idx_timeline_timestamp ON timeline(timestamp);
            """)
            await self._conn.commit()
            self._initialized = True

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            self._initialized = False

    async def _execute_write(self, query: str, params: tuple[Any, ...]) -> None:
        """Execute a write query with locking and retry logic."""
        await self._initialize()
        assert self._conn is not None  # nosec B101
        max_retries = 10
        base_delay = 0.005  # 5ms

        for attempt in range(max_retries):
            try:
                async with self._get_lock():
                    await self._conn.execute(query, params)
                    await self._conn.commit()
                return
            except Exception as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2**attempt) + (asyncio.get_event_loop().time() % 0.005)
                    await asyncio.sleep(delay)
                    continue
                raise

    async def _execute_read(self, query: str, params: tuple[Any, ...]) -> list[aiosqlite.Row]:
        """Execute a read query."""
        await self._initialize()
        assert self._conn is not None  # nosec B101
        async with self._conn.execute(query, params) as cursor:
            return list(await cursor.fetchall())

    async def record_event(
        self,
        execution_id: str,
        event_type: str,
        payload: dict[str, Any],
        node_name: str | None = None,
        iteration: int = 0,
    ) -> EventRecord:
        """Record a single event."""
        event = EventRecord(
            event_id=str(uuid.uuid4()),
            execution_id=execution_id,
            event_type=event_type,
            node_name=node_name,
            payload=payload,
            timestamp=datetime.utcnow(),
            iteration=iteration,
        )

        await self._execute_write(
            """
            INSERT INTO events (event_id, execution_id, event_type, node_name, payload, timestamp, iteration)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                event.event_id,
                event.execution_id,
                event.event_type,
                event.node_name,
                json.dumps(event.payload, default=str),
                event.timestamp.isoformat(),
                event.iteration,
            ),
        )

        return event

    async def record_events_batch(self, events: list[EventRecord]) -> list[EventRecord]:
        """Record multiple events in a single transaction."""
        if not events:
            return []

        await self._initialize()
        assert self._conn is not None  # nosec B101
        async with self._write_lock, self._conn.execute("BEGIN TRANSACTION"):
            for event in events:
                await self._conn.execute(
                    """
                    INSERT INTO events (event_id, execution_id, event_type, node_name, payload, timestamp, iteration)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        event.event_id,
                        event.execution_id,
                        event.event_type,
                        event.node_name,
                        json.dumps(event.payload, default=str),
                        event.timestamp.isoformat(),
                        event.iteration,
                    ),
                )
            await self._conn.commit()

        return events

    async def get_events(
        self,
        execution_id: str,
        event_type: str | None = None,
        node_name: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[EventRecord]:
        """Retrieve events for an execution with optional filters."""
        query = "SELECT * FROM events WHERE execution_id = ?"
        params: list[Any] = [execution_id]

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if node_name:
            query += " AND node_name = ?"
            params.append(node_name)

        query += " ORDER BY timestamp ASC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = await self._execute_read(query, tuple(params))

        return [
            EventRecord(
                event_id=row["event_id"],
                execution_id=row["execution_id"],
                event_type=row["event_type"],
                node_name=row["node_name"],
                payload=json.loads(row["payload"]),
                timestamp=datetime.fromisoformat(row["timestamp"]),
                iteration=row["iteration"],
            )
            for row in rows
        ]

    async def get_events_by_iteration(self, execution_id: str, iteration: int) -> list[EventRecord]:
        """Get all events for a specific iteration."""
        rows = await self._execute_read(
            """
                SELECT * FROM events
                WHERE execution_id = ? AND iteration = ?
                ORDER BY timestamp ASC
            """,
            (execution_id, iteration),
        )

        return [
            EventRecord(
                event_id=row["event_id"],
                execution_id=row["execution_id"],
                event_type=row["event_type"],
                node_name=row["node_name"],
                payload=json.loads(row["payload"]),
                timestamp=datetime.fromisoformat(row["timestamp"]),
                iteration=row["iteration"],
            )
            for row in rows
        ]

    async def record_node_metric(self, metric: NodeMetricRecord) -> NodeMetricRecord:
        """Record node execution metrics."""
        await self._execute_write(
            """
                INSERT INTO node_metrics
                (metric_id, execution_id, node_name, start_time, end_time, duration_ms, exit_code, tokens_consumed, api_calls, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metric.metric_id,
                metric.execution_id,
                metric.node_name,
                metric.start_time.isoformat(),
                metric.end_time.isoformat() if metric.end_time else None,
                metric.duration_ms,
                metric.exit_code,
                metric.tokens_consumed,
                metric.api_calls,
                metric.success,
                metric.error_message,
            ),
        )

        return metric

    async def get_node_metrics(
        self, execution_id: str, node_name: str | None = None
    ) -> list[NodeMetricRecord]:
        """Retrieve node metrics for an execution."""
        query = "SELECT * FROM node_metrics WHERE execution_id = ?"
        params: list[Any] = [execution_id]

        if node_name:
            query += " AND node_name = ?"
            params.append(node_name)

        query += " ORDER BY start_time ASC"

        rows = await self._execute_read(query, tuple(params))

        return [
            NodeMetricRecord(
                metric_id=row["metric_id"],
                execution_id=row["execution_id"],
                node_name=row["node_name"],
                start_time=datetime.fromisoformat(row["start_time"]),
                end_time=datetime.fromisoformat(row["end_time"]) if row["end_time"] else None,
                duration_ms=row["duration_ms"],
                exit_code=row["exit_code"],
                tokens_consumed=row["tokens_consumed"],
                api_calls=row["api_calls"],
                success=bool(row["success"]),
                error_message=row["error_message"],
            )
            for row in rows
        ]

    async def snapshot_state(self, execution_id: str, iteration: int, state_json: str) -> str:
        """Create a state snapshot."""
        snapshot_id = str(uuid.uuid4())

        await self._execute_write(
            """
            INSERT INTO state_snapshots (snapshot_id, execution_id, iteration, state_json, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """,
            (snapshot_id, execution_id, iteration, state_json, datetime.utcnow().isoformat()),
        )

        return snapshot_id

    async def get_latest_snapshot(self, execution_id: str) -> dict[str, Any] | None:
        """Get the most recent state snapshot for an execution."""
        rows = await self._execute_read(
            """
                SELECT * FROM state_snapshots
                WHERE execution_id = ?
                ORDER BY iteration DESC LIMIT 1
            """,
            (execution_id,),
        )

        if not rows:
            return None

        row = rows[0]
        return {
            "snapshot_id": row["snapshot_id"],
            "execution_id": row["execution_id"],
            "iteration": row["iteration"],
            "state": json.loads(row["state_json"]),
            "timestamp": datetime.fromisoformat(row["timestamp"]),
        }

    async def get_snapshot_at_iteration(
        self, execution_id: str, iteration: int
    ) -> dict[str, Any] | None:
        """Get state snapshot at a specific iteration."""
        rows = await self._execute_read(
            """
                SELECT * FROM state_snapshots
                WHERE execution_id = ? AND iteration = ?
                LIMIT 1
            """,
            (execution_id, iteration),
        )

        if not rows:
            return None

        row = rows[0]
        return {
            "snapshot_id": row["snapshot_id"],
            "execution_id": row["execution_id"],
            "iteration": row["iteration"],
            "state": json.loads(row["state_json"]),
            "timestamp": datetime.fromisoformat(row["timestamp"]),
        }

    async def get_execution_summary(self, execution_id: str) -> dict[str, Any]:
        """Get a summary of an execution."""
        event_row = await self._execute_read(
            "SELECT COUNT(*) as count FROM events WHERE execution_id = ?", (execution_id,)
        )
        event_count = event_row[0]["count"] if event_row else 0

        metric_rows = await self._execute_read(
            """
            SELECT node_name, COUNT(*) as calls, SUM(duration_ms) as total_ms,
                   SUM(tokens_consumed) as total_tokens, SUM(api_calls) as total_calls,
                   SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes,
                   SUM(CASE WHEN success THEN 0 ELSE 1 END) as failures
            FROM node_metrics WHERE execution_id = ?
            GROUP BY node_name
        """,
            (execution_id,),
        )

        first_event = await self._execute_read(
            """
            SELECT timestamp FROM events WHERE execution_id = ? ORDER BY timestamp ASC LIMIT 1
        """,
            (execution_id,),
        )

        last_event = await self._execute_read(
            """
            SELECT timestamp FROM events WHERE execution_id = ? ORDER BY timestamp DESC LIMIT 1
        """,
            (execution_id,),
        )

        node_metrics = {}
        for row in metric_rows:
            node_metrics[row["node_name"]] = {
                "calls": row["calls"],
                "total_duration_ms": row["total_ms"] or 0,
                "total_tokens": row["total_tokens"] or 0,
                "total_api_calls": row["total_calls"] or 0,
                "successes": row["successes"],
                "failures": row["failures"],
            }

        return {
            "execution_id": execution_id,
            "total_events": event_count,
            "node_metrics": node_metrics,
            "started_at": datetime.fromisoformat(first_event[0]["timestamp"])
            if first_event
            else None,
            "completed_at": datetime.fromisoformat(last_event[0]["timestamp"])
            if last_event
            else None,
        }

    async def delete_execution(self, execution_id: str) -> bool:
        """Delete all data for an execution (for testing/cleanup)."""
        await self._execute_write("DELETE FROM events WHERE execution_id = ?", (execution_id,))
        await self._execute_write(
            "DELETE FROM node_metrics WHERE execution_id = ?", (execution_id,)
        )
        await self._execute_write(
            "DELETE FROM state_snapshots WHERE execution_id = ?", (execution_id,)
        )
        return True

    # =========================================================================
    # Session Recovery API (nemo-rl-session-memory pattern)
    # =========================================================================

    async def create_session(
        self,
        session_id: str,
        goal: str,
        current_subtask: str = "Initializing",
        loaded_skills: list[str] | None = None,
        status: str = "Starting",
        plan: list[str] | None = None,
        assumptions: list[str] | None = None,
        blockers: list[str] | None = None,
        handoff_summary: str = "",
        next_actions: list[str] | None = None,
        watch_outs: list[str] | None = None,
    ) -> SessionRecord:
        """Initialize a new session with goal; returns SessionRecord."""
        now = datetime.now()
        record = SessionRecord(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            goal=goal,
            current_subtask=current_subtask,
            loaded_skills=loaded_skills or [],
            status=status,
            plan=plan or [],
            assumptions=assumptions or [],
            blockers=blockers or [],
            handoff_summary=handoff_summary,
            next_actions=next_actions or [],
            watch_outs=watch_outs or [],
        )
        await self._execute_write(
            """
            INSERT INTO sessions (
                session_id, created_at, updated_at, goal, current_subtask,
                loaded_skills, status, plan, assumptions, blockers,
                handoff_summary, next_actions, watch_outs
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.session_id,
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
                record.goal,
                record.current_subtask,
                json.dumps(record.loaded_skills),
                record.status,
                json.dumps(record.plan),
                json.dumps(record.assumptions),
                json.dumps(record.blockers),
                record.handoff_summary,
                json.dumps(record.next_actions),
                json.dumps(record.watch_outs),
            ),
        )
        return record

    async def update_session_state(self, session_id: str, **fields) -> SessionRecord | None:
        """Update session_state.md fields (subtask, status, plan, etc.)."""
        # Get current session
        current = await self.get_session(session_id)
        if current is None:
            return None

        # Update fields
        allowed_fields = {
            "goal",
            "current_subtask",
            "loaded_skills",
            "status",
            "plan",
            "assumptions",
            "blockers",
            "handoff_summary",
            "next_actions",
            "watch_outs",
        }
        updates = {k: v for k, v in fields.items() if k in allowed_fields}
        if not updates:
            return current

        # Build updated record
        updated = SessionRecord(
            session_id=current.session_id,
            created_at=current.created_at,
            updated_at=datetime.now(),
            goal=updates.get("goal", current.goal),
            current_subtask=updates.get("current_subtask", current.current_subtask),
            loaded_skills=updates.get("loaded_skills", current.loaded_skills),
            status=updates.get("status", current.status),
            plan=updates.get("plan", current.plan),
            assumptions=updates.get("assumptions", current.assumptions),
            blockers=updates.get("blockers", current.blockers),
            handoff_summary=updates.get("handoff_summary", current.handoff_summary),
            next_actions=updates.get("next_actions", current.next_actions),
            watch_outs=updates.get("watch_outs", current.watch_outs),
        )

        await self._execute_write(
            """
            UPDATE sessions SET
                updated_at = ?,
                goal = ?,
                current_subtask = ?,
                loaded_skills = ?,
                status = ?,
                plan = ?,
                assumptions = ?,
                blockers = ?,
                handoff_summary = ?,
                next_actions = ?,
                watch_outs = ?
            WHERE session_id = ?
            """,
            (
                updated.updated_at.isoformat(),
                updated.goal,
                updated.current_subtask,
                json.dumps(updated.loaded_skills),
                updated.status,
                json.dumps(updated.plan),
                json.dumps(updated.assumptions),
                json.dumps(updated.blockers),
                updated.handoff_summary,
                json.dumps(updated.next_actions),
                json.dumps(updated.watch_outs),
                session_id,
            ),
        )
        return updated

    async def append_timeline(self, session_id: str, entry: TimelineEntry) -> None:
        """Append to timeline.md (append-only, never mutate)."""
        await self._execute_write(
            """
            INSERT INTO timeline (
                entry_id, session_id, timestamp, user_request,
                context_gathered, decision, result
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.entry_id,
                entry.session_id,
                entry.timestamp.isoformat(),
                entry.user_request,
                entry.context_gathered,
                entry.decision,
                entry.result,
            ),
        )

    async def get_session(self, session_id: str) -> SessionRecord | None:
        """Load session_state.md for recovery."""
        rows = await self._execute_read(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        )
        if not rows:
            return None
        row = rows[0]
        return SessionRecord(
            session_id=row["session_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            goal=row["goal"],
            current_subtask=row["current_subtask"],
            loaded_skills=json.loads(row["loaded_skills"]),
            status=row["status"],
            plan=json.loads(row["plan"]),
            assumptions=json.loads(row["assumptions"]),
            blockers=json.loads(row["blockers"]),
            handoff_summary=row["handoff_summary"],
            next_actions=json.loads(row["next_actions"]),
            watch_outs=json.loads(row["watch_outs"]),
        )

    async def get_recent_timeline(self, session_id: str, limit: int = 20) -> list[TimelineEntry]:
        """Load recent timeline.md entries for recovery context."""
        rows = await self._execute_read(
            """
            SELECT * FROM timeline
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (session_id, limit),
        )
        entries = []
        for row in reversed(rows):  # Return in chronological order
            entries.append(
                TimelineEntry(
                    entry_id=row["entry_id"],
                    session_id=row["session_id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    user_request=row["user_request"],
                    context_gathered=row["context_gathered"],
                    decision=row["decision"],
                    result=row["result"],
                )
            )
        return entries

    async def generate_handoff(self, session_id: str) -> str:
        """Produce handoff.md content from session + recent timeline."""
        session = await self.get_session(session_id)
        if session is None:
            return f"# Handoff: Session {session_id} not found\n"

        timeline = await self.get_recent_timeline(session_id, limit=10)

        lines = [
            f"# Handoff: {session.session_id}",
            "",
            f"**Session Created:** {session.created_at.isoformat()}",
            f"**Last Updated:** {session.updated_at.isoformat()}",
            "",
            "## One-Paragraph Resume Summary",
            session.handoff_summary or "No summary available.",
            "",
            "## Current Goal",
            session.goal,
            "",
            "## Current Subtask",
            session.current_subtask,
            "",
            "## Status",
            session.status,
            "",
            "## Next Actions (Prioritized)",
        ]
        if session.next_actions:
            lines.extend(f"- {a}" for a in session.next_actions)
        else:
            lines.append("- None")
        lines.extend(
            [
                "",
                "## Watch Outs / Risks",
            ]
        )
        if session.watch_outs:
            lines.extend(f"- {w}" for w in session.watch_outs)
        else:
            lines.append("- None")
        lines.extend(
            [
                "",
                "## Recent Timeline (Last 10 Entries)",
                "",
            ]
        )
        for entry in timeline:
            lines.append(entry.to_markdown())

        return "\n".join(lines)

    async def list_sessions(self, limit: int = 50) -> list[SessionRecord]:
        """List recent sessions for 'ls -dt session/* | head' equivalent."""
        rows = await self._execute_read(
            """
            SELECT * FROM sessions
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        sessions = []
        for row in rows:
            sessions.append(
                SessionRecord(
                    session_id=row["session_id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    goal=row["goal"],
                    current_subtask=row["current_subtask"],
                    loaded_skills=json.loads(row["loaded_skills"]),
                    status=row["status"],
                    plan=json.loads(row["plan"]),
                    assumptions=json.loads(row["assumptions"]),
                    blockers=json.loads(row["blockers"]),
                    handoff_summary=row["handoff_summary"],
                    next_actions=json.loads(row["next_actions"]),
                    watch_outs=json.loads(row["watch_outs"]),
                )
            )
        return sessions

    async def export_session_artifacts(self, session_id: str, out_dir: Path) -> None:
        """Write session_state.md, timeline.md, files.md, handoff.md to disk
        (for human inspection / git commit / handoff to another agent)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        session = await self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        timeline = await self.get_recent_timeline(session_id, limit=100)

        # session_state.md
        (out_dir / "session_state.md").write_text(session.to_markdown())

        # timeline.md
        timeline_md = [
            f"# Timeline: {session_id}",
            "",
            f"Session created: {session.created_at.isoformat()}",
            f"Last updated: {session.updated_at.isoformat()}",
            "",
            "---",
            "",
        ]
        for entry in timeline:
            timeline_md.append(entry.to_markdown())
        (out_dir / "timeline.md").write_text("\n".join(timeline_md))

        # handoff.md
        (out_dir / "handoff.md").write_text(await self.generate_handoff(session_id))

        # files.md - list files from events (best effort)
        event_rows = await self._execute_read(
            """
            SELECT DISTINCT json_extract(payload, '$.file_path') as file_path,
                   json_extract(payload, '$.operation') as operation,
                   timestamp
            FROM events
            WHERE execution_id = ? AND json_extract(payload, '$.file_path') IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 100
            """,
            (session_id,),
        )
        files_md = [
            f"# Files Touched: {session_id}",
            "",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "| Timestamp | Operation | File |",
            "|-----------|-----------|------|",
        ]
        for row in event_rows:
            files_md.append(
                f"| {row['timestamp']} | {row['operation'] or 'unknown'} | {row['file_path']} |"
            )
        (out_dir / "files.md").write_text("\n".join(files_md))

    async def delete_session(self, session_id: str) -> bool:
        """Delete all data for a session (for testing/cleanup)."""
        await self._execute_write("DELETE FROM timeline WHERE session_id = ?", (session_id,))
        await self._execute_write("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        return True


# Global instance for convenience
_default_store: EventStore | None = None


async def get_event_store(db_path: str = "data/events.db") -> EventStore:
    """Get or create the default event store instance."""
    global _default_store
    if _default_store is None:
        _default_store = EventStore(db_path)
        await _default_store._initialize()

        # NEW: Auto-configure observability from NeMo Relay integration
        try:
            from ..infrastructure.nemo_relay_integration import get_nemo_relay_integration

            nemo = get_nemo_relay_integration()
            if nemo.config.observability and nemo.config.observability.enabled:
                # Ensure ATOF output directory exists
                from pathlib import Path

                Path(nemo.config.observability.atof.output_directory).mkdir(
                    parents=True, exist_ok=True
                )
                if nemo.config.observability.atif.enabled:
                    Path(nemo.config.observability.atif.output_directory).mkdir(
                        parents=True, exist_ok=True
                    )
        except ImportError:
            pass

    return _default_store


def set_event_store(store: EventStore) -> None:
    """Set the default event store instance (useful for testing)."""
    global _default_store
    _default_store = store


__all__ = [
    "EventRecord",
    "NodeMetricRecord",
    "SessionRecord",
    "TimelineEntry",
    "EventStore",
    "get_event_store",
    "set_event_store",
]
