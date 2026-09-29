"""Append-only JSONL plus a SQLite index for aggregation.

Two files, two jobs. The JSONL is the record of truth: one line per request,
never rewritten, trivially greppable and diffable, and the thing every number in
`docs/BENCHMARKS.md` traces back to. The SQLite file is a derived index that
exists so `/metrics` can group and sum without replaying the whole log; it can be
deleted and rebuilt from the JSONL at any time.

Both paths are configurable and both are gitignored.

Nothing here swallows errors: a store that cannot write says so, and the
`Recorder` is the single place that decides a failed write must not fail the
user's request.
"""

import json
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from conduit.telemetry.recorder import TelemetryRecord

__all__ = [
    "GROUPABLE_COLUMNS",
    "UNSLICED_KEY",
    "JsonlSqliteStore",
    "SliceAggregate",
    "TelemetryQuery",
    "TelemetryStore",
    "read_jsonl",
]

#: Columns `/metrics` may slice by. Anything else is rejected rather than
#: interpolated into SQL.
GROUPABLE_COLUMNS = frozenset({"provider", "team", "tier", "model", "workflow", "status"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS telemetry (
    record_id           TEXT PRIMARY KEY,
    trace_id            TEXT NOT NULL,
    ts                  TEXT NOT NULL,
    ts_epoch            REAL NOT NULL,
    team                TEXT NOT NULL,
    workflow            TEXT NOT NULL,
    tier                TEXT,
    provider            TEXT,
    model               TEXT,
    prompt_tokens       INTEGER NOT NULL,
    completion_tokens   INTEGER NOT NULL,
    cache_read_tokens   INTEGER NOT NULL,
    cache_write_tokens  INTEGER NOT NULL,
    cost_usd            REAL NOT NULL,
    latency_ms          INTEGER NOT NULL,
    provider_latency_ms INTEGER,
    guard_categories    TEXT NOT NULL,
    fallback_from       TEXT NOT NULL,
    status              TEXT NOT NULL,
    error               TEXT,
    prompt_text         TEXT,
    completion_text     TEXT
);
CREATE INDEX IF NOT EXISTS telemetry_ts ON telemetry (ts_epoch);
CREATE INDEX IF NOT EXISTS telemetry_trace ON telemetry (trace_id);
CREATE INDEX IF NOT EXISTS telemetry_slice ON telemetry (provider, team, tier);
"""

#: Group key used when a query is not sliced by any column.
UNSLICED_KEY = "all"


#: Columns a query may filter by with an equality test. `trace_id` is here but
#: deliberately absent from `GROUPABLE_COLUMNS`: grouping by it would produce
#: one bucket per request, which is a list, not an aggregate.
_FILTER_COLUMNS = ("provider", "team", "tier", "model", "workflow", "status", "trace_id")


class TelemetryQuery(BaseModel):
    """A time window plus equality filters, shared by every read path."""

    since: datetime | None = None
    until: datetime | None = None
    provider: str | None = None
    team: str | None = None
    tier: str | None = None
    model: str | None = None
    workflow: str | None = None
    status: str | None = None
    #: One request. The SQLite index `telemetry_trace` already exists for it.
    trace_id: str | None = None
    limit: int | None = Field(default=None, gt=0)

    #: Row order for `records()`. Aggregates ignore it — they have their own
    #: ORDER BY — but a console feed wants the newest rows, and `limit` is
    #: applied by SQLite *after* the sort, so ascending order with a limit
    #: would return the oldest N rows rather than the most recent ones.
    newest_first: bool = False

    def sql_where(self) -> tuple[str, list[Any]]:
        """Render this query as a WHERE clause plus bound parameters."""
        clauses: list[str] = []
        params: list[Any] = []
        if self.since is not None:
            clauses.append("ts_epoch >= ?")
            params.append(self.since.timestamp())
        if self.until is not None:
            clauses.append("ts_epoch < ?")
            params.append(self.until.timestamp())
        for column in _FILTER_COLUMNS:
            value = getattr(self, column)
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


class SliceAggregate(BaseModel):
    """Counts and sums for one slice, straight out of SQL."""

    key: str
    requests: int
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float
    errors: int


@runtime_checkable
class TelemetryStore(Protocol):
    """What the recorder writes to and `/metrics` reads from."""

    def write(self, record: TelemetryRecord) -> None: ...
    def records(self, query: TelemetryQuery | None = None) -> list[TelemetryRecord]: ...
    def aggregate(
        self, query: TelemetryQuery | None = None, group_by: str | None = None
    ) -> list[SliceAggregate]: ...
    def latencies(
        self, query: TelemetryQuery | None = None, group_by: str | None = None
    ) -> dict[str, list[int]]: ...
    def close(self) -> None: ...


def read_jsonl(path: Path) -> Iterator[TelemetryRecord]:
    """Replay the append-only log. Blank lines are tolerated, bad ones are not."""
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield TelemetryRecord.model_validate_json(line)


class JsonlSqliteStore:
    """The shipped store: append to JSONL, mirror into SQLite, read from SQLite."""

    def __init__(self, jsonl_path: Path, sqlite_path: Path) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.sqlite_path = Path(sqlite_path)
        for path in (self.jsonl_path, self.sqlite_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        # Writes arrive from the request path, reads from `/metrics` handlers;
        # one connection guarded by a lock is simpler than a pool and fast
        # enough for a single-process gateway.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_SCHEMA)

    # -- write ------------------------------------------------------------ #

    def write(self, record: TelemetryRecord) -> None:
        """Append to the log, then index. The log is written first on purpose."""
        row = record.model_dump(mode="json")
        line = json.dumps(row, separators=(",", ":"), sort_keys=True)
        with self._lock:
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            self._index(record, row)

    def rebuild_from_jsonl(self) -> int:
        """Reconstruct the SQLite index from the log. Returns rows reinserted."""
        with self._lock:
            with self._conn:
                self._conn.execute("DELETE FROM telemetry")
            count = 0
            for record in read_jsonl(self.jsonl_path):
                self._index(record, record.model_dump(mode="json"))
                count += 1
        return count

    def _index(self, record: TelemetryRecord, row: dict[str, Any]) -> None:
        """Insert one row into SQLite. Caller holds the lock."""
        with self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO telemetry VALUES (
                    :record_id, :trace_id, :ts, :ts_epoch, :team, :workflow, :tier,
                    :provider, :model, :prompt_tokens, :completion_tokens,
                    :cache_read_tokens, :cache_write_tokens, :cost_usd, :latency_ms,
                    :provider_latency_ms, :guard_categories, :fallback_from, :status,
                    :error, :prompt_text, :completion_text
                )
                """,
                {
                    **row,
                    "ts_epoch": record.ts.timestamp(),
                    "tier": record.tier.value if record.tier else None,
                    "guard_categories": json.dumps(record.guard_categories),
                    "fallback_from": json.dumps(record.fallback_from),
                },
            )

    # -- read ------------------------------------------------------------- #

    def records(self, query: TelemetryQuery | None = None) -> list[TelemetryRecord]:
        query = query or TelemetryQuery()
        where, params = query.sql_where()
        order = "DESC" if query.newest_first else "ASC"
        sql = f"SELECT * FROM telemetry{where} ORDER BY ts_epoch {order}"
        if query.limit is not None:
            sql += " LIMIT ?"
            params = [*params, query.limit]
        with self._lock:
            rows: Sequence[sqlite3.Row] = self._conn.execute(sql, params).fetchall()
        return [_record_from_row(row) for row in rows]

    def aggregate(
        self, query: TelemetryQuery | None = None, group_by: str | None = None
    ) -> list[SliceAggregate]:
        """Counts and sums, grouped by one column or over everything."""
        query = query or TelemetryQuery()
        where, params = query.sql_where()
        key_expr = _group_expr(group_by)
        sql = f"""
            SELECT {key_expr} AS key,
                   COUNT(*)                       AS requests,
                   COALESCE(SUM(prompt_tokens), 0)      AS prompt_tokens,
                   COALESCE(SUM(completion_tokens), 0)  AS completion_tokens,
                   COALESCE(SUM(cache_read_tokens), 0)  AS cache_read_tokens,
                   COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens,
                   COALESCE(SUM(cost_usd), 0.0)         AS cost_usd,
                   COALESCE(SUM(status != 'ok'), 0)     AS errors
            FROM telemetry{where}
            GROUP BY key
            ORDER BY cost_usd DESC, key ASC
        """
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [SliceAggregate(**dict(row)) for row in rows]

    def latencies(
        self, query: TelemetryQuery | None = None, group_by: str | None = None
    ) -> dict[str, list[int]]:
        """Ascending latency samples per slice, for percentile math upstream."""
        query = query or TelemetryQuery()
        where, params = query.sql_where()
        key_expr = _group_expr(group_by)
        sql = f"""
            SELECT {key_expr} AS key, latency_ms
            FROM telemetry{where}
            ORDER BY key ASC, latency_ms ASC
        """
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        buckets: dict[str, list[int]] = {}
        for row in rows:
            buckets.setdefault(str(row["key"]), []).append(int(row["latency_ms"]))
        return buckets

    # -- lifecycle -------------------------------------------------------- #

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "JsonlSqliteStore":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _group_expr(group_by: str | None) -> str:
    if group_by is None:
        return f"'{UNSLICED_KEY}'"
    if group_by not in GROUPABLE_COLUMNS:
        raise ValueError(
            f"cannot group telemetry by {group_by!r}; expected one of "
            f"{', '.join(sorted(GROUPABLE_COLUMNS))}"
        )
    # A NULL provider/model (a request that died before reaching one) is still a
    # row that has to be counted somewhere.
    return f"COALESCE({group_by}, 'unknown')"


def _record_from_row(row: sqlite3.Row) -> TelemetryRecord:
    data = dict(row)
    data.pop("ts_epoch", None)
    data["guard_categories"] = json.loads(data["guard_categories"])
    data["fallback_from"] = json.loads(data["fallback_from"])
    return TelemetryRecord.model_validate(data)
