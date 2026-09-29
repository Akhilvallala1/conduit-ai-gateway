"""Builders shared by the telemetry tests. No network, no real clock."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from conduit.contracts import (
    CompletionRequest,
    CompletionResponse,
    Complexity,
    Message,
    Usage,
)
from conduit.telemetry.recorder import RecordStatus, TelemetryRecord
from conduit.telemetry.store import JsonlSqliteStore

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config"
EPOCH = datetime(2026, 9, 7, 12, 0, 0, tzinfo=UTC)

HAIKU = "claude-haiku-4-5-20251001"


class StubClock:
    """A monotonic clock that advances a fixed step per read, in seconds."""

    def __init__(self, step: float = 0.25, start: float = 1000.0) -> None:
        self.step = step
        self.value = start

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        return current


def make_request(
    *,
    trace_id: str = "trace-1",
    team: str | None = "gtm",
    workflow: str | None = "quote_approval",
    text: str = "How much discount can I give on this quote?",
) -> CompletionRequest:
    metadata = {"trace_id": trace_id}
    if team is not None:
        metadata["team"] = team
    if workflow is not None:
        metadata["workflow"] = workflow
    return CompletionRequest(messages=[Message(role="user", content=text)], metadata=metadata)


def make_response(
    *,
    model: str = HAIKU,
    provider: str = "anthropic",
    tier: Complexity = Complexity.TRIVIAL,
    prompt_tokens: int = 1000,
    completion_tokens: int = 500,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cost_usd: float = 0.0035,
    latency_ms: int = 42,
    fallback_from: Sequence[str] = (),
    text: str = "Up to 15% without approval.",
) -> CompletionResponse:
    return CompletionResponse(
        text=text,
        model=model,
        provider=provider,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        ),
        latency_ms=latency_ms,
        routed_tier=tier,
        fallback_from=list(fallback_from),
    )


def make_record(
    *,
    index: int = 0,
    trace_id: str | None = None,
    team: str = "gtm",
    workflow: str = "quote_approval",
    tier: Complexity | None = Complexity.TRIVIAL,
    provider: str | None = "anthropic",
    model: str | None = HAIKU,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    cost_usd: float = 0.001,
    latency_ms: int = 100,
    status: RecordStatus = "ok",
    minutes_ago: float = 0.0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    provider_latency_ms: int | None = None,
    guard_categories: Sequence[str] = (),
    fallback_from: Sequence[str] = (),
    error: str | None = None,
    prompt_text: str | None = None,
    completion_text: str | None = None,
) -> TelemetryRecord:
    """A row seeded straight into the store, bypassing the recorder.

    The verdict, failover and text columns are settable because the console's
    read surface projects them, and a row built through the recorder cannot
    reach every shape of them — a blocked request never gets a provider
    latency, an exhausted chain never gets a winning model.
    """
    return TelemetryRecord(
        record_id=f"rec-{index:04d}",
        trace_id=trace_id or f"trace-{index:04d}",
        ts=EPOCH - timedelta(minutes=minutes_ago),
        team=team,
        workflow=workflow,
        tier=tier,
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        provider_latency_ms=provider_latency_ms,
        guard_categories=list(guard_categories),
        fallback_from=list(fallback_from),
        status=status,
        error=error,
        prompt_text=prompt_text,
        completion_text=completion_text,
    )


def open_store(tmp_path: Path) -> JsonlSqliteStore:
    return JsonlSqliteStore(tmp_path / "events.jsonl", tmp_path / "events.db")


def seed(store: JsonlSqliteStore, records: Sequence[TelemetryRecord]) -> None:
    for record in records:
        store.write(record)
