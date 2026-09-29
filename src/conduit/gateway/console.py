"""`/api/console/*` — the read surface the operator console renders.

Five GET routes over data the gateway already holds: recent requests, one
request in detail, provider and breaker state, the eval plane's committed
baselines, and the same aggregates `/metrics` computes. Nothing here measures,
prices or decides anything of its own; every number traces to a telemetry row,
a config file, or a live circuit breaker.

**This is a PII surface, and that is the design constraint that shapes it.**
An operator console that renders whatever a caller typed is an exfiltration
endpoint wearing a dashboard costume — the more so here, because telemetry's
`prompt_text` is the *original* prompt, captured in `Recorder.begin` before the
ingress guard has run. The vendor never receives that string; redaction happens
between the guard and the provider call. So the console is built on three rules:

*Text is absent, not merely redacted.* Every default response omits prompt and
completion text entirely. A field that is never populated cannot leak through a
serialisation bug, a log line or an over-eager client cache.

*There is no raw mode.* `?reveal=true` returns text with the guard chain's
redaction applied **at read time**, and the console offers no way to reach the
stored original. If the vendor is not trusted with a customer's email address,
a browser tab is not either. This also means the console stays safe on a
deployment that has flipped `persist_text: true` without thinking about it.

*Access is a grant, not an inheritance.* Traces are cross-team data, so a key
that can call `/v1/chat/completions` reaches none of this by default. The key's
row carries `console_scope`, and reveal is a second, stronger grant than read;
every authorized reveal is audit-logged whether or not it found anything.

Team scoping is enforced on every route: a key sees its own team's rows unless
its row sets `console_all_teams`. A trace belonging to another team is reported
as absent rather than forbidden, so the status code is not an existence oracle.
"""

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal

import structlog
from fastapi import APIRouter, Depends, Header, Query
from fastapi import Path as PathParam
from pydantic import BaseModel, Field

from conduit.contracts import Complexity
from conduit.eval.gate import Baseline, BaselineMissingError, load_baseline
from conduit.eval.settings import EvalSettings, load_eval
from conduit.gateway.auth import ApiKeyDirectory, Principal, extract_key
from conduit.gateway.errors import ForbiddenError, NotFoundError
from conduit.gateway.pipeline import Pipeline
from conduit.guards import GuardChain
from conduit.providers.failover import BreakerState
from conduit.providers.registry import ModelRegistry
from conduit.router.policy import TIER_LADDER, PolicyError
from conduit.router.service import Router
from conduit.telemetry.metrics import MetricsSnapshot, collect
from conduit.telemetry.recorder import TelemetryRecord
from conduit.telemetry.store import TelemetryQuery, TelemetryStore

__all__ = [
    "CONSOLE_PREFIX",
    "ConsoleProvider",
    "EvalSuiteView",
    "EvalsView",
    "FailoverAttempt",
    "GuardSummary",
    "ProvidersView",
    "RevealedText",
    "TierChainView",
    "TraceDetail",
    "TraceList",
    "TraceSummary",
    "TraceTimings",
    "create_console_router",
]

log = structlog.get_logger("conduit.gateway")

#: Everything this module serves lives under one prefix, so a deployment can
#: put the whole console behind a network policy with one rule.
CONSOLE_PREFIX = "/api/console"

#: Default and maximum page size for the trace feed. The cap is not politeness:
#: the store materialises every row it returns, and an uncapped `limit` turns a
#: read endpoint into a way to pull the entire telemetry log in one request.
DEFAULT_TRACE_LIMIT = 50
MAX_TRACE_LIMIT = 500

#: Prefixes `Pipeline` writes into `TelemetryRecord.guard_categories`. Parsing
#: them back apart is presentation of a stored fact, not a new judgement:
#: `_egress` namespaces its categories and `complete` marks the blocking guard.
_EGRESS_PREFIX = "egress:"
_BLOCKER_PREFIX = "guard:"


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #


class GuardSummary(BaseModel):
    """What the guards said about one request, split by the stage that said it."""

    ingress_categories: list[str] = Field(default_factory=list)
    egress_categories: list[str] = Field(default_factory=list)
    blocked: bool = False
    #: The guard that blocked, when one did. `None` on a request that passed.
    blocked_by: str | None = None


class TraceSummary(BaseModel):
    """One row of the trace feed. Metadata only — never any text."""

    trace_id: str
    ts: datetime
    team: str
    workflow: str
    tier: Complexity | None = None
    provider: str | None = None
    model: str | None = None
    status: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: int
    guards: GuardSummary
    #: Models tried and abandoned before the one that answered.
    failover_hops: int = 0


class TraceList(BaseModel):
    generated_at: datetime
    #: Echoed so a console can show what it is looking at, and so a reviewer can
    #: tell an empty window from an empty gateway.
    filters: TelemetryQuery
    traces: list[TraceSummary]


class TraceTimings(BaseModel):
    """The only two latencies that were actually measured, plus their gap.

    Conduit does not time individual pipeline stages, so this deliberately does
    not pretend to: `total_ms` is the recorder's wall clock over the whole
    request and `provider_ms` is what the adapter reported for its own call.
    Their difference is everything Conduit itself did — guards, routing, budget
    accounting — which is the number worth showing, and it is honest because
    both of its terms are stored rather than modelled.
    """

    total_ms: int
    provider_ms: int | None = None
    #: `None` when no provider answered. Floored at zero: the two clocks are
    #: taken by different processes and can disagree by a millisecond, and a
    #: negative overhead would be a measurement artefact rendered as a fact.
    gateway_overhead_ms: int | None = None


class FailoverAttempt(BaseModel):
    """One hop of the chain, reconstructed from `fallback_from` plus the winner."""

    model: str
    outcome: Literal["failed", "served"]


class RevealedText(BaseModel):
    """The text block. Populated only by an explicit, audited reveal.

    `available: false` with a reason is returned rather than an error, because
    "this deployment does not persist text" and "you may not see it" are
    different answers and a console should be able to say which one it got.
    """

    requested: bool = False
    available: bool = False
    #: One of: `not_requested`, `not_persisted`, `redacted`.
    reason: str = "not_requested"
    prompt: str | None = None
    completion: str | None = None
    #: How many spans the guard chain replaced on the way out. Zero with
    #: `reason: redacted` means the stored text held nothing the guards detect.
    redacted_entities: int = 0


class TraceDetail(TraceSummary):
    """One request, in full. Still no text unless a reveal was authorized."""

    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    timings: TraceTimings
    #: Chain order: every model tried, ending in the one that served.
    failover: list[FailoverAttempt] = Field(default_factory=list)
    error: str | None = None
    text: RevealedText = Field(default_factory=RevealedText)


class ConsoleProvider(BaseModel):
    name: str
    #: Live probe, exactly as `/readyz` does it. `False` also covers a probe
    #: that raised — an adapter is not healthy if asking costs an exception.
    reachable: bool
    breaker: BreakerState
    models: list[str] = Field(default_factory=list)


class TierChainView(BaseModel):
    """A tier's configured chain beside the part of it that can actually run."""

    tier: Complexity
    configured: list[str] = Field(default_factory=list)
    #: `configured` minus models whose provider is dormant or which are not
    #: eligible for the tier. The gap is the failover story: a chain can look
    #: three deep in YAML and be one deep in a checkout with no credentials.
    usable: list[str] = Field(default_factory=list)


class ProvidersView(BaseModel):
    generated_at: datetime
    providers: list[ConsoleProvider] = Field(default_factory=list)
    chains: list[TierChainView] = Field(default_factory=list)


class EvalSuiteView(BaseModel):
    """One suite's configuration and its committed baseline."""

    suite: str
    scorer: str
    description: str = ""
    pass_threshold: float
    baseline: Baseline | None = None
    #: A suite with no committed baseline fails the gate by construction, so
    #: this is a red panel rather than a missing one.
    baseline_missing: bool = False


class EvalsView(BaseModel):
    """The eval plane as committed, not as freshly run.

    These are the numbers `evals/baselines/*.json` holds — the reference points
    CI compares a run against. The gate's verdict comes from a run in CI; this
    endpoint deliberately does not start one, because a suite is minutes of work
    and a GET that executes it would be a denial-of-service lever.
    """

    generated_at: datetime
    configured: bool = True
    #: Set when `config/eval.yaml` could not be read, so the console can show
    #: "not configured" rather than an empty list that looks like zero suites.
    detail: str = ""
    model_under_test: str = ""
    judge_model: str = ""
    regression_threshold: float = 0.0
    suites: list[EvalSuiteView] = Field(default_factory=list)
    #: True when every configured suite has a baseline to be gated against.
    gate_ready: bool = False


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #


def _summarise_guards(categories: Sequence[str]) -> GuardSummary:
    """Split a stored category list back into the stages that produced it."""
    summary = GuardSummary()
    for category in categories:
        if category.startswith(_EGRESS_PREFIX):
            summary.egress_categories.append(category[len(_EGRESS_PREFIX) :])
        elif category.startswith(_BLOCKER_PREFIX):
            summary.blocked_by = category[len(_BLOCKER_PREFIX) :]
        else:
            summary.ingress_categories.append(category)
    return summary


def _to_summary(record: TelemetryRecord) -> TraceSummary:
    guards = _summarise_guards(record.guard_categories)
    guards.blocked = record.status == "blocked"
    return TraceSummary(
        trace_id=record.trace_id,
        ts=record.ts,
        team=record.team,
        workflow=record.workflow,
        tier=record.tier,
        provider=record.provider,
        model=record.model,
        status=record.status,
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        total_tokens=record.total_tokens,
        cost_usd=record.cost_usd,
        latency_ms=record.latency_ms,
        guards=guards,
        failover_hops=len(record.fallback_from),
    )


def _timings(record: TelemetryRecord) -> TraceTimings:
    overhead = (
        max(0, record.latency_ms - record.provider_latency_ms)
        if record.provider_latency_ms is not None
        else None
    )
    return TraceTimings(
        total_ms=record.latency_ms,
        provider_ms=record.provider_latency_ms,
        gateway_overhead_ms=overhead,
    )


def _failover(record: TelemetryRecord) -> list[FailoverAttempt]:
    """The chain as walked: every abandoned model, then the one that answered.

    `fallback_from` is the failover layer's own list of models tried and not
    answered, so a served request's chain is that list plus `model`. A request
    that never reached a provider has whatever hops it managed and no winner.
    """
    attempts = [FailoverAttempt(model=model, outcome="failed") for model in record.fallback_from]
    if record.status == "ok" and record.model:
        attempts.append(FailoverAttempt(model=record.model, outcome="served"))
    return attempts


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #


def create_console_router(
    *,
    pipeline: Pipeline,
    directory: ApiKeyDirectory,
    registry: ModelRegistry,
    router: Router,
    store: TelemetryStore,
    config_dir: Path | None = None,
) -> APIRouter:
    """Build the console router with its collaborators closed over.

    Same closure style as `create_routes`, and for the same reason: two apps in
    one process must be able to hold two independent configurations.

    Note what is *not* passed in — this router has no provider clients, no
    recorder and no way to write. It reads the store, probes health, and asks
    the pipeline's own guard chain to redact. A console cannot alter the system
    it is watching.
    """
    console = APIRouter(prefix=CONSOLE_PREFIX, tags=["console"])

    async def authenticate(
        x_api_key: Annotated[str | None, Header()] = None,
        authorization: Annotated[str | None, Header()] = None,
    ) -> Principal:
        return directory.resolve(extract_key(x_api_key, authorization))

    async def reader(principal: Annotated[Principal, Depends(authenticate)]) -> Principal:
        """Every console route depends on this, so none can forget the grant.

        Deliberately *not* rate-limited alongside the completion routes: a
        console polls, and sharing a per-key budget with completions would mean
        an open dashboard tab could 429 the traffic it is there to watch.
        """
        if not principal.console_scope.may_read:
            raise ForbiddenError(
                "This API key does not have operator-console access. Grant it by "
                "setting `console_scope: read` on the key's row in config/gateway.yaml."
            )
        return principal

    Reader = Annotated[Principal, Depends(reader)]  # noqa: N806

    def _team_of(principal: Principal) -> str | None:
        """The team filter for this caller, or None for a whole-gateway key."""
        return None if principal.console_all_teams else principal.team

    # -- traces ------------------------------------------------------------ #

    @console.get(
        "/traces",
        response_model=TraceList,
        summary="Recent requests: tier, cost, latency and guard verdicts",
    )
    def traces(
        principal: Reader,
        limit: int = Query(default=DEFAULT_TRACE_LIMIT, gt=0, le=MAX_TRACE_LIMIT),
        window_minutes: int | None = Query(default=None, gt=0),
        status: str | None = None,
        tier: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> TraceList:
        since = (
            datetime.now(UTC) - timedelta(minutes=window_minutes)
            if window_minutes is not None
            else None
        )
        query = TelemetryQuery(
            since=since,
            team=_team_of(principal),
            status=status,
            tier=tier,
            provider=provider,
            model=model,
            limit=limit,
            newest_first=True,
        )
        return TraceList(
            generated_at=datetime.now(UTC),
            filters=query,
            traces=[_to_summary(record) for record in store.records(query)],
        )

    @console.get(
        "/traces/{trace_id}",
        response_model=TraceDetail,
        summary="One request in detail, with an optional audited text reveal",
    )
    async def trace_detail(
        principal: Reader,
        trace_id: Annotated[str, PathParam(min_length=1, max_length=128)],
        reveal: bool = Query(
            default=False,
            description=(
                "Return prompt and completion text, redacted at read time by the "
                "gateway's own guard chain. Requires `console_scope: reveal` and is "
                "audit-logged. There is no way to retrieve the unredacted original."
            ),
        ),
    ) -> TraceDetail:
        record = _one(store, trace_id, _team_of(principal))
        detail = TraceDetail(
            **_to_summary(record).model_dump(),
            cache_read_tokens=record.cache_read_tokens,
            cache_write_tokens=record.cache_write_tokens,
            timings=_timings(record),
            failover=_failover(record),
            error=record.error,
        )
        detail.text = await _reveal(record, principal, pipeline.guards, requested=reveal)
        return detail

    # -- providers --------------------------------------------------------- #

    @console.get(
        "/providers",
        response_model=ProvidersView,
        summary="Registry state, circuit breakers and the per-tier failover chains",
    )
    async def providers(principal: Reader) -> ProvidersView:
        # Probes run concurrently and never propagate, exactly as `/readyz`
        # does it: one console refresh should cost one provider timeout, not
        # the sum of them, and a broken adapter must not take the panel down.
        adapters = registry.providers
        results = await asyncio.gather(
            *(adapter.health() for adapter in adapters), return_exceptions=True
        )
        breakers = pipeline.failover.breakers
        states = [
            ConsoleProvider(
                name=adapter.name,
                reachable=result is True,
                breaker=breakers.state(adapter.name),
                models=[
                    model_id
                    for model_id in registry.model_ids
                    if registry.spec(model_id).provider == adapter.name
                ],
            )
            for adapter, result in zip(adapters, results, strict=True)
        ]
        return ProvidersView(
            generated_at=datetime.now(UTC),
            providers=states,
            chains=[_chain_view(router, registry, tier) for tier in TIER_LADDER],
        )

    # -- evals ------------------------------------------------------------- #

    @console.get(
        "/evals",
        response_model=EvalsView,
        summary="Committed golden-set baselines and whether the gate can run",
    )
    def evals(principal: Reader) -> EvalsView:
        try:
            settings = load_eval(config_dir)
        except Exception as exc:
            # Config problems are an operator's to fix, and the console should
            # say so rather than render an empty panel that reads as "no
            # suites". The exception text is ours (a `ConfigError` or a
            # pydantic message over our own schema), not a vendor's.
            log.warning("console.evals_unavailable", error=str(exc))
            return EvalsView(
                generated_at=datetime.now(UTC),
                configured=False,
                detail=f"config/eval.yaml could not be loaded: {exc}",
            )
        return _evals_view(settings)

    # -- metrics ----------------------------------------------------------- #

    @console.get(
        "/metrics",
        response_model=MetricsSnapshot,
        summary="The same aggregates as /metrics, team-scoped and auth-gated",
    )
    def metrics(
        principal: Reader,
        window_minutes: int | None = Query(default=None, gt=0),
        provider: str | None = None,
        tier: str | None = None,
    ) -> MetricsSnapshot:
        """`collect` is telemetry's, not reimplemented here.

        The unauthenticated `/metrics` this mirrors sums the whole gateway. This
        one applies the caller's team filter first, so a tenant key's console
        cannot infer another team's spend from an aggregate it was refused at
        row level.
        """
        since = (
            datetime.now(UTC) - timedelta(minutes=window_minutes)
            if window_minutes is not None
            else None
        )
        return collect(
            store,
            TelemetryQuery(since=since, team=_team_of(principal), provider=provider, tier=tier),
        )

    return console


# --------------------------------------------------------------------------- #
# Handlers' helpers
# --------------------------------------------------------------------------- #


def _one(store: TelemetryStore, trace_id: str, team: str | None) -> TelemetryRecord:
    """Fetch one trace, scoped to a team, or 404.

    `trace_id` is caller-supplied — `Recorder.begin` takes it from request
    metadata and only mints one when it is absent — so it is not guaranteed
    unique. The newest matching row wins. It is also not a leak vector: the
    `team` column is stamped from the key by `Pipeline._attribute`, never from
    the caller, so guessing another team's id still matches nothing here.
    """
    rows = store.records(TelemetryQuery(trace_id=trace_id, team=team, limit=1, newest_first=True))
    if not rows:
        raise NotFoundError(f"No trace {trace_id!r} in this gateway's telemetry.")
    return rows[0]


async def _reveal(
    record: TelemetryRecord,
    principal: Principal,
    guards: GuardChain,
    *,
    requested: bool,
) -> RevealedText:
    """Resolve the text block for one trace detail.

    The redaction is applied *here*, on the way out, rather than trusted to have
    been applied on the way in — because it was not. `Recorder.begin` captures
    the prompt before the ingress guard runs, so `record.prompt_text` holds the
    original a human typed, not the string the vendor received. Re-inspecting it
    is what makes this endpoint safe on a deployment that turned `persist_text`
    on without reasoning about who can read the console.
    """
    if not requested:
        return RevealedText()
    if not principal.console_scope.may_reveal:
        raise ForbiddenError(
            "Revealing request text requires `console_scope: reveal` on this API "
            "key. Read access alone does not include prompt or completion text."
        )

    prompt, prompt_entities = await _redact(guards, record.prompt_text)
    completion, completion_entities = await _redact(guards, record.completion_text)
    available = prompt is not None or completion is not None

    # The audit event is the *attempt*, not the hit: a reveal that found nothing
    # is still someone asking to see a customer's prompt, and the question of
    # who asked is the one an incident review starts from. Logged at warning so
    # it survives a default log level. No text and no entity map, ever — the
    # guards package's own contract forbids persisting one.
    log.warning(
        "console.trace_revealed",
        trace_id=record.trace_id,
        key=principal.fingerprint,
        team=principal.team,
        record_team=record.team,
        text_available=available,
        redacted_entities=prompt_entities + completion_entities,
    )

    if not available:
        return RevealedText(
            requested=True,
            available=False,
            reason="not_persisted",
            # `persist_text` is false in the shipped config, so this is the
            # ordinary answer and not an error condition.
        )
    return RevealedText(
        requested=True,
        available=True,
        reason="redacted",
        prompt=prompt,
        completion=completion,
        redacted_entities=prompt_entities + completion_entities,
    )


async def _redact(guards: GuardChain, text: str | None) -> tuple[str | None, int]:
    """Run the guard chain over stored text and return its redacted form.

    The verdict's `allowed` flag is ignored on purpose: this is not an admission
    decision, the request was admitted or refused long ago. Only the redaction
    is wanted. A chain that found nothing returns `redacted_text=None`, which
    means the text was already clean rather than that redaction failed.
    """
    if text is None:
        return None, 0
    verdict = await guards.inspect(text)
    return (verdict.redacted_text or text), len(verdict.entity_map)


def _chain_view(router: Router, registry: ModelRegistry, tier: Complexity) -> TierChainView:
    """A tier's configured chain, and the part of it this process can run."""
    try:
        configured = list(router.policy.chain_for(tier).models)
    except PolicyError:  # pragma: no cover - defensive
        # `RoutingConfig` already refuses a config that leaves a tier without a
        # non-empty chain, so this is unreachable from a validated one. It stays
        # because the alternative is a 500 on the whole panel if that ever
        # changes, and an empty row is the more useful failure: omitting the
        # tier would read as "this tier does not exist".
        configured = []
    return TierChainView(
        tier=tier,
        configured=configured,
        usable=list(registry.usable_chain(configured, tier)),
    )


def _evals_view(settings: EvalSettings) -> EvalsView:
    suites: list[EvalSuiteView] = []
    for name, spec in settings.suites.items():
        baseline: Baseline | None = None
        try:
            baseline = load_baseline(settings.baseline_path(name))
        except (BaselineMissingError, ValueError):
            # Missing or unparseable both mean the gate has nothing to compare
            # against, which is the same red panel for an operator.
            baseline = None
        suites.append(
            EvalSuiteView(
                suite=name,
                scorer=spec.scorer,
                description=spec.description.strip(),
                pass_threshold=spec.pass_threshold,
                baseline=baseline,
                baseline_missing=baseline is None,
            )
        )
    return EvalsView(
        generated_at=datetime.now(UTC),
        configured=True,
        model_under_test=settings.model_under_test,
        judge_model=settings.judge_model,
        regression_threshold=settings.regression_threshold,
        suites=suites,
        gate_ready=bool(suites) and all(not suite.baseline_missing for suite in suites),
    )
