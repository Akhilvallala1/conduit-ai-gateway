"""The FastAPI application: configure structlog, wire wave 2, serve.

`create_app` is where the four independently-built packages are assembled into
one object graph, once, at startup:

    registry  (#3) ── FailoverChain ──┐
    Router    (#4) ───────────────────┼── Pipeline ── routes
    GuardChain(#5) ───────────────────┤
    Recorder  (#7) ───────────────────┘  and /metrics, mounted from #7

Every collaborator is an optional constructor argument. That is not test
scaffolding for its own sake: it is the only way to exercise failover, budget
rejection and circuit-breaker behaviour without a network, which AC-17 and the
no-network testing rule both require.

The telemetry store is a process-lifetime resource, so it opens in the lifespan
and closes on shutdown. Everything else is stateless or in-memory.
"""

import logging
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI

from conduit.config import ConduitConfig, load_config
from conduit.guards import GuardChain, build_chain
from conduit.providers.failover import CircuitBreakers, FailoverChain, FailoverEvent
from conduit.providers.registry import ModelRegistry, build_registry
from conduit.router.cost import SpendLedger
from conduit.router.service import Router
from conduit.telemetry import build_recorder, create_metrics_router
from conduit.telemetry.recorder import Recorder

from .auth import ApiKeyDirectory
from .console import create_console_router
from .errors import install_error_handlers
from .pipeline import Pipeline
from .ratelimit import RateLimiter
from .routes import create_routes
from .settings import GatewaySettings, load_gateway

__all__ = ["configure_logging", "create_app"]

log = structlog.get_logger("conduit.gateway")


def configure_logging(*, level: int = logging.INFO, json_logs: bool = True) -> None:
    """Structured logging, shared by the gateway and the packages it wires.

    The guards package logs verdicts through the stdlib `logging` module with
    `extra=` fields, so stdlib records are routed through the same structlog
    formatter — otherwise half the request's log lines would be JSON and half
    would be plain text.
    """
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.StackInfoRenderer(),
    ]
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            *shared,
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def _failover_logger() -> Callable[[FailoverEvent], None]:
    """Turn #3's `FailoverEvent` structs into log lines.

    The providers package deliberately emits a struct rather than logging, so
    that it never depends on structlog. This is the other half of that bargain:
    the circuit-breaker event AC-3 asks to see logged is logged here.
    """

    def emit(event: FailoverEvent) -> None:
        if event.outcome == "success":
            return
        log.warning(
            "gateway.failover",
            model=event.model,
            provider=event.provider,
            outcome=event.outcome,
            breaker=event.breaker_state.value,
            error=event.error,
        )

    return emit


def create_app(
    *,
    config: ConduitConfig | None = None,
    gateway: GatewaySettings | None = None,
    config_dir: Path | None = None,
    registry: ModelRegistry | None = None,
    guards: GuardChain | None = None,
    recorder: Recorder | None = None,
    router: Router | None = None,
    breakers: CircuitBreakers | None = None,
    limiter: RateLimiter | None = None,
    setup_logging: bool = True,
) -> FastAPI:
    """Assemble the gateway. Every dependency is injectable; none is required."""
    if setup_logging:
        configure_logging()

    config = config if config is not None else load_config(config_dir)
    gateway = gateway if gateway is not None else load_gateway(config_dir)
    directory = ApiKeyDirectory(gateway)

    registry = registry if registry is not None else build_registry(models=config.models)

    # Per-key daily limits come from the API-key table, but the ledger that
    # enforces them belongs to the router. Seeding it here is what connects
    # `gateway.yaml`'s `daily_budget_usd` to #4's budget engine.
    if router is None:
        ledger = SpendLedger(limits=directory.budget_limits())
        router = Router(config, ledger=ledger)
    else:
        for key, limit in directory.budget_limits().items():
            router.budget.ledger.set_limit(key, limit)

    guards = guards if guards is not None else build_chain(config.guards)
    recorder = recorder if recorder is not None else build_recorder(config_dir)
    failover = FailoverChain(registry, breakers=breakers, on_event=_failover_logger())
    limiter = limiter if limiter is not None else RateLimiter()

    pipeline = Pipeline(router=router, failover=failover, guards=guards, recorder=recorder)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        log.info(
            "gateway.startup",
            providers=list(registry.provider_names),
            models=len(registry),
            keys=len(directory.principals),
            require_auth=directory.require_auth,
        )
        if not directory.require_auth:
            log.warning("gateway.auth_disabled", detail="require_auth is false; keys not checked")
        try:
            yield
        finally:
            recorder.store.close()
            for provider in registry.providers:
                aclose = getattr(provider, "aclose", None)
                if aclose is not None:
                    await aclose()
            log.info("gateway.shutdown")

    app = FastAPI(
        title="Conduit",
        version="0.1.0",
        summary="A proxy-first AI gateway: routing, guards, failover and cost telemetry.",
        lifespan=lifespan,
    )
    install_error_handlers(app)
    app.include_router(
        create_routes(pipeline=pipeline, directory=directory, limiter=limiter, registry=registry)
    )
    # Telemetry owns the read side of its own data (`/metrics`, `/dashboard`).
    app.include_router(create_metrics_router(recorder.store))
    # The operator console reads the same store, but auth-gated and team-scoped:
    # `/metrics` sums the whole gateway for anyone who can reach the port, which
    # is defensible for a scrape target inside a network policy and is not
    # defensible for a browser surface that also renders prompts.
    app.include_router(
        create_console_router(
            pipeline=pipeline,
            directory=directory,
            registry=registry,
            router=router,
            store=recorder.store,
            config_dir=config_dir,
        )
    )

    # Held for tests and for anything that needs to reach past the HTTP surface.
    app.state.pipeline = pipeline
    app.state.registry = registry
    app.state.router = router
    app.state.directory = directory
    app.state.limiter = limiter
    app.state.recorder = recorder
    return app
