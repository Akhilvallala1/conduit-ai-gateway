"""The request pipeline: auth, guards, router, providers, guards, telemetry.

The order is fixed by the epic and enforced here rather than merely documented.
`StageLog.enter` refuses a stage that does not come strictly after the last one,
so moving the router call above the ingress guard raises `PipelineOrderError`
instead of quietly shipping an unguarded prompt to a vendor (AC-5).

Four seams are worth reading before changing anything in this module, because
each is a place two independently-built packages disagree until the gateway
makes a choice:

**One telemetry row per request.** `recorder.record(req)` wraps the *entire*
request — every failover hop, the guards on both ends, the router. Wrapping the
provider call instead would emit one row per hop and silently break #7's "exactly
one row per request" invariant.

**Redaction is per-request, not per-message.** `GuardChain` inspects one string
and `redact()` restarts its `<PERSON_n>` counters on every call, so inspecting
each message separately would mint two different `<PERSON_1>`s and merge them
into one entity map — rehydrating the wrong name into the caller's answer. The
chain therefore sees the whole prompt joined, and the resulting map is applied
back to each message by substituting originals for placeholders.

**The egress guard observes; it does not redact.** Re-scanning the completion is
worth doing, but applying that scan's `redacted_text` would hand the caller
fresh placeholders that the ingress entity map cannot resolve — the exact thing
AC-8's round trip forbids. So egress uses the verdict for risk and categories,
then rehydrates from the ingress map.

**Spend is priced from the registry.** `response.usage.cost_usd` is the
adapter's own figure; the ledger is fed `RoutingPolicy.price()` instead, which
reads `config/models.yaml`. The recorder reprices identically. Adding a third
path here is how the cost column and `/metrics` start to disagree.
"""

from dataclasses import dataclass, field
from enum import Enum

import structlog

from conduit.contracts import (
    CompletionRequest,
    CompletionResponse,
    Complexity,
    GuardVerdict,
    Message,
)
from conduit.gateway.auth import Principal
from conduit.gateway.errors import GuardRejectedError
from conduit.guards import GuardChain, rehydrate
from conduit.providers.base import TIER_METADATA_KEY
from conduit.providers.failover import AllProvidersFailedError, FailoverChain
from conduit.router.policy import TIER_LADDER, PolicyError
from conduit.router.service import Router, RouteResult
from conduit.telemetry.recorder import Recorder, RequestTrace

__all__ = [
    "PIPELINE_ORDER",
    "Pipeline",
    "PipelineOrderError",
    "PipelineOutcome",
    "Stage",
    "StageLog",
    "blocking_guard",
]

log = structlog.get_logger("conduit.gateway")


class Stage(str, Enum):
    """The six stages of the epic's pipeline, in their only legal order."""

    AUTH = "auth"
    INGRESS_GUARD = "ingress_guard"
    ROUTER = "router"
    PROVIDER = "provider"
    EGRESS_GUARD = "egress_guard"
    TELEMETRY = "telemetry"


#: The frozen order. `routes.py` and `test_pipeline_order.py` both read this.
PIPELINE_ORDER: tuple[Stage, ...] = (
    Stage.AUTH,
    Stage.INGRESS_GUARD,
    Stage.ROUTER,
    Stage.PROVIDER,
    Stage.EGRESS_GUARD,
    Stage.TELEMETRY,
)

_STAGE_INDEX = {stage: index for index, stage in enumerate(PIPELINE_ORDER)}


class PipelineOrderError(RuntimeError):
    """A stage ran out of order. A bug in this module, never in a request."""


class StageLog:
    """Records stage entry and refuses anything but forward progress.

    Skips are legal — a guard rejection goes straight from `ingress_guard` to
    `telemetry` — but repeats and reversals are not, which is precisely the
    difference between short-circuiting and reordering.
    """

    def __init__(self) -> None:
        self._stages: list[Stage] = []

    @property
    def stages(self) -> tuple[Stage, ...]:
        return tuple(self._stages)

    def enter(self, stage: Stage) -> None:
        if self._stages:
            previous = self._stages[-1]
            if _STAGE_INDEX[stage] <= _STAGE_INDEX[previous]:
                raise PipelineOrderError(
                    f"pipeline stage {stage.value!r} cannot run after {previous.value!r}; "
                    f"the fixed order is {' -> '.join(s.value for s in PIPELINE_ORDER)}"
                )
        self._stages.append(stage)


def blocking_guard(verdict: GuardVerdict, default: str = "chain") -> str:
    """Name the guard behind a block.

    `GuardVerdict` has no `name`, and the composed verdict a `GuardChain`
    returns is named "chain", so telemetry cannot attribute a block to the PII
    guard rather than the injection guard without the gateway working it out.
    Categories are `"<family>:<subtype>"`, and the family *is* the guard name
    for the two shipped guards, so the first category answers it.
    """
    for category in verdict.categories:
        family, _, _ = category.partition(":")
        if family and family != "guard":
            return family
    return default


@dataclass
class PipelineOutcome:
    """What one trip through the pipeline produced.

    A guard block is a value, not an exception, so it can travel back out of the
    telemetry context manager without `Recorder.record` recording the request as
    a failure: `trace.observe_guard` already set the status to "blocked", and an
    exception escaping the block would overwrite that with "error".
    """

    trace_id: str
    stages: tuple[Stage, ...] = ()
    response: CompletionResponse | None = None
    rejection: GuardRejectedError | None = None
    route: RouteResult | None = None
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.rejection is not None

    def unwrap(self) -> CompletionResponse:
        """The response, or the guard rejection raised as an HTTP 400."""
        if self.rejection is not None:
            raise self.rejection
        if self.response is None:  # pragma: no cover - defensive
            raise RuntimeError("pipeline produced neither a response nor a rejection")
        return self.response


def _redact_messages(messages: list[Message], entity_map: dict[str, str]) -> list[Message]:
    """Swap originals for placeholders in each message, per the ingress map.

    Substituting by value rather than splitting the redacted join keeps message
    boundaries exact: the chain inspected a joined string, but the vendor must
    receive the same turn structure the caller sent.

    Longest original first, so a name that contains another entity's text cannot
    be half-replaced. Every occurrence is redacted, not just the analyzed one —
    redacting more than was detected is the safe direction, and rehydration
    reverses all of it.
    """
    if not entity_map:
        return messages
    ordered = sorted(entity_map.items(), key=lambda item: len(item[1]), reverse=True)
    redacted: list[Message] = []
    for message in messages:
        content = message.content
        for placeholder, original in ordered:
            if original:
                content = content.replace(original, placeholder)
        redacted.append(message.model_copy(update={"content": content}))
    return redacted


def _cap_tier(tier: Complexity, ceiling: Complexity) -> Complexity:
    return ceiling if TIER_LADDER.index(tier) > TIER_LADDER.index(ceiling) else tier


class Pipeline:
    """Holds the four wave-2 components and runs a request through them once."""

    def __init__(
        self,
        *,
        router: Router,
        failover: FailoverChain,
        guards: GuardChain,
        recorder: Recorder,
    ) -> None:
        self._router = router
        self._failover = failover
        self._guards = guards
        self._recorder = recorder

    @property
    def router(self) -> Router:
        return self._router

    @property
    def recorder(self) -> Recorder:
        return self._recorder

    @property
    def guards(self) -> GuardChain:
        """The same chain the request path uses.

        The console re-runs it over stored text to redact a reveal, so it has to
        be this chain and not a freshly built one: a gateway configured with a
        narrower guard set must not get a wider redaction in its console, nor a
        narrower one.
        """
        return self._guards

    @property
    def failover(self) -> FailoverChain:
        """The chain, and through it the registry and the circuit breakers.

        The console's provider panel needs breaker state, and the breakers are
        created inside `create_app` and handed here. Exposing them through the
        pipeline keeps that single instance authoritative — a second
        `CircuitBreakers` would report a healthy system while the one on the
        request path had every provider open.
        """
        return self._failover

    async def complete(self, req: CompletionRequest, principal: Principal) -> PipelineOutcome:
        """Run the fixed pipeline. Exactly one telemetry row, whatever happens."""
        attributed = self._attribute(req, principal)
        stages = StageLog()

        # `record` opens before the first stage and writes on exit, so the row
        # covers the whole request: guards, routing, every failover hop.
        with self._recorder.record(attributed) as trace:
            # AUTH already happened — the caller cannot construct a `Principal`
            # without it — so entering the stage records a fact rather than
            # performing one. Rate-limit admission also ran, inside this stage.
            stages.enter(Stage.AUTH)
            outcome = PipelineOutcome(trace_id=trace.trace_id)

            verdict = await self._ingress(attributed, stages, trace)
            if not verdict.allowed:
                guard = blocking_guard(verdict)
                trace.add_categories([f"guard:{guard}"])
                stages.enter(Stage.TELEMETRY)
                outcome.rejection = GuardRejectedError(verdict, guard, stage="ingress")
                outcome.stages = stages.stages
                log.warning(
                    "gateway.guard_blocked",
                    guard=guard,
                    stage="ingress",
                    categories=list(verdict.categories),
                    risk_score=round(verdict.risk_score, 4),
                    team=principal.team,
                    key=principal.fingerprint,
                    trace_id=trace.trace_id,
                )
                return outcome

            vendor_req = attributed.model_copy(
                update={"messages": _redact_messages(attributed.messages, verdict.entity_map)}
            )

            route = await self._route(vendor_req, principal, stages, trace)
            outcome.route = route
            outcome.headers.update(route.headers())

            vendor_req = vendor_req.model_copy(
                update={
                    "metadata": {**vendor_req.metadata, TIER_METADATA_KEY: route.tier.value},
                }
            )

            response = await self._call(vendor_req, route, stages, trace)
            await self._record_spend(principal, response)

            response, rejection = await self._egress(response, verdict, stages, trace)
            stages.enter(Stage.TELEMETRY)
            outcome.stages = stages.stages
            if rejection is not None:
                outcome.rejection = rejection
                log.warning(
                    "gateway.guard_blocked",
                    guard=rejection.guard,
                    stage="egress",
                    categories=list(rejection.verdict.categories),
                    team=principal.team,
                    key=principal.fingerprint,
                    trace_id=trace.trace_id,
                )
                return outcome
            outcome.response = response

        log.info(
            "gateway.completed",
            trace_id=outcome.trace_id,
            team=principal.team,
            key=principal.fingerprint,
            tier=route.tier.value,
            model=response.model,
            provider=response.provider,
            fallback_from=list(response.fallback_from),
            stages=[stage.value for stage in outcome.stages],
        )
        return outcome

    # -- stages ------------------------------------------------------------ #

    def _attribute(self, req: CompletionRequest, principal: Principal) -> CompletionRequest:
        """Stamp the key's team onto the request before telemetry reads it.

        `Recorder.begin` takes `team` and `workflow` from `metadata`, so a row
        is only attributable to a team if the gateway puts it there. The
        caller's own `team` is not trusted over the key's.
        """
        metadata = {**req.metadata, "team": principal.team}
        return req.model_copy(update={"metadata": metadata})

    async def _ingress(
        self, req: CompletionRequest, stages: StageLog, trace: RequestTrace
    ) -> GuardVerdict:
        stages.enter(Stage.INGRESS_GUARD)
        verdict = await self._guards.inspect("\n\n".join(m.content for m in req.messages))
        trace.observe_guard(verdict)
        return verdict

    async def _route(
        self,
        req: CompletionRequest,
        principal: Principal,
        stages: StageLog,
        trace: RequestTrace,
    ) -> RouteResult:
        stages.enter(Stage.ROUTER)
        # An unmetered anonymous caller passes `None`, which is how the budget
        # enforcer is told not to account for this request at all.
        route = await self._router.route(req, api_key=principal.api_key or None)
        route = self._apply_tier_ceiling(route, principal)
        trace.set_tier(route.tier)
        return route

    def _apply_tier_ceiling(self, route: RouteResult, principal: Principal) -> RouteResult:
        """Enforce the key's `max_tier` policy on the router's verdict."""
        capped = _cap_tier(route.tier, principal.max_tier)
        if capped is route.tier:
            return route
        chain = self._router.policy.chain_for(capped).models
        log.info(
            "gateway.tier_capped",
            requested=route.tier.value,
            capped_to=capped.value,
            key=principal.fingerprint,
        )
        return route.model_copy(update={"tier": capped, "chain": chain})

    async def _call(
        self,
        req: CompletionRequest,
        route: RouteResult,
        stages: StageLog,
        trace: RequestTrace,
    ) -> CompletionResponse:
        """Always through the failover chain, never a bare adapter.

        A `Provider` cannot know which tier it was routed to, so calling one
        directly would return a guessed `routed_tier`. `FailoverChain.complete`
        overwrites it with the real one and owns `fallback_from`.
        """
        stages.enter(Stage.PROVIDER)
        # Name the first model before calling it, so a chain that fails
        # outright still leaves a row pointing at something.
        try:
            primary = self._failover.registry.provider_for(route.primary)
            trace.set_attempt(provider=primary.name, model=route.primary)
        except LookupError:
            trace.set_attempt(provider="unknown", model=route.primary)
        try:
            response = await self._failover.complete(req, route.chain, route.tier)
        except AllProvidersFailedError as exc:
            # The trace learns `fallback_from` from a *successful* response, so
            # a chain that never produced one recorded no hops at all — the row
            # for a total upstream outage was the one row with no failover
            # story on it, which is precisely backwards. The chain knows what it
            # tried; `record_failover` has existed for this since #7 and was
            # simply never wired up here.
            for attempted in exc.attempted:
                trace.record_failover(attempted)
            raise
        trace.set_response(response)
        return response

    async def _record_spend(self, principal: Principal, response: CompletionResponse) -> None:
        """Charge the ledger the registry's price, not the adapter's."""
        if not principal.api_key:
            return
        try:
            cost = self._router.policy.price(response.model, response.usage)
        except PolicyError:
            # A model the registry does not price — the adapter's own figure is
            # all there is. The recorder makes the same choice for the same row.
            cost = response.usage.cost_usd
        await self._router.record_spend(principal.api_key, cost)

    async def _egress(
        self,
        response: CompletionResponse,
        ingress: GuardVerdict,
        stages: StageLog,
        trace: RequestTrace,
    ) -> tuple[CompletionResponse, GuardRejectedError | None]:
        """Re-scan the completion, then rehydrate the caller's own entities.

        The scan runs on the pre-rehydration text so it sees what the vendor
        actually produced, and its redaction is deliberately discarded: the only
        substitution applied on the way out is the ingress map, reversed.

        A block comes back as a value for the same reason an ingress block does
        — raising inside the telemetry block would rewrite the row's "blocked"
        status to "error".
        """
        stages.enter(Stage.EGRESS_GUARD)
        verdict = await self._guards.inspect(response.text)
        # Namespaced so an egress category is never mistaken for the ingress
        # verdict on the same row.
        trace.add_categories(f"egress:{category}" for category in verdict.categories)
        if not verdict.allowed:
            guard = blocking_guard(verdict)
            trace.observe_guard(verdict)
            trace.add_categories([f"guard:{guard}"])
            return response, GuardRejectedError(verdict, guard, stage="egress")

        text = rehydrate(response.text, ingress.entity_map)
        return response.model_copy(update={"text": text}), None
