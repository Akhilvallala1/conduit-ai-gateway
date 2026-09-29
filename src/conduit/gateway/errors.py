"""Structured error envelopes, and the rule that nothing internal escapes.

Every failure the caller can see is a `GatewayError` rendered as JSON in the
OpenAI error shape — `{"error": {"message", "type", "code", ...}}` — because a
drop-in for the OpenAI SDK has to fail in a shape that SDK already parses
(AC-1). Conduit-specific detail rides in an `error.conduit` sub-object, which
unknown-field-tolerant clients ignore.

AC-8 is the load-bearing rule here: **no stack traces and no provider internals
reach a caller.** Upstream exception messages routinely quote vendor response
bodies — `AllProvidersFailedError` embeds the last provider error verbatim — so
the handlers below deliberately log the exception with its traceback and return
a message built from Conduit's own vocabulary instead of `str(exc)`.
"""

import logging
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from conduit.contracts import GuardVerdict
from conduit.providers.failover import AllProvidersFailedError
from conduit.router.cost import BudgetExceededError

__all__ = [
    "BadRequestError",
    "ForbiddenError",
    "GatewayError",
    "GuardRejectedError",
    "NotFoundError",
    "RateLimitExceededError",
    "StreamingUnsupportedError",
    "UnauthorizedError",
    "UpstreamUnavailableError",
    "error_payload",
    "install_error_handlers",
]

log = structlog.get_logger("conduit.gateway")

#: What a caller sees when something we did not anticipate went wrong. It names
#: no module, no vendor and no exception type on purpose.
INTERNAL_MESSAGE = "The gateway failed to process this request."


def error_payload(
    message: str,
    error_type: str,
    *,
    code: str | None = None,
    conduit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The one envelope shape. OpenAI-compatible, with room for our own fields."""
    body: dict[str, Any] = {
        "message": message,
        "type": error_type,
        "param": None,
        "code": code or error_type,
    }
    if conduit:
        body["conduit"] = conduit
    return {"error": body}


class GatewayError(Exception):
    """A failure with a caller-safe rendering already decided.

    Subclasses set the status and the vocabulary. The message passed in is
    assumed safe to return; anything derived from an upstream exception must be
    summarised by the raiser, not forwarded.
    """

    status_code = 500
    error_type = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        headers: dict[str, str] | None = None,
        conduit: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.headers = dict(headers or {})
        self.conduit = dict(conduit or {})

    def payload(self) -> dict[str, Any]:
        return error_payload(self.message, self.error_type, code=self.code, conduit=self.conduit)

    def response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status_code, content=self.payload(), headers=self.headers
        )


class UnauthorizedError(GatewayError):
    """No key, or a key that is not in the table. Raised before any other work."""

    status_code = 401
    error_type = "invalid_request_error"

    def __init__(self, message: str = "Missing or invalid API key.") -> None:
        # `WWW-Authenticate` is what makes a 401 a 401 rather than a 403 in
        # disguise, and some HTTP clients retry differently on its presence.
        super().__init__(message, code="invalid_api_key", headers={"WWW-Authenticate": "Bearer"})


class RateLimitExceededError(GatewayError):
    """Per-key request rate exceeded. Always carries `Retry-After` (AC-4)."""

    status_code = 429
    error_type = "rate_limit_error"

    def __init__(self, retry_after_s: int, limit_per_minute: int) -> None:
        super().__init__(
            f"Rate limit of {limit_per_minute} requests/minute exceeded. "
            f"Retry in {retry_after_s}s.",
            code="rate_limit_exceeded",
            headers={"Retry-After": str(retry_after_s)},
            conduit={"limit_per_minute": limit_per_minute, "retry_after_s": retry_after_s},
        )


class GuardRejectedError(GatewayError):
    """A guard blocked the request. 400 with the verdict category (AC-6)."""

    status_code = 400
    error_type = "invalid_request_error"

    def __init__(self, verdict: GuardVerdict, guard: str, stage: str = "ingress") -> None:
        categories = list(verdict.categories)
        super().__init__(
            f"Request blocked by the {guard} guard: {', '.join(categories) or 'policy'}.",
            code="guard_rejected",
            conduit={
                "guard": guard,
                "stage": stage,
                "categories": categories,
                # The score is ours, not a vendor's, and it is what makes a
                # block auditable without echoing the text that caused it.
                "risk_score": round(verdict.risk_score, 4),
            },
        )
        self.verdict = verdict
        self.guard = guard


class StreamingUnsupportedError(GatewayError):
    """`stream: true`. Out of scope by decision, so it is named, not ignored."""

    status_code = 400
    error_type = "invalid_request_error"

    def __init__(self) -> None:
        super().__init__(
            "Streaming is not supported by this gateway. Conduit's completion "
            "contract is unary, so the guard chain and cost accounting cannot "
            "run over a partial response. Retry with `stream: false`.",
            code="streaming_unsupported",
        )


class BadRequestError(GatewayError):
    """A malformed request the caller can fix, described without internals."""

    status_code = 400
    error_type = "invalid_request_error"

    def __init__(self, message: str, *, code: str = "invalid_request") -> None:
        super().__init__(message, code=code)


class ForbiddenError(GatewayError):
    """Authenticated, but this key does not hold the grant the route needs.

    Distinct from 401 on purpose: a 401 tells a caller to present a key, and a
    caller who already presented a valid one would retry forever on it. The
    message names the grant rather than the resource, so a refusal never
    confirms that the thing asked for exists.
    """

    status_code = 403
    error_type = "invalid_request_error"

    def __init__(self, message: str, *, code: str = "insufficient_scope") -> None:
        super().__init__(message, code=code)


class NotFoundError(GatewayError):
    """No such resource — or none this caller is allowed to know about.

    Rows outside the caller's team are reported as absent rather than
    forbidden. A 403 here would be an existence oracle: it would let one team
    confirm another team's trace ids by the status code alone.
    """

    status_code = 404
    error_type = "invalid_request_error"

    def __init__(self, message: str, *, code: str = "not_found") -> None:
        super().__init__(message, code=code)


class UpstreamUnavailableError(GatewayError):
    """Every model in the chain failed. Names the models tried, nothing else.

    Model ids come from `config/models.yaml`, which is ours, so listing them
    leaks nothing. The underlying provider errors — which quote vendor response
    bodies — stay in the log.
    """

    status_code = 502
    error_type = "api_error"

    def __init__(self, attempted: list[str]) -> None:
        super().__init__(
            "No model in the routing chain could serve this request.",
            code="upstream_unavailable",
            conduit={"attempted": attempted},
        )


def _budget_response(exc: BudgetExceededError) -> JSONResponse:
    """429 in `router/cost.py`'s documented shape — not a second invention.

    The header format and status code are #4's, documented at the top of
    `router/cost.py`. The gateway renders them; it does not redefine them.
    """
    decision = exc.decision
    return JSONResponse(
        status_code=exc.status_code,
        headers=exc.headers,
        content=error_payload(
            f"Daily budget of ${decision.limit_usd:.2f} exhausted "
            f"(${decision.spent_usd:.2f} spent on {decision.window.isoformat()}).",
            "rate_limit_error",
            code="budget_exceeded",
            conduit={
                "limit_usd": decision.limit_usd,
                "spent_usd": decision.spent_usd,
                "window": decision.window.isoformat(),
                "action": "reject",
            },
        ),
    )


def install_error_handlers(app: FastAPI) -> None:
    """Wire every failure mode onto the one envelope."""

    @app.exception_handler(GatewayError)
    async def _gateway_error(_: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, GatewayError)
        log.warning(
            "gateway.rejected",
            status=exc.status_code,
            code=exc.code,
            error_type=exc.error_type,
            **exc.conduit,
        )
        return exc.response()

    @app.exception_handler(BudgetExceededError)
    async def _budget_exceeded(_: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, BudgetExceededError)
        log.warning(
            "gateway.budget_exceeded",
            key=exc.key_fingerprint,
            limit_usd=exc.decision.limit_usd,
            spent_usd=exc.decision.spent_usd,
        )
        return _budget_response(exc)

    @app.exception_handler(AllProvidersFailedError)
    async def _all_providers_failed(_: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, AllProvidersFailedError)
        # `str(exc)` embeds the last provider's error, which for an HTTP adapter
        # is up to 400 characters of vendor response body. It goes to the log.
        log.error(
            "gateway.upstream_exhausted",
            attempted=list(exc.attempted),
            last_error=str(exc.last_error),
        )
        return UpstreamUnavailableError(list(exc.attempted)).response()

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, RequestValidationError)
        # Pydantic's own error list is caller-facing detail about the caller's
        # own payload, but `ctx` can carry exception objects, so only the
        # location and message survive the trip.
        messages = [str(err.get("msg", "")) for err in exc.errors()]
        fields: list[dict[str, Any]] = [
            {"loc": [str(part) for part in err.get("loc", ())], "msg": message}
            for err, message in zip(exc.errors(), messages, strict=True)
        ]
        first = messages[0] if messages else "Request body failed validation."
        return JSONResponse(
            status_code=400,
            content=error_payload(
                first, "invalid_request_error", code="invalid_request", conduit={"fields": fields}
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        # The traceback belongs in the log, and only in the log (AC-8).
        log.error("gateway.unhandled", exc_info=exc, error_class=type(exc).__name__)
        logging.getLogger("conduit.gateway").debug("unhandled gateway error", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=error_payload(INTERNAL_MESSAGE, "api_error", code="internal_error"),
        )
