"""Static API key -> team, quota, policy.

The shape is `mcp-gateway`'s: a key is a row in a config table, checked on every
request, with no token exchange and no directory to reach. Two differences, both
forced by what Conduit does with the answer:

- The row resolves to a `Principal` rather than to the key string, because the
  pipeline needs the team (telemetry attribution), the quota (rate limit and
  daily budget) and the policy (tier ceiling) — not just "yes".
- `Authorization: Bearer <key>` is accepted alongside `X-API-Key`, because the
  OpenAI SDK sends the former and AC-1 says an unmodified SDK client works.

Resolution does no I/O and is the first thing that happens to a request, so a
bad key costs a dict lookup and nothing else (AC-3).
"""

import hmac

from pydantic import BaseModel

from conduit.contracts import Complexity
from conduit.gateway.errors import UnauthorizedError
from conduit.gateway.settings import ApiKeySettings, ConsoleScope, GatewaySettings

# The router already has a non-reversible key id for logging. Reusing it means
# a key fingerprint in a gateway log line and one in a budget log line are the
# same string, which is the whole point of having one.
from conduit.router.cost import key_fingerprint

__all__ = [
    "ANONYMOUS",
    "API_KEY_HEADER",
    "ApiKeyDirectory",
    "Principal",
    "extract_key",
]

API_KEY_HEADER = "X-API-Key"
_BEARER_PREFIX = "bearer "


class Principal(BaseModel):
    """An authenticated caller and everything the pipeline is allowed to know."""

    api_key: str
    team: str
    rate_limit_per_minute: int
    daily_budget_usd: float | None = None
    max_tier: Complexity = Complexity.COMPLEX

    #: Operator-console grant, carried from the key's row. The console router
    #: reads it; nothing on the completion path does.
    console_scope: ConsoleScope = ConsoleScope.NONE
    console_all_teams: bool = False

    @property
    def fingerprint(self) -> str:
        """Short, stable, non-reversible id. Log this, never `api_key`."""
        return key_fingerprint(self.api_key) if self.api_key else "anonymous"


#: The caller used when `require_auth: false`. Unmetered by construction: the
#: router treats an empty key as "no budget accounting" and the rate limiter
#: buckets every anonymous caller together.
#: `require_auth: false` means the whole gateway is open — completions and the
#: already-unauthenticated `/metrics` included — so withholding the console's
#: read panels would protect nothing while breaking the zero-key demo. Reveal is
#: withheld anyway: an open gateway is a local convenience, and returning a
#: human's prompt text to an unauthenticated caller is not one.
ANONYMOUS = Principal(
    api_key="",
    team="anonymous",
    rate_limit_per_minute=60,
    max_tier=Complexity.COMPLEX,
    console_scope=ConsoleScope.READ,
    console_all_teams=True,
)


def extract_key(x_api_key: str | None, authorization: str | None) -> str | None:
    """Pull the presented key out of either header. `X-API-Key` wins.

    A bearer token is treated as the key itself, not as a JWT: these are static
    keys, and pretending otherwise would imply a validation this does not do.
    """
    if x_api_key:
        return x_api_key.strip() or None
    if authorization and authorization.lower().startswith(_BEARER_PREFIX):
        return authorization[len(_BEARER_PREFIX) :].strip() or None
    return None


class ApiKeyDirectory:
    """The key table, resolved to principals."""

    def __init__(self, settings: GatewaySettings) -> None:
        if settings.require_auth and not settings.api_keys:
            # mcp-gateway fails loudly at startup rather than serving an open
            # proxy, and an AI gateway with a spend ledger has more to lose.
            raise ValueError(
                "gateway.yaml configures no api_keys and require_auth is true; "
                "add a key or set require_auth: false for a local run"
            )
        self._settings = settings
        self._principals: dict[str, Principal] = {
            entry.key: self._principal(entry, settings.rate_limit_per_minute)
            for entry in settings.api_keys
        }

    @staticmethod
    def _principal(entry: ApiKeySettings, default_rpm: int) -> Principal:
        return Principal(
            api_key=entry.key,
            team=entry.team,
            rate_limit_per_minute=entry.rate_limit_per_minute or default_rpm,
            daily_budget_usd=entry.daily_budget_usd,
            max_tier=entry.max_tier,
            console_scope=entry.console_scope,
            console_all_teams=entry.console_all_teams,
        )

    @property
    def require_auth(self) -> bool:
        return self._settings.require_auth

    @property
    def principals(self) -> tuple[Principal, ...]:
        return tuple(self._principals.values())

    def budget_limits(self) -> dict[str, float]:
        """Per-key daily limits, in the shape `SpendLedger(limits=...)` wants."""
        return {
            principal.api_key: principal.daily_budget_usd
            for principal in self._principals.values()
            if principal.daily_budget_usd is not None
        }

    def resolve(self, presented: str | None) -> Principal:
        """Map a presented key to its principal, or raise `UnauthorizedError`.

        The dict lookup finds the candidate row; `compare_digest` then confirms
        it, so a match is not decided by a short-circuiting `==` on secret
        material.
        """
        if not self.require_auth:
            return ANONYMOUS
        if not presented:
            raise UnauthorizedError(
                f"Missing API key. Send it as {API_KEY_HEADER} or Authorization: Bearer <key>."
            )
        principal = self._principals.get(presented)
        if principal is None or not hmac.compare_digest(principal.api_key, presented):
            raise UnauthorizedError()
        return principal
