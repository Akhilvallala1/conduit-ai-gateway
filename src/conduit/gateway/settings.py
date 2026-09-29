"""`config/gateway.yaml` — the static API-key table and its defaults.

Auth is a lookup table, exactly as in `mcp-gateway`: a key is a row, and the row
carries the three things the rest of the request needs — which team to attribute
the traffic to, what quota the key holds, and what tier ceiling its policy
allows. Nothing here interprets those values; the pipeline does.
"""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

# The same YAML + `CONDUIT_<SECTION>__<KEY>` override engine every other config
# file uses. Telemetry reuses it for the same reason: a second loader would let
# gateway config drift from the rest of the system's.
from conduit.config import _load_section
from conduit.contracts import Complexity

__all__ = ["ApiKeySettings", "ConsoleScope", "GatewaySettings", "load_gateway"]


class ConsoleScope(str, Enum):
    """How much of the operator console a key may reach.

    The console is an admin surface over *cross-team* data, so console access is
    a separate grant rather than something every completion key inherits. A key
    that can call `/v1/chat/completions` gets `none` unless its row says
    otherwise, which is why the default is the empty grant and not a useful one.

    `reveal` is split from `read` because looking at a request's metadata and
    looking at the text a human typed are different acts with different blast
    radii. `read` can answer every question the console's panels ask — tier,
    cost, guard verdicts, failover — without any prompt text existing in the
    response at all.
    """

    NONE = "none"
    READ = "read"
    REVEAL = "reveal"

    @property
    def may_read(self) -> bool:
        return self is not ConsoleScope.NONE

    @property
    def may_reveal(self) -> bool:
        return self is ConsoleScope.REVEAL


class ApiKeySettings(BaseModel):
    """One key: who it bills to, how fast it may go, how far up it may route."""

    key: str = Field(min_length=1)
    team: str = Field(min_length=1)

    #: None means "use the gateway-wide default".
    rate_limit_per_minute: int | None = Field(default=None, gt=0)

    #: None means "use `routing.yaml`'s `budgets.default_daily_usd`". Set, it
    #: becomes a per-key limit in the router's spend ledger.
    daily_budget_usd: float | None = Field(default=None, ge=0.0)

    #: Routing ceiling. A key capped at `trivial` never reaches an opus model
    #: however complex its prompt classifies.
    max_tier: Complexity = Complexity.COMPLEX

    #: Operator-console grant. Defaults to the empty one: adding a key for a
    #: team must not hand that team a window onto everyone else's traffic.
    console_scope: ConsoleScope = ConsoleScope.NONE

    #: False confines the console to this key's own team. True is the operator
    #: case — one key that can see the whole gateway.
    console_all_teams: bool = False

    @model_validator(mode="after")
    def _console_grant_is_coherent(self) -> "ApiKeySettings":
        """Refuse a row that looks like it grants console access and does not.

        `console_all_teams: true` beside the default `console_scope: none` reads
        like a broad grant and is in fact no grant at all. Silently ignoring it
        is how an operator ends up debugging an empty console instead of a typo.
        """
        if self.console_all_teams and not self.console_scope.may_read:
            raise ValueError(
                f"api key for team {self.team!r} sets console_all_teams but its "
                f"console_scope is {self.console_scope.value!r}; set console_scope "
                "to 'read' or 'reveal', or drop console_all_teams"
            )
        return self


class GatewaySettings(BaseModel):
    """`config/gateway.yaml` — keys, default quota, and the auth switch."""

    api_keys: list[ApiKeySettings] = Field(default_factory=list)
    rate_limit_per_minute: int = Field(default=60, gt=0)

    #: False disables the key check entirely. Only sane for a local single-user
    #: run, so it defaults to on and the app logs loudly when it is off.
    require_auth: bool = True

    @field_validator("api_keys")
    @classmethod
    def _keys_are_unique(cls, value: list[ApiKeySettings]) -> list[ApiKeySettings]:
        seen: set[str] = set()
        for entry in value:
            if entry.key in seen:
                raise ValueError(f"duplicate api key for team {entry.team!r}")
            seen.add(entry.key)
        return value


def load_gateway(
    config_dir: Path | None = None, env: dict[str, str] | None = None
) -> GatewaySettings:
    """Load `gateway.yaml`, applying `CONDUIT_GATEWAY__*` overrides."""
    return _load_section(GatewaySettings, "gateway", config_dir, env)
