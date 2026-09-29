"""Tests for `/api/console/*`: the grant, the team scope, and the reveal.

Most of this file is about what the console *refuses*. That is the right weight
for it: the panels are a projection of rows the telemetry tests already cover,
whereas the access rules are new policy, and every one of them is the only thing
standing between an operator dashboard and a data-exfiltration endpoint.

The store is seeded directly rather than driven through the pipeline, for the
same reason `test_telemetry_metrics.py` does it — a row with an open circuit
breaker, an exhausted failover chain and persisted prompt text cannot be
produced by one offline request, and the projection has to be tested against all
three. `test_console_integration.py` covers the other direction: a real request
through the real pipeline, read back through the console.
"""

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from conduit.contracts import Complexity
from conduit.gateway.app import create_app
from conduit.gateway.console import CONSOLE_PREFIX, MAX_TRACE_LIMIT
from conduit.gateway.settings import ApiKeySettings, ConsoleScope, GatewaySettings
from conduit.telemetry.recorder import Recorder, TelemetryRecord
from conduit.telemetry.settings import TelemetrySettings
from conduit.telemetry.store import JsonlSqliteStore
from tests.gateway_fixtures import ECHO, SECONDARY, make_config, make_registry
from tests.telemetry_fixtures import make_record

#: Keys covering the three grants plus the tenant case. `ops` is the operator:
#: whole-gateway sight and the reveal grant. `auditor` may read everything but
#: never text. `gtm` is an ordinary tenant with read over its own team only.
#: `plain` holds no console grant at all, which is the shipped default.
OPS_KEY = "ops-key"
AUDITOR_KEY = "auditor-key"
GTM_KEY = "gtm-key"
PLAIN_KEY = "plain-key"


def console_settings(*, require_auth: bool = True) -> GatewaySettings:
    return GatewaySettings(
        api_keys=[
            ApiKeySettings(
                key=OPS_KEY,
                team="ops",
                console_scope=ConsoleScope.REVEAL,
                console_all_teams=True,
            ),
            ApiKeySettings(
                key=AUDITOR_KEY,
                team="ops",
                console_scope=ConsoleScope.READ,
                console_all_teams=True,
            ),
            ApiKeySettings(key=GTM_KEY, team="gtm", console_scope=ConsoleScope.READ),
            ApiKeySettings(key=PLAIN_KEY, team="gtm"),
        ],
        require_auth=require_auth,
    )


def build_console_app(
    tmp_path: Path,
    records: Sequence[TelemetryRecord] = (),
    *,
    require_auth: bool = True,
    persist_text: bool = False,
) -> TestClient:
    """A gateway over a store seeded with `records`, and nothing on the wire."""
    config = make_config()
    store = JsonlSqliteStore(tmp_path / "events.jsonl", tmp_path / "events.db")
    for record in records:
        store.write(record)
    recorder = Recorder(
        store,
        models=config.models.models,
        settings=TelemetrySettings(enabled=True, dir=tmp_path, persist_text=persist_text),
    )
    app = create_app(
        config=config,
        gateway=console_settings(require_auth=require_auth),
        registry=make_registry(config),
        recorder=recorder,
        setup_logging=False,
    )
    return TestClient(app)


def headers(key: str) -> dict[str, str]:
    return {"X-API-Key": key}


@pytest.fixture
def audit() -> Iterator[list[dict[str, Any]]]:
    """The structured events the gateway logged inside the block.

    `caplog` cannot see these. The gateway logs through structlog, and a test
    app is built with `setup_logging=False`, so structlog is either unconfigured
    — writing to stdout through `PrintLogger`, never touching stdlib `logging` —
    or configured by whichever earlier module built an app with logging on. Both
    make a stdlib handler the wrong place to listen.

    `capture_logs` intercepts at the processor chain instead, which works under
    either configuration and hands back the event dict rather than rendered
    text, so the audit trail can be asserted field by field.
    """
    with capture_logs() as events:
        yield events


# --------------------------------------------------------------------------- #
# The grant
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "route",
    ["/traces", "/traces/trace-0000", "/providers", "/evals", "/metrics"],
)
def test_every_console_route_refuses_a_key_without_the_grant(tmp_path: Path, route: str) -> None:
    """The default key table grants nothing, and it must apply to all five.

    Parametrized rather than spot-checked because the failure mode is a *new*
    route added later without the dependency — which would be invisible in a
    test that only names the routes that existed when it was written.
    """
    with build_console_app(tmp_path, [make_record()]) as client:
        response = client.get(f"{CONSOLE_PREFIX}{route}", headers=headers(PLAIN_KEY))
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "insufficient_scope"


@pytest.mark.parametrize(
    "route",
    ["/traces", "/traces/trace-0000", "/providers", "/evals", "/metrics"],
)
def test_every_console_route_refuses_an_unauthenticated_caller(tmp_path: Path, route: str) -> None:
    with build_console_app(tmp_path, [make_record()]) as client:
        response = client.get(f"{CONSOLE_PREFIX}{route}")
    assert response.status_code == 401


def test_a_console_grant_does_not_come_with_the_reveal_grant(tmp_path: Path) -> None:
    record = make_record(team="gtm", prompt_text="call me on 555-0100")
    with build_console_app(tmp_path, [record], persist_text=True) as client:
        response = client.get(
            f"{CONSOLE_PREFIX}/traces/{record.trace_id}?reveal=true",
            headers=headers(AUDITOR_KEY),
        )
    assert response.status_code == 403
    assert "console_scope: reveal" in response.json()["error"]["message"]


def test_a_key_claiming_all_teams_without_a_console_scope_is_a_config_error() -> None:
    """A row that reads like a broad grant and grants nothing is a footgun."""
    with pytest.raises(ValueError, match="console_all_teams"):
        ApiKeySettings(key="k", team="t", console_all_teams=True)


def test_an_open_gateway_gets_console_reads_but_still_no_reveal(tmp_path: Path) -> None:
    """`require_auth: false` opens the panels, not the prompt text.

    The completion routes and `/metrics` are already open in this mode, so
    withholding the console's metadata would protect nothing. Text is different:
    it is the one thing no other open surface exposes.
    """
    record = make_record(prompt_text="call me on 555-0100")
    with build_console_app(tmp_path, [record], require_auth=False, persist_text=True) as client:
        assert client.get(f"{CONSOLE_PREFIX}/traces").status_code == 200
        revealed = client.get(f"{CONSOLE_PREFIX}/traces/{record.trace_id}?reveal=true")
    assert revealed.status_code == 403


# --------------------------------------------------------------------------- #
# Team scoping
# --------------------------------------------------------------------------- #


def test_a_tenant_key_sees_only_its_own_teams_traces(tmp_path: Path) -> None:
    records = [
        make_record(index=0, team="gtm"),
        make_record(index=1, team="finance"),
        make_record(index=2, team="gtm"),
    ]
    with build_console_app(tmp_path, records) as client:
        body = client.get(f"{CONSOLE_PREFIX}/traces", headers=headers(GTM_KEY)).json()
    assert {trace["team"] for trace in body["traces"]} == {"gtm"}
    assert len(body["traces"]) == 2


def test_an_operator_key_sees_every_team(tmp_path: Path) -> None:
    records = [make_record(index=0, team="gtm"), make_record(index=1, team="finance")]
    with build_console_app(tmp_path, records) as client:
        body = client.get(f"{CONSOLE_PREFIX}/traces", headers=headers(OPS_KEY)).json()
    assert {trace["team"] for trace in body["traces"]} == {"gtm", "finance"}


def test_another_teams_trace_is_absent_rather_than_forbidden(tmp_path: Path) -> None:
    """404, not 403 — a 403 would confirm the id exists (an existence oracle)."""
    record = make_record(team="finance")
    with build_console_app(tmp_path, [record]) as client:
        response = client.get(
            f"{CONSOLE_PREFIX}/traces/{record.trace_id}", headers=headers(GTM_KEY)
        )
        # ...and the operator key, for whom it is not another team's row, gets it.
        allowed = client.get(f"{CONSOLE_PREFIX}/traces/{record.trace_id}", headers=headers(OPS_KEY))
    assert response.status_code == 404
    assert allowed.status_code == 200


def test_console_metrics_are_team_scoped(tmp_path: Path) -> None:
    """The aggregate must not become a side channel around the row filter."""
    records = [
        make_record(index=0, team="gtm", cost_usd=1.0),
        make_record(index=1, team="finance", cost_usd=99.0),
    ]
    with build_console_app(tmp_path, records) as client:
        tenant = client.get(f"{CONSOLE_PREFIX}/metrics", headers=headers(GTM_KEY)).json()
        operator = client.get(f"{CONSOLE_PREFIX}/metrics", headers=headers(OPS_KEY)).json()
    assert tenant["overall"]["requests"] == 1
    assert tenant["overall"]["cost_usd"] == pytest.approx(1.0)
    assert operator["overall"]["cost_usd"] == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# The trace feed
# --------------------------------------------------------------------------- #


def test_the_feed_returns_the_newest_rows_not_the_oldest(tmp_path: Path) -> None:
    """`limit` is applied by SQLite after the sort, so the order is load-bearing.

    Ascending order with a limit would return the oldest N rows — a console
    that silently shows the start of the log instead of the end.
    """
    records = [make_record(index=i, minutes_ago=100 - i) for i in range(10)]
    with build_console_app(tmp_path, records) as client:
        body = client.get(f"{CONSOLE_PREFIX}/traces?limit=3", headers=headers(OPS_KEY)).json()
    assert [trace["trace_id"] for trace in body["traces"]] == [
        "trace-0009",
        "trace-0008",
        "trace-0007",
    ]


def test_the_feed_caps_its_page_size(tmp_path: Path) -> None:
    """An uncapped limit is a way to pull the whole telemetry log in one GET."""
    with build_console_app(tmp_path, [make_record()]) as client:
        response = client.get(
            f"{CONSOLE_PREFIX}/traces?limit={MAX_TRACE_LIMIT + 1}", headers=headers(OPS_KEY)
        )
    assert response.status_code == 400


def test_the_feed_filters_by_status_and_tier(tmp_path: Path) -> None:
    records = [
        make_record(index=0, status="ok", tier=Complexity.TRIVIAL),
        make_record(index=1, status="blocked", tier=Complexity.TRIVIAL),
        make_record(index=2, status="ok", tier=Complexity.COMPLEX),
    ]
    with build_console_app(tmp_path, records) as client:
        blocked = client.get(
            f"{CONSOLE_PREFIX}/traces?status=blocked", headers=headers(OPS_KEY)
        ).json()
        complex_tier = client.get(
            f"{CONSOLE_PREFIX}/traces?tier=complex", headers=headers(OPS_KEY)
        ).json()
    assert [trace["trace_id"] for trace in blocked["traces"]] == ["trace-0001"]
    assert [trace["trace_id"] for trace in complex_tier["traces"]] == ["trace-0002"]


def test_no_feed_row_carries_any_text_field(tmp_path: Path) -> None:
    """The summary schema has no text in it, even where the store does.

    Asserted over the serialised body rather than the model, because the risk
    is a field appearing in JSON that nobody meant to put there.
    """
    record = make_record(prompt_text="ssn 123-45-6789", completion_text="noted")
    with build_console_app(tmp_path, [record], persist_text=True) as client:
        body = client.get(f"{CONSOLE_PREFIX}/traces", headers=headers(OPS_KEY)).json()
    assert "123-45-6789" not in str(body)
    assert set(body["traces"][0]) == {
        "trace_id",
        "ts",
        "team",
        "workflow",
        "tier",
        "provider",
        "model",
        "status",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost_usd",
        "latency_ms",
        "guards",
        "failover_hops",
    }


# --------------------------------------------------------------------------- #
# Trace detail: the projection
# --------------------------------------------------------------------------- #


def test_guard_categories_are_split_back_into_their_stages(tmp_path: Path) -> None:
    """`Pipeline` namespaces egress categories and marks the blocking guard."""
    record = make_record(
        status="blocked",
        guard_categories=["pii:email", "injection:instruction_override", "guard:injection"],
    )
    with build_console_app(tmp_path, [record]) as client:
        body = client.get(
            f"{CONSOLE_PREFIX}/traces/{record.trace_id}", headers=headers(OPS_KEY)
        ).json()
    assert body["guards"] == {
        "ingress_categories": ["pii:email", "injection:instruction_override"],
        "egress_categories": [],
        "blocked": True,
        "blocked_by": "injection",
    }


def test_an_egress_category_is_not_reported_as_an_ingress_one(tmp_path: Path) -> None:
    record = make_record(guard_categories=["pii:email", "egress:pii:phone"])
    with build_console_app(tmp_path, [record]) as client:
        guards = client.get(
            f"{CONSOLE_PREFIX}/traces/{record.trace_id}", headers=headers(OPS_KEY)
        ).json()["guards"]
    assert guards["ingress_categories"] == ["pii:email"]
    assert guards["egress_categories"] == ["pii:phone"]
    assert guards["blocked"] is False


def test_the_failover_chain_ends_in_the_model_that_served(tmp_path: Path) -> None:
    record = make_record(model=SECONDARY, fallback_from=[ECHO, "mock:other"])
    with build_console_app(tmp_path, [record]) as client:
        body = client.get(
            f"{CONSOLE_PREFIX}/traces/{record.trace_id}", headers=headers(OPS_KEY)
        ).json()
    assert body["failover"] == [
        {"model": ECHO, "outcome": "failed"},
        {"model": "mock:other", "outcome": "failed"},
        {"model": SECONDARY, "outcome": "served"},
    ]
    assert body["failover_hops"] == 2


def test_an_exhausted_chain_has_no_serving_model(tmp_path: Path) -> None:
    """Every hop failed, so nothing in the chain may be labelled as serving."""
    record = make_record(
        status="error",
        model=SECONDARY,
        fallback_from=[ECHO, SECONDARY],
        error="AllProvidersFailedError: exhausted chain",
    )
    with build_console_app(tmp_path, [record]) as client:
        body = client.get(
            f"{CONSOLE_PREFIX}/traces/{record.trace_id}", headers=headers(OPS_KEY)
        ).json()
    assert [hop["outcome"] for hop in body["failover"]] == ["failed", "failed"]
    assert body["error"] == "AllProvidersFailedError: exhausted chain"


def test_gateway_overhead_is_the_gap_between_the_two_measured_clocks(
    tmp_path: Path,
) -> None:
    record = make_record(latency_ms=250, provider_latency_ms=180)
    with build_console_app(tmp_path, [record]) as client:
        timings = client.get(
            f"{CONSOLE_PREFIX}/traces/{record.trace_id}", headers=headers(OPS_KEY)
        ).json()["timings"]
    assert timings == {"total_ms": 250, "provider_ms": 180, "gateway_overhead_ms": 70}


def test_overhead_is_absent_when_no_provider_answered(tmp_path: Path) -> None:
    """A blocked request never reached one, so there is no gap to report."""
    record = make_record(status="blocked", provider_latency_ms=None, latency_ms=4)
    with build_console_app(tmp_path, [record]) as client:
        timings = client.get(
            f"{CONSOLE_PREFIX}/traces/{record.trace_id}", headers=headers(OPS_KEY)
        ).json()["timings"]
    assert timings["provider_ms"] is None
    assert timings["gateway_overhead_ms"] is None


def test_overhead_never_goes_negative(tmp_path: Path) -> None:
    """Two clocks in two processes; a negative gap is an artefact, not a fact."""
    record = make_record(latency_ms=10, provider_latency_ms=12)
    with build_console_app(tmp_path, [record]) as client:
        timings = client.get(
            f"{CONSOLE_PREFIX}/traces/{record.trace_id}", headers=headers(OPS_KEY)
        ).json()["timings"]
    assert timings["gateway_overhead_ms"] == 0


def test_an_unknown_trace_is_a_404(tmp_path: Path) -> None:
    with build_console_app(tmp_path, [make_record()]) as client:
        response = client.get(f"{CONSOLE_PREFIX}/traces/nope", headers=headers(OPS_KEY))
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


# --------------------------------------------------------------------------- #
# The reveal
# --------------------------------------------------------------------------- #


def test_detail_carries_no_text_unless_a_reveal_is_asked_for(tmp_path: Path) -> None:
    record = make_record(prompt_text="ssn 123-45-6789", completion_text="noted")
    with build_console_app(tmp_path, [record], persist_text=True) as client:
        body = client.get(
            f"{CONSOLE_PREFIX}/traces/{record.trace_id}", headers=headers(OPS_KEY)
        ).json()
    assert body["text"] == {
        "requested": False,
        "available": False,
        "reason": "not_requested",
        "prompt": None,
        "completion": None,
        "redacted_entities": 0,
    }
    assert "123-45-6789" not in str(body)


def test_a_reveal_redacts_the_stored_original_at_read_time(tmp_path: Path) -> None:
    """The stored prompt is the *original*, so the console must redact it itself.

    `Recorder.begin` captures the prompt before the ingress guard runs, which
    means `prompt_text` is what the human typed and not what the vendor was
    sent. If the console ever returns that string verbatim, it is the only
    place in the system where unredacted PII crosses a network boundary.
    """
    record = make_record(prompt_text="email me at ada@example.com about the quote")
    with build_console_app(tmp_path, [record], persist_text=True) as client:
        text = client.get(
            f"{CONSOLE_PREFIX}/traces/{record.trace_id}?reveal=true",
            headers=headers(OPS_KEY),
        ).json()["text"]
    assert text["available"] is True
    assert text["reason"] == "redacted"
    assert "ada@example.com" not in text["prompt"]
    assert text["redacted_entities"] >= 1
    # The surrounding prompt survives — redaction, not suppression.
    assert "about the quote" in text["prompt"]


def test_a_reveal_with_nothing_stored_says_so_rather_than_erroring(
    tmp_path: Path,
) -> None:
    """`persist_text` is off in the shipped config, so this is the normal answer."""
    record = make_record()
    with build_console_app(tmp_path, [record]) as client:
        text = client.get(
            f"{CONSOLE_PREFIX}/traces/{record.trace_id}?reveal=true",
            headers=headers(OPS_KEY),
        ).json()["text"]
    assert text == {
        "requested": True,
        "available": False,
        "reason": "not_persisted",
        "prompt": None,
        "completion": None,
        "redacted_entities": 0,
    }


def test_every_authorized_reveal_is_audit_logged(
    tmp_path: Path, audit: list[dict[str, Any]]
) -> None:
    record = make_record(team="gtm", prompt_text="ada@example.com")
    with build_console_app(tmp_path, [record], persist_text=True) as client:
        client.get(
            f"{CONSOLE_PREFIX}/traces/{record.trace_id}?reveal=true",
            headers=headers(OPS_KEY),
        )
    events = [event for event in audit if event.get("event") == "console.trace_revealed"]
    assert len(events) == 1
    entry = events[0]
    assert entry["trace_id"] == record.trace_id
    assert entry["team"] == "ops"
    assert entry["record_team"] == "gtm"
    assert entry["text_available"] is True
    # The caller is identified by fingerprint, never by the key itself...
    assert entry["key"] and OPS_KEY not in str(entry)
    # ...and neither the revealed text nor an entity map reaches the log.
    assert "ada@example.com" not in str(entry)


def test_a_reveal_that_finds_nothing_is_still_audit_logged(
    tmp_path: Path, audit: list[dict[str, Any]]
) -> None:
    """The auditable act is asking to see a prompt, not succeeding at it."""
    record = make_record()
    with build_console_app(tmp_path, [record]) as client:
        client.get(
            f"{CONSOLE_PREFIX}/traces/{record.trace_id}?reveal=true",
            headers=headers(OPS_KEY),
        )
    events = [event for event in audit if event.get("event") == "console.trace_revealed"]
    assert len(events) == 1
    assert events[0]["text_available"] is False


def test_a_refused_reveal_is_not_recorded_as_one(
    tmp_path: Path, audit: list[dict[str, Any]]
) -> None:
    """A 403 leaves no `trace_revealed` event, because nothing was revealed.

    The refusal is logged by the error handler under its own name. Emitting the
    reveal event here too would put an entry in the audit trail for a prompt
    nobody saw, which is the fastest way to make the trail untrustworthy.
    """
    record = make_record(team="gtm", prompt_text="ada@example.com")
    with build_console_app(tmp_path, [record], persist_text=True) as client:
        client.get(
            f"{CONSOLE_PREFIX}/traces/{record.trace_id}?reveal=true",
            headers=headers(AUDITOR_KEY),
        )
    assert not [event for event in audit if event.get("event") == "console.trace_revealed"]


def test_a_refused_reveal_returns_no_text_at_all(tmp_path: Path) -> None:
    """403 bodies are a classic place for the payload to leak anyway."""
    record = make_record(team="gtm", prompt_text="ada@example.com")
    with build_console_app(tmp_path, [record], persist_text=True) as client:
        response = client.get(
            f"{CONSOLE_PREFIX}/traces/{record.trace_id}?reveal=true",
            headers=headers(AUDITOR_KEY),
        )
    assert response.status_code == 403
    assert "ada@example.com" not in response.text


# --------------------------------------------------------------------------- #
# Providers and evals
# --------------------------------------------------------------------------- #


def test_providers_reports_breaker_state_and_the_usable_chain(tmp_path: Path) -> None:
    with build_console_app(tmp_path) as client:
        body = client.get(f"{CONSOLE_PREFIX}/providers", headers=headers(OPS_KEY)).json()
    by_name = {entry["name"]: entry for entry in body["providers"]}
    assert by_name["mock"]["reachable"] is True
    # Nothing has failed, so every breaker is closed rather than absent.
    assert {entry["breaker"] for entry in body["providers"]} == {"closed"}
    assert by_name["mock"]["models"] == [ECHO]
    chains = {entry["tier"]: entry for entry in body["chains"]}
    assert chains["trivial"]["configured"] == [ECHO, SECONDARY]
    assert chains["trivial"]["usable"] == [ECHO, SECONDARY]


def test_providers_shows_an_open_breaker(tmp_path: Path) -> None:
    """The panel must read the breakers on the request path, not a fresh set."""
    with build_console_app(tmp_path) as client:
        breakers = client.app.state.pipeline.failover.breakers
        for _ in range(10):
            breakers["mock"].record_failure()
        body = client.get(f"{CONSOLE_PREFIX}/providers", headers=headers(OPS_KEY)).json()
    states = {entry["name"]: entry["breaker"] for entry in body["providers"]}
    assert states["mock"] == "open"
    assert states["scripted"] == "closed"


def test_evals_returns_the_committed_baselines(tmp_path: Path) -> None:
    with build_console_app(tmp_path) as client:
        body = client.get(f"{CONSOLE_PREFIX}/evals", headers=headers(OPS_KEY)).json()
    assert body["configured"] is True
    suites = {entry["suite"]: entry for entry in body["suites"]}
    assert {"routing", "guards", "l2c_quality"} <= set(suites)
    assert suites["routing"]["baseline"]["cases"] > 0
    assert body["gate_ready"] is True
    # AC-14: the judge is never the model under test, and the console shows both
    # so that claim is checkable from the dashboard.
    assert body["judge_model"] != body["model_under_test"]


def test_a_suite_without_a_committed_baseline_is_not_gate_ready(tmp_path: Path) -> None:
    """The gate fails on a missing baseline, so the console must show it as red.

    This is the state a suite is in between being added and having its first
    baseline recorded — `gate()` reports it under `missing_baselines` and
    returns a non-zero exit code. A console that rendered it as a suite with no
    number yet would invert the meaning of the one panel CI blocks on.
    """
    empty = tmp_path / "baselines"
    empty.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "eval.yaml").write_text(
        "model_under_test: mock:echo\n"
        "judge_model: offline-judge:rubric\n"
        f'baseline_dir: "{empty.as_posix()}"\n'
        "suites:\n"
        "  routing:\n"
        "    path: evals/golden/routing.jsonl\n"
        "    scorer: tier\n",
        encoding="utf-8",
    )

    config = make_config()
    store = JsonlSqliteStore(tmp_path / "events.jsonl", tmp_path / "events.db")
    app = create_app(
        config=config,
        gateway=console_settings(),
        registry=make_registry(config),
        recorder=Recorder(store, models=config.models.models),
        config_dir=config_dir,
        setup_logging=False,
    )
    with TestClient(app) as client:
        body = client.get(f"{CONSOLE_PREFIX}/evals", headers=headers(OPS_KEY)).json()

    assert body["configured"] is True
    suite = body["suites"][0]
    assert suite["suite"] == "routing"
    assert suite["baseline"] is None
    assert suite["baseline_missing"] is True
    assert body["gate_ready"] is False


def test_evals_reports_a_missing_config_rather_than_an_empty_panel(
    tmp_path: Path,
) -> None:
    """An empty suite list would read as "zero suites", which is a different bug."""
    config = make_config()
    store = JsonlSqliteStore(tmp_path / "events.jsonl", tmp_path / "events.db")
    app = create_app(
        config=config,
        gateway=console_settings(),
        registry=make_registry(config),
        recorder=Recorder(store, models=config.models.models),
        config_dir=tmp_path / "no-such-config",
        setup_logging=False,
    )
    with TestClient(app) as client:
        body = client.get(f"{CONSOLE_PREFIX}/evals", headers=headers(OPS_KEY)).json()
    assert body["configured"] is False
    assert body["suites"] == []
    assert "eval.yaml" in body["detail"]
