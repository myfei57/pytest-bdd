"""Tests for the versioned BDD step-registry audit flow."""

from __future__ import annotations

import base64
import json
import linecache
import textwrap
import threading

import pytest

import pytest_bdd.audit as audit_mod
from pytest_bdd import given, parsers, then
from pytest_bdd.audit import (
    AUDIT_SCHEMA_VERSION,
    AuditResultCategory,
    DuplicateCursorError,
    InvalidCursorError,
    StepRegistryAudit,
    reset_step_registry_audit,
)
from pytest_bdd.utils import collect_dumped_objects


@pytest.fixture(autouse=True)
def global_audit():
    """Provide a fresh global audit stream and reset it after each test."""
    stream = reset_step_registry_audit()
    yield stream
    reset_step_registry_audit()


# ----------------------------------------------------------- stable event records


def test_registration_records_stable_seq_source_and_summary(global_audit):
    @given("first audited step", target_fixture="first")
    def first_step():  # pragma: no cover - must never be executed by auditing
        raise AssertionError("step function must not be executed during auditing")

    @then("second audited step")
    def second_step():  # pragma: no cover - must never be executed by auditing
        raise AssertionError("step function must not be executed during auditing")

    events = global_audit.snapshot()
    assert [event.seq for event in events] == [1, 2]

    first, second = events
    assert first.schema_version == AUDIT_SCHEMA_VERSION
    assert first.category is AuditResultCategory.REGISTERED
    assert second.category is AuditResultCategory.REGISTERED

    # Stable event numbers survive repeated snapshots.
    assert global_audit.snapshot() == events
    assert global_audit.snapshot()[0] is first

    # Source location points at the decorator application site.
    assert first.source is not None
    assert first.source.path == __file__
    assert first.source.function == "test_registration_records_stable_seq_source_and_summary"
    decorator_line = linecache.getline(__file__, first.source.line)
    assert "@given(" in decorator_line

    # Only static registration metadata is recorded.
    assert first.details == {
        "step_type": "given",
        "step_name": "first audited step",
        "parser_type": "string",
        "step_function": "first_step",
        "target_fixture": "first",
        "converter_count": "0",
    }
    assert second.details["step_type"] == "then"
    assert second.details["target_fixture"] == ""

    assert first.summary.ok is True
    assert all(first.summary.checks)
    assert first.recorded_at


def test_audit_never_executes_step_or_converters_and_never_reads_inputs(global_audit):
    class ExplodingParser(parsers.StepParser):
        def parse_arguments(self, name):  # type: ignore[no-untyped-def]
            raise AssertionError("audit must not parse test input values")

        def is_matching(self, name):  # type: ignore[no-untyped-def]
            raise AssertionError("audit must not evaluate step matching")

    def forbidden_converter(value):  # type: ignore[no-untyped-def]
        raise AssertionError("converter must not be executed during auditing")

    def step_function(value):  # type: ignore[no-untyped-def]
        raise AssertionError("step function must not be executed during auditing")

    decorated = given(
        ExplodingParser("templated {value}"),
        converters={"value": forbidden_converter},
        target_fixture="thing",
    )(step_function)

    # The main flow returns the original function untouched.
    assert decorated is step_function

    events = global_audit.snapshot()
    assert len(events) == 1
    event = events[0]
    assert event.category is AuditResultCategory.REGISTERED
    assert event.details["step_name"] == "templated {value}"  # template, not an input
    assert event.details["parser_type"] == "ExplodingParser"
    assert event.details["converter_count"] == "1"  # counted, never inspected/called


def test_validation_failure_is_recorded_without_blocking_registration(global_audit):
    @given("")
    def blank_step():  # pragma: no cover - main flow must still work
        pass

    event = global_audit.snapshot()[0]
    assert event.category is AuditResultCategory.VALIDATION_FAILED
    assert event.summary.ok is False
    failed_checks = [name for name, passed in event.summary.checks if not passed]
    assert "step_name_present" in failed_checks
    assert "parser_name_present" in failed_checks

    # Invalid metadata can be audited directly as well.
    stream = StepRegistryAudit()
    stream.record_registration(
        step_name=" ",
        step_type="sideways",  # type: ignore[arg-type]
        parser_name=" ",
        parser_type="string",
        target_fixture="not an identifier!",
        converter_count=-1,
        function_name="f",
    )
    direct = stream.snapshot()[0]
    assert direct.category is AuditResultCategory.VALIDATION_FAILED
    assert dict(direct.summary.checks) == {
        "step_name_present": False,
        "step_type_valid": False,
        "parser_name_present": False,
        "target_fixture_identifier": False,
        "converter_count_non_negative": False,
    }


def test_error_category_and_audit_failures_are_contained(global_audit, monkeypatch):
    event = global_audit.record_error("boom", details={"where": "registry"})
    assert event.category is AuditResultCategory.ERROR
    assert event.summary.ok is False
    assert event.summary.error == "boom"
    assert event.details["where"] == "registry"

    # Even if source capture itself breaks, an error event is produced and the
    # exception never reaches the caller.
    def explode(frame):  # type: ignore[no-untyped-def]
        raise RuntimeError("frame capture exploded")

    monkeypatch.setattr(audit_mod, "source_from_frame", explode)
    recovered = global_audit.record_registration(
        step_name="x",
        step_type="given",
        parser_name="x",
        parser_type="string",
        target_fixture=None,
        converter_count=0,
        function_name="f",
        frame=object(),
    )
    assert recovered.category is AuditResultCategory.ERROR
    assert "audit failure" in (recovered.summary.error or "")


def test_main_flow_unaffected_when_auditing_raises(monkeypatch):
    stream = reset_step_registry_audit()

    def raise_always(**kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("audit subsystem down")

    monkeypatch.setattr(stream, "record_registration", raise_always)

    def step_function():  # type: ignore[no-untyped-def]
        return "ok"

    decorated = given("registration proceeds regardless")(step_function)
    assert decorated is step_function
    assert decorated() == "ok"


# ----------------------------------------------------------------------- empty flow


def test_empty_flow_pages_exports_and_recovery():
    stream = StepRegistryAudit()

    page = stream.read_page()
    assert page.events == ()
    assert page.next_cursor is None
    assert page.exhausted is True

    export = stream.export_all()
    assert export["schema_version"] == AUDIT_SCHEMA_VERSION
    assert export["count"] == 0
    assert export["events"] == []
    assert export["stream_id"]

    paged = stream.export_page(limit=10)
    assert paged["count"] == 0
    assert paged["next_cursor"] is None

    report = stream.recover_order()
    assert report.count == 0
    assert report.reordered is False
    assert report.missing_seqs == ()
    assert report.duplicate_seqs == ()


# ------------------------------------------------------------- cursor pagination


def _seed(stream: StepRegistryAudit, amount: int) -> None:
    for index in range(amount):
        stream.append(AuditResultCategory.REGISTERED, details={"index": str(index)})


def test_cursor_pagination_walks_every_event_in_order():
    stream = StepRegistryAudit()
    _seed(stream, 5)

    cursor = None
    collected: list[int] = []
    cursors = []
    expected_pages = [[1, 2], [3, 4], [5]]
    for expected_seqs in expected_pages:
        page = stream.read_page(cursor, limit=2)
        assert [event.seq for event in page.events] == expected_seqs
        assert page.next_cursor is not None
        assert page.exhausted is False
        cursors.append(page.next_cursor)
        cursor = page.next_cursor
        collected.extend(expected_seqs)

    # The cursor after the last event yields an empty, exhausted final page.
    final_page = stream.read_page(cursor, limit=2)
    assert final_page.events == ()
    assert final_page.next_cursor is None
    assert final_page.exhausted is True
    assert collected == [1, 2, 3, 4, 5]

    # All cursors are unique opaque tokens.
    assert len(set(cursors)) == len(cursors) == 3
    for token in cursors:
        decoded = json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))
        assert decoded["v"] == AUDIT_SCHEMA_VERSION
        assert decoded["n"]  # a single-use nonce is embedded


def test_duplicate_cursor_is_detected():
    stream = StepRegistryAudit()
    _seed(stream, 4)

    first = stream.read_page(limit=2)
    cursor = first.next_cursor
    assert cursor is not None

    second = stream.read_page(cursor, limit=2)
    assert [event.seq for event in second.events] == [3, 4]

    with pytest.raises(DuplicateCursorError):
        stream.read_page(cursor, limit=2)

    # A fresh traversal (cursor=None) is not a replay.
    assert [event.seq for event in stream.read_page(None, limit=1).events] == [1]


@pytest.mark.parametrize(
    "bad_cursor",
    [
        "not a cursor!!!",
        base64.urlsafe_b64encode(b"not-json").decode().rstrip("="),
        base64.urlsafe_b64encode(json.dumps({"v": 999, "s": "x", "a": 0, "n": "n"}).encode()).decode().rstrip("="),
        base64.urlsafe_b64encode(json.dumps({"v": 1, "s": "x", "a": -3, "n": "n"}).encode()).decode().rstrip("="),
        base64.urlsafe_b64encode(json.dumps({"v": 1, "s": "x", "a": 0}).encode()).decode().rstrip("="),
    ],
)
def test_malformed_cursors_are_rejected(bad_cursor):
    stream = StepRegistryAudit()
    with pytest.raises(InvalidCursorError):
        stream.read_page(bad_cursor)


def test_foreign_stream_cursor_is_rejected():
    stream = StepRegistryAudit()
    other = StepRegistryAudit()
    _seed(other, 2)

    foreign_cursor = other.read_page(limit=1).next_cursor
    assert foreign_cursor is not None
    with pytest.raises(InvalidCursorError, match="different audit stream"):
        stream.read_page(foreign_cursor)


@pytest.mark.parametrize("bad_limit", [0, -1, "2", 1.5, True])
def test_invalid_page_limit(bad_limit):
    stream = StepRegistryAudit()
    with pytest.raises(ValueError, match="limit"):
        stream.read_page(limit=bad_limit)  # type: ignore[arg-type]


# --------------------------------------------------------------- sensitive fields


def test_sensitive_fields_are_masked_only_in_exports():
    stream = StepRegistryAudit()
    stream.append(
        AuditResultCategory.REGISTERED,
        details={
            "password": "hunter2",
            "user_token": "token-value",
            "apiKey": "key-value",
            "normal": "visible",
        },
    )
    event = stream.snapshot()[0]

    # Values remain intact at rest.
    assert event.details["password"] == "hunter2"
    assert event.details["user_token"] == "token-value"

    export = stream.export_all()
    details = export["events"][0]["details"]
    assert details["password"] == "***"
    assert details["user_token"] == "***"
    assert details["apiKey"] == "***"
    assert details["normal"] == "visible"


def test_mask_sensitive_fields_handles_nested_mappings():
    nested = {
        "outer": "ok",
        "credentials": {"user": "bob", "secret": "s3cr3t"},
        "items": [{"token": "t"}, {"label": "keep"}],
    }
    masked = audit_mod.mask_sensitive_fields(nested)
    assert masked["outer"] == "ok"
    assert masked["credentials"] == "***"
    assert masked["items"][0] == {"token": "***"}
    assert masked["items"][1] == {"label": "keep"}


# ------------------------------------------------------------- export purity/safety


def test_export_is_json_serializable_and_does_not_mutate_stream():
    stream = StepRegistryAudit()
    _seed(stream, 3)

    first_export = stream.export_all()
    json.dumps(first_export)  # must not raise
    assert first_export["count"] == 3

    first_export["events"][0]["details"]["index"] = "tampered"
    first_export["schema_version"] = 999

    second_export = stream.export_all()
    assert second_export["schema_version"] == AUDIT_SCHEMA_VERSION
    assert second_export["events"][0]["details"]["index"] == "0"


def test_bulk_export_does_not_consume_pagination_cursors():
    stream = StepRegistryAudit()
    _seed(stream, 5)

    first = stream.read_page(None, limit=2)
    cursor = first.next_cursor
    assert cursor is not None

    # Bulk export must not disturb the paged consumer.
    stream.export_all()

    second = stream.read_page(cursor, limit=2)
    assert [event.seq for event in second.events] == [3, 4]


def test_snapshot_does_not_consume_cursors():
    stream = StepRegistryAudit()
    _seed(stream, 2)
    cursor = stream.read_page(limit=1).next_cursor
    assert cursor is not None

    for _ in range(3):
        snapshot = stream.snapshot()
        assert [event.seq for event in snapshot] == [1, 2]

    page = stream.read_page(cursor, limit=1)
    assert [event.seq for event in page.events] == [2]


# ------------------------------------------------------------- concurrent snapshots


def test_concurrent_appends_produce_contiguous_ordered_snapshots():
    stream = StepRegistryAudit()
    thread_count = 8
    reader_count = 2
    per_thread = 250
    barrier = threading.Barrier(thread_count + reader_count)
    observed_snapshots: list[tuple[int, ...]] = []
    snapshots_lock = threading.Lock()
    stop = threading.Event()

    def writer() -> None:
        barrier.wait()
        for _ in range(per_thread):
            stream.append(AuditResultCategory.REGISTERED)

    def reader() -> None:
        barrier.wait()
        while not stop.is_set():
            seqs = tuple(event.seq for event in stream.snapshot())
            # Every concurrent snapshot is ordered and contiguous from 1.
            assert seqs == tuple(range(1, len(seqs) + 1))
            with snapshots_lock:
                observed_snapshots.append(seqs)

    readers = [threading.Thread(target=reader) for _ in range(reader_count)]
    writers = [threading.Thread(target=writer) for _ in range(thread_count)]
    for thread in readers + writers:
        thread.start()
    for thread in writers:
        thread.join()
    stop.set()
    for thread in readers:
        thread.join()

    final = stream.snapshot()
    total = thread_count * per_thread
    assert len(final) == total
    assert [event.seq for event in final] == list(range(1, total + 1))
    assert observed_snapshots  # readers actually observed something


# ------------------------------------------------------------- order event recovery


def test_order_recovery_reorders_reports_gaps_and_dedupes():
    stream = StepRegistryAudit()

    # Out-of-order ingestion from an external/versioned source.
    stream.append(AuditResultCategory.REGISTERED, seq=3, details={"n": "third"})
    stream.append(AuditResultCategory.REGISTERED, seq=1, details={"n": "first"})
    stream.append(AuditResultCategory.REGISTERED, seq=5, details={"n": "fifth"})

    # Reads recover sequence order immediately.
    assert [event.seq for event in stream.snapshot()] == [1, 3, 5]

    report = stream.recover_order()
    assert report.reordered is True
    assert report.count == 3
    assert report.missing_seqs == (2, 4)
    assert report.duplicate_seqs == ()

    # Fill the gaps.
    stream.append(AuditResultCategory.REGISTERED, seq=2)
    stream.append(AuditResultCategory.REGISTERED, seq=4)
    report = stream.recover_order()
    assert report.reordered is True
    assert report.missing_seqs == ()

    # Duplicate sequence is dropped and reported.
    stream.append(AuditResultCategory.REGISTERED, seq=3, details={"n": "duplicate"})
    report = stream.recover_order()
    assert report.duplicate_seqs == (3,)
    assert report.count == 5
    assert [event.seq for event in stream.snapshot()] == [1, 2, 3, 4, 5]
    assert stream.snapshot()[2].details["n"] == "third"  # first occurrence wins

    # Automatic numbering continues after the highest retained event.
    next_event = stream.append(AuditResultCategory.REGISTERED)
    assert next_event.seq == 6


@pytest.mark.parametrize("bad_seq", [0, -1, "1", True])
def test_invalid_explicit_seq(bad_seq):
    stream = StepRegistryAudit()
    with pytest.raises(ValueError, match="seq"):
        stream.append(AuditResultCategory.REGISTERED, seq=bad_seq)  # type: ignore[arg-type]


# ------------------------------------------------------------------ integration


def test_audit_export_does_not_change_main_flow(pytester):
    """End-to-end: scenarios still run, step functions still execute normally,
    and the audit export only contains versioned, masked registration metadata."""
    pytester.makefile(
        ".feature",
        audit=textwrap.dedent(
            """\
            Feature: Audited registry
                Scenario: A normal scenario still executes
                    Given I have 42 coins
                    Then I should have 42 coins
            """
        ),
    )
    pytester.makepyfile(
        textwrap.dedent(
            """\
            from pytest_bdd import given, then, scenarios, parsers
            from pytest_bdd.utils import dump_obj
            import pytest_bdd.audit as audit

            @given(parsers.parse("I have {amount:d} coins"), target_fixture="coins")
            def _(amount):
                dump_obj(("step-ran", amount))
                return amount

            @then(parsers.parse("I should have {amount:d} coins"))
            def _(coins, amount):
                assert coins == amount

            scenarios("audit.feature")

            dump_obj(("audit", audit.step_registry_audit.export_all()))
            """
        )
    )
    result = pytester.runpytest("-s")
    result.assert_outcomes(passed=1)

    objects = collect_dumped_objects(result)
    # The export is produced at import/collection time, the step runs later.
    assert objects[0][0] == "audit"
    export = objects[0][1]
    assert objects[1] == ("step-ran", 42)

    assert export["schema_version"] == 1
    assert export["count"] >= 2
    audited_steps = {event["details"]["step_name"] for event in export["events"]}
    assert "I have {amount:d} coins" in audited_steps  # template, never the value 42
    assert "I should have {amount:d} coins" in audited_steps

    allowed_detail_keys = {
        "step_type",
        "step_name",
        "parser_type",
        "step_function",
        "target_fixture",
        "converter_count",
    }
    for event in export["events"]:
        assert event["category"] == "registered"
        assert set(event["details"]) == allowed_detail_keys
        assert event["source"]["path"].endswith(".py")
        assert event["source"]["line"] > 0
        assert event["summary"]["ok"] is True
        assert "42" not in json.dumps(event["details"])
