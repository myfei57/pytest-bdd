"""Tests for the concurrent read model of the BDD step registry.

The journal is meant to carry step definitions as opaque payloads. These tests
therefore never execute a payload and never compare payloads by value: every
payload is an inert stand-in whose ``__call__``/``__eq__`` would fail the test,
and all assertions use object identity (``is`` / ``id``) plus version metadata.
"""

from __future__ import annotations

import random
import threading
from collections.abc import Callable, Sequence
from dataclasses import FrozenInstanceError

import pytest

from pytest_bdd.exceptions import (
    CursorLaggedError,
    ReadCancelledError,
    StepRegistryClosedError,
)
from pytest_bdd.step_registry import (
    CancellationToken,
    Cursor,
    Increment,
    StepRegistryJournal,
    VersionedEntry,
)


class OpaquePayload:
    """Stand-in for a step definition: it must never be called or value-compared."""

    def __call__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("the journal must never execute a registered step function")

    def __eq__(self, other: object) -> bool:
        raise AssertionError("the journal must never read a payload value")

    __hash__ = None  # type: ignore[assignment]


def payloads(entries: Sequence[VersionedEntry[OpaquePayload]]) -> list[OpaquePayload]:
    return [entry.payload for entry in entries]


def assert_same_payloads(observed: Sequence[OpaquePayload], expected: Sequence[OpaquePayload]) -> None:
    """Identity-only sequence comparison (never invokes payload ``__eq__``)."""
    assert len(observed) == len(expected)
    assert all(observed is expected for observed, expected in zip(observed, expected, strict=True))


def versions(entries: Sequence[VersionedEntry[object]]) -> list[tuple[int, int]]:
    return [(entry.epoch, entry.version) for entry in entries]


def drain_until_closed(
    journal: StepRegistryJournal[OpaquePayload], cursor: Cursor
) -> tuple[list[VersionedEntry[OpaquePayload]], Cursor]:
    """Blocking drain loop used from reader threads."""
    collected: list[VersionedEntry[OpaquePayload]] = []
    while True:
        increment = journal.read(cursor, timeout=10)
        collected.extend(increment.entries)
        cursor = increment.cursor
        if increment.closed:
            return collected, cursor


def start_thread(target: Callable[[], None]) -> tuple[threading.Thread, list[Exception]]:
    errors: list[Exception] = []

    def wrapper() -> None:
        try:
            target()
        except Exception as exc:  # noqa: BLE001 - test harness surfaces any thread failure
            errors.append(exc)

    # Daemon threads: a failed assertion in the test must never leave a
    # blocked reader keeping the whole pytest process alive.
    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()
    return thread, errors


def join(thread: threading.Thread, timeout: float = 10) -> None:
    thread.join(timeout)
    assert not thread.is_alive(), f"thread {thread.name} did not finish within {timeout}s"


def test_publish_advances_monotonic_immutable_versions() -> None:
    journal = StepRegistryJournal[OpaquePayload](capacity=4)
    first = OpaquePayload()
    second = OpaquePayload()

    entry_one = journal.publish(first)
    entry_two = journal.publish(second)

    assert (entry_one.epoch, entry_one.version) == (0, 1)
    assert (entry_two.epoch, entry_two.version) == (0, 2)
    assert entry_one.payload is first
    assert entry_two.payload is second
    assert journal.current_cursor == Cursor(0, 2)

    with pytest.raises(FrozenInstanceError):
        entry_one.version = 99  # type: ignore[misc]


def test_publication_results_are_immutable_values() -> None:
    journal = StepRegistryJournal[OpaquePayload]()
    increment = journal.read(Cursor.initial(), timeout=0)

    assert isinstance(increment, Increment)
    with pytest.raises(FrozenInstanceError):
        increment.closed = True  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        increment.cursor = Cursor(0, 1)  # type: ignore[misc]


def test_journal_never_calls_or_value_compares_payloads() -> None:
    # If the journal touched the payload in any forbidden way, OpaquePayload
    # would raise; publishing, reading, snapshotting and wrapping must not.
    journal = StepRegistryJournal[OpaquePayload](capacity=2, version_mod=4)
    payload = OpaquePayload()

    journal.publish(payload)
    journal.publish(OpaquePayload())
    journal.publish(OpaquePayload())  # evicts the first version
    journal.publish(OpaquePayload())  # forces a wrap (v3 is the last in epoch 0)

    snapshot = journal.snapshot()
    resnapshot = journal.snapshot()
    assert snapshot.entries[0].payload is resnapshot.entries[0].payload
    assert snapshot.entries[0].payload is not payload  # identity only, no value read


def test_multiple_readers_consume_independent_increments() -> None:
    journal = StepRegistryJournal[OpaquePayload](capacity=10)
    first = OpaquePayload()
    second = OpaquePayload()
    third = OpaquePayload()

    journal.publish(first)
    increment_a = journal.read(Cursor.initial(), timeout=0)
    cursor_a = increment_a.cursor
    assert_same_payloads(payloads(increment_a.entries), [first])

    journal.publish(second)
    journal.publish(third)

    # Reader A gets only its increment; reader B independently gets all three.
    increment_a = journal.read(cursor_a, timeout=0)
    increment_b = journal.read(Cursor.initial(), timeout=0)

    assert_same_payloads(payloads(increment_a.entries), [second, third])
    assert_same_payloads(payloads(increment_b.entries), [first, second, third])
    assert increment_a.cursor == increment_b.cursor == Cursor(0, 3)
    assert increment_a.closed is False


def test_read_chunking_does_not_change_observed_sequence() -> None:
    journal = StepRegistryJournal[OpaquePayload](capacity=10)
    published = [OpaquePayload() for _ in range(5)]

    # Reader A polls after every publication (chunks of one entry).
    entries_a: list[VersionedEntry[OpaquePayload]] = []
    cursor = Cursor.initial()
    for payload in published:
        journal.publish(payload)
        increment = journal.read(cursor, timeout=0)
        entries_a.extend(increment.entries)
        cursor = increment.cursor

    # Reader B consumes the whole history in one increment.
    increment_b = journal.read(Cursor.initial(), timeout=0)

    assert_same_payloads(payloads(entries_a), published)
    assert_same_payloads(payloads(increment_b.entries), published)
    assert versions(entries_a) == versions(list(increment_b.entries))
    assert increment_b.cursor == cursor == Cursor(0, 5)


def test_slow_reader_gets_recoverable_lag_error_and_resyncs() -> None:
    journal = StepRegistryJournal[OpaquePayload](capacity=2)
    first = OpaquePayload()
    stalled_entry = journal.publish(first)
    stalled_cursor = Cursor(stalled_entry.epoch, stalled_entry.version)

    retained = [OpaquePayload(), OpaquePayload(), OpaquePayload()]
    journal.publish(retained[0])
    journal.publish(retained[1])  # evicts version 1: the stalled cursor is stale
    journal.publish(retained[2])  # evicts version 2

    with pytest.raises(CursorLaggedError) as exc_info:
        journal.read(stalled_cursor, timeout=0)
    error = exc_info.value
    assert error.requested_cursor == Cursor(0, 1)
    assert error.retained_cursor == Cursor(0, 3)
    assert error.current_cursor == Cursor(0, 4)

    # Recovery: resync from a snapshot, then keep following increments.
    snapshot = journal.snapshot()
    assert_same_payloads(payloads(snapshot.entries), retained[1:])
    assert snapshot.cursor == Cursor(0, 4)

    later = OpaquePayload()
    journal.publish(later)
    increment = journal.read(snapshot.cursor, timeout=0)
    assert_same_payloads(payloads(increment.entries), [later])
    assert increment.cursor == Cursor(0, 5)


def test_lag_error_does_not_block_the_publisher() -> None:
    journal = StepRegistryJournal[OpaquePayload](capacity=1)
    ready = threading.Event()
    read_result: list[Increment[OpaquePayload]] = []

    def reader() -> None:
        ready.set()
        # Blocks with no timeout; publication must still happen and wake it.
        read_result.append(journal.read(Cursor.initial()))

    thread, errors = start_thread(reader)
    assert ready.wait(5)
    entry = journal.publish(OpaquePayload())  # must return promptly
    assert entry.version == 1
    join(thread)
    assert errors == []
    assert read_result[0].entries[0].version == 1

    # And eviction while nobody reads also never blocks, however far behind.
    for _ in range(100):
        journal.publish(OpaquePayload())
    with pytest.raises(CursorLaggedError):
        journal.read(Cursor(0, 1), timeout=0)


def test_close_wakes_blocked_reader_with_empty_increment() -> None:
    journal = StepRegistryJournal[OpaquePayload]()
    ready = threading.Event()
    result: list[Increment[OpaquePayload]] = []

    def reader() -> None:
        ready.set()
        result.append(journal.read(Cursor.initial()))

    thread, errors = start_thread(reader)
    assert ready.wait(5)
    journal.close()
    join(thread)

    assert errors == []
    assert result[0].entries == ()
    assert result[0].cursor == Cursor.initial()
    assert result[0].closed is True


def test_close_allows_draining_pending_versions() -> None:
    journal = StepRegistryJournal[OpaquePayload]()
    published = [OpaquePayload() for _ in range(3)]
    for payload in published:
        journal.publish(payload)

    collected: list[VersionedEntry[OpaquePayload]] = []

    def reader() -> None:
        entries, _ = drain_until_closed(journal, Cursor.initial())
        collected.extend(entries)

    thread, errors = start_thread(reader)
    journal.close()
    join(thread)

    assert errors == []
    assert_same_payloads(payloads(collected), published)


def test_close_is_idempotent_and_terminal_for_publication() -> None:
    journal = StepRegistryJournal[OpaquePayload]()
    journal.close()
    journal.close()
    assert journal.closed is True

    with pytest.raises(StepRegistryClosedError):
        journal.publish(OpaquePayload())

    # Fully caught-up reads after close are terminal and never block.
    increment = journal.read(Cursor.initial(), timeout=0)
    assert increment.entries == ()
    assert increment.closed is True


def test_lag_on_a_closed_journal_is_still_recoverable() -> None:
    journal = StepRegistryJournal[OpaquePayload](capacity=2)
    journal.publish(OpaquePayload())
    consumed = Cursor(0, 1)  # a reader that consumed version 1
    journal.publish(OpaquePayload())
    journal.publish(OpaquePayload())  # version 1 evicted
    journal.close()

    with pytest.raises(CursorLaggedError):
        journal.read(consumed, timeout=0)

    snapshot = journal.snapshot()
    assert snapshot.closed is True
    assert len(snapshot.entries) == 2
    increment = journal.read(snapshot.cursor, timeout=0)
    assert increment.entries == ()
    assert increment.closed is True


def test_initial_cursor_subscribes_to_retained_history_without_lag() -> None:
    journal = StepRegistryJournal[OpaquePayload](capacity=2)
    payloads_published = [OpaquePayload() for _ in range(4)]
    for payload in payloads_published:
        journal.publish(payload)

    # A fresh cursor is not "slow": it gets the live retained history.
    increment = journal.read(Cursor.initial(), timeout=0)
    assert_same_payloads(payloads(increment.entries), payloads_published[-2:])
    assert increment.cursor == Cursor(0, 4)


def test_cancelled_read_raises_and_journal_keeps_working() -> None:
    journal = StepRegistryJournal[OpaquePayload]()
    token = CancellationToken()
    ready = threading.Event()
    cancelled: list[Exception] = []

    def reader() -> None:
        ready.set()
        try:
            journal.read(Cursor.initial(), cancel=token)
        except ReadCancelledError as exc:
            cancelled.append(exc)

    thread, errors = start_thread(reader)
    assert ready.wait(5)
    token.cancel()
    join(thread)

    assert errors == []
    assert isinstance(cancelled[0], ReadCancelledError)

    payload = OpaquePayload()
    journal.publish(payload)
    increment = journal.read(Cursor.initial(), timeout=0)
    assert increment.entries[0].payload is payload
    assert increment.closed is False


def test_cancellation_is_scoped_to_the_cancelled_reader() -> None:
    journal = StepRegistryJournal[OpaquePayload]()
    token = CancellationToken()
    first_ready = threading.Event()
    second_ready = threading.Event()
    cancelled: list[Exception] = []
    second_result: list[Increment[OpaquePayload]] = []

    def first_reader() -> None:
        first_ready.set()
        try:
            journal.read(Cursor.initial(), cancel=token)
        except ReadCancelledError as exc:
            cancelled.append(exc)

    def second_reader() -> None:
        second_ready.set()
        second_result.append(journal.read(Cursor.initial(), timeout=10))

    first_thread, first_errors = start_thread(first_reader)
    second_thread, second_errors = start_thread(second_reader)
    assert first_ready.wait(5)
    assert second_ready.wait(5)

    token.cancel()
    join(first_thread)
    payload = OpaquePayload()
    journal.publish(payload)  # wakes only the reader still waiting
    join(second_thread)

    assert first_errors == []
    assert second_errors == []
    assert isinstance(cancelled[0], ReadCancelledError)
    assert second_result[0].entries[0].payload is payload


def test_pre_set_cancellation_only_interrupts_waiting() -> None:
    journal = StepRegistryJournal[OpaquePayload]()
    token = CancellationToken()
    token.cancel()

    # Nothing to read and already cancelled: fail immediately, do not block.
    with pytest.raises(ReadCancelledError):
        journal.read(Cursor.initial(), cancel=token)

    # Available increments are delivered despite the stale token.
    payload = OpaquePayload()
    journal.publish(payload)
    increment = journal.read(Cursor.initial(), cancel=token, timeout=0)
    assert increment.entries[0].payload is payload


def test_version_wrap_invalidates_previous_epoch_cursors() -> None:
    journal = StepRegistryJournal[OpaquePayload](capacity=3, version_mod=8)
    epoch_zero = [OpaquePayload() for _ in range(7)]
    for payload in epoch_zero:
        journal.publish(payload)

    caught_up = journal.snapshot().cursor
    assert caught_up == Cursor(0, 7)

    wrapped_entry = journal.publish(OpaquePayload())
    assert (wrapped_entry.epoch, wrapped_entry.version) == (1, 1)

    # Every cursor from the finished epoch lags -- even a fully caught-up one,
    # and even the initial cursor of a brand-new reader.
    for stale_cursor in (caught_up, Cursor(0, 1), Cursor.initial()):
        with pytest.raises(CursorLaggedError) as exc_info:
            journal.read(stale_cursor, timeout=0)
        assert exc_info.value.requested_cursor == stale_cursor
        assert exc_info.value.current_cursor == Cursor(1, 1)

    # Recovery into the new epoch; following increments work as usual.
    snapshot = journal.snapshot()
    assert snapshot.cursor == Cursor(1, 1)
    assert_same_payloads(payloads(snapshot.entries), [wrapped_entry.payload])

    next_entry = journal.publish(OpaquePayload())
    assert (next_entry.epoch, next_entry.version) == (1, 2)
    increment = journal.read(snapshot.cursor, timeout=0)
    assert_same_payloads(payloads(increment.entries), [next_entry.payload])
    assert increment.cursor == Cursor(1, 2)

    # Cursors ahead of the journal are programming errors, not lag.
    with pytest.raises(ValueError, match="ahead"):
        journal.read(Cursor(1, 9), timeout=0)
    with pytest.raises(ValueError, match="ahead"):
        journal.read(Cursor(9, 0), timeout=0)


def test_multiple_wraps_keep_advancing_epochs() -> None:
    journal = StepRegistryJournal[OpaquePayload](capacity=3, version_mod=8)
    for _ in range(7):
        journal.publish(OpaquePayload())
    assert journal.publish(OpaquePayload()).epoch == 1
    for _ in range(6):  # versions 2..7 of epoch 1
        journal.publish(OpaquePayload())

    wrapped = journal.publish(OpaquePayload())
    assert (wrapped.epoch, wrapped.version) == (2, 1)

    with pytest.raises(CursorLaggedError):
        journal.read(Cursor(1, 7), timeout=0)
    assert journal.snapshot().cursor == Cursor(2, 1)


def test_invalid_construction_parameters() -> None:
    with pytest.raises(ValueError, match="version_mod"):
        StepRegistryJournal(version_mod=1)
    with pytest.raises(ValueError, match="capacity"):
        StepRegistryJournal(capacity=0)
    with pytest.raises(ValueError, match="capacity"):
        StepRegistryJournal(capacity=8, version_mod=8)


def test_timeout_polls_without_blocking() -> None:
    journal = StepRegistryJournal[OpaquePayload]()
    increment = journal.read(Cursor.initial(), timeout=0)
    assert increment.entries == ()
    assert increment.cursor == Cursor.initial()
    assert increment.closed is False

    payload = OpaquePayload()
    journal.publish(payload)
    increment = journal.read(Cursor.initial(), timeout=0)
    assert increment.entries[0].payload is payload

    with pytest.raises(ValueError, match="timeout"):
        journal.read(Cursor.initial(), timeout=-1)


def test_concurrent_input_reordering_converges_for_every_reader() -> None:
    """Shuffled, multi-threaded publication yields one total order that every
    reader reproduces exactly, regardless of read scheduling; reordering the
    input permutation preserves the aggregate (identity multiset)."""
    publisher_count = 8
    per_publisher = 50
    total = publisher_count * per_publisher
    reader_count = 4
    all_payloads = [OpaquePayload() for _ in range(total)]

    plan = list(range(total))
    random.Random(1234).shuffle(plan)
    partitions = [plan[i::publisher_count] for i in range(publisher_count)]

    journal = StepRegistryJournal[OpaquePayload](capacity=total)
    observations: list[list[VersionedEntry[OpaquePayload]]] = [[] for _ in range(reader_count)]
    started = threading.Barrier(reader_count + 1)

    def publisher(indices: list[int]) -> None:
        for index in indices:
            journal.publish(all_payloads[index])

    def reader(bucket: list[VersionedEntry[OpaquePayload]]) -> None:
        started.wait()
        entries, _ = drain_until_closed(journal, Cursor.initial())
        bucket.extend(entries)

    reader_threads: list[threading.Thread] = []
    reader_errors: list[list[Exception]] = []
    for bucket in observations:
        thread, errors = start_thread(lambda bucket=bucket: reader(bucket))
        reader_threads.append(thread)
        reader_errors.append(errors)
    started.wait()

    publisher_threads = [start_thread(lambda indices=part: publisher(indices)) for part in partitions]
    for thread, _ in publisher_threads:
        join(thread)
    journal.close()
    for thread in reader_threads:
        join(thread)

    canonical = list(journal.snapshot().entries)
    assert len(canonical) == total
    # The committed order is a gap-free, duplicate-free sequence of versions.
    assert versions(canonical) == [(0, version) for version in range(1, total + 1)]

    for bucket, errors in zip(observations, reader_errors, strict=True):
        assert errors == []
        assert len(bucket) == total
        assert versions(bucket) == versions(canonical)
        assert all(observed.payload is expected.payload for observed, expected in zip(bucket, canonical, strict=True))

    # Reordering the inputs produces a different order but the same aggregate.
    reordered = StepRegistryJournal[OpaquePayload](capacity=total)
    for payload in reversed(all_payloads):
        reordered.publish(payload)
    assert sorted(id(entry.payload) for entry in reordered.snapshot().entries) == sorted(
        id(payload) for payload in all_payloads
    )
