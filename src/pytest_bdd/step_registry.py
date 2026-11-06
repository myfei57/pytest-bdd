"""Concurrent read model for the BDD step registry.

The writer side publishes *immutable versions* into a journal; any number of
readers follow the journal with opaque :class:`Cursor` values and receive only
the increments produced after their cursor.

Guarantees
----------

* **Immutable publications.** Every :meth:`StepRegistryJournal.publish` returns
  an immutable :class:`VersionedEntry`. Published payloads (step definitions)
  are stored and handed back by reference; the journal never calls them and
  never inspects their values.
* **Incremental cursor reads.** A read returns precisely the entries with a
  version greater than the supplied cursor, together with the advanced cursor.
  Readers are independent: one reader advancing never affects another.
* **Bounded retention, non-blocking writer.** The journal keeps at most
  ``capacity`` versions per epoch. Eviction never blocks a publication. A
  reader whose cursor falls behind the oldest retained version gets a
  recoverable :class:`~pytest_bdd.exceptions.CursorLaggedError` instead of
  blocking the writer; it resyncs through
  :meth:`StepRegistryJournal.snapshot` and continues.
* **Close.** :meth:`StepRegistryJournal.close` is idempotent and terminal:
  further publications raise
  :class:`~pytest_bdd.exceptions.StepRegistryClosedError`, waiting readers are
  woken, pending increments can be drained, and a fully caught-up read returns
  ``closed=True``.
* **Cancellation.** A blocking read accepts a :class:`CancellationToken`;
  cancelling it raises :class:`~pytest_bdd.exceptions.ReadCancelledError` for
  that waiter only. The journal, its writer and every other reader keep
  working. Cancellation interrupts waiting only: increments already available
  are still delivered.
* **Version wrap-around.** Versions are fixed-width, cycling through
  ``1 .. version_mod - 1`` within an *epoch*. Reaching the last version of an
  epoch starts a new epoch on the next publication and discards retained
  history; every cursor from a previous epoch (including
  :meth:`Cursor.initial`) then fails with ``CursorLaggedError`` and must
  resync. Epoch tags on entries and cursors make wrap-around unambiguous.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Generic, TypeVar

from .exceptions import (
    CursorLaggedError,
    ReadCancelledError,
    StepRegistryClosedError,
)

T = TypeVar("T")

DEFAULT_CAPACITY = 1024
DEFAULT_VERSION_MOD = 1 << 64


@dataclass(frozen=True)
class Cursor:
    """Opaque, immutable reader position in the journal.

    ``Cursor(epoch=0, version=0)`` (see :meth:`initial`) is the position before
    anything was published. Cursors are values: they are safe to share and
    store, and advancing a read never mutates the cursor passed in.
    """

    epoch: int
    version: int

    @classmethod
    def initial(cls) -> Cursor:
        """Position before the first published version."""
        return cls(epoch=0, version=0)


@dataclass(frozen=True)
class VersionedEntry(Generic[T]):
    """One immutable publication: a payload tagged with epoch and version."""

    epoch: int
    version: int
    payload: T


@dataclass(frozen=True)
class Increment(Generic[T]):
    """Result of a cursor read.

    ``entries`` are all versions newer than the request cursor, in publication
    order; ``cursor`` is the advanced cursor (identical to the request cursor
    for an empty poll); ``closed`` reports the journal state at return time.
    """

    entries: tuple[VersionedEntry[T], ...]
    cursor: Cursor
    closed: bool


@dataclass(frozen=True)
class Snapshot(Generic[T]):
    """Immutable resync point: all currently retained versions and a cursor."""

    entries: tuple[VersionedEntry[T], ...]
    cursor: Cursor
    closed: bool


class CancellationToken:
    """One-shot cancellation handle for blocking :meth:`StepRegistryJournal.read` calls.

    Cancelling a token interrupts every read currently waiting on it; it does
    not touch the journal, the writer, or readers waiting on other tokens.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._wake_callbacks: list[Callable[[], None]] = []

    @property
    def is_cancelled(self) -> bool:
        """Whether :meth:`cancel` was called."""
        return self._event.is_set()

    def cancel(self) -> None:
        """Cancel waiting reads. Idempotent."""
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            callbacks = list(self._wake_callbacks)
        # Invoke callbacks without holding our lock: a callback acquires the
        # journal lock, and journal->token is the only allowed lock order.
        for callback in callbacks:
            callback()

    def _add_wake_callback(self, callback: Callable[[], None]) -> None:
        with self._lock:
            if self._event.is_set():
                raise ReadCancelledError()
            self._wake_callbacks.append(callback)

    def _remove_wake_callback(self, callback: Callable[[], None]) -> None:
        with self._lock:
            try:
                self._wake_callbacks.remove(callback)
            except ValueError:
                pass


class StepRegistryJournal(Generic[T]):
    """Append-only journal of immutable versions with cursor-based readers.

    :param capacity: Number of most recent versions retained per epoch. Must
        satisfy ``1 <= capacity < version_mod``.
    :param version_mod: Fixed-width version space; versions cycle through
        ``1 .. version_mod - 1`` and wrap starts a new epoch. Defaults to a
        64-bit space; smaller values are intended for testing.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY, *, version_mod: int = DEFAULT_VERSION_MOD) -> None:
        if version_mod < 2:
            raise ValueError("version_mod must be at least 2 so that version 1 exists")
        if not 1 <= capacity < version_mod:
            raise ValueError("capacity must satisfy 1 <= capacity < version_mod")
        self._capacity = capacity
        self._version_mod = version_mod
        self._condition = threading.Condition()
        self._entries: deque[VersionedEntry[T]] = deque(maxlen=capacity)
        self._epoch = 0
        self._version = 0
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether the journal was closed."""
        with self._condition:
            return self._closed

    @property
    def current_cursor(self) -> Cursor:
        """Cursor pointing at the latest published version."""
        with self._condition:
            return Cursor(self._epoch, self._version)

    def publish(self, payload: T) -> VersionedEntry[T]:
        """Publish one immutable version and wake all waiting readers.

        Never blocks on readers. Raises
        :class:`~pytest_bdd.exceptions.StepRegistryClosedError` after
        :meth:`close`. At a version-space boundary the publication wraps into a
        new epoch (see module docstring).
        """
        with self._condition:
            if self._closed:
                raise StepRegistryClosedError()
            if self._version >= self._version_mod - 1:
                # Wrap-around: start a fresh epoch. Previous-epoch cursors
                # cannot be continued and have to resync; retained history of
                # the finished epoch is discarded.
                self._epoch += 1
                self._entries.clear()
                self._version = 1
            else:
                self._version += 1
            entry = VersionedEntry(epoch=self._epoch, version=self._version, payload=payload)
            self._entries.append(entry)
            self._condition.notify_all()
        return entry

    def read(
        self,
        cursor: Cursor,
        *,
        timeout: float | None = None,
        cancel: CancellationToken | None = None,
    ) -> Increment[T]:
        """Read versions newer than ``cursor``.

        If increments are available (or the journal is closed) they are
        returned immediately. Otherwise the call blocks until a new version is
        published, the journal is closed, ``timeout`` (seconds) elapses -- in
        which case an empty increment with the unchanged cursor is returned --
        or ``cancel`` fires, raising
        :class:`~pytest_bdd.exceptions.ReadCancelledError`.

        An established cursor behind the oldest retained version, or any
        cursor belonging to a wrapped epoch, raises recoverable
        :class:`~pytest_bdd.exceptions.CursorLaggedError`. The epoch-origin
        cursor (:meth:`Cursor.initial`, version 0) instead simply subscribes
        to the oldest retained history. A cursor ahead of the journal (future
        version or epoch) is a programming error and raises
        :class:`ValueError`.
        """
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        deadline = None if timeout is None else monotonic() + timeout

        with self._condition:

            def wake() -> None:
                # Invoked from CancellationToken.cancel() without holding the
                # journal lock; acquire it before notifying.
                with self._condition:
                    self._condition.notify_all()

            cancellation_registered = False
            try:
                while True:
                    increment = self._collect(cursor)
                    if increment is not None:
                        # Available increments are delivered even when a
                        # pre-cancelled token was passed: cancellation only
                        # interrupts waiting.
                        return increment
                    if cancel is not None:
                        if not cancellation_registered:
                            cancel._add_wake_callback(wake)
                            cancellation_registered = True
                        if cancel.is_cancelled:
                            raise ReadCancelledError()
                    if timeout is None:
                        self._condition.wait()
                    else:
                        remaining = (deadline - monotonic()) if deadline is not None else 0.0
                        if remaining <= 0:
                            return Increment((), cursor, closed=self._closed)
                        self._condition.wait(remaining)
                    if cancel is not None and cancel.is_cancelled:
                        raise ReadCancelledError()
            finally:
                if cancel is not None and cancellation_registered:
                    cancel._remove_wake_callback(wake)

    def snapshot(self) -> Snapshot[T]:
        """Return all retained versions and the matching current cursor.

        This is the recovery path after
        :class:`~pytest_bdd.exceptions.CursorLaggedError` and a way for a new
        reader to obtain the live state before following increments.
        """
        with self._condition:
            return Snapshot(
                entries=tuple(self._entries),
                cursor=Cursor(self._epoch, self._version),
                closed=self._closed,
            )

    def close(self) -> None:
        """Close the journal. Idempotent.

        Rejects further publications, wakes every blocked read, and lets
        readers drain already published versions before receiving
        ``closed=True``.
        """
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _collect(self, cursor: Cursor) -> Increment[T] | None:
        """Build the increment for ``cursor``; ``None`` means "block and retry"."""
        if cursor.epoch > self._epoch:
            raise ValueError(f"cursor epoch {cursor.epoch} is ahead of the current epoch {self._epoch}")
        if cursor.epoch < self._epoch:
            # Cursor belongs to a wrapped-away epoch.
            raise self._lag_error(cursor)

        if cursor.version > self._version:
            raise ValueError(f"cursor version {cursor.version} is ahead of the current version {self._version}")
        oldest_version = self._entries[0].version if self._entries else self._version
        if cursor.version < oldest_version and cursor.version != 0:
            # Version 0 is the epoch origin: a fresh cursor subscribes to the
            # oldest retained history, it has not been evicted from under it.
            raise self._lag_error(cursor)

        entries = tuple(entry for entry in self._entries if entry.version > cursor.version)
        if not entries and not self._closed:
            return None
        new_cursor = Cursor(
            epoch=self._epoch,
            version=entries[-1].version if entries else cursor.version,
        )
        return Increment(entries=entries, cursor=new_cursor, closed=self._closed)

    def _lag_error(self, cursor: Cursor) -> CursorLaggedError:
        oldest_version = self._entries[0].version if self._entries else self._version
        return CursorLaggedError(
            requested=cursor,
            retained=Cursor(epoch=self._epoch, version=oldest_version),
            current=Cursor(epoch=self._epoch, version=self._version),
        )
