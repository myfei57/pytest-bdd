"""Versioned audit flow for the BDD step registry.

This module records an append-only, versioned audit trail of step registry
operations (i.e. calls to the :func:`pytest_bdd.given`, :func:`pytest_bdd.when`,
:func:`pytest_bdd.then` and :func:`pytest_bdd.step` decorators).

Design guarantees
-----------------

* Auditing is a *side channel*: it never changes the step registration main
  flow. Every failure of the audit machinery is contained and, at worst,
  recorded as an ``error`` event.
* The audited step function is never executed, its converters are never called
  and test input values (parsed step arguments) are never read. Only static
  registration metadata is observed: the step name *template*, the step type,
  the parser class, the target fixture name and the number of converters.
* Events carry a stable, gap-free sequence number and are always returned in
  sequence order, even when they were ingested out of order concurrently.
* Reads are cursor-paginated. Every server-issued cursor is single-use:
  replaying a consumed cursor raises :class:`DuplicateCursorError`.
* Exports are immutable, JSON-serializable copies and sensitive fields
  (passwords, tokens, credentials, ...) are redacted.
"""

from __future__ import annotations

import base64
import binascii
import enum
import json
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from inspect import getframeinfo
from types import FrameType
from typing import Any
from uuid import uuid4

#: Version of the audit event/export schema. Bump on incompatible changes.
AUDIT_SCHEMA_VERSION = 1

#: Value substituted for sensitive fields in exports.
MASK = "***"

#: Default page size for cursor pagination.
DEFAULT_PAGE_LIMIT = 100

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*$")

#: Field names that must be redacted in exports. Matched case-insensitively as
#: substrings, e.g. ``api_key``, ``userToken`` or ``client_secret``.
_SENSITIVE_FIELD_RE = re.compile(
    r"(password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key|authorization|credentials?|session[_-]?id)",
    re.IGNORECASE,
)


class AuditError(Exception):
    """Base class for audit-flow errors."""


class InvalidCursorError(AuditError):
    """Raised when a pagination cursor is malformed, tampered or foreign."""


class DuplicateCursorError(InvalidCursorError):
    """Raised when an already consumed cursor is presented again."""


@enum.unique
class AuditResultCategory(enum.Enum):
    """Category of the audited registry operation result."""

    #: The step definition was registered; all validation checks passed.
    REGISTERED = "registered"
    #: The step definition was registered, but metadata validation failed.
    VALIDATION_FAILED = "validation_failed"
    #: An unexpected error occurred while producing the audit record.
    ERROR = "error"


@dataclass(frozen=True)
class SourceLocation:
    """Source location of a registry operation.

    :ivar path: File in which the step decorator was applied.
    :ivar line: Line at which the step decorator was applied.
    :ivar function: Name of the enclosing function (``<module>`` at module level).
    """

    path: str
    line: int
    function: str | None = None

    def to_dict(self) -> dict[str, str | int]:
        result: dict[str, str | int] = {"path": self.path, "line": self.line}
        if self.function is not None:
            result["function"] = self.function
        return result


@dataclass(frozen=True)
class ValidationSummary:
    """Result of the static validation performed while auditing an operation.

    :ivar ok: ``True`` when every check passed and no error was recorded.
    :ivar checks: Ordered mapping of check name -> check result.
    :ivar error: Optional error message for the ``error`` result category.
    """

    ok: bool
    checks: tuple[tuple[str, bool], ...] = ()
    error: str | None = None

    @classmethod
    def from_checks(cls, checks: Mapping[str, bool], error: str | None = None) -> ValidationSummary:
        check_items = tuple(checks.items())
        return cls(ok=error is None and all(checks.values()), checks=check_items, error=error)

    @classmethod
    def failure(cls, error: str, checks: Mapping[str, bool] | None = None) -> ValidationSummary:
        check_items = tuple((checks or {}).items())
        return cls(ok=False, checks=check_items, error=error)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": dict(self.checks), "error": self.error}


@dataclass(frozen=True)
class AuditEvent:
    """A single, immutable audit record.

    :ivar seq: Stable, positive and (within a stream) unique event number.
    :ivar category: Result category of the registry operation.
    :ivar source: Source location where the operation originated, if available.
    :ivar summary: Static validation summary.
    :ivar details: Flat string mapping of non-executable registration metadata.
    :ivar recorded_at: UTC ISO-8601 timestamp of when the event was recorded.
    :ivar schema_version: Version of the event schema.
    """

    seq: int
    category: AuditResultCategory
    source: SourceLocation | None
    summary: ValidationSummary
    details: Mapping[str, str]
    recorded_at: str
    schema_version: int = AUDIT_SCHEMA_VERSION

    def to_dict(self, *, mask: bool = True) -> dict[str, Any]:
        """Serialize the event to a plain, JSON-serializable dict.

        :param mask: Redact sensitive fields when ``True``.
        """
        details = mask_sensitive_fields(self.details) if mask else dict(self.details)
        return {
            "seq": self.seq,
            "category": self.category.value,
            "source": self.source.to_dict() if self.source is not None else None,
            "summary": self.summary.to_dict(),
            "details": details,
            "recorded_at": self.recorded_at,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class AuditPage:
    """A single page of an audit stream.

    :ivar events: Events of the page, in ascending sequence order.
    :ivar next_cursor: Opaque cursor to fetch the next page, or ``None`` when
                       the stream is exhausted (no event follows the cursor).
    """

    events: tuple[AuditEvent, ...]
    next_cursor: str | None

    @property
    def exhausted(self) -> bool:
        """``True`` when no further page can be fetched from this position."""
        return self.next_cursor is None


@dataclass(frozen=True)
class RecoveryReport:
    """Result of :meth:`StepRegistryAudit.recover_order`.

    :ivar count: Number of events retained after dropping duplicate sequences.
    :ivar reordered: ``True`` when storage order differed from sequence order.
    :ivar missing_seqs: Sequence numbers missing between 1 and the highest seq.
    :ivar duplicate_seqs: Sequence numbers that appeared more than once.
    """

    count: int
    reordered: bool
    missing_seqs: tuple[int, ...]
    duplicate_seqs: tuple[int, ...]


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def is_sensitive_field(name: str) -> bool:
    """Return ``True`` when a field name denotes a sensitive value."""
    return bool(_SENSITIVE_FIELD_RE.search(name))


def mask_sensitive_fields(details: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``details`` with sensitive fields redacted.

    Nested mappings and lists are traversed so that redaction cannot be bypassed
    by nesting a sensitive value.
    """
    redacted: dict[str, Any] = {}
    for key, value in details.items():
        if is_sensitive_field(str(key)):
            redacted[str(key)] = MASK
        elif isinstance(value, Mapping):
            redacted[str(key)] = mask_sensitive_fields(value)
        elif isinstance(value, list):
            redacted[str(key)] = [mask_sensitive_fields(item) if isinstance(item, Mapping) else item for item in value]
        else:
            redacted[str(key)] = value
    return redacted


def source_from_frame(frame: FrameType | None) -> SourceLocation | None:
    """Build a :class:`SourceLocation` from a frame, without retaining it.

    Returns ``None`` when no frame is available. The frame is never stored on
    the returned object (only copied filename/line/function strings), so no
    reference cycle is created.
    """
    if frame is None:
        return None
    info = getframeinfo(frame, context=0)
    return SourceLocation(path=info.filename, line=info.lineno or 0, function=frame.f_code.co_name)


def _encode_cursor(*, stream_id: str, after_seq: int, nonce: str) -> str:
    payload = json.dumps(
        {"v": AUDIT_SCHEMA_VERSION, "s": stream_id, "a": after_seq, "n": nonce},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str, *, stream_id: str) -> tuple[int, str]:
    """Decode and validate an opaque cursor.

    :raises InvalidCursorError: for any malformed, tampered or foreign cursor.
    """
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        raw = base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
        payload = json.loads(raw.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise InvalidCursorError("Malformed audit cursor") from exc

    if not isinstance(payload, dict):
        raise InvalidCursorError("Malformed audit cursor")

    version = payload.get("v")
    cursor_stream = payload.get("s")
    after_seq = payload.get("a")
    nonce = payload.get("n")

    if version != AUDIT_SCHEMA_VERSION or not isinstance(cursor_stream, str):
        raise InvalidCursorError("Incompatible audit cursor version")
    if not isinstance(after_seq, int) or isinstance(after_seq, bool) or after_seq < 0:
        raise InvalidCursorError("Invalid cursor position")
    if not isinstance(nonce, str) or not nonce:
        raise InvalidCursorError("Invalid cursor nonce")
    if cursor_stream != stream_id:
        raise InvalidCursorError("Cursor belongs to a different audit stream")

    return after_seq, nonce


class StepRegistryAudit:
    """Thread-safe, append-only, versioned audit stream for step registrations."""

    def __init__(self, stream_id: str | None = None) -> None:
        #: Stable identifier of this stream; embedded in cursors and exports.
        self.stream_id = stream_id or uuid4().hex
        self._lock = threading.Lock()
        self._events: list[AuditEvent] = []
        self._next_seq = 1
        #: Nonces of cursors already consumed (replay detection).
        self._consumed_nonces: set[str] = set()

    # ------------------------------------------------------------------ recording

    def record_registration(
        self,
        *,
        step_name: str,
        step_type: str | None,
        parser_name: str,
        parser_type: str,
        target_fixture: str | None,
        converter_count: int,
        function_name: str,
        frame: FrameType | None = None,
    ) -> AuditEvent:
        """Audit a step registration.

        Only static metadata is accepted: the step function itself, its
        converters and the parsed test input values are intentionally absent
        from the signature and are never accessed.

        This method never raises into the registration main flow: if auditing
        fails, an :attr:`AuditResultCategory.ERROR` event is produced instead.
        """
        try:
            checks = {
                "step_name_present": bool(step_name.strip()),
                "step_type_valid": step_type is None or step_type in ("given", "when", "then"),
                "parser_name_present": bool(parser_name.strip()),
                "target_fixture_identifier": target_fixture is None or bool(_IDENTIFIER_RE.fullmatch(target_fixture)),
                "converter_count_non_negative": converter_count >= 0,
            }
            summary = ValidationSummary.from_checks(checks)
            category = AuditResultCategory.REGISTERED if summary.ok else AuditResultCategory.VALIDATION_FAILED
            details = {
                "step_type": step_type or "*",
                "step_name": step_name,
                "parser_type": parser_type,
                "step_function": function_name,
                "target_fixture": target_fixture or "",
                "converter_count": str(converter_count),
            }
            return self.append(category, summary=summary, details=details, source=source_from_frame(frame))
        except Exception as exc:  # noqa: BLE001 - auditing must never break the main flow
            # Last-resort fallback: record the audit failure itself.
            # Source capture may itself be the failing component, so do not retry it.
            return self.append(
                AuditResultCategory.ERROR,
                summary=ValidationSummary.failure(f"audit failure: {exc!r}"),
            )

    def record_error(
        self,
        error: str,
        *,
        details: Mapping[str, str] | None = None,
        source: SourceLocation | None = None,
    ) -> AuditEvent:
        """Record an error-category event without raising."""
        return self.append(
            AuditResultCategory.ERROR,
            summary=ValidationSummary.failure(error),
            details=details,
            source=source,
        )

    def append(
        self,
        category: AuditResultCategory,
        *,
        summary: ValidationSummary | None = None,
        details: Mapping[str, Any] | None = None,
        source: SourceLocation | None = None,
        seq: int | None = None,
        recorded_at: str | None = None,
    ) -> AuditEvent:
        """Append an event.

        :param seq: Explicit event number. When omitted, the next contiguous
                    number is assigned. Explicit numbers support ingesting
                    events from a versioned export; out-of-order ingestion is
                    repaired on read via :meth:`recover_order`.
        """
        if not isinstance(category, AuditResultCategory):
            raise TypeError("category must be an AuditResultCategory")
        frozen_details: Mapping[str, str] = _freeze_details(details)
        event_summary = summary if summary is not None else ValidationSummary(ok=True)

        with self._lock:
            if seq is None:
                assigned_seq = self._next_seq
                self._next_seq += 1
            else:
                if not isinstance(seq, int) or isinstance(seq, bool) or seq < 1:
                    raise ValueError("seq must be a positive integer")
                assigned_seq = seq
                if assigned_seq >= self._next_seq:
                    self._next_seq = assigned_seq + 1

            event = AuditEvent(
                seq=assigned_seq,
                category=category,
                source=source,
                summary=event_summary,
                details=frozen_details,
                recorded_at=recorded_at or _utc_now_iso(),
            )
            self._events.append(event)
        return event

    # -------------------------------------------------------------------- reading

    def snapshot(self) -> tuple[AuditEvent, ...]:
        """Return an immutable snapshot of all events in sequence order.

        The snapshot is taken under the stream lock, so concurrent appends
        either are fully visible or not visible at all. It does not consume
        pagination cursors.
        """
        with self._lock:
            return self._ordered_locked()

    def read_page(self, cursor: str | None = None, *, limit: int = DEFAULT_PAGE_LIMIT) -> AuditPage:
        """Read one page of events starting after ``cursor``.

        :param cursor: Cursor returned by a previous call, or ``None`` to start
                       from the beginning.
        :param limit: Maximum number of events to return (>= 1).

        :raises ValueError: if ``limit`` is not a positive integer.
        :raises InvalidCursorError: if the cursor is malformed or foreign.
        :raises DuplicateCursorError: if the cursor was already consumed.
        """
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer")

        with self._lock:
            if cursor is None:
                after_seq = 0
            else:
                after_seq, nonce = _decode_cursor(cursor, stream_id=self.stream_id)
                if nonce in self._consumed_nonces:
                    raise DuplicateCursorError("Audit cursor has already been consumed")
                self._consumed_nonces.add(nonce)

            ordered = self._ordered_locked()
            page = tuple(event for event in ordered if event.seq > after_seq)[:limit]
            if not page:
                return AuditPage(events=(), next_cursor=None)
            next_cursor = _encode_cursor(
                stream_id=self.stream_id,
                after_seq=page[-1].seq,
                nonce=uuid4().hex,
            )
            return AuditPage(events=page, next_cursor=next_cursor)

    # -------------------------------------------------------------------- export

    def export_page(self, cursor: str | None = None, *, limit: int = DEFAULT_PAGE_LIMIT) -> dict[str, Any]:
        """Return a masked, JSON-serializable export of one page.

        Exporting consumes the supplied cursor exactly like
        :meth:`read_page`; it never modifies the underlying events.
        """
        page = self.read_page(cursor, limit=limit)
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "stream_id": self.stream_id,
            "generated_at": _utc_now_iso(),
            "count": len(page.events),
            "next_cursor": page.next_cursor,
            "events": [event.to_dict(mask=True) for event in page.events],
        }

    def export_all(self) -> dict[str, Any]:
        """Export every event as a masked, JSON-serializable envelope.

        Uses :meth:`snapshot` internally: bulk export does not consume
        pagination cursors and never mutates the stream.
        """
        events = self.snapshot()
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "stream_id": self.stream_id,
            "generated_at": _utc_now_iso(),
            "count": len(events),
            "events": [event.to_dict(mask=True) for event in events],
        }

    # ------------------------------------------------------------- order recovery

    def recover_order(self) -> RecoveryReport:
        """Recover sequence order after out-of-order or duplicate ingestion.

        Events are sorted by sequence number, duplicate sequences are dropped
        (the first occurrence wins) and gaps between 1 and the highest sequence
        number are reported. Future automatic sequence numbers continue after
        the highest retained event.
        """
        with self._lock:
            before = self._events
            ordered = sorted(before, key=lambda event: event.seq)

            seen: set[int] = set()
            unique: list[AuditEvent] = []
            duplicates: list[int] = []
            for event in ordered:
                if event.seq in seen:
                    duplicates.append(event.seq)
                    continue
                seen.add(event.seq)
                unique.append(event)

            highest = unique[-1].seq if unique else 0
            missing = [seq for seq in range(1, highest + 1) if seq not in seen]

            reordered = [event.seq for event in before] != [event.seq for event in unique]
            self._events = unique
            if self._next_seq <= highest:
                self._next_seq = highest + 1

            return RecoveryReport(
                count=len(unique),
                reordered=reordered,
                missing_seqs=tuple(missing),
                duplicate_seqs=tuple(duplicates),
            )

    # ------------------------------------------------------------------- helpers

    def _ordered_locked(self) -> tuple[AuditEvent, ...]:
        """Return events ordered by seq; stable for duplicate explicit seqs."""
        return tuple(sorted(self._events, key=lambda event: event.seq))


def _freeze_details(details: Mapping[str, Any] | None) -> Mapping[str, str]:
    if details is None:
        return {}
    return {str(key): str(value) for key, value in details.items()}


#: Process-wide audit stream observed by the step registry decorators.
step_registry_audit = StepRegistryAudit()


def reset_step_registry_audit() -> StepRegistryAudit:
    """Replace the global step-registry audit stream with an empty one.

    Mainly intended for tests. The decorators read the module-level singleton
    lazily, so the replacement takes effect immediately.
    """
    global step_registry_audit
    step_registry_audit = StepRegistryAudit()
    return step_registry_audit
