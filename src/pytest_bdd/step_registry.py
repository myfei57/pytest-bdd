"""Two-phase change protocol for the BDD step registry.

Registering a batch of steps is split into an explicit prepare/commit pair:

1. :meth:`StepRegistry.prepare` validates the requested steps and returns a
   frozen :class:`ChangePlan` -- the *candidate plan* -- stamped with the
   registry version it was built against (the input snapshot version) and a
   unique ``plan_id``.  It never mutates the registry.
2. :meth:`StepRegistry.commit` writes the plan into the registry only if:

   * the snapshot version is still current (optimistic concurrency check),
   * none of the step identifiers already exists,
   * the batch itself contains no duplicate identifier.

   Writes are applied atomically: a new registry mapping is built in a local
   variable and installed with a single swap, so any failure rolls back to the
   previous snapshot and can never leave the registry half-populated.

Retrying :meth:`commit` with the same plan is idempotent: the ``plan_id``
identifies a plan that was already committed, so the original result is
returned without applying anything twice.

The protocol treats step functions as opaque callables: they are stored by
reference and never executed.  Step identifiers are derived from declaration
metadata only (step type and the declared parser name); no parser is ever run
against test input values.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .exceptions import DuplicateStepError, StaleSnapshotError, StepAlreadyExistsError, StepRegistryError
from .steps import StepNamePrefix

if TYPE_CHECKING:
    from .steps import StepFunctionContext

StepType = Literal["given", "when", "then"]

_ALLOWED_STEP_TYPES = ("given", "when", "then")


@dataclass(frozen=True)
class StepKey:
    """Identifier of a registered step: its type and declared name."""

    step_type: StepType | None
    name: str

    def __post_init__(self) -> None:
        if self.step_type is not None and self.step_type not in _ALLOWED_STEP_TYPES:
            raise ValueError(f"Invalid step type: {self.step_type!r}")
        if not isinstance(self.name, str):
            raise TypeError(f"Step name must be a string, got {type(self.name).__name__}")

    @property
    def fixture_name(self) -> str:
        """Fixture name conventionally derived from this step identifier."""
        return f"{StepNamePrefix.step_impl.value}_{self.step_type or '*'}_{self.name}"


@dataclass(frozen=True)
class StepEntry:
    """A single step candidate: its identifier and the (opaque) step function."""

    key: StepKey
    func: Callable[..., object]

    def __post_init__(self) -> None:
        if not isinstance(self.key, StepKey):
            raise TypeError(f"Step key must be a StepKey, got {type(self.key).__name__}")
        if not callable(self.func):
            raise TypeError(f"Step function must be callable, got {type(self.func).__name__}")

    @classmethod
    def from_context(cls, context: StepFunctionContext) -> StepEntry:
        """Build an entry from a :class:`~pytest_bdd.steps.StepFunctionContext`.

        Only declaration metadata is read (the step type and the parser's
        declared ``name`` pattern).  The parser is never executed against test
        input and the step function is never called.
        """
        return cls(key=StepKey(step_type=context.type, name=context.parser.name), func=context.step_func)


@dataclass(frozen=True)
class ChangePlan:
    """Candidate plan returned by :meth:`StepRegistry.prepare`.

    :param base_version: Registry snapshot version the plan was prepared against.
    :param plan_id: Unique idempotency token for this plan.
    :param entries: Validated, de-duplicated candidate entries.
    """

    base_version: int
    plan_id: str
    entries: tuple[StepEntry, ...]

    @property
    def keys(self) -> tuple[StepKey, ...]:
        return tuple(entry.key for entry in self.entries)


@dataclass(frozen=True)
class CommitResult:
    """Result of a successful :meth:`StepRegistry.commit`."""

    plan_id: str
    version: int
    added: tuple[StepKey, ...]


class StepRegistry:
    """In-memory BDD step registry with a two-phase prepare/commit protocol."""

    def __init__(self) -> None:
        self._version: int = 0
        self._entries: dict[StepKey, Callable[..., object]] = {}
        self._committed: dict[str, tuple[tuple[StepEntry, ...], CommitResult]] = {}

    @property
    def version(self) -> int:
        """Current snapshot version; bumped on every commit that changes state."""
        return self._version

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __iter__(self) -> Iterator[StepKey]:
        return iter(tuple(self._entries))

    def get(self, key: StepKey) -> Callable[..., object] | None:
        """Return the step function registered for ``key``, or ``None``."""
        return self._entries.get(key)

    def keys(self) -> tuple[StepKey, ...]:
        """Return a snapshot copy of the registered step identifiers."""
        return tuple(self._entries)

    def prepare(self, entries: Iterable[StepEntry | StepFunctionContext]) -> ChangePlan:
        """Validate ``entries`` and return a candidate plan without writing.

        :raises TypeError: if an entry is not a valid :class:`StepEntry` / context.
        :raises DuplicateStepError: if the batch contains a repeated identifier.
        """
        normalized = tuple(self._coerce_entry(entry) for entry in entries)
        self._check_duplicates(normalized)
        return ChangePlan(base_version=self._version, plan_id=uuid.uuid4().hex, entries=normalized)

    def commit(self, plan: ChangePlan) -> CommitResult:
        """Commit a candidate plan into the registry.

        :raises TypeError: if ``plan`` is not a :class:`ChangePlan`.
        :raises DuplicateStepError: if the plan itself contains duplicates.
        :raises StepRegistryError: if ``plan_id`` was already committed with a
            different payload.
        :raises StaleSnapshotError: if the registry moved on since prepare.
        :raises StepAlreadyExistsError: if an identifier already exists.
        """
        if not isinstance(plan, ChangePlan):
            raise TypeError(f"Expected a ChangePlan, got {type(plan).__name__}")

        entries = tuple(self._coerce_entry(entry) for entry in plan.entries)
        self._check_duplicates(entries)

        already_committed = self._committed.get(plan.plan_id)
        if already_committed is not None:
            # Idempotent retry: return the original result without re-applying.
            committed_entries, committed_result = already_committed
            if committed_entries != entries:
                raise StepRegistryError(f"Plan id {plan.plan_id!r} was already committed with a different payload")
            return committed_result

        if plan.base_version != self._version:
            raise StaleSnapshotError(expected_version=plan.base_version, actual_version=self._version)

        conflicts = tuple(key for key in plan.keys if key in self._entries)
        if conflicts:
            raise StepAlreadyExistsError(tuple(key.fixture_name for key in conflicts))

        # Build the new mapping outside the registry state, then swap it in.
        new_entries = dict(self._entries)
        for entry in entries:
            new_entries[entry.key] = entry.func
        new_version = self._version + 1 if entries else self._version

        previous_entries = self._entries
        previous_version = self._version
        try:
            self._install(new_entries, new_version)
        except BaseException:  # noqa: BLE002 - roll back *any* swap failure, then re-raise
            self._entries = previous_entries
            self._version = previous_version
            raise

        result = CommitResult(plan_id=plan.plan_id, version=new_version, added=plan.keys)
        self._committed[plan.plan_id] = (entries, result)
        return result

    def _install(self, entries: dict[StepKey, Callable[..., object]], version: int) -> None:
        """Install a new registry snapshot.

        Extension point for subclasses (e.g. persisting to disk). The base
        implementation performs a single state swap; any raised exception must
        leave the previous snapshot intact -- :meth:`commit` restores it.
        """
        self._entries = entries
        self._version = version

    @staticmethod
    def _coerce_entry(raw: StepEntry | StepFunctionContext) -> StepEntry:
        if isinstance(raw, StepEntry):
            return raw
        step_func = getattr(raw, "step_func", None)
        parser = getattr(raw, "parser", None)
        if callable(step_func) and parser is not None and hasattr(parser, "name"):
            return StepEntry.from_context(raw)  # type: ignore[arg-type]
        raise TypeError(f"Expected a StepEntry or StepFunctionContext, got {type(raw).__name__}")

    @staticmethod
    def _check_duplicates(entries: tuple[StepEntry, ...]) -> None:
        seen: set[StepKey] = set()
        duplicates: list[str] = []
        for entry in entries:
            if entry.key in seen and entry.key.fixture_name not in duplicates:
                duplicates.append(entry.key.fixture_name)
            seen.add(entry.key)
        if duplicates:
            raise DuplicateStepError(duplicates)
