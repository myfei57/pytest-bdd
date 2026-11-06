"""Tests for the step registry two-phase change protocol (prepare/commit)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from pytest_bdd import parsers
from pytest_bdd.exceptions import (
    DuplicateStepError,
    StaleSnapshotError,
    StepAlreadyExistsError,
    StepRegistryError,
)
from pytest_bdd.step_registry import ChangePlan, CommitResult, StepEntry, StepKey, StepRegistry
from pytest_bdd.steps import StepFunctionContext


def never_called(*args: Any, **kwargs: Any) -> None:
    """A stand-in step function that fails loudly if ever executed."""
    raise AssertionError("step functions must never be executed by the registry")


@pytest.fixture
def registry() -> StepRegistry:
    return StepRegistry()


def key(step_type: str | None, name: str) -> StepKey:
    return StepKey(step_type=step_type, name=name)  # type: ignore[arg-type]


def entry(step_type: str | None, name: str, func: Any = never_called) -> StepEntry:
    return StepEntry(key=key(step_type, name), func=func)


# --------------------------------------------------------------------------- #
# Basic prepare/commit behaviour
# --------------------------------------------------------------------------- #


def test_prepare_returns_versioned_plan_without_mutating_registry(registry: StepRegistry) -> None:
    plan = registry.prepare([entry("given", "a"), entry("when", "b")])

    assert isinstance(plan, ChangePlan)
    assert plan.base_version == 0
    assert isinstance(plan.plan_id, str) and plan.plan_id
    assert [e.key for e in plan.entries] == [key("given", "a"), key("when", "b")]
    assert plan.keys == (key("given", "a"), key("when", "b"))
    # Prepare is read-only: nothing was written and the snapshot did not move.
    assert len(registry) == 0
    assert registry.version == 0


def test_commit_applies_plan_and_bumps_version(registry: StepRegistry) -> None:
    plan = registry.prepare([entry("given", "a"), entry("then", "b")])
    result = registry.commit(plan)

    assert isinstance(result, CommitResult)
    assert result.plan_id == plan.plan_id
    assert result.version == 1
    assert result.added == (key("given", "a"), key("then", "b"))
    assert registry.version == 1
    assert len(registry) == 2
    assert key("given", "a") in registry
    assert registry.get(key("then", "b")) is never_called
    assert registry.keys() == (key("given", "a"), key("then", "b"))


def test_versions_advance_monotonically_across_commits(registry: StepRegistry) -> None:
    assert registry.commit(registry.prepare([entry("given", "a")])).version == 1
    assert registry.commit(registry.prepare([entry("when", "b")])).version == 2
    assert registry.commit(registry.prepare([entry("then", "c")])).version == 3


def test_step_key_fixture_name_convention() -> None:
    assert key("given", "foo").fixture_name == "pytestbdd_stepimpl_given_foo"
    assert key(None, "any").fixture_name == "pytestbdd_stepimpl_*_any"


# --------------------------------------------------------------------------- #
# Empty input
# --------------------------------------------------------------------------- #


def test_empty_input_is_a_noop_plan(registry: StepRegistry) -> None:
    plan = registry.prepare(())

    assert plan.entries == ()
    assert plan.keys == ()
    assert plan.base_version == 0

    result = registry.commit(plan)

    assert result.added == ()
    assert result.version == 0
    assert registry.version == 0
    assert len(registry) == 0


def test_multiple_empty_plans_all_commit(registry: StepRegistry) -> None:
    first = registry.prepare(())
    second = registry.prepare(())
    assert first.plan_id != second.plan_id

    assert registry.commit(first) == CommitResult(plan_id=first.plan_id, version=0, added=())
    # The snapshot does not move on empty commits, so a second empty plan is
    # not stale.
    assert registry.commit(second) == CommitResult(plan_id=second.plan_id, version=0, added=())
    assert registry.version == 0


def test_empty_plan_commit_is_idempotent(registry: StepRegistry) -> None:
    plan = registry.prepare([])
    first = registry.commit(plan)
    second = registry.commit(plan)
    assert first is second


# --------------------------------------------------------------------------- #
# Duplicate identifiers
# --------------------------------------------------------------------------- #


def test_duplicate_identifiers_rejected_at_prepare(registry: StepRegistry) -> None:
    candidates = [entry("given", "a"), entry("when", "b"), entry("given", "a")]

    with pytest.raises(DuplicateStepError) as exc_info:
        registry.prepare(candidates)

    assert exc_info.value.duplicate_keys == ("pytestbdd_stepimpl_given_a",)
    # No plan materialised any state.
    assert registry.version == 0
    assert len(registry) == 0


def test_all_duplicated_identifiers_are_reported(registry: StepRegistry) -> None:
    candidates = [
        entry("given", "a"),
        entry("given", "a"),
        entry("when", "b"),
        entry("when", "b"),
    ]
    with pytest.raises(DuplicateStepError) as exc_info:
        registry.prepare(candidates)
    assert exc_info.value.duplicate_keys == (
        "pytestbdd_stepimpl_given_a",
        "pytestbdd_stepimpl_when_b",
    )


def test_same_name_with_different_type_is_not_a_duplicate(registry: StepRegistry) -> None:
    plan = registry.prepare([entry("given", "x"), entry("when", "x"), entry("then", "x")])
    result = registry.commit(plan)
    assert result.version == 1
    assert len(registry) == 3


def test_duplicate_identifiers_rejected_defensively_at_commit(registry: StepRegistry) -> None:
    # A plan object is normally only produced by prepare(), but commit() must
    # still defend itself against a hand-built plan carrying duplicates.
    forged = ChangePlan(
        base_version=0,
        plan_id="forged",
        entries=(entry("given", "a"), entry("given", "a")),
    )
    with pytest.raises(DuplicateStepError):
        registry.commit(forged)
    assert registry.version == 0
    assert len(registry) == 0


# --------------------------------------------------------------------------- #
# Partial failure: conflicts must never leave half-finished state
# --------------------------------------------------------------------------- #


def test_conflicting_batch_writes_nothing(registry: StepRegistry) -> None:
    registry.commit(registry.prepare([entry("given", "existing")]))

    plan = registry.prepare([entry("when", "new-1"), entry("given", "existing"), entry("then", "new-2")])
    with pytest.raises(StepAlreadyExistsError) as exc_info:
        registry.commit(plan)

    assert exc_info.value.existing_keys == ("pytestbdd_stepimpl_given_existing",)
    # The non-conflicting entries of the batch must not have been written.
    assert len(registry) == 1
    assert key("when", "new-1") not in registry
    assert key("then", "new-2") not in registry
    assert registry.version == 1


def test_failed_install_rolls_back_to_previous_snapshot(registry: StepRegistry) -> None:
    registry.commit(registry.prepare([entry("given", "already-there")]))
    snapshot_keys = registry.keys()
    snapshot_version = registry.version

    class BrokenRegistry(StepRegistry):
        def __init__(self) -> None:
            super().__init__()
            self.install_count = 0

        def _install(self, entries: Any, version: int) -> None:
            self.install_count += 1
            if self.install_count == 2:
                # Simulate a failure halfway through installing the new snapshot.
                self._entries = entries
                raise RuntimeError("storage backend blew up")
            super()._install(entries, version)

    broken = BrokenRegistry()
    broken.commit(broken.prepare([entry("given", "already-there")]))
    plan = broken.prepare([entry("when", "would-be")])

    with pytest.raises(RuntimeError, match="storage backend blew up"):
        broken.commit(plan)

    # The half-installed mapping was rolled back: no trace of the failed plan.
    assert broken.version == snapshot_version
    assert broken.keys() == snapshot_keys
    assert key("when", "would-be") not in broken

    # The registry is still usable after the rollback.
    result = broken.commit(broken.prepare([entry("then", "healthy")]))
    assert result.version == 2
    assert key("then", "healthy") in broken


# --------------------------------------------------------------------------- #
# Stale snapshot detection
# --------------------------------------------------------------------------- #


def test_commit_rejects_plan_prepared_against_stale_snapshot(registry: StepRegistry) -> None:
    stale_plan = registry.prepare([entry("given", "a")])
    # Another change lands first and advances the snapshot.
    registry.commit(registry.prepare([entry("when", "b")]))

    with pytest.raises(StaleSnapshotError) as exc_info:
        registry.commit(stale_plan)

    assert exc_info.value.expected_version == 0
    assert exc_info.value.actual_version == 1
    # The committed change is intact, the stale change leaked nothing.
    assert registry.version == 1
    assert key("when", "b") in registry
    assert key("given", "a") not in registry


def test_stale_plan_can_be_reprepared_after_snapshot_move(registry: StepRegistry) -> None:
    wanted = [entry("given", "a")]
    stale_plan = registry.prepare(wanted)
    registry.commit(registry.prepare([entry("when", "b")]))

    with pytest.raises(StaleSnapshotError):
        registry.commit(stale_plan)

    fresh_plan = registry.prepare(wanted)
    assert fresh_plan.base_version == 1
    result = registry.commit(fresh_plan)
    assert result.version == 2
    assert registry.get(key("given", "a")) is never_called


# --------------------------------------------------------------------------- #
# Idempotent retries
# --------------------------------------------------------------------------- #


def test_committing_same_plan_twice_is_idempotent(registry: StepRegistry) -> None:
    plan = registry.prepare([entry("given", "a"), entry("when", "b")])
    first = registry.commit(plan)
    second = registry.commit(plan)

    assert first is second
    assert registry.version == 1
    assert len(registry) == 2


def test_idempotent_retry_after_an_intervening_commit(registry: StepRegistry) -> None:
    first_plan = registry.prepare([entry("given", "a")])
    first_result = registry.commit(first_plan)
    # The snapshot moves because of an unrelated commit before the retry...
    registry.commit(registry.prepare([entry("when", "b")]))
    assert registry.version == 2

    # ...but retrying the already-committed plan must still succeed (classic
    # retry-after-timeout) instead of being reported as stale.
    retried = registry.commit(first_plan)
    assert retried is first_result
    assert registry.version == 2
    assert len(registry) == 2


def test_plan_id_reuse_with_different_payload_is_rejected(registry: StepRegistry) -> None:
    first_plan = registry.prepare([entry("given", "a")])
    registry.commit(first_plan)

    other_plan = registry.prepare([entry("when", "b")])
    forged = replace(other_plan, plan_id=first_plan.plan_id)
    with pytest.raises(StepRegistryError, match="already committed with a different payload"):
        registry.commit(forged)
    # The forged payload was not applied.
    assert key("when", "b") not in registry
    assert registry.version == 1


# --------------------------------------------------------------------------- #
# Step functions are opaque: never executed, test inputs never read
# --------------------------------------------------------------------------- #


def test_step_functions_are_never_executed(registry: StepRegistry) -> None:
    calls: list[str] = []

    def recording_step(label: str) -> Any:
        def _step(*args: Any, **kwargs: Any) -> None:
            calls.append(label)
            raise AssertionError("must not run")

        return _step

    plan = registry.prepare(
        [
            entry("given", "a", recording_step("a")),
            entry("when", "b", recording_step("b")),
        ]
    )
    registry.commit(plan)
    registry.commit(plan)  # idempotent replay
    registry.get(key("given", "a"))
    list(registry)
    assert calls == []


def test_declared_pattern_is_kept_literal_without_parsing_inputs(registry: StepRegistry) -> None:
    # The name is declaration metadata (a parse pattern), not a test input:
    # it is stored verbatim and no parser is run against any value.
    pattern = "I have {n:d} apples"
    plan = registry.prepare([entry("given", pattern)])
    assert plan.entries[0].key.name == pattern
    registry.commit(plan)
    assert registry.get(key("given", pattern)) is never_called


def test_entry_from_step_function_context_reads_metadata_only(registry: StepRegistry) -> None:
    pattern = "I have {n:d} apples"
    context = StepFunctionContext(
        type="given",
        step_func=never_called,
        parser=parsers.parse(pattern),
        converters={},
    )

    step_entry = StepEntry.from_context(context)
    assert step_entry.key == key("given", pattern)
    assert step_entry.func is never_called

    result = registry.commit(registry.prepare([context]))
    assert result.added == (key("given", pattern),)


def test_duck_typed_contexts_are_accepted_without_invoking_anything() -> None:
    context = SimpleNamespace(
        type="then",
        step_func=never_called,
        parser=SimpleNamespace(name="backend is healthy"),
    )
    registry = StepRegistry()
    plan = registry.prepare([context])
    assert plan.entries == (entry("then", "backend is healthy"),)
    registry.commit(plan)
    assert registry.get(key("then", "backend is healthy")) is never_called


# --------------------------------------------------------------------------- #
# Invalid inputs
# --------------------------------------------------------------------------- #


def test_invalid_entries_are_rejected_without_state_change(registry: StepRegistry) -> None:
    with pytest.raises(TypeError):
        registry.prepare([object()])
    with pytest.raises(TypeError, match="callable"):
        StepEntry(key=key("given", "x"), func=42)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Invalid step type"):
        StepKey(step_type="later", name="x")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="string"):
        StepKey(step_type="given", name=123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="StepKey"):
        StepEntry(key="given:x", func=never_called)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ChangePlan"):
        registry.commit([entry("given", "x")])  # type: ignore[arg-type]

    assert registry.version == 0
    assert len(registry) == 0
