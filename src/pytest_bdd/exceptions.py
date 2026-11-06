"""pytest-bdd Exceptions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .step_registry import Cursor


class StepImplementationError(Exception):
    """Step implementation error."""


class ScenarioIsDecoratorOnly(Exception):
    """Scenario can be only used as decorator."""


class ScenarioValidationError(Exception):
    """Base class for scenario validation."""


class ScenarioNotFound(ScenarioValidationError):
    """Scenario Not Found."""


class StepDefinitionNotFoundError(Exception):
    """Step definition not found."""


class NoScenariosFound(Exception):
    """No scenarios found."""


class GherkinParseError(Exception):
    """Base class for all Gherkin parsing errors."""

    def __init__(self, message: str, line: int, line_content: str, filename: str) -> None:
        super().__init__(message)
        self.message = message
        self.line = line
        self.line_content = line_content
        self.filename = filename

    def __str__(self) -> str:
        return f"{self.message}\nLine number: {self.line}\nLine: {self.line_content}\nFile: {self.filename}"


class FeatureError(GherkinParseError):
    pass


class BackgroundError(GherkinParseError):
    pass


class ScenarioError(GherkinParseError):
    pass


class StepError(GherkinParseError):
    pass


class RuleError(GherkinParseError):
    pass


class TokenError(GherkinParseError):
    pass


class StepRegistryError(Exception):
    """Base class for errors of the concurrent step registry read model."""


class StepRegistryClosedError(StepRegistryError):
    """Published into a registry journal that was already closed.

    Closing is a terminal state: further publications are rejected, while
    readers may still drain the versions published before the close.
    """

    def __init__(self, message: str = "step registry journal is closed") -> None:
        super().__init__(message)


class CursorLaggedError(StepRegistryError):
    """Reader cursor points at history the journal no longer retains.

    This error is recoverable and never indicates a problem on the writer
    side: the writer is not blocked by slow readers. Resync with
    :meth:`pytest_bdd.step_registry.StepRegistryJournal.snapshot` and continue
    reading from the returned cursor. It is raised both when versions have
    been evicted due to bounded retention and when the cursor belongs to an
    epoch invalidated by version wrap-around.
    """

    def __init__(self, requested: Cursor, retained: Cursor, current: Cursor) -> None:
        self.requested_cursor = requested
        self.retained_cursor = retained
        self.current_cursor = current
        super().__init__(
            f"reader cursor {requested} lagged beyond the retained history "
            f"(oldest retained cursor: {retained}, current cursor: {current}); "
            "recover via StepRegistryJournal.snapshot()"
        )


class ReadCancelledError(StepRegistryError):
    """A blocking registry read was cancelled through its cancellation token."""

    def __init__(self, message: str = "step registry journal read was cancelled") -> None:
        super().__init__(message)
