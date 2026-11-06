"""pytest-bdd Exceptions."""

from __future__ import annotations

from collections.abc import Sequence


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
    """Base class for step registry two-phase change protocol errors."""


class DuplicateStepError(StepRegistryError):
    """The same step identifier appears more than once in a candidate plan.

    Raised while preparing a plan (and re-checked defensively at commit time).
    Nothing is written to the registry when this error is raised.
    """

    def __init__(self, duplicate_keys: Sequence[str]) -> None:
        self.duplicate_keys = tuple(duplicate_keys)
        super().__init__(f"Duplicate step identifiers in candidate plan: {', '.join(self.duplicate_keys)}")


class StepAlreadyExistsError(StepRegistryError):
    """A plan tries to register a step identifier that is already committed."""

    def __init__(self, existing_keys: Sequence[str]) -> None:
        self.existing_keys = tuple(existing_keys)
        super().__init__(f"Steps already registered: {', '.join(self.existing_keys)}")


class StaleSnapshotError(StepRegistryError):
    """The plan was prepared against a registry version that is no longer current."""

    def __init__(self, expected_version: int, actual_version: int) -> None:
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"Stale change plan: it was prepared against version {expected_version}, "
            f"but the registry is now at version {actual_version}"
        )
