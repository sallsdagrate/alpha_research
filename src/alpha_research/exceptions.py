"""Project-specific exceptions."""


class AlphaResearchError(Exception):
    """Base exception for project errors."""


class ConfigurationError(AlphaResearchError):
    """Raised when configuration is invalid."""


class DataValidationError(AlphaResearchError):
    """Raised when data fails a validation gate."""


class MissingDependencyError(AlphaResearchError):
    """Raised when an optional runtime dependency is required but unavailable."""
