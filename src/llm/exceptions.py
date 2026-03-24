"""
Custom exceptions for the LLM package.

Provides standardized exception types for better error handling and debugging.
"""


class LLMError(Exception):
    """Base exception for all LLM-related errors."""

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        """
        Initialize LLMError with message and optional details.

        Args:
            message: Error message
            details: Additional context information
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class ModelError(LLMError):
    """Exception raised for model-related errors."""

    pass


class ConfigError(LLMError):
    """Exception raised for configuration errors."""

    pass


class TokenizationError(LLMError):
    """Exception raised for tokenization errors."""

    pass


class DataError(LLMError):
    """Exception raised for data loading/processing errors."""

    pass


class TrainingError(LLMError):
    """Exception raised for training errors."""

    pass


class InferenceError(LLMError):
    """Exception raised for inference errors."""

    pass


class ServingError(LLMError):
    """Exception raised for serving/API errors."""

    pass


class ValidationError(LLMError):
    """Exception raised for validation errors."""

    pass
