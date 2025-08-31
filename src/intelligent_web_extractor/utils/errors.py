class ExtractionError(Exception):
    """Base class for extraction-related errors."""

class NavigationError(ExtractionError):
    """Raised when navigation or interaction fails."""

class PageLoadTimeout(ExtractionError):
    """Raised when a page load or wait operation times out."""

class IframeExtractionError(ExtractionError):
    """Raised when iframe content cannot be collected or merged."""

class SchemaFormatError(ExtractionError):
    """Raised when AI schema formatting fails irrecoverably."""

class AIRequestError(ExtractionError):
    """Raised when AI request fails persistently.""" 