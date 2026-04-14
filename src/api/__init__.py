"""DocIQ REST API (FastAPI).

Convenience imports::

    from src.api import app
    from src.api.models import HealthResponse, ReadinessResponse, ProcessingResponse
"""

from src.api.models import (
    HealthResponse,
    ProcessingResponse,
    ReadinessResponse,
)

__all__ = [
    "HealthResponse",
    "ProcessingResponse",
    "ReadinessResponse",
]
