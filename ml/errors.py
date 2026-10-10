"""Exceptions of the ML public interface (re-exported by ml.api)."""
from __future__ import annotations


class LoanNotFoundError(KeyError):
    """loan_id does not exist in the portfolio. Backend → HTTP 404."""


class InvalidImageError(ValueError):
    """Image is empty, corrupted, not an image, or too large. Backend → HTTP 400/413."""
