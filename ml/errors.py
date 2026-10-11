"""Exceptions of the ML public interface (re-exported by ml.api)."""
from __future__ import annotations


class LoanNotFoundError(KeyError):
    """loan_id does not exist in the portfolio. Backend → HTTP 404."""


class InvalidImageError(ValueError):
    """Image is empty, corrupted, not an image, or too large. Backend → HTTP 400/413."""


class _TranslatedError(ValueError):
    """str(e) is the Uzbek message; `e.messages` is {uz, en, ru} (the backend answers in the caller's language)."""

    def __init__(self, uz: str, en: str | None = None, ru: str | None = None):
        super().__init__(uz)
        self.messages = {"uz": uz, "en": en or uz, "ru": ru or uz}


class InvalidRequestError(_TranslatedError):
    """A request argument is invalid (e.g. a malformed sandbox_id); message in uz / en / ru. Backend -> HTTP 400."""


class ScenarioError(_TranslatedError):
    """The scenario given to score_scenario is invalid (message in uz / en / ru). Backend -> HTTP 400."""
