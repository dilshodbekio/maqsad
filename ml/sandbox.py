"""sandbox_id validation shared by the photo module and the portfolio views."""
from __future__ import annotations

import re

from ml.errors import InvalidRequestError

MAX_LEN = 64
_OK = re.compile(r"[A-Za-z0-9_.:-]+")


def validate_sandbox_id(sandbox_id):
    """None (portfolio mode) or a clean id: str of 1..64 characters from [A-Za-z0-9_.:-]. Anything else raises
    InvalidRequestError (Uzbek message) - never a raw TypeError / KeyError."""
    if sandbox_id is None:
        return None
    if not isinstance(sandbox_id, str):
        raise InvalidRequestError("sandbox_id matn (str) bo'lishi kerak")
    if not sandbox_id:
        raise InvalidRequestError("sandbox_id bo'sh bo'lishi mumkin emas")
    if len(sandbox_id) > MAX_LEN:
        raise InvalidRequestError(f"sandbox_id {MAX_LEN} belgidan uzun bo'lishi mumkin emas (kiritildi: {len(sandbox_id)})")
    if not _OK.fullmatch(sandbox_id):
        raise InvalidRequestError("sandbox_id faqat lotin harflari, raqamlar va _ . : - belgilaridan iborat bo'lishi kerak")
    return sandbox_id
