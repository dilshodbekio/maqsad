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
        raise InvalidRequestError("sandbox_id matn (str) bo'lishi kerak", "sandbox_id must be a string (str)", "sandbox_id должен быть строкой (str)")
    if not sandbox_id:
        raise InvalidRequestError("sandbox_id bo'sh bo'lishi mumkin emas", "sandbox_id must not be empty", "sandbox_id не должен быть пустым")
    if len(sandbox_id) > MAX_LEN:
        raise InvalidRequestError(f"sandbox_id {MAX_LEN} belgidan uzun bo'lishi mumkin emas (kiritildi: {len(sandbox_id)})",
                                  f"sandbox_id must not be longer than {MAX_LEN} characters (entered: {len(sandbox_id)})",
                                  f"sandbox_id не должен быть длиннее {MAX_LEN} символов (введено: {len(sandbox_id)})")
    if not _OK.fullmatch(sandbox_id):
        raise InvalidRequestError("sandbox_id faqat lotin harflari, raqamlar va _ . : - belgilaridan iborat bo'lishi kerak",
                                  "sandbox_id may only contain Latin letters, digits and the characters _ . : -",
                                  "sandbox_id может содержать только латинские буквы, цифры и символы _ . : -")
    return sandbox_id
